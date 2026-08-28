"""In-memory, same-policy replay for score rollouts.

The replay buffer is deliberately stricter than a cache.  A cache miss may be
recomputed; a replay miss during an optimizer step changes the experiment by
sampling a second response group.  Therefore every marked training request is
served exactly from the score payload or fails closed.

Only backend-native generation payloads (VERL ``DataProto`` slices in
production) are retained.  Decoded text is useful for audit logs but is not a
substitute for token ids, response masks, log probabilities, and reward scores
consumed by GRPO.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Callable, Sequence

from .backends import PolicyIdentity


REPLAY_SCHEMA = "score-rollout-replay-v1"


class ReplayContractError(RuntimeError):
    """A marked optimizer batch cannot be served without changing semantics."""


class ReplayBatchUnavailable(ReplayContractError):
    """There are not exactly enough eligible, resident groups for one update."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    """SHA-256 of a typed, canonical JSON value."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def prompt_hash(messages: Sequence[dict[str, str]]) -> str:
    return content_hash(list(messages))


def answer_hash(answer: str) -> str:
    # Hash the exact reward target.  Normalisation belongs in the verifier, not
    # in the replay identity: changing even whitespace here must be explicit.
    return content_hash({"answer": str(answer)})


def verifier_hash(verifier: dict[str, Any]) -> str:
    return content_hash({"verifier": verifier})


def policy_hash(identity: PolicyIdentity) -> str:
    return content_hash(asdict(identity))


def payload_row_count(payload: Any) -> int:
    """Best-effort row count for a VERL DataProto (and small test doubles).

    DataProto versions expose their size through slightly different paths.  We
    collect every usable report and require them to agree.  Disagreement is a
    contract error rather than choosing the largest value and risking a silent
    row swap.
    """

    if payload is None:
        return 0
    sizes: list[int] = []
    tensor_batch = getattr(payload, "batch", None)
    if tensor_batch is not None:
        try:
            sizes.append(int(tensor_batch.batch_size[0]))
        except (AttributeError, IndexError, TypeError, ValueError):
            try:
                sizes.append(int(len(tensor_batch)))
            except (TypeError, ValueError):
                pass
    non_tensor = getattr(payload, "non_tensor_batch", None)
    if isinstance(non_tensor, dict):
        for value in non_tensor.values():
            try:
                sizes.append(int(len(value)))
            except (TypeError, ValueError):
                continue
    # Lightweight test doubles may expose rows directly.
    if not sizes and hasattr(payload, "rows"):
        try:
            sizes.append(int(len(payload.rows)))
        except (TypeError, ValueError):
            pass
    positive = {size for size in sizes if size > 0}
    if not positive and any(size == 0 for size in sizes):
        return 0
    if not positive:
        raise ReplayContractError(
            f"cannot determine replay payload row count for {type(payload).__name__}"
        )
    if len(positive) != 1:
        raise ReplayContractError(
            f"replay payload reports inconsistent row counts: {sorted(positive)}"
        )
    return positive.pop()


def slice_score_payloads(
    output: Any,
    *,
    num_groups: int,
    group_size: int,
) -> list[Any]:
    """Split an instance-major generation output into exact score groups.

    VERL lays out ``repeat(interleave=True)`` output as ``[p0 x G, p1 x G,
    ...]``.  This helper intentionally raises for a partial group; partial
    denominator scoring and partial replay are both forbidden in v0.2.
    """

    num_groups = int(num_groups)
    group_size = int(group_size)
    if num_groups < 1 or group_size < 1:
        raise ValueError("num_groups and group_size must be positive")
    expected = num_groups * group_size
    actual = payload_row_count(output)
    if actual != expected:
        raise ReplayContractError(
            f"score output has {actual} rows, expected {num_groups} x "
            f"{group_size} = {expected}; discard and rescore the full request"
        )
    if not hasattr(output, "slice"):
        raise ReplayContractError(
            f"{type(output).__name__} cannot produce backend-native row slices"
        )
    groups: list[Any] = []
    for index in range(num_groups):
        start = index * group_size
        payload = output.slice(start, start + group_size)
        size = payload_row_count(payload)
        if size != group_size:
            raise ReplayContractError(
                f"score payload slice {index} has {size} rows, expected {group_size}"
            )
        groups.append(payload)
    return groups


@dataclass(frozen=True, slots=True)
class ReplayKey:
    """Everything that must remain identical between scoring and training."""

    score_observation_id: str
    problem_id: str
    score_iteration: int
    policy: PolicyIdentity
    prompt_sha256: str
    answer_sha256: str
    verifier_sha256: str
    group_size: int
    schema: str = REPLAY_SCHEMA

    @property
    def policy_sha256(self) -> str:
        return policy_hash(self.policy)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["policy_sha256"] = self.policy_sha256
        return value


@dataclass(slots=True)
class ScoreReplayGroup:
    """One complete score observation and its native rollout payload."""

    key: ReplayKey
    messages: tuple[dict[str, str], ...]
    answer: str
    verifier: dict[str, Any]
    payload: Any
    purpose: str = "score"

    def validate(self) -> None:
        if self.purpose != "score":
            raise ReplayContractError(
                "only score rollout payloads may enter replay; label/crossover "
                f"payload purpose was {self.purpose!r}"
            )
        if not self.key.score_observation_id.strip():
            raise ReplayContractError("score_observation_id must be non-empty")
        if not self.key.problem_id.strip():
            raise ReplayContractError("problem_id must be non-empty")
        if self.key.group_size < 1:
            raise ReplayContractError("group_size must be positive")
        if payload_row_count(self.payload) != self.key.group_size:
            raise ReplayContractError(
                "replay payload row count does not equal the fixed score group size"
            )
        if prompt_hash(self.messages) != self.key.prompt_sha256:
            raise ReplayContractError("stored prompt does not match prompt_sha256")
        if answer_hash(self.answer) != self.key.answer_sha256:
            raise ReplayContractError("stored answer does not match answer_sha256")
        if verifier_hash(self.verifier) != self.key.verifier_sha256:
            raise ReplayContractError("stored verifier does not match verifier_sha256")

    @classmethod
    def capture(
        cls,
        *,
        score_observation_id: str,
        problem_id: str,
        score_iteration: int,
        policy: PolicyIdentity,
        messages: Sequence[dict[str, str]],
        answer: str,
        verifier: dict[str, Any],
        payload: Any,
        group_size: int,
        purpose: str = "score",
    ) -> "ScoreReplayGroup":
        frozen_messages = tuple(dict(message) for message in messages)
        frozen_verifier = dict(verifier)
        group = cls(
            key=ReplayKey(
                score_observation_id=str(score_observation_id),
                problem_id=str(problem_id),
                score_iteration=int(score_iteration),
                policy=policy,
                prompt_sha256=prompt_hash(frozen_messages),
                answer_sha256=answer_hash(str(answer)),
                verifier_sha256=verifier_hash(frozen_verifier),
                group_size=int(group_size),
            ),
            messages=frozen_messages,
            answer=str(answer),
            verifier=frozen_verifier,
            payload=payload,
            purpose=str(purpose),
        )
        group.validate()
        return group

    @classmethod
    def from_generated(
        cls,
        *,
        problem: Any,
        score: Any,
        generated_group: Any,
        policy: PolicyIdentity,
        messages: Sequence[dict[str, str]],
        group_size: int,
        purpose: str = "score",
    ) -> "ScoreReplayGroup":
        """Capture from the engine's ProblemRecord/ScoreEvidence/backend group.

        The method uses structural attributes rather than importing model
        classes, so replay remains independent of archive serialization.  The
        score request id and generated request id must agree; otherwise an
        asynchronous result could be attached to the wrong problem.
        """

        observation_id = str(getattr(score, "observation_id", "") or "")
        score_request_id = str(getattr(score, "request_id", "") or "")
        generated_request_id = str(getattr(generated_group, "request_id", "") or "")
        if not observation_id:
            raise ReplayContractError(
                "ScoreEvidence.observation_id is required for replay capture"
            )
        if int(getattr(score, "num_rollouts", -1)) != int(group_size):
            raise ReplayContractError(
                "ScoreEvidence rollout count differs from replay group size"
            )
        if int(getattr(score, "policy_version", -1)) != policy.policy_version:
            raise ReplayContractError("ScoreEvidence policy version is stale")
        score_run_uuid = str(getattr(score, "policy_run_uuid", "") or "")
        if score_run_uuid and score_run_uuid != policy.run_uuid:
            raise ReplayContractError("ScoreEvidence run UUID is stale")
        score_adapter = int(getattr(score, "adapter_version", -1))
        if score_adapter >= 0 and score_adapter != policy.adapter_version:
            raise ReplayContractError("ScoreEvidence adapter version is stale")
        score_step = int(getattr(score, "global_step", -1))
        if score_step >= 0 and score_step != policy.global_step:
            raise ReplayContractError("ScoreEvidence global step is stale")
        nonempty_request_ids = {
            value for value in (score_request_id, generated_request_id) if value
        }
        if len(nonempty_request_ids) > 1:
            raise ReplayContractError(
                "ScoreEvidence and GeneratedGroup request ids do not match"
            )
        samples = list(getattr(generated_group, "samples", ()) or ())
        if len(samples) != int(group_size):
            raise ReplayContractError(
                f"generated score group has {len(samples)} samples; "
                f"expected {group_size}"
            )
        if any(
            getattr(sample, "status", "accepted") != "accepted" for sample in samples
        ):
            raise ReplayContractError(
                "a rejected/partial score group cannot enter replay; rescore it"
            )
        payload = getattr(generated_group, "payload", None)
        tensor_batch = getattr(payload, "batch", None)
        if tensor_batch is not None and hasattr(tensor_batch, "get"):
            rm_scores = tensor_batch.get("rm_scores")
            if rm_scores is None:
                raise ReplayContractError(
                    "native score payload has no rm_scores; replay would train "
                    "on a different or missing reward"
                )
            reward_flags: list[bool] = []
            try:
                for row in rm_scores:
                    value = row.sum().item() if hasattr(row, "sum") else row
                    reward_flags.append(float(value) > 0.5)
            except (TypeError, ValueError, AttributeError) as exc:
                raise ReplayContractError(
                    "cannot audit score payload rm_scores"
                ) from exc
            score_rollouts = list(getattr(score, "rollouts", ()) or ())
            score_flags = [
                bool(getattr(row, "correct", False)) for row in score_rollouts
            ]
            if len(reward_flags) != int(group_size) or reward_flags != score_flags:
                raise ReplayContractError(
                    "score payload rm_scores disagree with ScoreEvidence grading"
                )
        return cls.capture(
            score_observation_id=observation_id,
            problem_id=str(getattr(problem, "problem_id")),
            score_iteration=int(getattr(score, "iteration")),
            policy=policy,
            messages=messages,
            answer=str(getattr(problem, "pseudo_gold")),
            verifier=dict(getattr(problem, "verifier")),
            payload=payload,
            group_size=int(group_size),
            purpose=purpose,
        )

    def replay_metadata(self) -> dict[str, Any]:
        """Metadata copied into each dataset row and repeated by VERL."""

        return {
            "replay_required": True,
            "replay_schema": self.key.schema,
            "score_observation_id": self.key.score_observation_id,
            "problem_id": self.key.problem_id,
            "score_iteration": self.key.score_iteration,
            "policy_sha256": self.key.policy_sha256,
            "prompt_sha256": self.key.prompt_sha256,
            "answer_sha256": self.key.answer_sha256,
            "verifier_sha256": self.key.verifier_sha256,
            "replay_group_size": self.key.group_size,
        }


@dataclass(slots=True)
class ReplayStats:
    captured_groups: int = 0
    captured_rows: int = 0
    served_batches: int = 0
    served_groups: int = 0
    served_rows: int = 0
    passthrough_calls: int = 0
    failed_calls: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class ScoreReplayBuffer:
    """Ephemeral score payloads valid for one policy version and one update."""

    expected_group_size: int
    expected_training_groups: int = 32
    iteration: int = -1
    policy: PolicyIdentity | None = None
    groups: dict[str, ScoreReplayGroup] = field(default_factory=dict)
    consumed: bool = False
    authorized_batch_id: str | None = None
    authorized_observation_ids: frozenset[str] = field(default_factory=frozenset)
    stats: ReplayStats = field(default_factory=ReplayStats)

    def begin_cycle(self, *, iteration: int, policy: PolicyIdentity) -> None:
        self.groups.clear()
        self.iteration = int(iteration)
        self.policy = policy
        self.consumed = False
        self.authorized_batch_id = None
        self.authorized_observation_ids = frozenset()
        self.stats = ReplayStats()

    def _require_open(self) -> PolicyIdentity:
        if self.policy is None or self.iteration < 0:
            raise ReplayContractError(
                "begin_cycle must be called before replay capture"
            )
        if self.consumed:
            raise ReplayContractError("replay batch was already consumed by an update")
        return self.policy

    def store(self, group: ScoreReplayGroup) -> None:
        policy = self._require_open()
        group.validate()
        key = group.key
        if key.policy != policy:
            raise ReplayContractError(
                "score payload policy does not match the current replay cycle"
            )
        if key.score_iteration != self.iteration:
            raise ReplayContractError(
                f"score iteration {key.score_iteration} != replay cycle {self.iteration}"
            )
        if key.group_size != self.expected_group_size:
            raise ReplayContractError(
                f"score group size {key.group_size} != configured "
                f"{self.expected_group_size}"
            )
        observation_id = key.score_observation_id
        if observation_id in self.groups:
            raise ReplayContractError(
                f"duplicate score_observation_id in replay: {observation_id}"
            )
        self.groups[observation_id] = group
        self.stats.captured_groups += 1
        self.stats.captured_rows += key.group_size

    def get(self, score_observation_id: str) -> ScoreReplayGroup:
        self._require_open()
        try:
            return self.groups[str(score_observation_id)]
        except KeyError as exc:
            raise ReplayContractError(
                f"score observation is not resident: {score_observation_id}"
            ) from exc

    def exact_groups(
        self,
        observation_ids: Sequence[str],
        *,
        expected_count: int | None = None,
    ) -> tuple[ScoreReplayGroup, ...]:
        """Resolve one no-padding training batch and validate uniqueness."""

        self._require_open()
        expected = (
            self.expected_training_groups
            if expected_count is None
            else int(expected_count)
        )
        ids = tuple(str(value) for value in observation_ids)
        if len(ids) != expected:
            raise ReplayBatchUnavailable(
                f"optimizer update needs exactly {expected} score groups; got {len(ids)}"
            )
        if len(set(ids)) != expected:
            raise ReplayBatchUnavailable(
                "optimizer update cannot repeat a score observation to pad the batch"
            )
        groups = tuple(self.get(value) for value in ids)
        problem_ids = [group.key.problem_id for group in groups]
        if len(set(problem_ids)) != expected:
            raise ReplayBatchUnavailable(
                "optimizer update needs distinct concrete problems; duplicate problem_id"
            )
        return groups

    def mark_consumed(self) -> None:
        self._require_open()
        self.consumed = True

    def authorize_batch(self, *, batch_id: str, observation_ids: Sequence[str]) -> None:
        """Bind the hook to the one selection decision this cycle permits."""

        self._require_open()
        ids = tuple(str(value) for value in observation_ids)
        if not str(batch_id).strip():
            raise ReplayContractError("authorized replay batch id must be non-empty")
        if len(ids) != self.expected_training_groups or len(set(ids)) != len(ids):
            raise ReplayContractError(
                "authorized replay batch must contain the exact unique group count"
            )
        if self.authorized_batch_id is not None:
            if self.authorized_batch_id != str(
                batch_id
            ) or self.authorized_observation_ids != frozenset(ids):
                raise ReplayContractError(
                    "a different training batch is already authorized this cycle"
                )
            return
        self.authorized_batch_id = str(batch_id)
        self.authorized_observation_ids = frozenset(ids)

    def discard(self) -> None:
        """Erase native payloads; required after update and before policy change."""

        self.groups.clear()
        self.policy = None
        self.iteration = -1
        self.consumed = True
        self.authorized_batch_id = None
        self.authorized_observation_ids = frozenset()


def _metadata_dict(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


class FailClosedReplayHook:
    """Serve marked VERL training calls exclusively from score payloads.

    Unmarked calls (crossover, label, score generation, validation) pass through
    to the original rollout manager.  A call containing *any* replay marker is
    an optimizer call and may never fall through on mismatch.
    """

    def __init__(
        self,
        buffer: ScoreReplayBuffer,
        *,
        group_size: int,
        training_groups: int = 32,
        concat_fn: Callable[[list[Any]], Any] | None = None,
    ) -> None:
        self.buffer = buffer
        self.group_size = int(group_size)
        self.training_groups = int(training_groups)
        self.concat_fn = concat_fn
        self._manager: Any = None
        self._original: Callable[..., Any] | None = None

    def install(self, manager: Any) -> None:
        if manager is None or not hasattr(manager, "generate_sequences"):
            raise ReplayContractError(
                "VERL trainer has no async rollout manager to install replay on"
            )
        if self._manager is manager:
            return
        if self._manager is not None:
            raise ReplayContractError("replay hook is already installed elsewhere")
        self._manager = manager
        self._original = manager.generate_sequences

        def generate_sequences(gen_batch: Any, *args: Any, **kwargs: Any) -> Any:
            if not self._marked_for_replay(gen_batch):
                self.buffer.stats.passthrough_calls += 1
                assert self._original is not None
                return self._original(gen_batch, *args, **kwargs)
            try:
                return self.serve(gen_batch)
            except Exception:
                self.buffer.stats.failed_calls += 1
                raise

        manager.generate_sequences = generate_sequences

    def uninstall(self) -> None:
        if self._manager is not None and self._original is not None:
            self._manager.generate_sequences = self._original
        self._manager = None
        self._original = None

    @staticmethod
    def _non_tensor(batch: Any) -> dict[str, Any]:
        value = getattr(batch, "non_tensor_batch", None)
        return value if isinstance(value, dict) else {}

    def _marked_for_replay(self, batch: Any) -> bool:
        extras = self._non_tensor(batch).get("extra_info")
        if extras is None:
            return False
        markers = [
            bool(value.get("replay_required"))
            for value in extras
            if isinstance(value, dict)
        ]
        return any(markers)

    def _concat(self, payloads: list[Any]) -> Any:
        if self.concat_fn is not None:
            return self.concat_fn(payloads)
        try:
            from verl.protocol import DataProto  # type: ignore

            return DataProto.concat(payloads)
        except ImportError:
            cls = type(payloads[0])
            concat = getattr(cls, "concat", None)
            if concat is None:
                raise ReplayContractError(
                    "VERL is unavailable and replay payload type has no concat"
                )
            return concat(payloads)

    def serve(self, gen_batch: Any) -> Any:
        policy = self.buffer._require_open()
        non_tensor = self._non_tensor(gen_batch)
        extras = non_tensor.get("extra_info")
        prompts = non_tensor.get("raw_prompt")
        rewards = non_tensor.get("reward_model")
        if extras is None or prompts is None or rewards is None:
            raise ReplayContractError(
                "marked training batch needs extra_info, raw_prompt, and reward_model"
            )
        total = payload_row_count(gen_batch)
        expected_rows = self.training_groups * self.group_size
        if total != expected_rows:
            raise ReplayContractError(
                f"marked training call has {total} rows; expected exactly "
                f"{self.training_groups} x {self.group_size} = {expected_rows}"
            )
        if not (len(extras) == len(prompts) == len(rewards) == total):
            raise ReplayContractError("marked training metadata arrays are misaligned")
        markers = [
            bool(value.get("replay_required")) if isinstance(value, dict) else False
            for value in extras
        ]
        if not all(markers):
            raise ReplayContractError(
                "mixed replay-marked and unmarked rows in one optimizer call"
            )

        payloads: list[Any] = []
        seen_observations: set[str] = set()
        seen_problems: set[str] = set()
        for start in range(0, total, self.group_size):
            group_extras = extras[start : start + self.group_size]
            group_prompts = prompts[start : start + self.group_size]
            group_rewards = rewards[start : start + self.group_size]
            first = _metadata_dict(group_extras[0])
            if first is None:
                raise ReplayContractError("replay extra_info row must be a mapping")
            batch_id = str(first.get("batch_id", ""))
            if (
                self.buffer.authorized_batch_id is None
                or batch_id != self.buffer.authorized_batch_id
            ):
                raise ReplayContractError(
                    "optimizer call is not the replay batch authorized by selection"
                )
            if any(value != first for value in group_extras[1:]):
                raise ReplayContractError(
                    "repeated rollout rows have different metadata"
                )
            observation_id = str(first.get("score_observation_id", ""))
            group = self.buffer.get(observation_id)
            key = group.key
            if observation_id in seen_observations:
                raise ReplayContractError(
                    "duplicate score observation in optimizer call"
                )
            if key.problem_id in seen_problems:
                raise ReplayContractError(
                    "duplicate concrete problem in optimizer call"
                )
            seen_observations.add(observation_id)
            seen_problems.add(key.problem_id)

            expected_meta = group.replay_metadata()
            for name, expected in expected_meta.items():
                if first.get(name) != expected:
                    raise ReplayContractError(
                        f"replay metadata mismatch for {observation_id}: {name}"
                    )
            if key.policy != policy:
                raise ReplayContractError("stored replay group is from another policy")
            if any(value != group_prompts[0] for value in group_prompts[1:]):
                raise ReplayContractError(
                    "repeated rollout rows have different prompts"
                )
            if prompt_hash(group_prompts[0]) != key.prompt_sha256:
                raise ReplayContractError(
                    f"training prompt differs from score prompt: {observation_id}"
                )
            for reward in group_rewards:
                if not isinstance(reward, dict):
                    raise ReplayContractError("reward_model row must be a mapping")
                truth = str(reward.get("ground_truth", ""))
                verifier = reward.get("verifier")
                if answer_hash(truth) != key.answer_sha256:
                    raise ReplayContractError(
                        f"training ground truth differs from score target: {observation_id}"
                    )
                if (
                    not isinstance(verifier, dict)
                    or verifier_hash(verifier) != key.verifier_sha256
                ):
                    raise ReplayContractError(
                        f"training verifier differs from score verifier: {observation_id}"
                    )
            if payload_row_count(group.payload) != self.group_size:
                raise ReplayContractError("resident replay payload size changed")
            payloads.append(group.payload)

        if len(payloads) != self.training_groups:
            raise ReplayContractError(
                "optimizer call did not resolve every replay group"
            )
        if seen_observations != set(self.buffer.authorized_observation_ids):
            raise ReplayContractError(
                "optimizer call observation set differs from the authorized batch"
            )
        served = self._concat(payloads)
        if payload_row_count(served) != expected_rows:
            raise ReplayContractError("concatenated replay payload has wrong row count")
        served.meta_info = dict(getattr(gen_batch, "meta_info", {}) or {})
        served.meta_info.setdefault("timing", {})
        self.buffer.stats.served_batches += 1
        self.buffer.stats.served_groups += self.training_groups
        self.buffer.stats.served_rows += expected_rows
        return served
