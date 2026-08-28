"""Current-policy R_Q measurement and optional native-payload capture."""

from __future__ import annotations

from dataclasses import dataclass, field

from .archive import ConcreteMapArchive
from .backends import (
    GeneratedGroup,
    PolicyBackend,
    PolicyIdentity,
    SamplingSpec,
    merge_generated_groups,
    select_generated_samples,
)
from .config import ScoringConfig
from .grading import GraderClient
from .models import CandidateEvent, ProblemRecord, ScoreEvidence
from .prompts import PromptBook
from .replay import ScoreReplayGroup
from .scoring import score_group
from .storage import RunStore
from .utils import stable_id


@dataclass(slots=True)
class ScoreRunResult:
    scores: dict[str, ScoreEvidence] = field(default_factory=dict)
    replay_groups: dict[str, ScoreReplayGroup] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)


class ScoreRunner:
    def __init__(
        self,
        *,
        config: ScoringConfig,
        backend: PolicyBackend,
        prompts: PromptBook,
        grader: GraderClient,
        store: RunStore,
        archive: ConcreteMapArchive,
    ) -> None:
        self.config = config
        self.backend = backend
        self.prompts = prompts
        self.grader = grader
        self.store = store
        self.archive = archive

    def _event(
        self,
        *,
        iteration: int,
        problem_id: str,
        status: str,
        reason: str | None,
        details: dict | None = None,
    ) -> None:
        self.store.append_event(
            CandidateEvent(
                event_id=stable_id(
                    "event",
                    iteration,
                    "score",
                    problem_id,
                    status,
                    reason,
                    (details or {}).get("attempt", -1),
                    (details or {}).get("request_id", ""),
                ),
                iteration=iteration,
                candidate_id=None,
                phase="score",
                status=status,
                reason=reason,
                details={"problem_id": problem_id, **(details or {})},
            )
        )

    def run(
        self,
        records: list[ProblemRecord],
        *,
        iteration: int,
        frozen: PolicyIdentity,
        replay_problem_ids: set[str] | None = None,
    ) -> ScoreRunResult:
        replay_problem_ids = set(replay_problem_ids or ())
        if len({record.problem_id for record in records}) != len(records):
            raise ValueError("score runner received duplicate problem IDs")
        pending = list(records)
        fragments: dict[str, list[GeneratedGroup]] = {
            record.problem_id: [] for record in records
        }
        result = ScoreRunResult()
        for attempt in range(self.config.max_infrastructure_retries + 1):
            if not pending:
                break
            retry: list[ProblemRecord] = []
            by_needed: dict[int, list[ProblemRecord]] = {}
            for record in pending:
                collected = sum(
                    len(group.samples) for group in fragments[record.problem_id]
                )
                by_needed.setdefault(self.config.num_rollouts - collected, []).append(
                    record
                )
            for needed, bucket in sorted(by_needed.items()):
                messages = [
                    self.prompts.solver_messages(record.question) for record in bucket
                ]
                request_ids = [
                    stable_id(
                        "score_refill",
                        frozen.run_uuid,
                        frozen.policy_version,
                        iteration,
                        record.problem_id,
                        attempt,
                        needed,
                    )
                    for record in bucket
                ]
                groups = self.backend.generate(
                    messages,
                    request_ids=request_ids,
                    sampling=SamplingSpec(
                        n=needed,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        max_tokens=self.config.max_tokens,
                    ),
                    purpose="score",
                    ground_truths=[record.pseudo_gold for record in bucket],
                    verifiers=[record.verifier for record in bucket],
                )
                if self.backend.policy_identity != frozen:
                    raise RuntimeError(
                        "policy changed while score rollouts were in flight"
                    )
                by_id = {group.request_id: group for group in groups}
                if len(by_id) != len(groups):
                    raise RuntimeError("score backend returned duplicate request IDs")
                for record, messages_row, request_id in zip(
                    bucket, messages, request_ids, strict=True
                ):
                    raw = by_id.get(
                        request_id,
                        GeneratedGroup(request_id=request_id, samples=[]),
                    )
                    good_indices: list[int] = []
                    for index, sample in enumerate(raw.samples):
                        if sample.status != "accepted":
                            continue
                        if self.config.require_entropy:
                            try:
                                entropy = float(sample.entropy)  # type: ignore[arg-type]
                            except (TypeError, ValueError):
                                continue
                            if not 0.0 <= entropy <= 1.0:
                                continue
                        good_indices.append(index)
                        if len(good_indices) == needed:
                            break
                    if good_indices:
                        fragments[record.problem_id].append(
                            select_generated_samples(raw, good_indices)
                        )
                    assembled_id = stable_id(
                        "score_assembled",
                        frozen.run_uuid,
                        frozen.policy_version,
                        iteration,
                        record.problem_id,
                    )
                    group = merge_generated_groups(
                        assembled_id, fragments[record.problem_id]
                    )
                    if (
                        len(group.samples) < self.config.num_rollouts
                        and attempt < self.config.max_infrastructure_retries
                    ):
                        self._event(
                            iteration=iteration,
                            problem_id=record.problem_id,
                            status="retrying",
                            reason="incomplete_score_group",
                            details={
                                "attempt": attempt,
                                "request_id": raw.request_id,
                                "requested_in_attempt": needed,
                                "collected_total": len(group.samples),
                                "remaining": self.config.num_rollouts
                                - len(group.samples),
                                "received": len(raw.samples),
                                "sample_statuses": [
                                    sample.status for sample in raw.samples
                                ],
                            },
                        )
                        retry.append(record)
                        continue
                    score, reason = score_group(
                        group,
                        requested_rollouts=self.config.num_rollouts,
                        gold=record.pseudo_gold,
                        verifier=record.verifier,
                        grader=self.grader,
                        identity=frozen,
                        iteration=iteration,
                        require_entropy=self.config.require_entropy,
                        sampling_contract={
                            "n": self.config.num_rollouts,
                            "temperature": self.config.temperature,
                            "top_p": self.config.top_p,
                            "max_tokens": self.config.max_tokens,
                        },
                    )
                    if score is None:
                        final_reason = reason or "invalid_score_group"
                        result.failures[record.problem_id] = final_reason
                        self._event(
                            iteration=iteration,
                            problem_id=record.problem_id,
                            status="rejected",
                            reason=final_reason,
                            details={
                                "attempt": attempt,
                                "request_id": group.request_id,
                            },
                        )
                        continue
                    score.observation_id = stable_id(
                        "score_observation",
                        frozen.run_uuid,
                        frozen.policy_version,
                        iteration,
                        record.problem_id,
                    )
                    score.request_id = group.request_id
                    self.archive.apply_score(record.problem_id, score)
                    self.store.append_score(record.problem_id, score)
                    result.scores[record.problem_id] = score
                    if record.problem_id in replay_problem_ids:
                        result.replay_groups[record.problem_id] = (
                            ScoreReplayGroup.from_generated(
                                problem=record,
                                score=score,
                                generated_group=group,
                                policy=frozen,
                                messages=messages_row,
                                group_size=self.config.num_rollouts,
                                purpose="score",
                            )
                        )
                    self._event(
                        iteration=iteration,
                        problem_id=record.problem_id,
                        status="accepted",
                        reason=None,
                        details={
                            "score_observation_id": score.observation_id,
                            "s_hat": score.s_hat,
                            "rq_score": score.rq_score,
                            "attempt": attempt,
                            "refill_requests": attempt + 1,
                        },
                    )
            pending = retry
        return result
