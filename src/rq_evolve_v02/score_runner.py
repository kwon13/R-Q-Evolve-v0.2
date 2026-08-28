"""Current-policy R_Q measurement and optional native-payload capture."""

from __future__ import annotations

from dataclasses import dataclass, field

from .archive import ConcreteMapArchive
from .backends import GeneratedGroup, PolicyBackend, PolicyIdentity, SamplingSpec
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
        result = ScoreRunResult()
        for attempt in range(self.config.max_infrastructure_retries + 1):
            if not pending:
                break
            messages = [
                self.prompts.solver_messages(record.question) for record in pending
            ]
            request_ids = [
                stable_id(
                    "score_request",
                    frozen.run_uuid,
                    frozen.policy_version,
                    iteration,
                    record.problem_id,
                    attempt,
                )
                for record in pending
            ]
            groups = self.backend.generate(
                messages,
                request_ids=request_ids,
                sampling=SamplingSpec(
                    n=self.config.num_rollouts,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_tokens,
                ),
                purpose="score",
                ground_truths=[record.pseudo_gold for record in pending],
                verifiers=[record.verifier for record in pending],
            )
            if self.backend.policy_identity != frozen:
                raise RuntimeError("policy changed while score rollouts were in flight")
            by_id = {group.request_id: group for group in groups}
            if len(by_id) != len(groups):
                raise RuntimeError("score backend returned duplicate request IDs")
            retry: list[ProblemRecord] = []
            for record, messages_row, request_id in zip(
                pending, messages, request_ids, strict=True
            ):
                group = by_id.get(
                    request_id, GeneratedGroup(request_id=request_id, samples=[])
                )
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
                    if attempt < self.config.max_infrastructure_retries:
                        self._event(
                            iteration=iteration,
                            problem_id=record.problem_id,
                            status="retrying",
                            reason=reason or "invalid_score_group",
                            details={
                                "attempt": attempt,
                                "request_id": request_id,
                                "received": len(group.samples),
                                "sample_statuses": [
                                    sample.status for sample in group.samples
                                ],
                            },
                        )
                        retry.append(record)
                        continue
                    final_reason = reason or "invalid_score_group"
                    result.failures[record.problem_id] = final_reason
                    self._event(
                        iteration=iteration,
                        problem_id=record.problem_id,
                        status="rejected",
                        reason=final_reason,
                        details={"attempt": attempt, "request_id": request_id},
                    )
                    continue
                score.observation_id = stable_id(
                    "score_observation",
                    frozen.run_uuid,
                    frozen.policy_version,
                    iteration,
                    record.problem_id,
                )
                score.request_id = request_id
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
                    },
                )
            pending = retry
        return result
