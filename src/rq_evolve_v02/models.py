"""Serializable records crossing the generation, archive, and training boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .concepts import cell_key


@dataclass(slots=True)
class ParentPair:
    pair_id: str
    left_id: str
    right_id: str
    prompt_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ParentPair":
        return cls(**value)


@dataclass(slots=True)
class Candidate:
    candidate_id: str
    question: str
    domain: str
    proposed_answer: str
    parent_ids: tuple[str, str]
    pair_id: str
    iteration: int
    child_index: int
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["parent_ids"] = list(self.parent_ids)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Candidate":
        payload = dict(value)
        payload["parent_ids"] = tuple(payload["parent_ids"])
        return cls(**payload)


@dataclass(slots=True)
class RolloutSample:
    response: str
    predicted_answer: str | None
    correct: bool | None = None
    entropy: float = 0.0
    status: str = "accepted"
    reject_reason: str | None = None
    policy_version: int = -1
    adapter_version: int = -1
    global_step: int = -1
    source_checkpoint: str = ""
    sample_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RolloutSample":
        return cls(**value)


@dataclass(slots=True)
class LabelEvidence:
    pseudo_gold: str | None
    cluster_sizes: list[int]
    agreement: float
    proposed_matches: bool
    accepted: bool
    reason: str | None
    rollouts: list[RolloutSample] = field(default_factory=list)
    request_id: str = ""
    policy_run_uuid: str = ""
    policy_version: int = -1
    adapter_version: int = -1
    global_step: int = -1
    source_checkpoint: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LabelEvidence":
        payload = dict(value)
        payload["rollouts"] = [
            RolloutSample.from_dict(x) for x in payload.get("rollouts", [])
        ]
        return cls(**payload)


@dataclass(slots=True)
class ScoreEvidence:
    iteration: int
    policy_version: int
    s_hat: float
    learnability: float
    u_score: float
    rq_score: float
    num_rollouts: int
    num_correct: int
    policy_run_uuid: str = ""
    adapter_version: int = -1
    global_step: int = -1
    source_checkpoint: str = ""
    sampling_contract: dict[str, Any] = field(default_factory=dict)
    rollouts: list[RolloutSample] = field(default_factory=list)
    observation_id: str = ""
    request_id: str = ""

    def to_dict(self, *, include_rollouts: bool = True) -> dict[str, Any]:
        value = asdict(self)
        if not include_rollouts:
            value.pop("rollouts", None)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ScoreEvidence":
        payload = dict(value)
        payload["rollouts"] = [
            RolloutSample.from_dict(x) for x in payload.get("rollouts", [])
        ]
        return cls(**payload)


@dataclass(slots=True)
class ProblemRecord:
    problem_id: str
    question: str
    proposed_answer: str
    pseudo_gold: str
    verifier: dict[str, Any]
    domain: str
    problem_type: str
    parent_ids: tuple[str, ...]
    lineage_root_ids: tuple[str, ...]
    generation: int
    created_iteration: int
    created_policy_version: int
    label_evidence: dict[str, Any]
    domain_evidence: dict[str, Any]
    score_history: list[ScoreEvidence] = field(default_factory=list)
    novelty: dict[str, Any] = field(default_factory=dict)
    source: str = "crossover"

    @property
    def cell(self) -> str:
        return cell_key(self.domain, self.problem_type)

    @property
    def latest_score(self) -> ScoreEvidence | None:
        return self.score_history[-1] if self.score_history else None

    def to_dict(self, *, include_rollouts: bool = False) -> dict[str, Any]:
        value = asdict(self)
        value["parent_ids"] = list(self.parent_ids)
        value["lineage_root_ids"] = list(self.lineage_root_ids)
        value["cell"] = self.cell
        value["score_history"] = [
            score.to_dict(include_rollouts=include_rollouts)
            for score in self.score_history
        ]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProblemRecord":
        payload = dict(value)
        payload.pop("cell", None)
        payload["parent_ids"] = tuple(payload.get("parent_ids", ()))
        payload["lineage_root_ids"] = tuple(payload.get("lineage_root_ids", ()))
        payload["score_history"] = [
            ScoreEvidence.from_dict(x) for x in payload.get("score_history", [])
        ]
        return cls(**payload)


@dataclass(slots=True)
class CandidateEvent:
    event_id: str
    iteration: int
    candidate_id: str | None
    phase: str
    status: str
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainingExample:
    problem_id: str
    problem: str
    answer: str
    verifier: dict[str, Any]
    domain: str
    problem_type: str
    rq_score: float
    s_hat: float
    score_iteration: int
    selection_score_iteration: int
    policy_version: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
