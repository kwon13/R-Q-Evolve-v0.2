"""Model backend contracts; generation payloads remain opaque but replayable."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class SamplingSpec:
    n: int
    temperature: float
    top_p: float
    max_tokens: int
    top_k: int | None = None


@dataclass(slots=True)
class GeneratedSample:
    text: str
    # Normalized token entropy in [0, 1].  None means the backend did not
    # provide the quantity; a real zero is a valid deterministic rollout.
    entropy: float | None = None
    status: str = "accepted"
    reject_reason: str | None = None
    payload_row: Any = None


@dataclass(slots=True)
class GeneratedGroup:
    request_id: str
    samples: list[GeneratedSample]
    payload: Any = None
    prompt_fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class PolicyIdentity:
    run_uuid: str
    policy_version: int
    adapter_version: int
    global_step: int
    source_checkpoint: str


class PolicyBackend(Protocol):
    @property
    def policy_identity(self) -> PolicyIdentity: ...

    def generate(
        self,
        messages: Sequence[list[dict[str, str]]],
        *,
        request_ids: Sequence[str],
        sampling: SamplingSpec,
        purpose: str,
        ground_truths: Sequence[str] | None = None,
        verifiers: Sequence[dict] | None = None,
    ) -> list[GeneratedGroup]: ...

class TrainingBackend(Protocol):
    def apply_replay_batch(self, batch: "ReplayTrainingBatch") -> dict[str, Any]: ...

    def save_checkpoint(
        self, *, global_step: int, pipeline_state: dict[str, Any]
    ) -> str: ...


# Forward reference for static checkers without importing training.py here.
class ReplayTrainingBatch(Protocol):
    batch_id: str
