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


def select_generated_samples(
    group: GeneratedGroup,
    indices: Sequence[int],
    *,
    request_id: str | None = None,
) -> GeneratedGroup:
    """Select matching decoded and native payload rows from one generation.

    Score rows are later replayed by VERL, so retry code must never concatenate
    decoded text while silently retaining a different native payload.
    """

    chosen = [int(index) for index in indices]
    if any(index < 0 or index >= len(group.samples) for index in chosen):
        raise IndexError("generated sample index is out of range")
    payload = group.payload
    selected_payload = None
    if payload is not None:
        if not hasattr(payload, "slice"):
            raise TypeError(
                f"{type(payload).__name__} cannot slice native generation rows"
            )
        pieces = [payload.slice(index, index + 1) for index in chosen]
        if pieces:
            concat = getattr(type(payload), "concat", None)
            if concat is None:
                raise TypeError(
                    f"{type(payload).__name__} cannot concatenate native rows"
                )
            selected_payload = concat(pieces)
    return GeneratedGroup(
        request_id=str(request_id or group.request_id),
        samples=[group.samples[index] for index in chosen],
        payload=selected_payload,
        prompt_fingerprint=group.prompt_fingerprint,
    )


def merge_generated_groups(
    request_id: str,
    groups: Sequence[GeneratedGroup],
) -> GeneratedGroup:
    """Merge refill fragments while preserving one-to-one native row order."""

    fragments = [group for group in groups if group.samples]
    samples = [sample for group in fragments for sample in group.samples]
    payloads = [group.payload for group in fragments if group.payload is not None]
    if payloads and len(payloads) != len(fragments):
        raise TypeError("cannot merge a mixture of native and text-only fragments")
    payload = None
    if payloads:
        concat = getattr(type(payloads[0]), "concat", None)
        if concat is None or any(type(row) is not type(payloads[0]) for row in payloads):
            raise TypeError("native generation fragments cannot be concatenated")
        payload = concat(payloads)
    fingerprints = {
        group.prompt_fingerprint for group in fragments if group.prompt_fingerprint
    }
    if len(fingerprints) > 1:
        raise ValueError("cannot merge generation fragments from different prompts")
    return GeneratedGroup(
        request_id=str(request_id),
        samples=samples,
        payload=payload,
        prompt_fingerprint=next(iter(fingerprints), ""),
    )


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
