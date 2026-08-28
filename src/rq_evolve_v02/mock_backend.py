"""Deterministic CPU-only backend for pipeline and replay tests.

The mock deliberately mirrors both runtime boundaries instead of mocking
individual call sites.  Generation is a pure function of the request id,
purpose, sampling index, and optional scripts.  Each group carries an opaque
payload whose rows correspond one-for-one with its samples, allowing tests to
assert that training consumes the exact score rollout object.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .backends import (
    GeneratedGroup,
    GeneratedSample,
    PolicyIdentity,
    SamplingSpec,
)
from .utils import canonical_text, stable_id, stable_json


@dataclass(frozen=True, slots=True)
class MockPayloadRow:
    """One immutable generated row inside an otherwise opaque payload."""

    request_id: str
    purpose: str
    sample_index: int
    text: str
    entropy: float | None
    ground_truth: str | None
    verifier_json: str | None


@dataclass(frozen=True, slots=True)
class MockPayload:
    """Group payload used to test exact, non-reconstructed replay."""

    request_id: str
    purpose: str
    policy_identity: PolicyIdentity
    prompt_fingerprint: str
    rows: tuple[MockPayloadRow, ...]

    def __len__(self) -> int:
        return len(self.rows)

    def slice(self, start: int, stop: int) -> "MockPayload":
        return MockPayload(
            request_id=self.request_id,
            purpose=self.purpose,
            policy_identity=self.policy_identity,
            prompt_fingerprint=self.prompt_fingerprint,
            rows=self.rows[start:stop],
        )

    @classmethod
    def concat(cls, payloads: Sequence["MockPayload"]) -> "MockPayload":
        if not payloads:
            raise ValueError("cannot concatenate an empty payload list")
        first = payloads[0]
        if any(
            payload.purpose != first.purpose
            or payload.policy_identity != first.policy_identity
            or payload.prompt_fingerprint != first.prompt_fingerprint
            for payload in payloads
        ):
            raise ValueError("mock payload fragments have incompatible contracts")
        return cls(
            request_id=first.request_id,
            purpose=first.purpose,
            policy_identity=first.policy_identity,
            prompt_fingerprint=first.prompt_fingerprint,
            rows=tuple(row for payload in payloads for row in payload.rows),
        )


def _digest_int(*parts: object) -> int:
    data = stable_json(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def _last_user_text(messages: Sequence[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


class DeterministicMockBackend:
    """Combined policy/training backend that never touches a GPU.

    ``scripts`` maps either ``request_id`` or ``(purpose, request_id)`` to a
    sequence of strings or :class:`GeneratedSample` instances.  Short scripts
    repeat cyclically so a test only needs to specify the semantic outcomes it
    cares about.  Unscripted crossover requests produce small arithmetic
    problems; other generation requests return a deterministic boxed answer.
    """

    def __init__(
        self,
        *,
        identity: PolicyIdentity | None = None,
        scripts: Mapping[object, Sequence[str | GeneratedSample]] | None = None,
        answers: Mapping[str, str] | None = None,
    ) -> None:
        self._identity = identity or PolicyIdentity(
            run_uuid="mock-run",
            policy_version=0,
            adapter_version=0,
            global_step=0,
            source_checkpoint="mock://initial",
        )
        self.scripts = dict(scripts or {})
        self.answers = {
            canonical_text(question): str(answer)
            for question, answer in (answers or {}).items()
        }
        self.generation_calls: list[dict[str, Any]] = []
        self.applied_batches: list[Any] = []
        self.saved_checkpoints: list[tuple[int, dict[str, Any]]] = []

    @property
    def policy_identity(self) -> PolicyIdentity:
        return self._identity

    def _script_for(
        self, purpose: str, request_id: str
    ) -> Sequence[str | GeneratedSample] | None:
        return self.scripts.get((purpose, request_id)) or self.scripts.get(request_id)

    def _default_answer(
        self,
        messages: Sequence[dict[str, str]],
        *,
        request_id: str,
        purpose: str,
        sample_index: int,
        ground_truth: str | None,
    ) -> str:
        question = canonical_text(_last_user_text(messages))
        known = self.answers.get(question)
        answer = str(ground_truth) if ground_truth is not None else known
        if answer is None:
            arithmetic = re.search(
                r"what is\s+(\d+)\s*\+\s*(\d+)\s*\?",
                question,
                flags=re.IGNORECASE,
            )
            if arithmetic:
                answer = str(int(arithmetic.group(1)) + int(arithmetic.group(2)))
        if answer is None:
            answer = str(_digest_int(request_id, purpose) % 19)
        if (
            "score" in purpose.lower()
            and _digest_int(request_id, "mock-frontier") % 5 != 0
            and sample_index % 4 == 3
        ):
            lowered = answer.strip().lower()
            if lowered in {"yes", "true"}:
                return "No"
            if lowered in {"no", "false"}:
                return "Yes"
            return f"({answer})+1"
        return answer

    def _default_text(
        self,
        messages: Sequence[dict[str, str]],
        *,
        request_id: str,
        purpose: str,
        sample_index: int,
        ground_truth: str | None,
    ) -> str:
        if "cross" in purpose.lower() or "candidate" in purpose.lower():
            value = _digest_int(request_id, purpose, sample_index)
            left = 2 + value % 23
            right = 2 + (value // 23) % 23
            # A request-stable nonce keeps the CPU mock from manufacturing 64
            # parameter-only copies of one template. It is explicitly test
            # scaffolding, not a production crossover convention.
            nonce = hashlib.sha256(
                f"{request_id}:{sample_index}".encode("utf-8")
            ).hexdigest()[:12]
            return (
                "<question>"
                f"For the audit token {nonce}, what is {left}+{right}?"
                "</question>"
                "<domain>algebra</domain>"
                f"\\boxed{{{left + right}}}"
            )
        answer = self._default_answer(
            messages,
            request_id=request_id,
            purpose=purpose,
            sample_index=sample_index,
            ground_truth=ground_truth,
        )
        return f"Mock reasoning. \\boxed{{{answer}}}"

    def generate(
        self,
        messages: Sequence[list[dict[str, str]]],
        *,
        request_ids: Sequence[str],
        sampling: SamplingSpec,
        purpose: str,
        ground_truths: Sequence[str] | None = None,
        verifiers: Sequence[dict] | None = None,
    ) -> list[GeneratedGroup]:
        if len(messages) != len(request_ids):
            raise ValueError("messages and request_ids must have equal length")
        if ground_truths is not None and len(ground_truths) != len(messages):
            raise ValueError("ground_truths must have one value per request")
        if verifiers is not None and len(verifiers) != len(messages):
            raise ValueError("verifiers must have one value per request")
        if sampling.n <= 0:
            raise ValueError("sampling.n must be positive")

        self.generation_calls.append(
            {
                "request_ids": tuple(request_ids),
                "purpose": purpose,
                "sampling": sampling,
                "ground_truths": (
                    tuple(ground_truths) if ground_truths is not None else None
                ),
                "verifiers": tuple(verifiers) if verifiers is not None else None,
            }
        )
        groups: list[GeneratedGroup] = []
        for request_index, (request_messages, request_id) in enumerate(
            zip(messages, request_ids)
        ):
            ground_truth = (
                str(ground_truths[request_index]) if ground_truths is not None else None
            )
            verifier = verifiers[request_index] if verifiers is not None else None
            prompt_fingerprint = stable_id("prompt", request_messages, length=32)
            script = self._script_for(purpose, request_id)
            samples: list[GeneratedSample] = []
            rows: list[MockPayloadRow] = []
            for sample_index in range(sampling.n):
                scripted = script[sample_index % len(script)] if script else None
                if isinstance(scripted, GeneratedSample):
                    sample = GeneratedSample(
                        text=scripted.text,
                        entropy=(
                            float(scripted.entropy)
                            if scripted.entropy is not None
                            else None
                        ),
                        status=scripted.status,
                        reject_reason=scripted.reject_reason,
                    )
                else:
                    text = (
                        str(scripted)
                        if scripted is not None
                        else self._default_text(
                            request_messages,
                            request_id=request_id,
                            purpose=purpose,
                            sample_index=sample_index,
                            ground_truth=ground_truth,
                        )
                    )
                    entropy = (
                        0.1
                        + (
                            _digest_int(request_id, purpose, sample_index, "entropy")
                            % 800
                        )
                        / 1000.0
                    )
                    sample = GeneratedSample(text=text, entropy=entropy)
                row = MockPayloadRow(
                    request_id=request_id,
                    purpose=purpose,
                    sample_index=sample_index,
                    text=sample.text,
                    entropy=(
                        float(sample.entropy) if sample.entropy is not None else None
                    ),
                    ground_truth=ground_truth,
                    verifier_json=(
                        json.dumps(verifier, sort_keys=True, separators=(",", ":"))
                        if verifier is not None
                        else None
                    ),
                )
                sample.payload_row = row
                samples.append(sample)
                rows.append(row)
            payload = MockPayload(
                request_id=request_id,
                purpose=purpose,
                policy_identity=self._identity,
                prompt_fingerprint=prompt_fingerprint,
                rows=tuple(rows),
            )
            groups.append(
                GeneratedGroup(
                    request_id=request_id,
                    samples=samples,
                    payload=payload,
                    prompt_fingerprint=prompt_fingerprint,
                )
            )
        return groups

    def apply_replay_batch(self, batch: Any) -> dict[str, Any]:
        """Record the exact batch object and advance one mock policy step."""

        batch_id = getattr(batch, "batch_id", None)
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError("replay batch must expose a non-empty batch_id")
        self.applied_batches.append(batch)
        self._identity = PolicyIdentity(
            run_uuid=self._identity.run_uuid,
            policy_version=self._identity.policy_version + 1,
            adapter_version=self._identity.adapter_version + 1,
            global_step=self._identity.global_step + 1,
            source_checkpoint=f"mock://step/{self._identity.global_step + 1}",
        )
        return {
            "batch_id": batch_id,
            "applied": True,
            "global_step": self._identity.global_step,
            "exact_batch_object": self.applied_batches[-1] is batch,
        }

    def save_checkpoint(
        self,
        *,
        global_step: int,
        pipeline_state: dict[str, Any],
    ) -> str:
        state = dict(pipeline_state)
        self.saved_checkpoints.append((int(global_step), state))
        return f"mock://checkpoint/{int(global_step)}"
