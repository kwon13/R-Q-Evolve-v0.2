from __future__ import annotations

from dataclasses import dataclass

import pytest

from rq_evolve_v02.backends import GeneratedSample, SamplingSpec
from rq_evolve_v02.mock_backend import DeterministicMockBackend, MockPayload
from rq_evolve_v02.output_parser import parse_problem_response


SAMPLING = SamplingSpec(n=3, temperature=0.45, top_p=0.85, max_tokens=128)


def test_mock_generation_is_request_deterministic_and_payload_aligned() -> None:
    messages = [[{"role": "user", "content": "parents"}]]
    left = DeterministicMockBackend().generate(
        messages,
        request_ids=["candidate-7"],
        sampling=SAMPLING,
        purpose="crossover",
    )[0]
    right = DeterministicMockBackend().generate(
        messages,
        request_ids=["candidate-7"],
        sampling=SAMPLING,
        purpose="crossover",
    )[0]
    assert [sample.text for sample in left.samples] == [
        sample.text for sample in right.samples
    ]
    assert isinstance(left.payload, MockPayload)
    assert len(left.payload) == SAMPLING.n
    assert all(
        sample.payload_row is row
        for sample, row in zip(left.samples, left.payload.rows)
    )
    assert all(
        parse_problem_response(sample.text)[1] is None for sample in left.samples
    )


def test_mock_scripts_preserve_rejections_and_ground_truth_in_payload() -> None:
    backend = DeterministicMockBackend(
        scripts={
            ("score", "score-1"): [
                GeneratedSample(r"\boxed{5}", entropy=0.9),
                GeneratedSample(
                    "timeout", status="timeout", reject_reason="worker_lost"
                ),
            ]
        }
    )
    group = backend.generate(
        [[{"role": "user", "content": "What is 2+3?"}]],
        request_ids=["score-1"],
        sampling=SAMPLING,
        purpose="score",
        ground_truths=["5"],
        verifiers=[{"mode": "expression"}],
    )[0]
    assert [sample.status for sample in group.samples] == [
        "accepted",
        "timeout",
        "accepted",
    ]
    assert group.payload.rows[0].ground_truth == "5"
    assert group.payload.rows[0].verifier_json == '{"mode":"expression"}'


@dataclass
class Batch:
    batch_id: str
    payload: object


def test_mock_training_records_exact_batch_object_and_checkpoint() -> None:
    backend = DeterministicMockBackend()
    batch = Batch("batch-1", object())
    before = backend.policy_identity
    metrics = backend.apply_replay_batch(batch)
    assert backend.applied_batches == [batch]
    assert backend.applied_batches[0] is batch
    assert metrics["exact_batch_object"]
    assert backend.policy_identity.global_step == before.global_step + 1
    assert backend.policy_identity.policy_version == before.policy_version + 1
    assert backend.save_checkpoint(global_step=1, pipeline_state={"iteration": 2}) == (
        "mock://checkpoint/1"
    )
    assert backend.saved_checkpoints == [(1, {"iteration": 2})]


def test_mock_validates_batch_lengths() -> None:
    backend = DeterministicMockBackend()
    with pytest.raises(ValueError):
        backend.generate(
            [[{"role": "user", "content": "one"}]],
            request_ids=[],
            sampling=SAMPLING,
            purpose="score",
        )
