from __future__ import annotations

import pytest

from rq_evolve_v02.backends import GeneratedGroup, GeneratedSample, PolicyIdentity
from rq_evolve_v02.scoring import in_frontier, score_group


class ExactGrader:
    def grade(self, pred: str, gold: str, verifier: dict) -> bool:
        return pred.strip() == gold.strip()


IDENTITY = PolicyIdentity("run", 3, 2, 11, "checkpoint-11")


def test_rq_score_and_boxless_as_completed_wrong() -> None:
    samples = [GeneratedSample(r"\boxed{1}", entropy=0.8) for _ in range(4)]
    samples += [GeneratedSample(r"\boxed{0}", entropy=0.8) for _ in range(3)]
    samples += [GeneratedSample("completed without box", entropy=0.8)]
    score, error = score_group(
        GeneratedGroup("score-1", samples),
        requested_rollouts=8,
        gold="1",
        verifier={"mode": "expression"},
        grader=ExactGrader(),  # type: ignore[arg-type]
        identity=IDENTITY,
        iteration=7,
        require_entropy=True,
    )
    assert error is None
    assert score is not None
    assert score.s_hat == 0.5
    assert score.learnability == pytest.approx(2 / 7)
    assert score.u_score == pytest.approx(0.8)
    assert score.rq_score == pytest.approx(8 / 35)
    assert score.num_correct == 4
    assert score.policy_run_uuid == IDENTITY.run_uuid
    assert score.adapter_version == IDENTITY.adapter_version
    assert score.global_step == IDENTITY.global_step
    assert score.source_checkpoint == IDENTITY.source_checkpoint
    assert score.rollouts[-1].correct is False
    assert in_frontier(score, low=0.1, high=0.9)


def test_score_group_fails_closed_on_partial_or_missing_entropy() -> None:
    short, error = score_group(
        GeneratedGroup("short", [GeneratedSample(r"\boxed{1}", entropy=1)]),
        requested_rollouts=2,
        gold="1",
        verifier={"mode": "expression"},
        grader=ExactGrader(),  # type: ignore[arg-type]
        identity=IDENTITY,
        iteration=0,
        require_entropy=True,
    )
    assert short is None and error == "incomplete_score_group"

    rejected = [GeneratedSample(r"\boxed{1}", entropy=1) for _ in range(2)]
    rejected[1].status = "timeout"
    score, error = score_group(
        GeneratedGroup("rejected", rejected),
        requested_rollouts=2,
        gold="1",
        verifier={"mode": "expression"},
        grader=ExactGrader(),  # type: ignore[arg-type]
        identity=IDENTITY,
        iteration=0,
        require_entropy=True,
    )
    assert score is None and error == "score_group_contains_rejected_sample"

    score, error = score_group(
        GeneratedGroup(
            "entropy", [GeneratedSample(r"\boxed{1}", entropy=None) for _ in range(2)]
        ),
        requested_rollouts=2,
        gold="1",
        verifier={"mode": "expression"},
        grader=ExactGrader(),  # type: ignore[arg-type]
        identity=IDENTITY,
        iteration=0,
        require_entropy=True,
    )
    assert score is None and error == "missing_score_entropy"
