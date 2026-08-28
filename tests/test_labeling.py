from __future__ import annotations

from rq_evolve_v02.backends import GeneratedGroup, GeneratedSample
from rq_evolve_v02.labeling import build_pseudo_label


class ExactGrader:
    def grade(self, pred: str, gold: str, verifier: dict) -> bool:
        return pred.strip() == gold.strip()

    def equivalent(self, left: str, right: str, verifier: dict) -> bool:
        return left.strip() == right.strip()


class NonTransitiveGrader(ExactGrader):
    def equivalent(self, left: str, right: str, verifier: dict) -> bool:
        pair = frozenset((left.strip(), right.strip()))
        return pair in {frozenset(("a", "b")), frozenset(("b", "c"))} or len(pair) == 1


def group(*texts: str) -> GeneratedGroup:
    return GeneratedGroup(
        request_id="label-1",
        samples=[GeneratedSample(text=text) for text in texts],
    )


def test_labeling_uses_fixed_requested_denominator() -> None:
    evidence = build_pseudo_label(
        group(*([r"\boxed{2}"] * 5), *(["boxless"] * 4)),
        requested_rollouts=9,
        proposed_answer="2",
        verifier={"mode": "expression"},
        grader=ExactGrader(),  # type: ignore[arg-type]
        min_agreement=5 / 9,
        require_proposed_match=True,
    )
    assert evidence.accepted
    assert evidence.cluster_sizes == [5]
    assert evidence.agreement == 5 / 9
    assert len(evidence.rollouts) == 9


def test_labeling_rejects_incomplete_tie_and_proposed_mismatch() -> None:
    incomplete = build_pseudo_label(
        group(*([r"\boxed{2}"] * 8)),
        requested_rollouts=9,
        proposed_answer="2",
        verifier={"mode": "expression"},
        grader=ExactGrader(),  # type: ignore[arg-type]
        min_agreement=5 / 9,
        require_proposed_match=True,
    )
    assert incomplete.reason == "incomplete_label_group"

    tie = build_pseudo_label(
        group(*([r"\boxed{2}"] * 4), *([r"\boxed{3}"] * 4), "boxless"),
        requested_rollouts=9,
        proposed_answer="2",
        verifier={"mode": "expression"},
        grader=ExactGrader(),  # type: ignore[arg-type]
        min_agreement=4 / 9,
        require_proposed_match=True,
    )
    assert tie.reason == "label_tie"

    mismatch = build_pseudo_label(
        group(*([r"\boxed{2}"] * 9)),
        requested_rollouts=9,
        proposed_answer="3",
        verifier={"mode": "expression"},
        grader=ExactGrader(),  # type: ignore[arg-type]
        min_agreement=5 / 9,
        require_proposed_match=True,
    )
    assert mismatch.reason == "proposed_answer_mismatch"


def test_labeling_rejects_non_transitive_equivalence() -> None:
    evidence = build_pseudo_label(
        group(r"\boxed{a}", r"\boxed{b}", r"\boxed{c}"),
        requested_rollouts=3,
        proposed_answer="a",
        verifier={"mode": "expression"},
        grader=NonTransitiveGrader(),  # type: ignore[arg-type]
        min_agreement=2 / 3,
        require_proposed_match=False,
    )
    assert evidence.reason == "non_transitive_answer_equivalence"
