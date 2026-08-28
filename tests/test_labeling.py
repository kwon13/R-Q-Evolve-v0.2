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
    )
    assert evidence.accepted
    assert evidence.cluster_sizes == [5]
    assert evidence.agreement == 5 / 9
    assert len(evidence.rollouts) == 9


def test_labeling_rejects_incomplete_and_tied_groups() -> None:
    incomplete = build_pseudo_label(
        group(*([r"\boxed{2}"] * 8)),
        requested_rollouts=9,
        proposed_answer="2",
        verifier={"mode": "expression"},
        grader=ExactGrader(),  # type: ignore[arg-type]
    )
    assert incomplete.reason == "incomplete_label_group"

    tie = build_pseudo_label(
        group(*([r"\boxed{2}"] * 4), *([r"\boxed{3}"] * 4), "boxless"),
        requested_rollouts=9,
        proposed_answer="2",
        verifier={"mode": "expression"},
        grader=ExactGrader(),  # type: ignore[arg-type]
    )
    assert tie.reason == "label_tie"

def test_unique_plurality_is_accepted_and_proposed_mismatch_is_audit_only() -> None:
    mismatch = build_pseudo_label(
        group(*([r"\boxed{2}"] * 2), r"\boxed{3}", *(["boxless"] * 6)),
        requested_rollouts=9,
        proposed_answer="3",
        verifier={"mode": "expression"},
        grader=ExactGrader(),  # type: ignore[arg-type]
    )
    assert mismatch.accepted
    assert mismatch.reason is None
    assert mismatch.pseudo_gold == "2"
    assert mismatch.agreement == 2 / 9
    assert not mismatch.proposed_matches


def test_labeling_rejects_non_transitive_equivalence() -> None:
    evidence = build_pseudo_label(
        group(r"\boxed{a}", r"\boxed{b}", r"\boxed{c}"),
        requested_rollouts=3,
        proposed_answer="a",
        verifier={"mode": "expression"},
        grader=NonTransitiveGrader(),  # type: ignore[arg-type]
    )
    assert evidence.reason == "non_transitive_answer_equivalence"
