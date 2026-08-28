from __future__ import annotations

from rq_evolve_v02.archive import ConcreteMapArchive
from rq_evolve_v02.novelty import (
    NoveltyIndex,
    question_similarity,
    template_question,
)
from rq_evolve_v02.pairing import select_parent_pairs

from .helpers import make_record


def test_novelty_exact_parent_near_and_new() -> None:
    record = make_record("p1", "Find the remainder when 2^10 is divided by 7.")
    index = NoveltyIndex([record])
    exact = index.check(
        " Find   the remainder when 2^10 is divided by 7. ",
        parent_questions=[],
        near_threshold=0.85,
        parent_ceiling=0.95,
        parent_containment_ceiling=0.85,
        parent_containment_min_shared_shingles=8,
    )
    assert not exact.accepted and exact.reason == "exact_duplicate"
    assert exact.parent_max_similarity == 0.0
    assert exact.parent_max_containment == 0.0

    near = index.check(
        "Find the remainder when 3^11 is divided by 8.",
        parent_questions=[],
        near_threshold=0.8,
        parent_ceiling=0.99,
        parent_containment_ceiling=0.85,
        parent_containment_min_shared_shingles=8,
    )
    assert template_question(record.question) == template_question(
        "Find the remainder when 3^11 is divided by 8."
    )
    assert not near.accepted and near.reason == "near_duplicate"

    parent = index.check(
        "Find the remainder when 2^10 is divided by 9.",
        parent_questions=["Find the remainder when 2^10 is divided by 7."],
        near_threshold=1.1,
        parent_ceiling=0.9,
        parent_containment_ceiling=0.85,
        parent_containment_min_shared_shingles=8,
    )
    assert not parent.accepted and parent.reason == "parent_containment"

    short_parent_copy = NoveltyIndex().check(
        "What is 2+3?",
        parent_questions=["What is 2+2?"],
        near_threshold=1.1,
        parent_ceiling=0.9,
        parent_containment_ceiling=0.85,
        parent_containment_min_shared_shingles=8,
    )
    assert not short_parent_copy.accepted
    assert short_parent_copy.reason == "parent_copy"

    novel = index.check(
        "How many diagonals does a convex decagon have?",
        parent_questions=[record.question],
        near_threshold=0.85,
        parent_ceiling=0.95,
        parent_containment_ceiling=0.85,
        parent_containment_min_shared_shingles=8,
    )
    assert novel.accepted


def test_directional_parent_containment_rejects_appended_parent_tasks() -> None:
    parent = (
        "A data set of 10 measurements was reported to have mean 41. One "
        "measurement was recorded as 41, but its correct value is 11. What is "
        "the corrected mean?"
    )
    child = (
        f"{parent} Evaluate the definite integral from 0 to 2 of 20x^4. "
        "What is the sum of those two values?"
    )
    decision = NoveltyIndex().check(
        child,
        parent_questions=[parent],
        near_threshold=0.92,
        parent_ceiling=0.94,
        parent_containment_ceiling=0.85,
        parent_containment_min_shared_shingles=8,
    )
    assert not decision.accepted
    assert decision.reason == "parent_containment"
    assert decision.parent_max_containment == 1.0
    assert decision.parent_max_containment_shared_shingles >= 8
    assert decision.parent_max_similarity < 0.94

    number_mutation = NoveltyIndex().check(
        (
            "A data set of 12 measurements was reported to have mean 53. One "
            "measurement was recorded as 53, but its correct value is 13. What "
            "is the corrected mean? Also evaluate a separate definite integral."
        ),
        parent_questions=[parent],
        near_threshold=0.92,
        parent_ceiling=0.94,
        parent_containment_ceiling=0.85,
        parent_containment_min_shared_shingles=8,
    )
    assert not number_mutation.accepted
    assert number_mutation.reason == "parent_containment"

    short_parent = "What is 2+2?"
    guarded = NoveltyIndex().check(
        (
            f"{short_parent} Use that value as the side length of a regular "
            "polygon and determine a completely new requested quantity."
        ),
        parent_questions=[short_parent],
        near_threshold=0.92,
        parent_ceiling=0.94,
        parent_containment_ceiling=0.85,
        parent_containment_min_shared_shingles=8,
    )
    assert guarded.accepted
    assert guarded.parent_max_containment == 1.0
    assert guarded.parent_max_containment_shared_shingles < 8


def test_sibling_similarity_detects_observed_near_restatement() -> None:
    first = (
        "A data set of 10 measurements has a mean of 41. One measurement was "
        "incorrectly recorded as 41, but its correct value is 11. Additionally, "
        "evaluate the definite integral from 0 to 2 of 20x^4. What is the "
        "corrected mean and the value of the integral?"
    )
    second = (
        "A data set of 10 measurements was reported to have mean 41. One "
        "measurement was recorded as 41, but its correct value is 11. What is "
        "the corrected mean? Evaluate the definite integral from 0 to 2 of "
        "20x^4. What is the sum of the corrected mean and the integral?"
    )
    metrics = question_similarity(first, second)
    assert metrics["maximum"] >= 0.82
    assert question_similarity(first, "How many diagonals does a decagon have?")[
        "maximum"
    ] < 0.82

    contained = question_similarity(
        first,
        first
        + " Now append an unrelated second task with enough additional language "
        + "to dilute ordinary symmetric similarity well below the gate threshold.",
    )
    assert contained["sequence"] < 0.82
    assert contained["shingle_overlap"] == 1.0
    assert contained["shared_shingles"] >= 8
    assert contained["maximum"] == 1.0


def test_parent_pairing_is_deterministic_and_prefers_distinct_lineages() -> None:
    archive = ConcreteMapArchive(
        [
            make_record("a", "What is 1+1?", roots=("root-a",)),
            make_record(
                "b",
                "How many subsets does a three-element set have?",
                domain="discrete_mathematics",
                problem_type="counting",
                roots=("root-b",),
            ),
            make_record(
                "c",
                "Find the area of a unit square.",
                domain="geometry",
                roots=("root-c",),
            ),
        ]
    )
    first = select_parent_pairs(archive, count=12, seed=123)
    second = select_parent_pairs(archive, count=12, seed=123)
    assert [pair.to_dict() for pair in first] == [pair.to_dict() for pair in second]
    assert all(pair.left_id != pair.right_id for pair in first)
    assert all(
        archive.lineage_roots(pair.left_id).isdisjoint(
            archive.lineage_roots(pair.right_id)
        )
        for pair in first
    )
