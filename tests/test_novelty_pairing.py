from __future__ import annotations

from rq_evolve_v02.archive import ConcreteMapArchive
from rq_evolve_v02.novelty import NoveltyIndex, template_question
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
    )
    assert not exact.accepted and exact.reason == "exact_duplicate"

    near = index.check(
        "Find the remainder when 3^11 is divided by 8.",
        parent_questions=[],
        near_threshold=0.8,
        parent_ceiling=0.99,
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
    )
    assert not parent.accepted and parent.reason == "parent_copy"

    novel = index.check(
        "How many diagonals does a convex decagon have?",
        parent_questions=[record.question],
        near_threshold=0.85,
        parent_ceiling=0.95,
    )
    assert novel.accepted


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
