from __future__ import annotations

import pytest

from rq_evolve_v02.grading import GraderClient
from rq_evolve_v02.verifier import canonical_boolean, normalize_verifier


def test_normalize_verifier_is_data_only_and_strict() -> None:
    assert normalize_verifier(None, answer="2") == {"mode": "expression"}
    assert normalize_verifier({"mode": "boolean"}, answer=r"\text{Yes}") == {
        "mode": "boolean"
    }
    assert normalize_verifier({"mode": "set", "elements": ["1", "2"]}) == {
        "mode": "set",
        "elements": ["1", "2"],
    }
    with pytest.raises(ValueError):
        normalize_verifier({"mode": "expression", "code": "unsafe"})
    with pytest.raises(ValueError):
        normalize_verifier({"mode": "set", "elements": ["1", "1"]})
    with pytest.raises(ValueError):
        normalize_verifier({"mode": "boolean"}, answer="maybe")


def test_canonical_boolean() -> None:
    assert canonical_boolean(" TRUE ") == "Yes"
    assert canonical_boolean(r"\text{no}") == "No"
    assert canonical_boolean("unknown") is None


def test_grader_expression_boolean_and_unordered_set() -> None:
    grader = GraderClient(timeout_s=5)
    try:
        assert grader.grade(r"\frac{1}{2}", "0.5", {"mode": "expression"})
        assert not grader.grade("3", "4", {"mode": "expression"})
        assert grader.grade("true", "Yes", {"mode": "boolean"})
        assert grader.grade(
            r"\{2, 1/2\}",
            r"\{0.5, 2\}",
            {"mode": "set", "elements": ["0.5", "2"]},
        )
        assert not grader.grade(
            r"\{1,2,3\}",
            r"\{1,2\}",
            {"mode": "set", "elements": ["1", "2"]},
        )
    finally:
        grader.close()
