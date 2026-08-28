from __future__ import annotations

from pathlib import Path

from rq_evolve_v02.concepts import DOMAINS, PROBLEM_TYPES
from rq_evolve_v02.seeds import load_seed_records


def test_shipped_diagonal_seeds_cover_both_axes() -> None:
    root = Path(__file__).resolve().parents[1]
    records = load_seed_records(root / "seed_problems" / "diagonal_seeds.jsonl")
    assert len(records) == 7
    assert {record.domain for record in records} == set(DOMAINS)
    assert {record.problem_type for record in records} == set(PROBLEM_TYPES)
    assert len({record.cell for record in records}) == 7
