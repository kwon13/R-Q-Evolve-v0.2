from __future__ import annotations

import json
from pathlib import Path

import pytest

from rq_evolve_v02.concepts import DOMAINS, PROBLEM_TYPES
from rq_evolve_v02.seeds import load_seed_records


def test_shipped_diagonal_seeds_cover_both_axes() -> None:
    root = Path(__file__).resolve().parents[1]
    records = load_seed_records(root / "seed_problems" / "diagonal_seeds.jsonl")
    assert len(records) == 7
    assert {record.domain for record in records} == set(DOMAINS)
    assert {record.problem_type for record in records} == set(PROBLEM_TYPES)
    assert len({record.cell for record in records}) == 7


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_mode", "verifier does not match"),
        ("wrong_set_elements", "verifier does not match"),
        ("bad_proposed", "proposed_answer violates"),
        ("bad_pseudo_gold", "pseudo_gold violates"),
    ],
)
def test_seed_rows_cannot_bypass_live_answer_contracts(
    tmp_path: Path, mutation: str, message: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "seed_problems" / "diagonal_seeds.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    if mutation == "wrong_mode":
        rows[0]["verifier"] = {"mode": "expression"}
    elif mutation == "wrong_set_elements":
        rows[1]["verifier"]["elements"] = ["9"]
    elif mutation == "bad_proposed":
        rows[0]["proposed_answer"] = "maybe"
    elif mutation == "bad_pseudo_gold":
        rows[0]["pseudo_gold"] = "maybe"
    path = tmp_path / "seeds.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_seed_records(path)
