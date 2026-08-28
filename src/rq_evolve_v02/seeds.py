"""Certified diagonal seed loading."""

from __future__ import annotations

import json
from pathlib import Path

from .concepts import DOMAINS, PROBLEM_TYPES, validate_cell
from .models import ProblemRecord
from .problem_type import (
    annotate_problem_type,
    answer_contract_error,
    verifier_for_problem_type,
)
from .verifier import normalize_verifier, verifier_for_answer


def load_seed_records(path: str | Path) -> list[ProblemRecord]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"seed problem file does not exist: {source}")
    records: list[ProblemRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            record = ProblemRecord.from_dict(row)
        except Exception as exc:
            raise ValueError(f"invalid seed row {source}:{line_number}: {exc}") from exc
        if record.problem_id in seen:
            raise ValueError(f"duplicate seed problem_id: {record.problem_id}")
        errors = validate_cell(record.domain, record.problem_type)
        if errors:
            raise ValueError(f"invalid seed {record.problem_id}: {'; '.join(errors)}")
        annotation = annotate_problem_type(record.question)
        if (
            annotation.problem_type != record.problem_type
            or annotation.confidence != "high"
        ):
            raise ValueError(
                f"seed {record.problem_id} statement type {annotation.problem_type!r} "
                f"does not match {record.problem_type!r}"
            )
        for name, answer in (
            ("proposed_answer", record.proposed_answer),
            ("pseudo_gold", record.pseudo_gold),
        ):
            answer_error = answer_contract_error(record.problem_type, answer)
            if answer_error is not None:
                raise ValueError(
                    f"seed {record.problem_id} {name} violates its output contract: "
                    f"{answer_error}"
                )
        record.verifier = normalize_verifier(record.verifier, answer=record.pseudo_gold)
        expected_mode = verifier_for_problem_type(record.problem_type)["mode"]
        expected_verifier = verifier_for_answer(expected_mode, record.pseudo_gold)
        if record.verifier != expected_verifier:
            raise ValueError(
                f"seed {record.problem_id} verifier does not match its "
                f"{record.problem_type} answer contract"
            )
        if not record.lineage_root_ids:
            record.lineage_root_ids = (record.problem_id,)
        seen.add(record.problem_id)
        records.append(record)
    if len(records) != len(DOMAINS):
        raise ValueError(f"v0.2 requires exactly {len(DOMAINS)} diagonal seed problems")
    if {record.domain for record in records} != set(DOMAINS):
        raise ValueError(
            "diagonal seeds must cover every top-level domain exactly once"
        )
    if len({record.cell for record in records}) != len(records):
        raise ValueError("diagonal seeds must occupy distinct MAP cells")
    if {record.problem_type for record in records} != set(PROBLEM_TYPES):
        raise ValueError(
            "diagonal seeds must collectively cover all five problem types"
        )
    return records
