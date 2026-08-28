from __future__ import annotations

from rq_evolve_v02.models import ProblemRecord


def make_record(
    problem_id: str,
    question: str,
    *,
    answer: str = "1",
    domain: str = "algebra",
    problem_type: str = "function",
    roots: tuple[str, ...] | None = None,
) -> ProblemRecord:
    mode = (
        "boolean"
        if problem_type == "decision"
        else "set" if problem_type == "search" else "expression"
    )
    verifier = {"mode": mode}
    if mode == "set":
        verifier["elements"] = [answer]
    return ProblemRecord(
        problem_id=problem_id,
        question=question,
        proposed_answer=answer,
        pseudo_gold=answer,
        verifier=verifier,
        domain=domain,
        problem_type=problem_type,
        parent_ids=(),
        lineage_root_ids=roots or (problem_id,),
        generation=0,
        created_iteration=0,
        created_policy_version=0,
        label_evidence={},
        domain_evidence={},
        source="seed",
    )
