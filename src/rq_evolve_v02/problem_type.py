"""Deterministic statement-only annotation of computational output type."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .concepts import PROBLEM_TYPES


@dataclass(frozen=True, slots=True)
class ProblemTypeAnnotation:
    problem_type: str | None
    confidence: str
    evidence: str
    request_window: str
    review_reason: str | None = None


_SPACE_RE = re.compile(r"\s+")
_PROOF_RE = re.compile(
    r"(?:^|[.?!;:]\s*)(?:prove|show|demonstrate|establish)\b(?:\s+that)?",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(?:(?:determine|decide)\s+whether|prove\s+or\s+disprove|answer\s+yes\s+or\s+no)\b|"
    r"(?:^|[.?!;:]\s*)(?:is|are|does|do|can|must)\b[^?]{0,500}\?",
    re.IGNORECASE,
)
_OPTIMIZATION_RE = re.compile(
    r"\b(?:find|determine|compute|calculate|evaluate|what\s+(?:is|are))\s+"
    r"(?:the\s+)?(?:maximum|minimum|largest|smallest|greatest|least|optimal|best\s+possible)\b|"
    r"\b(?:maximize|minimize)\b|\bafter\s+what\s+(?:least|fewest|minimum)\b",
    re.IGNORECASE,
)
_COUNTING_RE = re.compile(
    r"\bhow\s+many\b|"
    r"\b(?:determine|find|compute|calculate|what\s+(?:is|are))\s+(?:the\s+)?number\s+of\b|"
    r"\b(?:determine|find|compute|calculate)\s+(?:the\s+)?cardinality\s+of\b|"
    r"(?:^|[.?!;:]\s*)count\s+the\b",
    re.IGNORECASE,
)
_SEARCH_RE = re.compile(
    r"\b(?:find|determine)\s+all\b|\bfor\s+(?:what|which)\b|"
    r"\b(?:construct|exhibit)\b|\bgive\s+(?:an?\s+)?(?:example|construction)\b|"
    r"\b(?:find|determine)\s+(?:an?|the)\s+[^.?!]{0,180}?\bsuch\s+that\b|"
    r"\bsolve\s+(?:the\s+)?(?:equation|system|congruence)\b",
    re.IGNORECASE,
)
_FUNCTION_RE = re.compile(
    r"\b(?:compute|calculate|evaluate)\b|\bwhat\s+(?:is|are)\b|"
    r"\b(?:find|determine)\s+(?:the\s+)?(?:value|values|sum|product|remainder|"
    r"residue|digit|digits|perimeter|area|volume|length|distance|ratio|"
    r"probability|coefficient|degree|order|measure)\b",
    re.IGNORECASE,
)
_GENERIC_RE = re.compile(r"\b(?:find|determine|give)\b", re.IGNORECASE)


def annotate_problem_type(
    statement: str, *, max_chars: int = 1200
) -> ProblemTypeAnnotation:
    request = _SPACE_RE.sub(" ", str(statement or "")).strip()[-max_chars:]
    if not request:
        return ProblemTypeAnnotation(None, "none", "", request, "empty_statement")
    proof = _PROOF_RE.search(request)
    if proof:
        return ProblemTypeAnnotation(
            None, "none", proof.group(0), request, "proof_or_justification"
        )
    for label, pattern in (
        ("decision", _DECISION_RE),
        ("optimization", _OPTIMIZATION_RE),
        ("counting", _COUNTING_RE),
        ("search", _SEARCH_RE),
        ("function", _FUNCTION_RE),
    ):
        match = pattern.search(request)
        if match:
            return ProblemTypeAnnotation(label, "high", match.group(0), request)
    generic = _GENERIC_RE.search(request)
    return ProblemTypeAnnotation(
        None,
        "none",
        generic.group(0) if generic else "",
        request,
        "generic_find_or_determine" if generic else "no_output_contract_cue",
    )


def verifier_for_problem_type(problem_type: str) -> dict:
    if problem_type not in PROBLEM_TYPES:
        raise ValueError(f"unknown problem type: {problem_type!r}")
    if problem_type == "decision":
        return {"mode": "boolean"}
    if problem_type == "search":
        return {"mode": "set", "elements": []}
    return {"mode": "expression"}
