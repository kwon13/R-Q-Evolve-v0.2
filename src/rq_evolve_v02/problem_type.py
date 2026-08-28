"""Deterministic statement-only annotation of computational output type."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .concepts import PROBLEM_TYPES
from .verifier import canonical_boolean, parse_finite_set


@dataclass(frozen=True, slots=True)
class ProblemTypeAnnotation:
    problem_type: str | None
    confidence: str
    evidence: str
    request_window: str
    review_reason: str | None = None


_SPACE_RE = re.compile(r"\s+")
_PROOF_RE = re.compile(
    r"(?:^|[.?!;:]\s*)(?:prove|show|demonstrate|establish|explain|justify)\b"
    r"(?:\s+(?:that|why))?|"
    r"\b(?:can|could)\s+you\s+(?:prove|show|explain|justify)\b|"
    r"\band\s+(?:prove|show|explain|justify)\b",
    re.IGNORECASE,
)
_WITNESS_RE = re.compile(
    r"\b(?:construct|exhibit)\b|"
    r"\bgive\s+(?:an?\s+)?(?:example|construction)\b",
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
_REQUEST_VERB = (
    r"(?:determine\s+whether|decide\s+whether|"
    r"find|determine|compute|calculate|evaluate|count|construct|exhibit|"
    r"give\s+(?:an?\s+)?(?:example|construction)|"
    r"solve|maximize|minimize|what\s+(?:is|are)|how\s+many|"
    r"(?:is|are|does|do|can|must)\b[^?]{0,500}\?)"
)
_REQUEST_START_RE = re.compile(
    rf"(?:^|[.?!;:]\s+)(?:please\s+)?({_REQUEST_VERB})",
    re.IGNORECASE,
)
_SECONDARY_REQUEST_RE = re.compile(
    rf"\b(?:and\s+also|and|also|additionally)\s*,?\s*({_REQUEST_VERB})",
    re.IGNORECASE,
)
_NONNEGATIVE_INTEGER_RE = re.compile(r"\+?(?:\d+|\d{1,3}(?:,\d{3})+)")
_GROUPED_INTEGER_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+")
_TEXT_BOOLEAN_RE = re.compile(
    r"(?:\\text\{\s*)?(?:yes|no|true|false)(?:\s*\})?",
    re.IGNORECASE,
)


def _distinct_request_cues(request: str) -> list[str]:
    """Extract explicit output requests without double-counting overlaps."""

    matches = [
        *_REQUEST_START_RE.finditer(request),
        *_SECONDARY_REQUEST_RE.finditer(request),
    ]
    matches.sort(key=lambda match: (match.start(1), match.end(1)))
    spans: list[tuple[int, int]] = []
    for match in matches:
        span = match.span(1)
        if any(span[0] < end and start < span[1] for start, end in spans):
            continue
        spans.append(span)
    cues: list[str] = []
    for index, (start, _end) in enumerate(spans):
        stop = spans[index + 1][0] if index + 1 < len(spans) else len(request)
        cues.append(request[start : min(stop, start + 300)].strip(" ,.;:"))
    return cues


def _cue_problem_type(cue: str) -> str | None:
    for label, pattern in (
        ("decision", _DECISION_RE),
        ("optimization", _OPTIMIZATION_RE),
        ("counting", _COUNTING_RE),
        ("search", _SEARCH_RE),
        ("function", _FUNCTION_RE),
    ):
        if pattern.search(cue):
            return label
    return None


def _has_explicit_set_envelope(answer: str) -> bool:
    text = str(answer).strip().replace(r"\left", "").replace(r"\right", "")
    if text in {r"\emptyset", r"\varnothing", "∅", "{}", r"\{\}", "[]"}:
        return True
    return any(
        text.startswith(left) and text.endswith(right)
        for left, right in ((r"\{", r"\}"), ("{", "}"), ("[", "]"))
    )


def _scalar_sequence_error(answer: str) -> str | None:
    text = str(answer).strip().replace(r"\left", "").replace(r"\right", "")
    if _GROUPED_INTEGER_RE.fullmatch(text):
        return None
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = set(opening.values())
    stack: list[str] = []
    for char in text:
        if char in opening:
            stack.append(opening[char])
        elif char in closing:
            if not stack or stack.pop() != char:
                return "malformed_scalar_answer"
        elif char in {",", ";"} and not stack:
            return "compound_scalar_answer"
    if stack:
        return "malformed_scalar_answer"
    if (
        re.search(r"\s+and\s+", text, re.IGNORECASE)
        or re.search(r"\\text\{\s*and\s*\}", text, re.IGNORECASE)
    ):
        return "compound_scalar_answer"
    return None


def answer_contract_error(problem_type: str, answer: str) -> str | None:
    """Validate the visible answer shape before expensive solver rollouts.

    This establishes only the output contract; mathematical correctness remains
    the pseudo-labeler's and typed grader's responsibility.
    """

    text = str(answer).strip()
    if not text:
        return "empty_answer"
    if problem_type == "decision":
        return None if canonical_boolean(text) is not None else "decision_answer_not_boolean"
    if problem_type == "search":
        elements = parse_finite_set(text)
        if elements is None:
            return "search_answer_not_finite_set"
        if len(elements) != len(set(elements)):
            return "search_answer_contains_duplicates"
        return None
    if problem_type == "counting":
        return (
            None
            if _NONNEGATIVE_INTEGER_RE.fullmatch(text)
            else "counting_answer_not_nonnegative_integer"
        )
    if problem_type in {"optimization", "function"}:
        if _TEXT_BOOLEAN_RE.fullmatch(text) or _has_explicit_set_envelope(text):
            return "scalar_answer_has_wrong_shape"
        # A comma-separated/semicolon-separated pair is two outputs, not one
        # scalar.  Conventional thousands separators such as 12,345 remain valid.
        return _scalar_sequence_error(text)
    return "unknown_problem_type"


def annotate_problem_type(
    statement: str, *, max_chars: int = 1200
) -> ProblemTypeAnnotation:
    full_request = _SPACE_RE.sub(" ", str(statement or "")).strip()
    request = full_request[-max_chars:]
    if not full_request:
        return ProblemTypeAnnotation(None, "none", "", request, "empty_statement")
    proof = _PROOF_RE.search(full_request)
    if proof:
        return ProblemTypeAnnotation(
            None, "none", proof.group(0), request, "proof_or_justification"
        )
    witness = _WITNESS_RE.search(full_request)
    if witness:
        return ProblemTypeAnnotation(
            None,
            "none",
            witness.group(0),
            request,
            "unsupported_witness_request",
        )
    request_cues = _distinct_request_cues(full_request)
    if full_request.count("?") > 1 or len(request_cues) > 1:
        cue_types = {
            label for cue in request_cues if (label := _cue_problem_type(cue))
        }
        return ProblemTypeAnnotation(
            None,
            "none",
            " | ".join(request_cues),
            request,
            "mixed_output_contract"
            if len(cue_types) > 1
            else "multiple_output_requests",
        )
    for label, pattern in (
        ("decision", _DECISION_RE),
        ("optimization", _OPTIMIZATION_RE),
        ("counting", _COUNTING_RE),
        ("search", _SEARCH_RE),
        ("function", _FUNCTION_RE),
    ):
        match = pattern.search(full_request)
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
