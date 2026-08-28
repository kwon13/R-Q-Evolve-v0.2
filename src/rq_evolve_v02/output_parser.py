"""Strict R-Zero-compatible question/answer envelope parser."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .concepts import DOMAINS
from .utils import canonical_text, stable_id

_THINK_RE = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)
_DOMAIN_WRAPPER_RE = re.compile(
    r"\A<([a-z_]+)>\s*(\\boxed\{.*)\s*</\1>\s*\Z", re.DOTALL
)
_PUBLIC_MARKER_RE = re.compile(r"<\/?(?:question|domain)>|\\boxed\{")


@dataclass(frozen=True, slots=True)
class ParsedProblem:
    question: str
    domain: str
    answer: str


def _extract_balanced_box(text: str, start: int) -> tuple[str, int] | None:
    marker = r"\boxed{"
    if not text.startswith(marker, start):
        return None
    depth = 1
    cursor = start + len(marker)
    content_start = cursor
    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:cursor].strip(), cursor + 1
        cursor += 1
    return None


def _parse_domain_prefix(text: str) -> tuple[str | None, str, str | None]:
    """Parse one exact domain block at the start of ``text``."""

    open_tag, close_tag = "<domain>", "</domain>"
    if not text.startswith(open_tag):
        return None, text, "missing_domain_open"
    close = text.find(close_tag, len(open_tag))
    if close < 0:
        return None, text, "missing_domain_close"
    domain = text[len(open_tag) : close].strip()
    if domain not in DOMAINS:
        return None, text, "invalid_domain"
    return domain, text[close + len(close_tag) :].strip(), None


def parse_problem_response(response: str) -> tuple[ParsedProblem | None, str | None]:
    """Require one question, one closed-vocabulary domain, and one answer.

    Qwen may emit one private ``<think>`` prefix even when instructed not to;
    that prefix is ignored.  Anything else before, between, or after the three
    public fields fails closed so fields from different copies can never be
    spliced together.  Domain tokens are exact and case-sensitive.  The domain
    and answer fields may appear in either order because small base models
    frequently preserve R-Zero's historical question-then-box envelope.
    """

    if not isinstance(response, str):
        return None, "response_not_string"
    text = _THINK_RE.sub("", response, count=1).strip()
    open_tag, close_tag = "<question>", "</question>"
    if not text.startswith(open_tag):
        return None, "missing_question_open"
    close = text.find(close_tag, len(open_tag))
    if close < 0:
        return None, "missing_question_close"
    if (
        text.find(open_tag, len(open_tag)) >= 0
        or text.find(close_tag, close + len(close_tag)) >= 0
    ):
        return None, "multiple_question_blocks"
    question = text[len(open_tag) : close].strip()
    tail = text[close + len(close_tag) :].strip()
    # A frequent small-model variant uses the selected closed-vocabulary token
    # itself as an XML tag. This is unambiguous only when the wrapper spans the
    # entire remaining envelope and names exactly one supported domain.
    wrapper = _DOMAIN_WRAPPER_RE.fullmatch(tail)
    if wrapper and wrapper.group(1) in DOMAINS:
        tail = f"<domain>{wrapper.group(1)}</domain>{wrapper.group(2)}"
    if tail.count("<domain>") > 1 or tail.count("</domain>") > 1:
        return None, "multiple_domain_blocks"
    if tail.startswith("<domain>"):
        domain, tail, error = _parse_domain_prefix(tail)
        if error is not None:
            return None, error
        if not tail.startswith(r"\boxed{"):
            return None, "missing_boxed_answer"
        parsed_box = _extract_balanced_box(tail, 0)
        if parsed_box is None:
            return None, "unclosed_boxed_answer"
        answer, end = parsed_box
        remainder = tail[end:].strip()
        if remainder and _PUBLIC_MARKER_RE.search(remainder):
            return None, "trailing_output"
    elif tail.startswith(r"\boxed{"):
        parsed_box = _extract_balanced_box(tail, 0)
        if parsed_box is None:
            return None, "unclosed_boxed_answer"
        answer, end = parsed_box
        domain, remainder, error = _parse_domain_prefix(tail[end:].strip())
        if error is not None:
            return None, error
        if remainder and _PUBLIC_MARKER_RE.search(remainder):
            return None, "trailing_output"
    else:
        return None, "missing_domain_open"
    assert domain is not None
    question = canonical_text(question)
    answer = answer.strip()
    if not question:
        return None, "empty_question"
    if not answer:
        return None, "empty_answer"
    return ParsedProblem(question=question, domain=domain, answer=answer), None


def candidate_id(question: str, answer: str) -> str:
    return stable_id("cand", canonical_text(question).lower(), answer.strip())


def extract_last_boxed(response: str) -> str | None:
    """Extract the last complete boxed answer from one Solver response."""

    text = str(response or "")
    marker = r"\boxed{"
    positions = [match.start() for match in re.finditer(re.escape(marker), text)]
    for start in reversed(positions):
        parsed = _extract_balanced_box(text, start)
        if parsed is not None:
            answer, _ = parsed
            return answer or None
    return None
