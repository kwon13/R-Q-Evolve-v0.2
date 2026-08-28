"""Strict R-Zero-compatible question/answer envelope parser."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .utils import canonical_text, stable_id

_THINK_RE = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)


@dataclass(frozen=True, slots=True)
class ParsedProblem:
    question: str
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


def parse_problem_response(response: str) -> tuple[ParsedProblem | None, str | None]:
    """Require exactly one question block and one final balanced box.

    Qwen may emit one private ``<think>`` prefix even when instructed not to;
    that prefix is ignored.  Anything else before, between, or after the two
    public fields fails closed so a question and answer from different copies
    can never be spliced together.
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
    if not tail.startswith(r"\boxed{"):
        return None, "missing_boxed_answer"
    parsed_box = _extract_balanced_box(tail, 0)
    if parsed_box is None:
        return None, "unclosed_boxed_answer"
    answer, end = parsed_box
    if tail[end:].strip():
        return None, "trailing_output"
    question = canonical_text(question)
    answer = answer.strip()
    if not question:
        return None, "empty_question"
    if not answer:
        return None, "empty_answer"
    return ParsedProblem(question=question, answer=answer), None


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
