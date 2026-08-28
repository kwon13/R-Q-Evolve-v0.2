"""Data-only answer contracts shared by labeling, scoring, and training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

VERIFIER_MODES = ("expression", "boolean", "set")
MAX_ITEMS = 32


def canonical_boolean(value: str) -> str | None:
    text = str(value).strip().lower()
    text = re.sub(r"^\\text\{(.*)\}$", r"\1", text)
    text = text.strip("{}[]() .")
    if text in {"yes", "true", "1"}:
        return "Yes"
    if text in {"no", "false", "0"}:
        return "No"
    return None


def parse_finite_set(value: str) -> list[str] | None:
    """Parse one shallow, finite set serialization without evaluating code.

    Elements may contain nested mathematical delimiters, so separators count
    only at depth zero.  Semantic equality of the elements remains the typed
    grader's job; this parser establishes the data-only verifier contract.
    """

    text = str(value).strip().replace(r"\left", "").replace(r"\right", "")
    if text in {r"\emptyset", r"\varnothing", "∅", "{}", r"\{\}", "[]"}:
        return []
    for left, right in ((r"\{", r"\}"), ("{", "}"), ("[", "]")):
        if text.startswith(left) and text.endswith(right):
            text = text[len(left) : -len(right)].strip()
            break
    if not text:
        return []
    opening = {"{": "}", "[": "]", "(": ")"}
    closing = set(opening.values())
    stack: list[str] = []
    parts: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if char in opening:
            stack.append(opening[char])
        elif char in closing:
            if not stack or stack.pop() != char:
                return None
        elif char in {",", ";"} and not stack:
            item = text[start:index].strip()
            if not item:
                return None
            parts.append(item)
            start = index + 1
    if stack:
        return None
    final = text[start:].strip()
    if not final:
        return None
    parts.append(final)
    if len(parts) > MAX_ITEMS:
        return None
    return parts


def verifier_for_answer(mode: str, answer: str) -> dict[str, Any]:
    """Finalize a verifier after the pseudo-gold answer is known."""

    if mode == "set":
        elements = parse_finite_set(answer)
        if elements is None or len(elements) != len(set(elements)):
            raise ValueError("search pseudo-gold is not a unique finite set")
        return normalize_verifier({"mode": "set", "elements": elements}, answer=answer)
    return normalize_verifier({"mode": mode}, answer=answer)


def normalize_verifier(
    verifier: Mapping[str, Any] | None, *, answer: str | None = None
) -> dict:
    if verifier is None:
        result: dict[str, Any] = {"mode": "expression"}
    elif not isinstance(verifier, Mapping):
        raise ValueError("verifier must be an object")
    else:
        mode = str(verifier.get("mode", "")).strip().lower()
        if mode not in VERIFIER_MODES:
            raise ValueError(f"unknown verifier mode: {mode!r}")
        allowed = {"mode", "elements"} if mode == "set" else {"mode"}
        if set(verifier) != allowed:
            raise ValueError(f"verifier fields must be exactly {sorted(allowed)}")
        result = {"mode": mode}
        if mode == "set":
            elements = verifier.get("elements")
            if isinstance(elements, (str, bytes)) or not isinstance(elements, Sequence):
                raise ValueError("set elements must be an array")
            values = [str(item).strip() for item in elements]
            if len(values) > MAX_ITEMS or any(not item for item in values):
                raise ValueError("invalid set elements")
            if len(values) != len(set(values)):
                raise ValueError("set elements must be unique")
            result["elements"] = values
    if answer is not None:
        text = str(answer).strip()
        if not text:
            raise ValueError("reference answer must not be empty")
        if result["mode"] == "boolean" and canonical_boolean(text) is None:
            raise ValueError("boolean reference must be Yes/No")
    return result
