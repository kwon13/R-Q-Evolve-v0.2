"""Scalable exact, template, near-duplicate, and parent-copy rejection."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from .models import ProblemRecord
from .utils import canonical_text

_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:/\d+)?")
_WORD_RE = re.compile(r"[A-Za-z]+|\d+|[^\w\s]", re.UNICODE)


def normalized_question(question: str) -> str:
    return canonical_text(question).lower()


def template_question(question: str) -> str:
    return _NUMBER_RE.sub("#", normalized_question(question))


def _shingles(text: str, width: int = 3) -> set[str]:
    tokens = _WORD_RE.findall(text)
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


def _simhash(shingles: set[str]) -> int:
    weights = [0] * 64
    for item in shingles:
        value = int.from_bytes(hashlib.sha256(item.encode()).digest()[:8], "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


@dataclass(slots=True)
class NoveltyDecision:
    accepted: bool
    reason: str | None
    nearest_problem_id: str | None
    nearest_similarity: float
    parent_max_similarity: float

    def to_dict(self) -> dict:
        return asdict(self)


class NoveltyIndex:
    def __init__(self, records: list[ProblemRecord] | None = None) -> None:
        self._exact: dict[str, str] = {}
        self._templates: dict[str, set[str]] = {}
        self._texts: dict[str, str] = {}
        self._shingles: dict[str, set[str]] = {}
        self._bands: dict[tuple[int, int], set[str]] = {}
        for record in records or []:
            self.add(record.problem_id, record.question)

    def add(self, problem_id: str, question: str) -> None:
        text = normalized_question(question)
        template = template_question(question)
        shingles = _shingles(template)
        signature = _simhash(shingles)
        self._exact[text] = problem_id
        self._templates.setdefault(template, set()).add(problem_id)
        self._texts[problem_id] = text
        self._shingles[problem_id] = shingles
        for band in range(4):
            key = (band, (signature >> (band * 16)) & 0xFFFF)
            self._bands.setdefault(key, set()).add(problem_id)

    def _shortlist(self, question: str) -> set[str]:
        template = template_question(question)
        shingles = _shingles(template)
        signature = _simhash(shingles)
        result = set(self._templates.get(template, set()))
        for band in range(4):
            result.update(
                self._bands.get((band, (signature >> (band * 16)) & 0xFFFF), set())
            )
        return result

    def check(
        self,
        question: str,
        *,
        parent_questions: list[str],
        near_threshold: float,
        parent_ceiling: float,
    ) -> NoveltyDecision:
        text = normalized_question(question)
        if text in self._exact:
            return NoveltyDecision(
                False, "exact_duplicate", self._exact[text], 1.0, 1.0
            )

        candidate_shingles = _shingles(template_question(question))
        nearest_id = None
        nearest = 0.0
        for problem_id in self._shortlist(question):
            jac = _jaccard(candidate_shingles, self._shingles[problem_id])
            if jac < max(0.0, near_threshold - 0.2):
                continue
            ratio = SequenceMatcher(
                None, text, self._texts[problem_id], autojunk=False
            ).ratio()
            similarity = max(jac, ratio)
            if similarity > nearest:
                nearest, nearest_id = similarity, problem_id
        parent_max = max(
            (
                SequenceMatcher(
                    None, text, normalized_question(parent), autojunk=False
                ).ratio()
                for parent in parent_questions
            ),
            default=0.0,
        )
        if parent_max >= parent_ceiling:
            return NoveltyDecision(
                False, "parent_copy", nearest_id, nearest, parent_max
            )
        if nearest >= near_threshold:
            return NoveltyDecision(
                False, "near_duplicate", nearest_id, nearest, parent_max
            )
        return NoveltyDecision(True, None, nearest_id, nearest, parent_max)
