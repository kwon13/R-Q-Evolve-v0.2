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


def question_similarity(
    left: str, right: str, *, min_shared_shingles: int = 8
) -> dict[str, float | int]:
    """Return deterministic lexical similarity metrics for two questions."""

    sequence = SequenceMatcher(
        None,
        normalized_question(left),
        normalized_question(right),
        autojunk=False,
    ).ratio()
    left_shingles = _shingles(template_question(left))
    right_shingles = _shingles(template_question(right))
    shared_shingles = len(left_shingles & right_shingles)
    shingle_jaccard = _jaccard(left_shingles, right_shingles)
    shingle_overlap = (
        shared_shingles / min(len(left_shingles), len(right_shingles))
        if left_shingles and right_shingles
        else 0.0
    )
    guarded_overlap = (
        shingle_overlap if shared_shingles >= min_shared_shingles else 0.0
    )
    return {
        "sequence": sequence,
        "shingle_jaccard": shingle_jaccard,
        "shingle_overlap": shingle_overlap,
        "shared_shingles": shared_shingles,
        "maximum": max(sequence, shingle_jaccard, guarded_overlap),
    }


@dataclass(slots=True)
class NoveltyDecision:
    accepted: bool
    reason: str | None
    nearest_problem_id: str | None
    nearest_similarity: float
    parent_max_similarity: float
    parent_max_containment: float
    parent_max_containment_shared_shingles: int
    parent_similarities: list[float]
    parent_containments: list[float]
    parent_shared_shingles: list[int]

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
        parent_containment_ceiling: float,
        parent_containment_min_shared_shingles: int,
    ) -> NoveltyDecision:
        text = normalized_question(question)
        candidate_shingles = _shingles(template_question(question))
        parent_similarities: list[float] = []
        parent_containments: list[float] = []
        parent_shared_shingles: list[int] = []
        for parent in parent_questions:
            parent_shingles = _shingles(template_question(parent))
            shared = len(candidate_shingles & parent_shingles)
            parent_similarities.append(
                SequenceMatcher(
                    None, text, normalized_question(parent), autojunk=False
                ).ratio()
            )
            parent_containments.append(
                shared / len(parent_shingles) if parent_shingles else 0.0
            )
            parent_shared_shingles.append(shared)
        parent_max_similarity = max(parent_similarities, default=0.0)
        if parent_containments:
            containment_parent = max(
                range(len(parent_containments)),
                key=lambda index: (
                    parent_containments[index],
                    parent_shared_shingles[index],
                    parent_similarities[index],
                ),
            )
            parent_max_containment = parent_containments[containment_parent]
            parent_containment_shared = parent_shared_shingles[containment_parent]
        else:
            parent_max_containment = 0.0
            parent_containment_shared = 0

        def decision(
            accepted: bool,
            reason: str | None,
            nearest_problem_id: str | None,
            nearest_similarity: float,
        ) -> NoveltyDecision:
            return NoveltyDecision(
                accepted=accepted,
                reason=reason,
                nearest_problem_id=nearest_problem_id,
                nearest_similarity=nearest_similarity,
                parent_max_similarity=parent_max_similarity,
                parent_max_containment=parent_max_containment,
                parent_max_containment_shared_shingles=(
                    parent_containment_shared
                ),
                parent_similarities=parent_similarities,
                parent_containments=parent_containments,
                parent_shared_shingles=parent_shared_shingles,
            )

        if text in self._exact:
            return decision(False, "exact_duplicate", self._exact[text], 1.0)

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
        if any(
            containment >= parent_containment_ceiling
            and shared >= parent_containment_min_shared_shingles
            for containment, shared in zip(
                parent_containments, parent_shared_shingles, strict=True
            )
        ):
            return decision(False, "parent_containment", nearest_id, nearest)
        if parent_max_similarity >= parent_ceiling:
            return decision(False, "parent_copy", nearest_id, nearest)
        if nearest >= near_threshold:
            return decision(False, "near_duplicate", nearest_id, nearest)
        return decision(True, None, nearest_id, nearest)
