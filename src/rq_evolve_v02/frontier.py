"""Lagged, cell-balanced frontier selection.

Selection uses only the previous cycle's score.  The chosen problems are then
remeasured under the current frozen policy, and those new payloads are what the
optimizer consumes.  This separation prevents score-selection noise from
choosing the very rollout batch used for learning.
"""

from __future__ import annotations

from collections import defaultdict
import random

from .archive import ConcreteMapArchive
from .models import ProblemRecord, ScoreEvidence


def score_at_iteration(record: ProblemRecord, iteration: int) -> ScoreEvidence | None:
    matches = [score for score in record.score_history if score.iteration == iteration]
    if not matches:
        return None
    return max(matches, key=lambda score: (score.policy_version, score.observation_id))


def select_lagged_frontier(
    archive: ConcreteMapArchive,
    *,
    iteration: int,
    selection_lag: int,
    batch_size: int,
    low: float,
    high: float,
    max_per_cell: int,
    seed: int,
) -> list[tuple[ProblemRecord, ScoreEvidence]]:
    """Return a complete selection or an empty list; never a partial batch."""

    evidence_iteration = iteration - selection_lag
    if evidence_iteration < 0:
        return []
    buckets: dict[str, list[tuple[ProblemRecord, ScoreEvidence]]] = defaultdict(list)
    for record in archive.records.values():
        score = score_at_iteration(record, evidence_iteration)
        if score is not None and low < score.s_hat < high:
            buckets[record.cell].append((record, score))

    rng = random.Random(int(seed))
    for cell, rows in buckets.items():
        rng.shuffle(rows)
        rows.sort(key=lambda row: (-row[1].rq_score, row[0].problem_id))
        if max_per_cell > 0:
            buckets[cell] = rows[:max_per_cell]

    selected: list[tuple[ProblemRecord, ScoreEvidence]] = []
    cells = sorted(buckets)
    # Rotate the first cell deterministically so alphabetic order cannot
    # systematically win the last seat of a full batch.
    if cells:
        offset = rng.randrange(len(cells))
        cells = cells[offset:] + cells[:offset]
    while cells and len(selected) < batch_size:
        next_cells: list[str] = []
        for cell in cells:
            if buckets[cell] and len(selected) < batch_size:
                selected.append(buckets[cell].pop(0))
            if buckets[cell]:
                next_cells.append(cell)
        cells = next_cells
    if len(selected) != batch_size:
        return []
    if len({record.problem_id for record, _ in selected}) != batch_size:
        raise RuntimeError("frontier selector produced duplicate problem IDs")
    return selected


def current_frontier_count(
    archive: ConcreteMapArchive,
    *,
    iteration: int,
    low: float,
    high: float,
) -> int:
    return sum(
        1
        for record in archive.records.values()
        if (score := score_at_iteration(record, iteration)) is not None
        and low < score.s_hat < high
    )


def current_frontier_capacity(
    archive: ConcreteMapArchive,
    *,
    iteration: int,
    low: float,
    high: float,
    max_per_cell: int,
) -> int:
    """How many distinct current frontier rows a balanced batch may consume."""

    counts: dict[str, int] = {}
    for record in archive.records.values():
        score = score_at_iteration(record, iteration)
        if score is not None and low < score.s_hat < high:
            counts[record.cell] = counts.get(record.cell, 0) + 1
    return sum(min(value, max_per_cell) for value in counts.values())
