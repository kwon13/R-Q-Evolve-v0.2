"""Coverage-balanced, descriptor-blind parent pairing."""

from __future__ import annotations

import random

from .archive import ConcreteMapArchive
from .models import ParentPair
from .utils import stable_id


def select_parent_pairs(
    archive: ConcreteMapArchive,
    *,
    count: int,
    seed: int,
    require_distinct_lineages: bool = True,
) -> list[ParentPair]:
    occupied = archive.occupied_cells()
    if len(archive.records) < 2 or not occupied:
        raise ValueError("at least two archived problems are required for crossover")
    rng = random.Random(int(seed))
    result: list[ParentPair] = []
    all_ids = sorted(archive.records)
    for index in range(int(count)):
        left_cell = rng.choice(occupied)
        left_id = rng.choice(archive.cells[left_cell])
        preferred_cells = [key for key in occupied if key != left_cell]
        rng.shuffle(preferred_cells)
        candidate_ids: list[str] = []
        for key in preferred_cells + [left_cell]:
            ids = list(archive.cells[key])
            rng.shuffle(ids)
            candidate_ids.extend(ids)
        if require_distinct_lineages:
            left_roots = archive.lineage_roots(left_id)
            candidate_ids = [
                pid
                for pid in candidate_ids
                if pid != left_id and left_roots.isdisjoint(archive.lineage_roots(pid))
            ]
        else:
            candidate_ids = [pid for pid in candidate_ids if pid != left_id]
        if not candidate_ids:
            candidate_ids = [pid for pid in all_ids if pid != left_id]
        right_id = candidate_ids[0] if candidate_ids else rng.choice(all_ids)
        pair_seed = rng.randrange(0, 2**31)
        result.append(
            ParentPair(
                pair_id=stable_id("pair", seed, index, left_id, right_id),
                left_id=left_id,
                right_id=right_id,
                prompt_seed=pair_seed,
            )
        )
    return result
