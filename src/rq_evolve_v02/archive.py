"""All-accepted concrete-problem MAP and balanced parent/frontier views."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .concepts import DOMAINS, PROBLEM_TYPES, cell_key
from .models import ProblemRecord, ScoreEvidence


class ConcreteMapArchive:
    def __init__(self, records: Iterable[ProblemRecord] = ()) -> None:
        self.records: dict[str, ProblemRecord] = {}
        self.cells: dict[str, list[str]] = {
            cell_key(domain, problem_type): []
            for domain in DOMAINS
            for problem_type in PROBLEM_TYPES
        }
        for record in records:
            self.add(record)

    def add(self, record: ProblemRecord) -> bool:
        if record.problem_id in self.records:
            return False
        self.records[record.problem_id] = record
        self.cells[record.cell].append(record.problem_id)
        return True

    def apply_score(self, problem_id: str, score: ScoreEvidence) -> None:
        record = self.records.get(problem_id)
        if record is None:
            raise KeyError(f"score references unknown problem: {problem_id}")
        prior = [
            item
            for item in record.score_history
            if not (
                item.iteration == score.iteration
                and item.policy_version == score.policy_version
            )
        ]
        prior.append(score)
        prior.sort(key=lambda x: (x.iteration, x.policy_version))
        record.score_history = prior

    def occupied_cells(self) -> list[str]:
        return [key for key, ids in self.cells.items() if ids]

    def representatives(self) -> dict[str, str | None]:
        result: dict[str, str | None] = {}
        for key, ids in self.cells.items():
            if not ids:
                result[key] = None
                continue
            result[key] = max(
                ids,
                key=lambda pid: (
                    (
                        self.records[pid].latest_score.rq_score
                        if self.records[pid].latest_score
                        else -1.0
                    ),
                    -self.records[pid].created_iteration,
                    pid,
                ),
            )
        return result

    def to_index(self) -> dict:
        return {
            "schema_version": "concrete-problem-map-v1",
            "axes": ["domain", "problem_type"],
            "domains": list(DOMAINS),
            "problem_types": list(PROBLEM_TYPES),
            "accepted_count": len(self.records),
            "occupied_count": len(self.occupied_cells()),
            "cells": {key: list(ids) for key, ids in self.cells.items()},
            "representatives": self.representatives(),
        }

    def lineage_roots(self, problem_id: str) -> frozenset[str]:
        record = self.records[problem_id]
        return frozenset(record.lineage_root_ids or (record.problem_id,))

    def eligible_previous_score(
        self,
        record: ProblemRecord,
        *,
        iteration: int,
        selection_lag: int,
    ) -> ScoreEvidence | None:
        cutoff = iteration - selection_lag
        candidates = [
            score for score in record.score_history if score.iteration <= cutoff
        ]
        return candidates[-1] if candidates else None

    def score_candidates_for_policy(
        self,
        *,
        policy_version: int,
        budget: int,
        frontier_low: float,
        frontier_high: float,
    ) -> list[ProblemRecord]:
        """Bounded, cell-balanced remeasurement view for the current policy."""

        if budget <= 0:
            return []
        buckets: dict[str, list[ProblemRecord]] = defaultdict(list)
        for record in self.records.values():
            latest = record.latest_score
            current = latest is not None and latest.policy_version == policy_version
            if current:
                continue
            buckets[record.cell].append(record)
        for records in buckets.values():
            records.sort(
                key=lambda rec: (
                    (
                        0
                        if rec.latest_score
                        and frontier_low < rec.latest_score.s_hat < frontier_high
                        else 1
                    ),
                    rec.latest_score.policy_version if rec.latest_score else -1,
                    rec.created_iteration,
                    rec.problem_id,
                )
            )
        selected: list[ProblemRecord] = []
        keys = sorted(buckets)
        while keys and len(selected) < budget:
            next_keys: list[str] = []
            for key in keys:
                if buckets[key] and len(selected) < budget:
                    selected.append(buckets[key].pop(0))
                if buckets[key]:
                    next_keys.append(key)
            keys = next_keys
        return selected
