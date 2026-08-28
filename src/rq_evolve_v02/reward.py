"""Typed VERL reward function shared with labeling and R_Q scoring."""

from __future__ import annotations

import os
import queue
from typing import Any

from rq_evolve_v02.grading import GraderClient
from rq_evolve_v02.output_parser import extract_last_boxed


class _Pool:
    def __init__(self, size: int) -> None:
        self.clients: queue.Queue[GraderClient] = queue.Queue()
        for _ in range(max(1, int(size))):
            self.clients.put(
                GraderClient(
                    timeout_s=float(os.environ.get("RQ_V02_GRADE_TIMEOUT", "8"))
                )
            )

    def grade(self, prediction: str, gold: str, verifier: dict | None) -> bool:
        client = self.clients.get()
        try:
            return client.grade(prediction, gold, verifier)
        finally:
            self.clients.put(client)


_GRADERS = _Pool(int(os.environ.get("RQ_V02_GRADE_WORKERS", "4")))


def _score_one(response: str, ground_truth: str, extra_info: Any) -> dict[str, float]:
    predicted = extract_last_boxed(str(response))
    verifier = extra_info.get("verifier") if isinstance(extra_info, dict) else None
    correct = bool(
        predicted is not None and _GRADERS.grade(predicted, str(ground_truth), verifier)
    )
    return {
        "score": float(correct),
        "overall": float(correct),
        "accuracy": float(correct),
        "format": float(predicted is not None),
    }


def compute_score(
    data_source: Any = None,
    solution_str: Any = None,
    ground_truth: Any = None,
    extra_info: Any = None,
    *,
    response_str_list: list[str] | None = None,
    ground_truth_list: list[str] | None = None,
    solution_strs: list[str] | None = None,
    ground_truths: list[str] | None = None,
    extra_infos: list[Any] | None = None,
    **_: Any,
) -> dict[str, float] | list[dict[str, float]]:
    """Support current single-row and legacy/batched VERL reward calls."""

    if (
        isinstance(data_source, (list, tuple))
        and isinstance(solution_str, (list, tuple))
        and ground_truth is None
    ):
        response_str_list = list(data_source)
        ground_truth_list = list(solution_str)
        solution_str = None
    responses = solution_strs if solution_strs is not None else response_str_list
    truths = ground_truths if ground_truths is not None else ground_truth_list
    if responses is not None or truths is not None:
        if responses is None or truths is None or len(responses) != len(truths):
            raise ValueError(
                "batch reward requires aligned responses and ground truths"
            )
        infos = extra_infos if extra_infos is not None else [None] * len(responses)
        if len(infos) != len(responses):
            raise ValueError("batch reward extra_infos must align with responses")
        return [
            _score_one(response, truth, info)
            for response, truth, info in zip(responses, truths, infos, strict=True)
        ]
    if solution_str is None or ground_truth is None:
        raise ValueError("compute_score requires solution_str and ground_truth")
    return _score_one(str(solution_str), str(ground_truth), extra_info)
