"""Exact frontier selection and no-padding replay training rows."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Sequence

from .backends import PolicyIdentity
from .replay import (
    ReplayBatchUnavailable,
    ReplayContractError,
    ScoreReplayBuffer,
    ScoreReplayGroup,
    content_hash,
)


@dataclass(frozen=True, slots=True)
class LaggedTrainingCandidate:
    """A current score payload selected by an earlier R_Q observation.

    ``score_observation_id`` names the *current-policy* rollouts that will be
    replayed. ``selection_rq_score`` comes from an earlier iteration and decides
    whether the problem enters the batch.  Keeping the fields distinct prevents
    accidental same-noise selection.
    """

    problem_id: str
    score_observation_id: str
    domain: str
    problem_type: str
    current_s_hat: float
    current_rq_score: float
    score_iteration: int
    selection_rq_score: float
    selection_score_iteration: int
    selection_s_hat: float | None = None

    @property
    def cell(self) -> str:
        return f"{self.domain}::{self.problem_type}"

    def validate(self, *, selection_lag: int) -> None:
        if not self.problem_id or not self.score_observation_id:
            raise ReplayContractError(
                "training candidate identifiers must be non-empty"
            )
        if not 0.0 <= float(self.current_s_hat) <= 1.0:
            raise ReplayContractError("current_s_hat must be in [0, 1]")
        if selection_lag == 1 and not (
            int(self.selection_score_iteration) < int(self.score_iteration)
        ):
            raise ReplayContractError(
                "lagged selection must use an observation earlier than the "
                "current replay payload"
            )
        if selection_lag == 1 and self.selection_s_hat is None:
            raise ReplayContractError(
                "lagged frontier selection requires selection_s_hat from the "
                "earlier score observation"
            )
        if selection_lag == 0 and (
            int(self.selection_score_iteration) != int(self.score_iteration)
        ):
            raise ReplayContractError(
                "zero-lag selection must name the current score iteration"
            )


@dataclass(frozen=True, slots=True)
class SelectedReplayGroup:
    group: ScoreReplayGroup
    candidate: LaggedTrainingCandidate


@dataclass(slots=True)
class ReplayTrainingBatch:
    """Exactly one optimizer step: 32 distinct problems x fixed G responses."""

    batch_id: str
    iteration: int
    policy: PolicyIdentity
    groups: tuple[SelectedReplayGroup, ...]
    group_size: int
    training_groups: int = 32

    @classmethod
    def build(
        cls,
        *,
        buffer: ScoreReplayBuffer,
        candidates: Sequence[LaggedTrainingCandidate],
        training_groups: int = 32,
        selection_lag: int = 1,
        frontier_s_hat_low: float = 0.1,
        frontier_s_hat_high: float = 0.9,
        max_per_cell: int = 8,
    ) -> "ReplayTrainingBatch":
        policy = buffer._require_open()
        selected = select_cell_balanced_frontier(
            candidates,
            buffer=buffer,
            training_groups=training_groups,
            selection_lag=selection_lag,
            frontier_s_hat_low=frontier_s_hat_low,
            frontier_s_hat_high=frontier_s_hat_high,
            max_per_cell=max_per_cell,
        )
        observation_ids = [item.score_observation_id for item in selected]
        resident = buffer.exact_groups(observation_ids, expected_count=training_groups)
        by_observation = {item.score_observation_id: item for item in selected}
        pairs = tuple(
            SelectedReplayGroup(
                group=group,
                candidate=by_observation[group.key.score_observation_id],
            )
            for group in resident
        )
        batch_id = content_hash(
            {
                "kind": "replay_training_batch",
                "iteration": buffer.iteration,
                "policy": (
                    buffer.policy and buffer.policy.__dict__
                    if hasattr(buffer.policy, "__dict__")
                    else {
                        "run_uuid": policy.run_uuid,
                        "policy_version": policy.policy_version,
                        "adapter_version": policy.adapter_version,
                        "global_step": policy.global_step,
                        "source_checkpoint": policy.source_checkpoint,
                    }
                ),
                "score_observation_ids": observation_ids,
            }
        )
        batch = cls(
            batch_id=batch_id,
            iteration=buffer.iteration,
            policy=policy,
            groups=pairs,
            group_size=buffer.expected_group_size,
            training_groups=int(training_groups),
        )
        batch.validate()
        buffer.authorize_batch(
            batch_id=batch.batch_id,
            observation_ids=batch.score_observation_ids,
        )
        return batch

    def validate(self) -> None:
        if len(self.groups) != self.training_groups:
            raise ReplayBatchUnavailable(
                f"training batch has {len(self.groups)} groups; "
                f"requires exactly {self.training_groups}"
            )
        observations = [item.group.key.score_observation_id for item in self.groups]
        problems = [item.group.key.problem_id for item in self.groups]
        if len(set(observations)) != self.training_groups:
            raise ReplayContractError("training batch repeats a score observation")
        if len(set(problems)) != self.training_groups:
            raise ReplayContractError("training batch repeats a concrete problem")
        for item in self.groups:
            item.group.validate()
            if item.group.key.policy != self.policy:
                raise ReplayContractError("training batch mixes policy versions")
            if item.group.key.score_iteration != self.iteration:
                raise ReplayContractError("training batch mixes score iterations")
            if item.group.key.group_size != self.group_size:
                raise ReplayContractError("training batch mixes rollout group sizes")
            if item.candidate.problem_id != item.group.key.problem_id:
                raise ReplayContractError("selection record points to another problem")
            if (
                item.candidate.score_observation_id
                != item.group.key.score_observation_id
            ):
                raise ReplayContractError("selection record points to another score")

    @property
    def score_observation_ids(self) -> tuple[str, ...]:
        return tuple(item.group.key.score_observation_id for item in self.groups)

    def audit_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "iteration": self.iteration,
            "policy": {
                "run_uuid": self.policy.run_uuid,
                "policy_version": self.policy.policy_version,
                "adapter_version": self.policy.adapter_version,
                "global_step": self.policy.global_step,
                "source_checkpoint": self.policy.source_checkpoint,
            },
            "training_groups": self.training_groups,
            "group_size": self.group_size,
            "score_observation_ids": list(self.score_observation_ids),
            "problem_ids": [item.group.key.problem_id for item in self.groups],
        }


def select_cell_balanced_frontier(
    candidates: Sequence[LaggedTrainingCandidate],
    *,
    buffer: ScoreReplayBuffer,
    training_groups: int,
    selection_lag: int,
    frontier_s_hat_low: float,
    frontier_s_hat_high: float,
    max_per_cell: int,
) -> list[LaggedTrainingCandidate]:
    """Select exactly one current payload per problem, without padding.

    Candidates are filtered and ranked only by the earlier score observation.
    The current score exists to provide an on-policy payload, not to decide
    whether its own noise earns selection.  Round-robin across non-empty cells
    avoids one dense cell consuming the batch. If fewer than
    ``training_groups`` survive, the caller must skip the optimizer update.
    """

    training_groups = int(training_groups)
    max_per_cell = int(max_per_cell)
    if training_groups < 1 or max_per_cell < 1:
        raise ValueError("training_groups and max_per_cell must be positive")
    if selection_lag not in {0, 1}:
        raise ValueError("selection_lag must be 0 or 1")
    low = float(frontier_s_hat_low)
    high = float(frontier_s_hat_high)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("frontier band must satisfy 0 <= low < high <= 1")

    # Reject contradictory duplicates instead of whichever happens to arrive
    # last.  Archive order must not alter the training batch.
    by_observation: dict[str, LaggedTrainingCandidate] = {}
    problem_to_observation: dict[str, str] = {}
    for candidate in candidates:
        candidate.validate(selection_lag=selection_lag)
        selection_s_hat = (
            candidate.selection_s_hat if selection_lag == 1 else candidate.current_s_hat
        )
        if selection_s_hat is None or not (low < float(selection_s_hat) < high):
            continue
        try:
            group = buffer.get(candidate.score_observation_id)
        except ReplayContractError:
            continue
        if group.key.problem_id != candidate.problem_id:
            raise ReplayContractError(
                "candidate and resident replay payload disagree on problem_id"
            )
        existing = by_observation.get(candidate.score_observation_id)
        if existing is not None and existing != candidate:
            raise ReplayContractError(
                "one score_observation_id has contradictory selection metadata"
            )
        prior_observation = problem_to_observation.get(candidate.problem_id)
        if (
            prior_observation is not None
            and prior_observation != candidate.score_observation_id
        ):
            raise ReplayContractError(
                "one concrete problem has multiple current score payloads"
            )
        by_observation[candidate.score_observation_id] = candidate
        problem_to_observation[candidate.problem_id] = candidate.score_observation_id

    per_cell: dict[str, list[LaggedTrainingCandidate]] = defaultdict(list)
    for candidate in by_observation.values():
        per_cell[candidate.cell].append(candidate)
    for rows in per_cell.values():
        rows.sort(
            key=lambda value: (
                -float(value.selection_rq_score),
                value.problem_id,
                value.score_observation_id,
            )
        )

    # Start denser/higher-priority cells first but take at most one item from a
    # cell per round.  Ties are stable and independent of input order.
    cells = sorted(
        per_cell,
        key=lambda cell: (
            -float(per_cell[cell][0].selection_rq_score),
            cell,
        ),
    )
    queues = {cell: deque(per_cell[cell]) for cell in cells}
    taken_per_cell: dict[str, int] = defaultdict(int)
    selected: list[LaggedTrainingCandidate] = []
    while len(selected) < training_groups:
        progressed = False
        for cell in cells:
            queue = queues[cell]
            if not queue or taken_per_cell[cell] >= max_per_cell:
                continue
            selected.append(queue.popleft())
            taken_per_cell[cell] += 1
            progressed = True
            if len(selected) == training_groups:
                break
        if not progressed:
            break
    if len(selected) != training_groups:
        raise ReplayBatchUnavailable(
            f"frontier has {len(selected)} eligible distinct resident groups; "
            f"requires exactly {training_groups}. Skip this optimizer update; "
            "do not modulo-pad or resample."
        )
    return selected


class ExactReplayDataset:
    """Mutable VERL dataset whose length is either zero or one exact batch.

    There is deliberately no ``min_size``, wraparound, or modulo indexing.
    Repeating a row here repeats the exact G responses and changes its weight.
    """

    def __init__(self, *, data_source: str = "rq_evolve_v02_score_replay") -> None:
        self.data_source = str(data_source)
        self._batch: ReplayTrainingBatch | None = None

    @property
    def batch(self) -> ReplayTrainingBatch | None:
        return self._batch

    def stage(self, batch: ReplayTrainingBatch) -> None:
        batch.validate()
        self._batch = batch

    def clear(self) -> None:
        self._batch = None

    def __len__(self) -> int:
        return 0 if self._batch is None else self._batch.training_groups

    def __getitem__(self, index: int) -> dict[str, Any]:
        batch = self._batch
        if batch is None:
            raise IndexError("no exact replay batch is staged")
        if isinstance(index, bool) or not 0 <= int(index) < len(batch.groups):
            raise IndexError(index)
        item = batch.groups[int(index)]
        group = item.group
        candidate = item.candidate
        extra = group.replay_metadata()
        extra.update(
            {
                "batch_id": batch.batch_id,
                "domain": candidate.domain,
                "problem_type": candidate.problem_type,
                "current_s_hat": float(candidate.current_s_hat),
                "current_rq_score": float(candidate.current_rq_score),
                "selection_rq_score": float(candidate.selection_rq_score),
                "selection_score_iteration": int(candidate.selection_score_iteration),
            }
        )
        try:
            import torch

            dummy = torch.tensor([0], dtype=torch.uint8)
        except ImportError:  # keeps CPU contract tests independent of torch
            dummy = 0
        return {
            "raw_prompt": [dict(message) for message in group.messages],
            "dummy_tensor": dummy,
            "data_source": self.data_source,
            "reward_model": {
                "ground_truth": group.answer,
                "verifier": dict(group.verifier),
            },
            "extra_info": extra,
            "index": int(index),
        }


class ResidentReplayDataset(ExactReplayDataset):
    """Blocking one-batch dataset used by a continuously resident trainer.

    The VERL ``fit`` loop runs once in a background driver thread. Between
    optimizer steps its dataloader blocks in ``__getitem__`` until the pipeline
    submits the next exact batch. ``dataloader_num_workers`` must be zero so the
    condition and native payload lifetime stay in this process.
    """

    def __init__(
        self,
        *,
        training_groups: int = 32,
        data_source: str = "rq_evolve_v02_score_replay",
    ) -> None:
        super().__init__(data_source=data_source)
        self.training_groups = int(training_groups)
        if self.training_groups < 1:
            raise ValueError("training_groups must be positive")
        self._condition = threading.Condition()
        self._waiting = False
        self._closed = False
        self._completed_batch_id: str | None = None
        self._issued_batch_id: str | None = None
        self._served_indices: set[int] = set()
        self._checkpoint_barrier_id: str | None = None

    def __len__(self) -> int:
        # VERL builds a fixed, drop-last dataloader before the first pipeline
        # cycle. Expose exactly one full batch even while no payload is staged;
        # __getitem__ blocks instead of manufacturing a placeholder row.
        return self.training_groups

    def stage(self, batch: ReplayTrainingBatch) -> None:
        batch.validate()
        if batch.training_groups != self.training_groups:
            raise ReplayContractError(
                f"resident dataset expects {self.training_groups} groups, got "
                f"{batch.training_groups}"
            )
        with self._condition:
            if self._closed:
                raise ReplayContractError("resident replay dataset is closed")
            if self._batch is not None:
                raise ReplayContractError("a replay batch is already staged")
            if self._issued_batch_id is not None:
                raise ReplayContractError(
                    "the previous replay batch was issued but the optimizer has "
                    "not requested its next batch yet"
                )
            self._completed_batch_id = None
            self._served_indices.clear()
            self._batch = batch
            self._condition.notify_all()

    def clear(self) -> None:
        with self._condition:
            self._batch = None
            self._condition.notify_all()

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self._condition:
            while self._batch is None and not self._closed:
                # Completion is published by the checkpoint-manager weight-sync
                # wrapper, before StatefulDataLoader can advance into this next
                # request. Reaching here with an issued id means that barrier
                # was not installed or did not fire: fail closed rather than
                # checkpoint an in-flight next-epoch cursor.
                if self._issued_batch_id is not None:
                    raise ReplayContractError(
                        "next replay fetch began before the post-update weight-sync "
                        "checkpoint barrier"
                    )
                self._waiting = True
                self._condition.notify_all()
                self._condition.wait()
            self._waiting = False
            if self._closed:
                raise RuntimeError("resident replay dataset closed")
            row = super().__getitem__(index)
            normalized_index = int(index)
            if normalized_index in self._served_indices:
                raise ReplayContractError(
                    f"resident dataloader requested index {normalized_index} twice"
                )
            self._served_indices.add(normalized_index)
            if len(self._served_indices) == self.training_groups:
                assert self._batch is not None
                self._issued_batch_id = self._batch.batch_id
                self._batch = None
                self._served_indices.clear()
                self._condition.notify_all()
            return row

    def wait_until_blocked(self, *, timeout_s: float = 300.0) -> None:
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while not self._waiting and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "resident VERL trainer did not reach the replay dataloader"
                    )
                self._condition.wait(min(remaining, 1.0))

    def complete_after_weight_sync(self) -> bool:
        """Publish completion at exact policy/data high-water mark K.

        Called by the resident backend's wrapper immediately after VERL's live
        actor-to-rollout weight sync and before logging, ``global_steps += 1``,
        or any next DataLoader fetch. Returns False for non-training syncs.
        """

        with self._condition:
            if self._issued_batch_id is None:
                return False
            self._completed_batch_id = self._issued_batch_id
            self._issued_batch_id = None
            self._checkpoint_barrier_id = self._completed_batch_id
            self._condition.notify_all()
            while self._checkpoint_barrier_id is not None and not self._closed:
                self._condition.wait()
            return True

    def on_batch_end(self, *, batch: Any) -> None:
        # The weight-sync barrier must already have completed and been released
        # by the external checkpoint before VERL reaches this callback.
        with self._condition:
            if self._issued_batch_id is not None:
                raise ReplayContractError(
                    "VERL reached on_batch_end without the weight-sync barrier"
                )
            self._condition.notify_all()

    def release_after_checkpoint(self, batch_id: str) -> None:
        with self._condition:
            if self._checkpoint_barrier_id is None:
                # The final configured VERL step returns before on_batch_end;
                # its clean fit-thread exit is the completion signal and there
                # is no callback to release.
                return
            if self._checkpoint_barrier_id != str(batch_id):
                raise ReplayContractError(
                    "checkpoint release names a different replay batch"
                )
            self._checkpoint_barrier_id = None
            self._condition.notify_all()

    def completed(self, batch_id: str) -> bool:
        with self._condition:
            return self._completed_batch_id == str(batch_id)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._batch = None
            self._issued_batch_id = None
            self._served_indices.clear()
            self._checkpoint_barrier_id = None
            self._condition.notify_all()


@dataclass(slots=True)
class ReplayUpdateGuard:
    """Verify that a resident trainer consumed exactly one replayed batch."""

    buffer: ScoreReplayBuffer
    dataset: ExactReplayDataset
    batch: ReplayTrainingBatch
    _served_before: int = field(init=False, default=0)

    def __enter__(self) -> "ReplayUpdateGuard":
        if self.buffer.consumed:
            raise ReplayContractError("cannot reuse a consumed replay cycle")
        self.batch.validate()
        self.dataset.stage(self.batch)
        self._served_before = self.buffer.stats.served_batches
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        served = self.buffer.stats.served_batches - self._served_before
        self.dataset.clear()
        if exc_type is not None:
            # The caller must stop/resume from the last checkpoint; retrying an
            # uncertain optimizer call could double-apply a partial update.
            return False
        if served != 1:
            raise ReplayContractError(
                f"optimizer step returned but replay hook served {served} batches; "
                "expected exactly one"
            )
        self.buffer.mark_consumed()
        return False
