"""End-to-end concrete-problem evolution and training cycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .archive import ConcreteMapArchive
from .backends import PolicyBackend, TrainingBackend
from .config import AppConfig
from .discovery import DiscoveryRunner
from .frontier import (
    current_frontier_capacity,
    current_frontier_count,
    select_lagged_frontier,
)
from .grading import GraderClient
from .models import ParentPair, ProblemRecord
from .novelty import NoveltyIndex
from .pairing import select_parent_pairs
from .prompts import PromptBook
from .replay import ReplayBatchUnavailable, ReplayContractError, ScoreReplayBuffer
from .score_runner import ScoreRunner
from .seeds import load_seed_records
from .storage import PipelineState, RunStore
from .training import LaggedTrainingCandidate, ReplayTrainingBatch
from .utils import derived_seed, stable_id


@dataclass(slots=True)
class CycleSummary:
    iteration: int
    policy_version_before: int
    policy_version_after: int
    accepted_before: int
    accepted_after: int
    attempted_candidates: int
    archived_children: int
    current_frontier_count: int
    lagged_frontier_count: int
    update_applied: bool
    update_skip_reason: str | None
    replay_batch_id: str | None
    checkpoint: str | None
    waves: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvolutionEngine:
    """A single-writer engine over one resident policy/trainer."""

    def __init__(
        self,
        *,
        config: AppConfig,
        policy_backend: PolicyBackend,
        training_backend: TrainingBackend,
        manifest: dict[str, Any],
        project_root: str | Path,
        resume: bool,
    ) -> None:
        self.config = config
        self.policy_backend = policy_backend
        self.training_backend = training_backend
        self.manifest = manifest
        self.project_root = Path(project_root).resolve()
        self.store = RunStore(
            config.archive.output_dir,
            fsync_jsonl=config.archive.fsync_jsonl,
            store_rollout_text=config.archive.store_rollout_text,
        )
        self.store.initialize(manifest, resume=resume)
        self.prompts = PromptBook(self.project_root / "prompt_templates")
        self.grader = GraderClient()
        self.state = self.store.load_state()
        self.store.verify_checkpoint_event_prefix(self.state)
        self._recover_incomplete_phase(resume=resume)
        self.archive = ConcreteMapArchive(self.store.load_problems())
        if not self.archive.records:
            seed_path = Path(config.archive.seed_file)
            if not seed_path.is_absolute():
                seed_path = self.project_root / seed_path
            for record in load_seed_records(seed_path):
                if not self.store.append_problem(record):
                    raise RuntimeError(f"duplicate initial seed: {record.problem_id}")
                self.archive.add(record)
        for problem_id, score in self.store.load_scores():
            self.archive.apply_score(problem_id, score)
        self.novelty = NoveltyIndex(list(self.archive.records.values()))
        self.store.write_map_index(self.archive.to_index())
        resident_replay = getattr(training_backend, "replay_buffer", None)
        self.replay = resident_replay or ScoreReplayBuffer(
            expected_group_size=config.scoring.num_rollouts,
            expected_training_groups=config.frontier.training_batch_size,
        )
        if (
            self.replay.expected_group_size != config.scoring.num_rollouts
            or self.replay.expected_training_groups
            != config.frontier.training_batch_size
        ):
            raise ValueError(
                "resident training replay buffer differs from pipeline config"
            )
        self.discovery = DiscoveryRunner(
            config=config,
            backend=policy_backend,
            prompts=self.prompts,
            grader=self.grader,
            store=self.store,
            archive=self.archive,
            novelty=self.novelty,
        )
        self.scorer = ScoreRunner(
            config=config.scoring,
            backend=policy_backend,
            prompts=self.prompts,
            grader=self.grader,
            store=self.store,
            archive=self.archive,
        )
        identity = self.policy_backend.policy_identity
        if identity.run_uuid != str(manifest["run_uuid"]):
            raise ValueError(
                f"backend run_uuid {identity.run_uuid!r} differs from manifest "
                f"{manifest['run_uuid']!r}"
            )
        if (
            self.state.policy_version != identity.policy_version
            or self.state.global_step != identity.global_step
        ):
            raise ValueError(
                "pipeline state and resident policy identity disagree; load the "
                "checkpoint named by the run state before resuming"
            )

    def close(self) -> None:
        self.grader.close()
        close = getattr(self.training_backend, "close", None)
        if callable(close):
            close()

    def _recover_incomplete_phase(self, *, resume: bool) -> None:
        if self.state.phase == "ready":
            return
        if not resume:
            raise ValueError(f"run is not at a committed boundary: {self.state.phase}")
        if self.state.phase in {"policy_frozen", "scoring", "discovery", "batch_ready"}:
            # Native rollout payloads are intentionally not persisted. Never
            # rerun the same stochastic namespace: candidate, label, and score
            # IDs are deterministic per iteration, while regenerated text is
            # not. Preserve complete rows already appended, mark this cycle
            # aborted, and continue under a fresh iteration namespace.
            interrupted_iteration = self.state.iteration
            interrupted_phase = self.state.phase
            aborted = {
                "event_id": stable_id(
                    "training-abort",
                    self.manifest["run_uuid"],
                    interrupted_iteration,
                    self.state.active_cycle_id,
                    interrupted_phase,
                ),
                "iteration": interrupted_iteration,
                "status": "aborted",
                "reason": "interrupted_before_optimizer_commit",
                "interrupted_phase": interrupted_phase,
                "active_cycle_id": self.state.active_cycle_id,
                "active_training_batch_id": self.state.active_training_batch_id,
                "policy_version": self.state.policy_version,
                "global_step": self.state.global_step,
            }
            self.store.append_training_event(aborted)
            self.store.write_cycle_artifact(interrupted_iteration, "aborted", aborted)
            self.state.iteration = interrupted_iteration + 1
            self.state.phase = "ready"
            self.state.active_cycle_id = None
            self.state.active_training_batch_id = None
            self.store.save_state(self.state)
            return
        if self.state.phase == "checkpointed":
            # The model checkpoint is durable; only the final projection/state
            # commit was interrupted.
            self.state.iteration += 1
            self.state.phase = "ready"
            self.state.active_cycle_id = None
            self.state.active_training_batch_id = None
            self.store.save_state(self.state)
            return
        raise RuntimeError(
            "resume stopped at update_applied before a durable checkpoint; automatic "
            "retry could double-apply an optimizer step. Restore the preceding "
            "checkpoint and set the state to batch_ready explicitly after audit."
        )

    def _save_phase(self, phase: str, *, batch_id: str | None = None) -> None:
        self.state.phase = phase
        if batch_id is not None:
            self.state.active_training_batch_id = batch_id
        self.store.save_state(self.state)

    def _planned_pairs(
        self,
        *,
        iteration: int,
        wave_index: int,
        count: int,
    ) -> list[ParentPair]:
        name = f"parent_pairs_wave_{wave_index:02d}"
        saved = self.store.read_cycle_artifact(iteration, name)
        if saved is not None:
            pairs = [ParentPair.from_dict(row) for row in saved]
            if len(pairs) != count:
                raise RuntimeError(
                    f"persisted {name} has {len(pairs)} pairs; expected {count}"
                )
            return pairs
        pairs = select_parent_pairs(
            self.archive,
            count=count,
            seed=derived_seed(
                self.config.run.master_seed, "parent-pairs", iteration, wave_index
            ),
            require_distinct_lineages=self.config.generation.require_distinct_lineages,
        )
        self.store.write_cycle_artifact(
            iteration, name, [pair.to_dict() for pair in pairs]
        )
        return pairs

    def _score_initial_archive_and_lagged(
        self,
        *,
        iteration: int,
        frozen: Any,
        lagged: list[tuple[ProblemRecord, Any]],
    ) -> tuple[dict[str, Any], list[LaggedTrainingCandidate]]:
        lagged_ids = {record.problem_id for record, _ in lagged}
        targets: list[ProblemRecord] = [record for record, _ in lagged]
        selected_ids = set(lagged_ids)
        remeasure = self.archive.score_candidates_for_policy(
            policy_version=frozen.policy_version,
            budget=self.config.frontier.remeasure_budget,
            frontier_low=self.config.scoring.frontier_s_hat_low,
            frontier_high=self.config.scoring.frontier_s_hat_high,
        )
        for record in remeasure:
            if record.problem_id not in selected_ids:
                targets.append(record)
                selected_ids.add(record.problem_id)
        result = self.scorer.run(
            targets,
            iteration=iteration,
            frozen=frozen,
            replay_problem_ids=lagged_ids if self.config.training.enabled else set(),
        )
        for group in result.replay_groups.values():
            self.replay.store(group)
        prior_by_problem = {record.problem_id: score for record, score in lagged}
        candidates: list[LaggedTrainingCandidate] = []
        for record, prior in lagged:
            current = result.scores.get(record.problem_id)
            if current is None or record.problem_id not in result.replay_groups:
                continue
            candidates.append(
                LaggedTrainingCandidate(
                    problem_id=record.problem_id,
                    score_observation_id=current.observation_id,
                    domain=record.domain,
                    problem_type=record.problem_type,
                    current_s_hat=current.s_hat,
                    current_rq_score=current.rq_score,
                    score_iteration=current.iteration,
                    selection_rq_score=prior_by_problem[record.problem_id].rq_score,
                    selection_score_iteration=prior_by_problem[
                        record.problem_id
                    ].iteration,
                    selection_s_hat=prior_by_problem[record.problem_id].s_hat,
                )
            )
        return result.scores, candidates

    def run_cycle(self) -> CycleSummary:
        if self.state.phase != "ready":
            raise RuntimeError(f"cannot start cycle from phase {self.state.phase!r}")
        iteration = self.state.iteration
        frozen = self.policy_backend.policy_identity
        cycle_id = stable_id(
            "cycle",
            frozen.run_uuid,
            iteration,
            frozen.policy_version,
            frozen.global_step,
        )
        self.state.active_cycle_id = cycle_id
        self._save_phase("policy_frozen")
        accepted_before = len(self.archive.records)
        lagged = select_lagged_frontier(
            self.archive,
            iteration=iteration,
            selection_lag=self.config.frontier.selection_lag,
            batch_size=self.config.frontier.training_batch_size,
            low=self.config.scoring.frontier_s_hat_low,
            high=self.config.scoring.frontier_s_hat_high,
            max_per_cell=self.config.frontier.max_per_cell_per_batch,
            seed=derived_seed(self.config.run.master_seed, "frontier", iteration),
        )
        self.replay.begin_cycle(iteration=iteration, policy=frozen)
        self._save_phase("scoring")
        _, training_candidates = self._score_initial_archive_and_lagged(
            iteration=iteration, frozen=frozen, lagged=lagged
        )

        self._save_phase("discovery")
        attempted = 0
        archived_children = 0
        waves: list[dict[str, Any]] = []
        wave_index = 0
        while attempted < self.config.generation.max_candidates:
            candidates_this_wave = (
                self.config.generation.initial_candidates
                if wave_index == 0
                else self.config.generation.refill_candidates
            )
            candidates_this_wave = min(
                candidates_this_wave,
                self.config.generation.max_candidates - attempted,
            )
            pair_count = (
                candidates_this_wave // self.config.generation.children_per_pair
            )
            if pair_count <= 0:
                break
            pairs = self._planned_pairs(
                iteration=iteration,
                wave_index=wave_index,
                count=pair_count,
            )
            wave = self.discovery.run_wave(
                iteration=iteration,
                wave_index=wave_index,
                pairs=pairs,
                frozen=frozen,
            )
            attempted += pair_count * self.config.generation.children_per_pair
            archived_children += len(wave.archived)
            if wave.archived:
                self.scorer.run(
                    wave.archived,
                    iteration=iteration,
                    frozen=frozen,
                    replay_problem_ids=set(),
                )
            wave_row = wave.to_dict()
            wave_row["wave_index"] = wave_index
            wave_row["attempted_total"] = attempted
            waves.append(wave_row)
            current_capacity = current_frontier_capacity(
                self.archive,
                iteration=iteration,
                low=self.config.scoring.frontier_s_hat_low,
                high=self.config.scoring.frontier_s_hat_high,
                max_per_cell=self.config.frontier.max_per_cell_per_batch,
            )
            if (
                wave_index == 0
                and attempted < self.config.generation.initial_candidates
            ):
                raise RuntimeError("initial candidate wave violated its fixed budget")
            if current_capacity >= self.config.frontier.training_batch_size:
                break
            wave_index += 1

        update_applied = False
        skip_reason: str | None = None
        replay_batch_id: str | None = None
        checkpoint: str | None = None
        if not self.config.training.enabled:
            skip_reason = "training_disabled"
        elif len(lagged) != self.config.frontier.training_batch_size:
            skip_reason = "insufficient_lagged_frontier"
        elif len(training_candidates) != self.config.frontier.training_batch_size:
            skip_reason = "incomplete_current_remeasurement"
        else:
            try:
                batch = ReplayTrainingBatch.build(
                    buffer=self.replay,
                    candidates=training_candidates,
                    training_groups=self.config.frontier.training_batch_size,
                    selection_lag=self.config.frontier.selection_lag,
                    frontier_s_hat_low=self.config.scoring.frontier_s_hat_low,
                    frontier_s_hat_high=self.config.scoring.frontier_s_hat_high,
                    max_per_cell=self.config.frontier.max_per_cell_per_batch,
                )
            except (ReplayBatchUnavailable, ReplayContractError) as exc:
                skip_reason = f"replay_batch_unavailable:{exc}"
            else:
                replay_batch_id = batch.batch_id
                self._save_phase("batch_ready", batch_id=batch.batch_id)
                # Failures beyond this boundary propagate. Treating an
                # uncertain optimizer call as a benign skip could double-apply
                # it on resume.
                metrics = self.training_backend.apply_replay_batch(batch)
                after = self.policy_backend.policy_identity
                if after == frozen:
                    refresh = getattr(
                        self.training_backend, "refresh_policy_identity", None
                    )
                    if callable(refresh):
                        refresh()
                        after = self.policy_backend.policy_identity
                if after == frozen:
                    raise RuntimeError(
                        "optimizer returned but policy identity did not advance"
                    )
                update_applied = True
                self.state.policy_version = after.policy_version
                self.state.global_step = after.global_step
                self._save_phase("update_applied", batch_id=batch.batch_id)
                event = {
                    "event_id": stable_id("training", iteration, batch.batch_id),
                    "iteration": iteration,
                    "status": "applied",
                    "batch": batch.audit_dict(),
                    "metrics": metrics,
                }
                self.store.append_training_event(event)
                checkpoint = self.training_backend.save_checkpoint(
                    global_step=after.global_step,
                    pipeline_state=self.state.to_dict(),
                )
                self.state.checkpoint_step = after.global_step
                position, digest = self.store.event_position()
                self.state.checkpoint_event_offset = position
                self.state.checkpoint_event_hash = digest
                self.store.append_training_event(
                    {
                        "event_id": stable_id(
                            "training-checkpoint", iteration, batch.batch_id
                        ),
                        "iteration": iteration,
                        "status": "checkpointed",
                        "batch_id": batch.batch_id,
                        "global_step": after.global_step,
                        "checkpoint": checkpoint,
                    }
                )
                self._save_phase("checkpointed", batch_id=batch.batch_id)

        if not update_applied:
            self.store.append_training_event(
                {
                    "event_id": stable_id("training-skip", iteration, skip_reason),
                    "iteration": iteration,
                    "status": "skipped",
                    "reason": skip_reason,
                    "lagged_frontier_count": len(lagged),
                    "resident_replay_groups": len(self.replay.groups),
                }
            )
        current_count = current_frontier_count(
            self.archive,
            iteration=iteration,
            low=self.config.scoring.frontier_s_hat_low,
            high=self.config.scoring.frontier_s_hat_high,
        )
        after_identity = self.policy_backend.policy_identity
        summary = CycleSummary(
            iteration=iteration,
            policy_version_before=frozen.policy_version,
            policy_version_after=after_identity.policy_version,
            accepted_before=accepted_before,
            accepted_after=len(self.archive.records),
            attempted_candidates=attempted,
            archived_children=archived_children,
            current_frontier_count=current_count,
            lagged_frontier_count=len(lagged),
            update_applied=update_applied,
            update_skip_reason=skip_reason,
            replay_batch_id=replay_batch_id,
            checkpoint=checkpoint,
            waves=waves,
        )
        self.store.write_cycle_artifact(iteration, "summary", summary.to_dict())
        self.store.write_map_index(self.archive.to_index())
        self.replay.discard()
        self.state.iteration = iteration + 1
        self.state.phase = "ready"
        self.state.active_cycle_id = None
        self.state.active_training_batch_id = None
        self.state.policy_version = after_identity.policy_version
        self.state.global_step = after_identity.global_step
        self.store.save_state(self.state)
        return summary

    def run(self) -> list[CycleSummary]:
        summaries: list[CycleSummary] = []
        while (
            self.state.iteration < self.config.run.max_iterations
            and self.state.global_step < self.config.training.total_training_steps
        ):
            summaries.append(self.run_cycle())
        return summaries
