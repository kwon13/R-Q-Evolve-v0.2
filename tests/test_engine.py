from __future__ import annotations

from pathlib import Path

from rq_evolve_v02.backends import PolicyIdentity
from rq_evolve_v02.config import AppConfig
from rq_evolve_v02.engine import EvolutionEngine
from rq_evolve_v02.mock_backend import DeterministicMockBackend


def test_two_cycle_discovery_then_exact_lagged_update(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = AppConfig()
    config.backend.kind = "mock"
    config.backend.model_path = "mock://model"
    config.backend.n_gpus = 1
    config.archive.output_dir = str(tmp_path / "run")
    config.archive.seed_file = str(root / "seed_problems" / "diagonal_seeds.jsonl")
    config.archive.fsync_jsonl = False
    config.archive.store_rollout_text = False
    config.frontier.remeasure_budget = 96
    # The mock crossover intentionally emits arithmetic/algebra children only.
    # Lift the per-cell cap here so this replay integration test exercises the
    # update path without fabricating semantic domain diversity.
    config.frontier.max_per_cell_per_batch = 32
    config.run.max_iterations = 2
    config.training.total_training_steps = 1
    config.verl_config = {
        "data": {"train_batch_size": 32},
        "actor_rollout_ref": {"rollout": {"n": 8}},
    }
    config.validate()
    identity = PolicyIdentity("engine-test", 0, 0, 0, "mock://initial")
    backend = DeterministicMockBackend(identity=identity)
    manifest = {
        "schema_version": config.run.schema_version,
        "run_uuid": identity.run_uuid,
        "resolved_config": {},
        "source": "unit-test",
    }
    engine = EvolutionEngine(
        config=config,
        policy_backend=backend,
        training_backend=backend,
        manifest=manifest,
        project_root=root,
        resume=False,
    )
    try:
        first = engine.run_cycle()
        assert first.accepted_before == 7
        assert first.archived_children == 64
        assert first.accepted_after == 71
        assert first.current_frontier_count < first.accepted_after
        assert not first.update_applied
        assert first.update_skip_reason == "insufficient_lagged_frontier"

        second = engine.run_cycle()
        assert second.lagged_frontier_count == 32
        assert second.update_applied
        assert second.policy_version_after == 1
        assert len(backend.applied_batches) == 1
        batch = backend.applied_batches[0]
        assert len(batch.groups) == 32
        assert len({item.group.key.problem_id for item in batch.groups}) == 32
        assert all(
            item.candidate.selection_score_iteration == 0
            and item.candidate.score_iteration == 1
            for item in batch.groups
        )
        assert all(item.group.purpose == "score" for item in batch.groups)
        label_calls = [
            call for call in backend.generation_calls if call["purpose"] == "label"
        ]
        score_calls = [
            call for call in backend.generation_calls if call["purpose"] == "score"
        ]
        assert label_calls and score_calls
        assert all(call["ground_truths"] is None for call in label_calls)
        assert all(call["ground_truths"] is not None for call in score_calls)
        assert len(backend.saved_checkpoints) == 1
        checkpoint_step, checkpoint_state = backend.saved_checkpoints[0]
        assert checkpoint_step == 1
        assert checkpoint_state["phase"] == "update_applied"
        assert checkpoint_state["active_training_batch_id"] == batch.batch_id
        committed = engine.store.load_state()
        assert committed.phase == "ready"
        assert committed.checkpoint_step == 1
        assert committed.global_step == 1
    finally:
        engine.close()
