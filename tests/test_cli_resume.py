from __future__ import annotations

import argparse
import builtins
import hashlib
from pathlib import Path

import pytest

from rq_evolve_v02 import cli
from rq_evolve_v02.backends import PolicyIdentity
from rq_evolve_v02.config import AppConfig
from rq_evolve_v02.engine import EvolutionEngine
from rq_evolve_v02.mock_backend import DeterministicMockBackend
from rq_evolve_v02.models import ScoreEvidence
from rq_evolve_v02.storage import PipelineState, RunStore
from rq_evolve_v02.storage import read_jsonl

from .helpers import make_record


ROOT = Path(__file__).resolve().parents[1]


def make_config(run_dir: Path) -> AppConfig:
    config = AppConfig()
    config.backend.kind = "mock"
    config.backend.model_path = "mock://model"
    config.backend.n_gpus = 1
    config.archive.output_dir = str(run_dir)
    config.archive.seed_file = str(ROOT / "seed_problems" / "diagonal_seeds.jsonl")
    config.archive.fsync_jsonl = False
    config.archive.store_rollout_text = False
    config.frontier.remeasure_budget = 64
    config.run.max_iterations = 3
    config.training.total_training_steps = 1
    config.verl_config = {
        "data": {"train_batch_size": 32},
        "actor_rollout_ref": {"rollout": {"n": 8}},
        "trainer": {"save_freq": 0},
    }
    config.validate()
    return config


def manifest(run_uuid: str = "resume-test") -> dict:
    return {
        "schema_version": "concrete-problem-map-v1",
        "run_uuid": run_uuid,
        "resolved_config": {},
        "source": "unit-test",
    }


def make_backend(
    state: PipelineState, run_uuid: str = "resume-test"
) -> DeterministicMockBackend:
    return DeterministicMockBackend(
        identity=PolicyIdentity(
            run_uuid,
            state.policy_version,
            state.policy_version,
            state.global_step,
            (
                f"mock://checkpoint/{state.checkpoint_step}"
                if state.checkpoint_step
                else "mock://initial"
            ),
        )
    )


@pytest.mark.parametrize(
    "phase",
    ["policy_frozen", "scoring", "discovery", "batch_ready"],
)
def test_resume_aborts_pre_update_namespace_and_advances_iteration(
    tmp_path: Path,
    phase: str,
) -> None:
    config = make_config(tmp_path / phase)
    store = RunStore(config.archive.output_dir, fsync_jsonl=False)
    store.initialize(manifest(), resume=False)
    state = PipelineState(
        iteration=2,
        phase=phase,
        active_cycle_id="cycle-in-flight",
        active_training_batch_id="batch-in-flight" if phase == "batch_ready" else None,
    )
    store.save_state(state)
    engine = EvolutionEngine(
        config=config,
        policy_backend=make_backend(state),
        training_backend=make_backend(state),
        manifest=manifest(),
        project_root=ROOT,
        resume=True,
    )
    try:
        assert engine.state.iteration == 3
        assert engine.state.phase == "ready"
        assert engine.state.active_cycle_id is None
        assert engine.state.active_training_batch_id is None
        aborted = engine.store.read_cycle_artifact(2, "aborted")
        assert aborted["status"] == "aborted"
        assert aborted["interrupted_phase"] == phase
        training_rows = list(read_jsonl(engine.store.training_path))
        assert len(training_rows) == 1
        assert training_rows[0]["event_id"] == aborted["event_id"]
    finally:
        engine.close()


def test_resume_checkpointed_phase_commits_iteration_exactly_once(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path / "checkpointed")
    store = RunStore(config.archive.output_dir, fsync_jsonl=False)
    store.initialize(manifest(), resume=False)
    state = PipelineState(
        iteration=1,
        phase="checkpointed",
        active_cycle_id="cycle-1",
        active_training_batch_id="batch-1",
        policy_version=1,
        global_step=1,
        checkpoint_step=1,
        checkpoint_event_hash=hashlib.sha256(b"").hexdigest(),
    )
    store.save_state(state)
    backend = make_backend(state)
    engine = EvolutionEngine(
        config=config,
        policy_backend=backend,
        training_backend=backend,
        manifest=manifest(),
        project_root=ROOT,
        resume=True,
    )
    engine.close()
    committed = store.load_state()
    assert committed.iteration == 2
    assert committed.phase == "ready"

    # Opening the already committed boundary again must not advance it twice.
    reopened = EvolutionEngine(
        config=config,
        policy_backend=make_backend(committed),
        training_backend=make_backend(committed),
        manifest=manifest(),
        project_root=ROOT,
        resume=True,
    )
    try:
        assert reopened.state.iteration == 2
    finally:
        reopened.close()


def test_aborted_cycle_keeps_complete_problem_and_score_rows(tmp_path: Path) -> None:
    config = make_config(tmp_path / "retained")
    store = RunStore(config.archive.output_dir, fsync_jsonl=False)
    store.initialize(manifest(), resume=False)
    record = make_record(
        "retained-child",
        "Find the value of 6+7.",
        answer="13",
    )
    record.created_iteration = 2
    record.source = "crossover"
    assert store.append_problem(record)
    score = ScoreEvidence(
        iteration=2,
        policy_version=0,
        s_hat=0.5,
        learnability=2 / 7,
        u_score=0.5,
        rq_score=1 / 7,
        num_rollouts=8,
        num_correct=4,
        observation_id="retained-score",
        request_id="retained-request",
    )
    assert store.append_score(record.problem_id, score)
    state = PipelineState(
        iteration=2,
        phase="discovery",
        active_cycle_id="interrupted-with-complete-rows",
    )
    store.save_state(state)
    backend = make_backend(state)
    engine = EvolutionEngine(
        config=config,
        policy_backend=backend,
        training_backend=backend,
        manifest=manifest(),
        project_root=ROOT,
        resume=True,
    )
    try:
        assert engine.state.iteration == 3
        retained = engine.archive.records[record.problem_id]
        assert retained.latest_score is not None
        assert retained.latest_score.observation_id == "retained-score"
    finally:
        engine.close()


def test_resume_refuses_uncertain_post_update_pre_checkpoint_state(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path / "uncertain")
    store = RunStore(config.archive.output_dir, fsync_jsonl=False)
    store.initialize(manifest(), resume=False)
    state = PipelineState(
        iteration=1,
        phase="update_applied",
        active_cycle_id="cycle-1",
        active_training_batch_id="batch-1",
        policy_version=1,
        global_step=1,
        checkpoint_step=0,
    )
    store.save_state(state)
    backend = make_backend(state)
    with pytest.raises(RuntimeError, match="could double-apply"):
        EvolutionEngine(
            config=config,
            policy_backend=backend,
            training_backend=backend,
            manifest=manifest(),
            project_root=ROOT,
            resume=True,
        )


def test_state_loader_rejects_inconsistent_checkpoint_high_water(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "bad-state", fsync_jsonl=False)
    store.initialize(manifest(), resume=False)
    with pytest.raises(ValueError, match="durable checkpoint high-water"):
        store.save_state(
            PipelineState(
                iteration=2,
                phase="ready",
                policy_version=1,
                global_step=1,
                checkpoint_step=0,
            )
        )


def test_manifest_allows_only_operational_resume_mode_change(tmp_path: Path) -> None:
    config = make_config(tmp_path / "manifest-run")
    config_path = ROOT / "configs" / "rq_evolve_v02_4gpu.yaml"
    first = cli._manifest_for_run(config, str(config_path), resume=False)
    RunStore(config.archive.output_dir, fsync_jsonl=False).initialize(
        first, resume=False
    )

    config.training.resume_mode = "auto"
    resumed = cli._manifest_for_run(config, str(config_path), resume=True)
    assert resumed == first


def test_cli_mock_resume_path_never_imports_verl_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path / "cli-mock")
    config.training.resume_mode = "auto"
    run_manifest = manifest("cli-mock-run")
    store = RunStore(config.archive.output_dir, fsync_jsonl=False)
    store.initialize(run_manifest, resume=False)
    store.save_state(PipelineState(iteration=config.run.max_iterations))

    monkeypatch.setattr(cli, "_load", lambda _path: config)
    monkeypatch.setattr(
        cli, "_manifest_for_run", lambda *_args, **_kwargs: run_manifest
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.endswith("verl_backend"):
            raise AssertionError("mock CLI path imported the VERL backend")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert cli.command_run(argparse.Namespace(config="mock-config.yaml")) == 0
    committed = store.load_state()
    assert committed.iteration == config.run.max_iterations
