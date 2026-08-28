"""Command-line entrypoint for preflight, execution, and archive inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .archive import ConcreteMapArchive
from .backends import PolicyIdentity
from .config import AppConfig, load_config
from .contract import build_run_manifest
from .engine import EvolutionEngine
from .mock_backend import DeterministicMockBackend
from .preflight import run_preflight
from .storage import PipelineState, RunStore, atomic_write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _absolute_runtime_paths(config: AppConfig) -> None:
    output = Path(config.archive.output_dir).expanduser()
    if not output.is_absolute():
        config.archive.output_dir = str((PROJECT_ROOT / output).resolve())
    seed = Path(config.archive.seed_file).expanduser()
    if not seed.is_absolute():
        config.archive.seed_file = str((PROJECT_ROOT / seed).resolve())
    trainer = config.verl_config.get("trainer")
    if isinstance(trainer, dict) and trainer.get("default_local_dir"):
        checkpoint = Path(str(trainer["default_local_dir"])).expanduser()
        if not checkpoint.is_absolute():
            trainer["default_local_dir"] = str((PROJECT_ROOT / checkpoint).resolve())


def _load(path: str) -> AppConfig:
    config = load_config(path)
    _absolute_runtime_paths(config)
    config.validate()
    return config


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _manifest_for_run(
    config: AppConfig, config_path: str, *, resume: bool
) -> dict[str, Any]:
    output = Path(config.archive.output_dir)
    manifest_path = output / "run_manifest.json"
    run_uuid: str | None = None
    if manifest_path.exists():
        if not resume:
            raise ValueError(
                f"run directory already has a manifest but resume is disabled: {output}"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_uuid = str(existing["run_uuid"])
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f"non-empty output directory has no run manifest: {output}")
    manifest = build_run_manifest(
        config,
        project_root=PROJECT_ROOT,
        config_path=config_path,
        run_uuid=run_uuid,
    )
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Git cleanliness/commit is provenance captured at run creation, not a
        # semantic compatibility key.  Source, prompt, seed, resolved config,
        # package, and platform hashes below remain strict.  This also permits
        # changing only the operational resume flag in a tracked YAML file.
        existing_identity = {
            key: value for key, value in existing.items() if key != "git"
        }
        current_identity = {
            key: value for key, value in manifest.items() if key != "git"
        }
        if existing_identity != current_identity:
            raise ValueError(
                "current config/prompts/source/runtime differ from the existing run manifest"
            )
        return existing
    return manifest


def _mock_runtime(
    config: AppConfig, manifest: dict[str, Any]
) -> DeterministicMockBackend:
    state_path = Path(config.archive.output_dir) / "pipeline_state.json"
    state = (
        PipelineState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
        if state_path.exists()
        else PipelineState()
    )
    return DeterministicMockBackend(
        identity=PolicyIdentity(
            run_uuid=str(manifest["run_uuid"]),
            policy_version=state.policy_version,
            adapter_version=state.policy_version,
            global_step=state.global_step,
            source_checkpoint=(
                f"mock://checkpoint/{state.checkpoint_step}"
                if state.checkpoint_step
                else "mock://initial"
            ),
        )
    )


def command_preflight(args: argparse.Namespace) -> int:
    config = _load(args.config)
    report = run_preflight(config, project_root=PROJECT_ROOT)
    _print_json({"status": "ok", **report})
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = _load(args.config)
    resume = config.training.resume_mode == "auto"
    run_preflight(config, project_root=PROJECT_ROOT)
    manifest = _manifest_for_run(config, args.config, resume=resume)
    runtime: Any | None = None
    if config.backend.kind == "mock":
        policy_backend = training_backend = _mock_runtime(config, manifest)
    else:
        from .verl_backend import build_verl_runtime

        runtime = build_verl_runtime(
            config,
            project_root=PROJECT_ROOT,
            run_uuid=str(manifest["run_uuid"]),
        )
        policy_backend = runtime.policy_backend
        training_backend = runtime.training_backend
    engine: EvolutionEngine | None = None
    try:
        engine = EvolutionEngine(
            config=config,
            policy_backend=policy_backend,
            training_backend=training_backend,
            manifest=manifest,
            project_root=PROJECT_ROOT,
            resume=resume,
        )
        while (
            engine.state.iteration < config.run.max_iterations
            and engine.state.global_step < config.training.total_training_steps
        ):
            summary = engine.run_cycle()
            _print_json({"cycle": summary.to_dict()})
    finally:
        if engine is not None:
            engine.close()
        elif runtime is not None:
            runtime.close()
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    root = Path(args.run_dir).expanduser().resolve()
    store = RunStore(root)
    archive = ConcreteMapArchive(store.load_problems())
    for problem_id, score in store.load_scores():
        archive.apply_score(problem_id, score)
    summaries = sorted((root / "iterations").glob("iter_*/summary.json"))
    _print_json(
        {
            "run_dir": str(root),
            "state": store.load_state().to_dict(),
            "accepted_count": len(archive.records),
            "occupied_cells": len(archive.occupied_cells()),
            "cell_counts": {
                cell: len(ids) for cell, ids in archive.cells.items() if ids
            },
            "last_summary": (
                json.loads(summaries[-1].read_text(encoding="utf-8"))
                if summaries
                else None
            ),
        }
    )
    return 0


def command_rebuild_index(args: argparse.Namespace) -> int:
    root = Path(args.run_dir).expanduser().resolve()
    store = RunStore(root)
    archive = ConcreteMapArchive(store.load_problems())
    for problem_id, score in store.load_scores():
        archive.apply_score(problem_id, score)
    store.write_map_index(archive.to_index())
    _print_json(
        {
            "status": "rebuilt",
            "path": str(store.map_path),
            "accepted_count": len(archive.records),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rq-evolve-v02")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in (
        ("preflight", command_preflight, "validate a config without reserving GPUs"),
        ("run", command_run, "run or configured-auto-resume the pipeline"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--config", required=True)
        sub.set_defaults(handler=handler)
    inspect = subparsers.add_parser("inspect", help="summarize one run directory")
    inspect.add_argument("--run-dir", required=True)
    inspect.set_defaults(handler=command_inspect)
    rebuild = subparsers.add_parser(
        "rebuild-index", help="rebuild map_index.json from append-only rows"
    )
    rebuild.add_argument("--run-dir", required=True)
    rebuild.set_defaults(handler=command_rebuild_index)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
