"""Read-only production contract checks before Ray reserves GPUs."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from .config import AppConfig
from .contract import validate_model_paths
from .prompts import PromptBook
from .seeds import load_seed_records
from .storage import PipelineState


_PATCH_MARKERS = (
    "# [rq-evolve-v0.2] per-call sampling overrides from meta_info v1",
    "# [rq-evolve] per-call sampling overrides from meta_info v2",
)


def _verl_patch_status() -> tuple[bool, str]:
    spec = importlib.util.find_spec("verl")
    if spec is None:
        return False, "verl is not installed"
    root = (
        Path(next(iter(spec.submodule_search_locations)))
        if spec.submodule_search_locations
        else Path(str(spec.origin)).parent
    )
    target = root / "experimental" / "agent_loop" / "agent_loop.py"
    if not target.exists():
        return False, f"async agent-loop source is missing: {target}"
    text = target.read_text(encoding="utf-8")
    return any(marker in text for marker in _PATCH_MARKERS), str(target)


def run_preflight(config: AppConfig, *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    PromptBook(root / "prompt_templates")
    seed_path = Path(config.archive.seed_file)
    if not seed_path.is_absolute():
        seed_path = root / seed_path
    seeds = load_seed_records(seed_path)
    validate_model_paths(config)
    report: dict[str, Any] = {
        "backend": config.backend.kind,
        "model_path": config.backend.model_path,
        "n_gpus": config.backend.n_gpus,
        "seed_count": len(seeds),
        "map_cells": 35,
        "generation_per_initial_wave": config.generation.initial_candidates,
        "label_rollouts": config.labeling.num_rollouts,
        "score_rollouts": config.scoring.num_rollouts,
        "training_groups": config.frontier.training_batch_size,
        "output_dir": str(Path(config.archive.output_dir).resolve()),
    }
    if config.backend.kind == "verl":
        trainer_cfg = config.verl_config.get("trainer", {})
        checkpoint_dir = Path(
            str(trainer_cfg.get("default_local_dir", ""))
        ).expanduser()
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = (root / checkpoint_dir).resolve()
        output_dir = Path(config.archive.output_dir).expanduser().resolve()
        state_path = output_dir / "pipeline_state.json"
        manifest_path = output_dir / "run_manifest.json"
        latest_path = checkpoint_dir / "latest_checkpointed_iteration.txt"
        if config.training.resume_mode == "disable":
            if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                raise ValueError(
                    "resume is disabled but the checkpoint directory is non-empty: "
                    f"{checkpoint_dir}"
                )
        elif manifest_path.exists():
            if not state_path.exists():
                raise ValueError(
                    "resume manifest exists but pipeline_state.json is missing"
                )
            state = PipelineState.from_dict(
                json.loads(state_path.read_text(encoding="utf-8"))
            )
            state.validate()
            if state.checkpoint_step > 0:
                if not latest_path.exists():
                    raise ValueError(
                        "resume state names a checkpoint but latest marker is missing"
                    )
                try:
                    latest_step = int(latest_path.read_text(encoding="utf-8").strip())
                except ValueError as exc:
                    raise ValueError("invalid latest checkpoint marker") from exc
                if latest_step != state.checkpoint_step:
                    raise ValueError(
                        "pipeline/checkpoint high-water mismatch: state="
                        f"{state.checkpoint_step}, latest={latest_step}"
                    )
                expected = checkpoint_dir / f"global_step_{state.checkpoint_step}"
                if not expected.is_dir():
                    raise ValueError(f"checkpoint directory is missing: {expected}")
            elif checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                raise ValueError(
                    "pipeline state is at step 0 but checkpoint directory is non-empty"
                )
        elif output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("resume output is non-empty but has no run manifest")
        elif checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise ValueError("new auto-resume run has a non-empty checkpoint directory")
        report["checkpoint_dir"] = str(checkpoint_dir)
        report["resume_mode"] = config.training.resume_mode
        applied, target = _verl_patch_status()
        report["verl_sampling_patch"] = {"applied": applied, "target": target}
        if not applied:
            raise RuntimeError(
                "the installed async VERL agent loop lacks the required per-call "
                "sampling patch. Run `python patches/verl_agent_loop_sampling.py` "
                "in this environment, then rerun preflight."
            )
        visible = [
            value
            for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if value
        ]
        if visible and len(visible) != config.backend.n_gpus:
            raise ValueError(
                f"CUDA_VISIBLE_DEVICES exposes {len(visible)} GPUs, config requires "
                f"{config.backend.n_gpus}"
            )
        reward_path = root / "src" / "rq_evolve_v02" / "reward.py"
        if not reward_path.exists():
            raise FileNotFoundError(f"typed reward function is missing: {reward_path}")
    elif config.backend.kind != "mock":
        raise ValueError(f"unknown backend.kind: {config.backend.kind!r}")
    return report
