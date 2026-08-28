"""Immutable run-manifest construction and preflight checks."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import uuid
from typing import Any

from .config import AppConfig, as_dict
from .prompts import PromptBook
from .utils import file_sha256, value_sha256


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "uncommitted"
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        dirty = True
    return {"commit": commit, "dirty": dirty}


def build_run_manifest(
    config: AppConfig,
    *,
    project_root: str | Path,
    config_path: str | Path,
    run_uuid: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    resolved_config = as_dict(config)
    # Resume is an invocation policy, not an experiment-semantic setting.  A
    # clean run may start with ``disable`` and, after interruption, be reopened
    # with ``auto``.  Pin every other resolved field while deliberately
    # normalizing this one flag so that safe resume does not require weakening
    # the manifest comparison.
    resolved_config["training"]["resume_mode"] = "<operational>"
    prompt_book = PromptBook(root / "prompt_templates")
    seed_path = Path(config.archive.seed_file)
    if not seed_path.is_absolute():
        seed_path = root / seed_path
    source_files = sorted((root / "src" / "rq_evolve_v02").glob("*.py"))
    source_hash = value_sha256({path.name: file_sha256(path) for path in source_files})
    prompts = {
        name: file_sha256(root / "prompt_templates" / name)
        for name in (
            "crossover_system.txt",
            "crossover_user.txt",
            "solver_system.txt",
        )
    }
    # Force prompt parsing during preflight; the variable is intentionally used.
    del prompt_book
    return {
        "schema_version": config.run.schema_version,
        "run_uuid": run_uuid or str(uuid.uuid4()),
        "run_name": config.run.name,
        "resolved_config": resolved_config,
        "config_file": str(Path(config_path).resolve()),
        "config_sha256": value_sha256(resolved_config),
        "seed_file": str(seed_path.resolve()),
        "seed_sha256": file_sha256(seed_path),
        "prompt_sha256": prompts,
        "source_sha256": source_hash,
        "git": _git_state(root),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in (
                    "torch",
                    "transformers",
                    "verl",
                    "vllm",
                    "ray",
                    "math-verify",
                )
            },
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    }


def validate_model_paths(config: AppConfig) -> None:
    if config.backend.kind == "mock":
        return
    model_path = Path(config.backend.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"backend.model_path does not exist: {model_path}")
    verl_model = (
        config.verl_config.get("actor_rollout_ref", {}).get("model", {}).get("path")
    )
    if (
        verl_model
        and Path(str(verl_model)).expanduser().resolve() != model_path.resolve()
    ):
        raise ValueError(
            "backend.model_path and verl_config.actor_rollout_ref.model.path must match"
        )
