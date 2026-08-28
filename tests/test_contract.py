from __future__ import annotations

from pathlib import Path

from rq_evolve_v02.config import load_config
from rq_evolve_v02.contract import build_run_manifest


def test_resume_mode_is_operational_but_semantic_config_remains_pinned() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "rq_evolve_v02_4gpu.yaml"
    config = load_config(config_path)
    initial = build_run_manifest(
        config,
        project_root=root,
        config_path=config_path,
        run_uuid="fixed-run",
    )

    config.training.resume_mode = "auto"
    resumed = build_run_manifest(
        config,
        project_root=root,
        config_path=config_path,
        run_uuid="fixed-run",
    )
    assert resumed == initial

    config.scoring.temperature = 0.9
    changed = build_run_manifest(
        config,
        project_root=root,
        config_path=config_path,
        run_uuid="fixed-run",
    )
    assert changed != initial
    assert changed["config_sha256"] != initial["config_sha256"]
