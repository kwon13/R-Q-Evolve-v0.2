from __future__ import annotations

from pathlib import Path

from rq_evolve_v02.config import load_config
from rq_evolve_v02 import preflight


ROOT = Path(__file__).resolve().parents[1]


def test_preflight_reports_resolved_ray_resource_limits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config(ROOT / "configs" / "rq_evolve_v02_4gpu.yaml")
    config.archive.output_dir = str(tmp_path / "pipeline")
    config.verl_config["trainer"]["default_local_dir"] = str(tmp_path / "checkpoints")

    # This test exercises the startup report, not installation/model
    # discovery.  Those independent gates are covered by production preflight.
    monkeypatch.setattr(preflight, "validate_model_paths", lambda _config: None)
    monkeypatch.setattr(
        preflight,
        "_verl_patch_status",
        lambda: (True, "/test/site-packages/verl/agent_loop.py"),
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    report = preflight.run_preflight(config, project_root=ROOT)

    assert report["ray_num_cpus"] == 16
    assert report["ray_object_store_memory"] == 16 * 1024**3
