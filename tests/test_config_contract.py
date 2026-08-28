from __future__ import annotations

from pathlib import Path

import pytest

from rq_evolve_v02.config import AppConfig, load_config


def assert_invalid(mutator, match: str) -> None:
    config = AppConfig()
    mutator(config)
    with pytest.raises(ValueError, match=match):
        config.validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pairs_per_cycle", 31),
        ("children_per_pair", 3),
        ("initial_candidates", 62),
        ("refill_candidates", 16),
        ("max_candidates", 256),
    ],
)
def test_generation_geometry_is_fixed(field: str, value: int) -> None:
    assert_invalid(
        lambda config: setattr(config.generation, field, value),
        "generation geometry is fixed",
    )


def test_label_score_and_training_geometry_is_fixed() -> None:
    assert_invalid(
        lambda config: setattr(config.labeling, "num_rollouts", 7),
        "exactly 9 label rollouts",
    )
    assert_invalid(
        lambda config: setattr(config.labeling, "min_agreement", 6 / 9),
        "min_agreement=5/9",
    )
    assert_invalid(
        lambda config: setattr(config.labeling, "require_proposed_answer_match", False),
        "proposed answer",
    )
    assert_invalid(
        lambda config: setattr(config.labeling, "count_invalid_in_denominator", False),
        "denominator",
    )
    assert_invalid(
        lambda config: setattr(config.scoring, "num_rollouts", 16),
        "exactly 8 score rollouts",
    )
    assert_invalid(
        lambda config: setattr(config.frontier, "training_batch_size", 16),
        "exactly 32 training",
    )
    assert_invalid(
        lambda config: setattr(config.frontier, "selection_lag", 0),
        "selection_lag=1",
    )


def test_checkpoint_ownership_is_fixed() -> None:
    assert_invalid(
        lambda config: setattr(config.training, "save_freq", 32),
        "externally checkpoints",
    )
    config = AppConfig()
    config.verl_config = {"trainer": {"save_freq": 32}}
    with pytest.raises(ValueError, match="periodic pre-weight-sync"):
        config.validate()


@pytest.mark.parametrize("num_cpus", [None, True, 0, 3, 3.5])
def test_verl_ray_cpu_budget_must_be_explicit_and_cover_all_gpus(
    num_cpus: object,
) -> None:
    config = AppConfig()
    config.backend.n_gpus = 4
    config.verl_config = {
        "ray_init": {
            "num_cpus": num_cpus,
            "object_store_memory": 16 * 1024**3,
        }
    }
    with pytest.raises(ValueError, match="ray_init.num_cpus"):
        config.validate()


@pytest.mark.parametrize(
    "object_store_memory",
    [None, True, 0, 1024**3 - 1, 1.5 * 1024**3],
)
def test_verl_ray_object_store_budget_is_explicit_integer_at_least_one_gib(
    object_store_memory: object,
) -> None:
    config = AppConfig()
    config.backend.n_gpus = 4
    config.verl_config = {
        "ray_init": {
            "num_cpus": 16,
            "object_store_memory": object_store_memory,
        }
    }
    with pytest.raises(ValueError, match="ray_init.object_store_memory"):
        config.validate()


def test_shipped_gpu_configs_obey_fixed_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    four = load_config(root / "configs" / "rq_evolve_v02_4gpu.yaml")
    eight = load_config(root / "configs" / "rq_evolve_v02_8gpu.yaml")
    for config, gpu_count in ((four, 4), (eight, 8)):
        assert config.backend.n_gpus == gpu_count
        assert config.generation.pairs_per_cycle == 32
        assert config.generation.children_per_pair == 2
        assert config.labeling.num_rollouts == 9
        assert config.scoring.num_rollouts == 8
        assert config.frontier.training_batch_size == 32
        assert config.frontier.selection_lag == 1
        assert config.training.save_freq == 0
        assert config.verl_config["trainer"]["save_freq"] == 0
        # Both launch shapes deliberately share the same bounded Ray driver
        # budget.  Leaving num_cpus null on a 112-core host eagerly creates a
        # Python-worker process storm before model loading begins.
        assert config.verl_config["ray_init"]["num_cpus"] == 16
        assert config.verl_config["ray_init"]["object_store_memory"] == 16 * 1024**3
        assert config.novelty.parent_shingle_containment_ceiling == 0.85
        assert config.novelty.parent_containment_min_shared_shingles == 8
        assert config.novelty.sibling_similarity_ceiling == 0.82


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("near_duplicate_threshold", True, "near_duplicate_threshold"),
        ("parent_similarity_ceiling", True, "parent_similarity_ceiling"),
        ("parent_shingle_containment_ceiling", 0.0, "containment_ceiling"),
        ("parent_shingle_containment_ceiling", 1.1, "containment_ceiling"),
        ("parent_shingle_containment_ceiling", True, "containment_ceiling"),
        ("sibling_similarity_ceiling", 0.0, "sibling_similarity_ceiling"),
        ("sibling_similarity_ceiling", 1.1, "sibling_similarity_ceiling"),
        ("sibling_similarity_ceiling", True, "sibling_similarity_ceiling"),
        (
            "parent_containment_min_shared_shingles",
            0,
            "min_shared_shingles",
        ),
        (
            "parent_containment_min_shared_shingles",
            True,
            "min_shared_shingles",
        ),
    ],
)
def test_novelty_gate_thresholds_are_validated(
    field: str, value: object, message: str
) -> None:
    assert_invalid(
        lambda config: setattr(config.novelty, field, value),
        message,
    )
