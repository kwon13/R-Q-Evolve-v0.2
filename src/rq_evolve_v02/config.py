"""Typed configuration and validation for the concrete-problem pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf


@dataclass(slots=True)
class GenerationConfig:
    pairs_per_cycle: int = 32
    children_per_pair: int = 2
    initial_candidates: int = 64
    refill_candidates: int = 32
    max_candidates: int = 128
    temperature: float = 0.45
    top_p: float = 0.85
    max_tokens: int = 4096
    require_distinct_lineages: bool = True


@dataclass(slots=True)
class LabelingConfig:
    num_rollouts: int = 9
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 40
    max_tokens: int = 4096
    count_invalid_in_denominator: bool = True
    max_infrastructure_retries: int = 2


@dataclass(slots=True)
class ScoringConfig:
    num_rollouts: int = 8
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 5000
    frontier_s_hat_low: float = 0.3
    frontier_s_hat_high: float = 0.8
    reject_overlong: bool = True
    reject_invalid_answer: bool = False
    reject_duplicates: bool = False
    require_entropy: bool = True
    max_infrastructure_retries: int = 2


@dataclass(slots=True)
class NoveltyConfig:
    exact: bool = True
    near_duplicate_threshold: float = 0.92
    parent_similarity_ceiling: float = 0.94
    # Directional coverage catches a parent copied into a longer child, which
    # symmetric edit similarity systematically misses.  The minimum shared
    # shingle count prevents short generic phrases from tripping the gate.
    parent_shingle_containment_ceiling: float = 0.85
    parent_containment_min_shared_shingles: int = 8
    # Two samples drawn for the same parent pair should be meaningfully
    # different candidates, not near-identical restatements.
    sibling_similarity_ceiling: float = 0.82
    min_question_chars: int = 20
    max_question_chars: int = 4000


@dataclass(slots=True)
class ArchiveConfig:
    output_dir: str = "./rq_output/concrete_4b_8gpu"
    seed_file: str = "seed_problems/diagonal_seeds.jsonl"
    selection_strategy: str = "uniform_cell"
    fsync_jsonl: bool = True
    store_rollout_text: bool = True


@dataclass(slots=True)
class FrontierConfig:
    training_batch_size: int = 32
    selection_lag: int = 1
    # Deliberately false: selecting a newly measured problem and training on
    # that same lucky rollout group produces winner's-curse bias.  A problem
    # first becomes selectable from its score in a later cycle.
    warmup_use_current_score: bool = False
    skip_update_when_short: bool = True
    # Apply a ready replay batch before spending the cycle budget on new
    # discovery.  This keeps an expensive generation wave from delaying an
    # optimizer update that is already fully remeasured and validated.
    update_before_discovery: bool = False
    remeasure_budget: int = 64
    max_per_cell_per_batch: int = 8


@dataclass(slots=True)
class TrainingConfig:
    enabled: bool = True
    replay_score_rollouts: bool = True
    total_training_steps: int = 256
    # Disable scheduled/pre-weight-sync saving. EvolutionEngine creates one
    # external durable checkpoint after every successful optimizer update.
    save_freq: int = 0
    resume_mode: str = "disable"


@dataclass(slots=True)
class BackendConfig:
    kind: str = "verl"
    model_path: str = "/data1/yhoon113/qwen3-4b-base"
    n_gpus: int = 8
    gpu_memory_utilization: float = 0.42
    request_chunk_size: int = 8


@dataclass(slots=True)
class RunConfig:
    name: str = "rq_evolve_v02_4b_8gpu"
    master_seed: int = 0
    # Cycle cap, not optimizer-step target. Extra room covers the mandatory
    # discovery-only first cycle and later insufficient-frontier cycles.
    max_iterations: int = 320
    schema_version: str = "concrete-problem-map-v1"


@dataclass(slots=True)
class AppConfig:
    run: RunConfig = field(default_factory=RunConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    frontier: FrontierConfig = field(default_factory=FrontierConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    verl_config: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        g = self.generation
        if (
            g.pairs_per_cycle,
            g.children_per_pair,
            g.initial_candidates,
            g.refill_candidates,
            g.max_candidates,
        ) != (32, 2, 64, 32, 128):
            raise ValueError(
                "v0.2 generation geometry is fixed at 32 pairs, 2 children per "
                "pair, 64 initial candidates, 32-candidate refills, and 128 "
                "maximum candidates"
            )
        for name, value in (
            ("generation.top_p", g.top_p),
            ("labeling.top_p", self.labeling.top_p),
            ("scoring.top_p", self.scoring.top_p),
        ):
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if min(g.temperature, self.labeling.temperature, self.scoring.temperature) < 0:
            raise ValueError("sampling temperatures must be nonnegative")
        if (
            min(
                g.max_tokens,
                self.labeling.max_tokens,
                self.scoring.max_tokens,
            )
            < 1
        ):
            raise ValueError("generation token limits must be positive")
        if self.labeling.num_rollouts != 9:
            raise ValueError("v0.2 requires exactly 9 label rollouts")
        if not self.labeling.count_invalid_in_denominator:
            raise ValueError(
                "count_invalid_in_denominator must remain true; boxless completed "
                "answers may not shrink the nine-rollout denominator"
            )
        if self.labeling.max_infrastructure_retries < 0:
            raise ValueError("labeling.max_infrastructure_retries must be nonnegative")
        if self.scoring.num_rollouts != 8:
            raise ValueError("v0.2 requires exactly 8 score rollouts")
        if not self.scoring.reject_overlong:
            raise ValueError("v0.2 rejects max-token-truncated score groups")
        if self.scoring.reject_invalid_answer:
            raise ValueError(
                "boxless completed score responses must count as wrong, not be dropped"
            )
        if self.scoring.reject_duplicates:
            raise ValueError("duplicate solver responses are valid independent samples")
        if not self.scoring.require_entropy:
            raise ValueError("v0.2 R_Q scoring requires normalized actor entropy")
        if self.scoring.max_infrastructure_retries < 0:
            raise ValueError("scoring.max_infrastructure_retries must be nonnegative")
        low, high = (
            self.scoring.frontier_s_hat_low,
            self.scoring.frontier_s_hat_high,
        )
        if not 0 <= low < high <= 1:
            raise ValueError(
                "frontier success-rate range must satisfy 0 <= low < high <= 1"
            )
        if self.frontier.training_batch_size != 32:
            raise ValueError("v0.2 requires exactly 32 training problem groups")
        if self.frontier.max_per_cell_per_batch < 1:
            raise ValueError("frontier.max_per_cell_per_batch must be positive")
        if (
            self.frontier.max_per_cell_per_batch * 35
            < self.frontier.training_batch_size
        ):
            raise ValueError("cell cap makes the configured training batch impossible")
        if self.frontier.selection_lag not in {0, 1}:
            raise ValueError("frontier.selection_lag currently supports only 0 or 1")
        if self.frontier.selection_lag != 1:
            raise ValueError(
                "v0.2 requires selection_lag=1 so selection evidence is independent "
                "of the rollout payload consumed for training"
            )
        if self.frontier.warmup_use_current_score:
            raise ValueError(
                "warmup_use_current_score must be false; the discovery cycle may skip training"
            )
        if not self.frontier.skip_update_when_short:
            raise ValueError(
                "skip_update_when_short must be true; partial batches are forbidden"
            )
        if self.training.enabled and not self.training.replay_score_rollouts:
            raise ValueError("v0.2 training requires exact score-rollout replay")
        if self.training.save_freq != 0:
            raise ValueError(
                "training.save_freq must be 0; the pipeline externally checkpoints "
                "every successful optimizer update"
            )
        if self.archive.selection_strategy != "uniform_cell":
            raise ValueError("v0.2 currently requires uniform_cell parent selection")
        if not self.novelty.exact:
            raise ValueError("v0.2 exact duplicate rejection cannot be disabled")
        if self.generation.refill_candidates % self.generation.children_per_pair:
            raise ValueError("refill_candidates must be divisible by children_per_pair")
        if self.generation.max_candidates % self.generation.children_per_pair:
            raise ValueError("max_candidates must be divisible by children_per_pair")
        if self.training.replay_score_rollouts and (
            self.frontier.training_batch_size
            != int(
                _select(
                    self.verl_config,
                    "data.train_batch_size",
                    self.frontier.training_batch_size,
                )
            )
        ):
            raise ValueError(
                "frontier.training_batch_size must equal verl_config.data.train_batch_size "
                "when score rollout replay is enabled"
            )
        rollout_n = int(
            _select(
                self.verl_config,
                "actor_rollout_ref.rollout.n",
                self.scoring.num_rollouts,
            )
        )
        if (
            self.training.replay_score_rollouts
            and rollout_n != self.scoring.num_rollouts
        ):
            raise ValueError(
                "scoring.num_rollouts must equal verl_config.actor_rollout_ref.rollout.n "
                "for exact replay"
            )
        verl_save_freq = int(_select(self.verl_config, "trainer.save_freq", 0))
        if self.training.replay_score_rollouts and verl_save_freq != 0:
            raise ValueError(
                "verl_config.trainer.save_freq must be 0; periodic pre-weight-sync "
                "checkpoints would race the pipeline-owned update barrier"
            )
        if self.backend.n_gpus < 1:
            raise ValueError("backend.n_gpus must be positive")
        if self.backend.kind not in {"verl", "mock"}:
            raise ValueError("backend.kind must be verl or mock")
        if not 0 < self.backend.gpu_memory_utilization < 1:
            raise ValueError("backend.gpu_memory_utilization must be in (0, 1)")
        if self.backend.request_chunk_size < 1:
            raise ValueError("backend.request_chunk_size must be positive")
        if self.backend.kind == "verl" and self.verl_config:
            ray_num_cpus = _select(self.verl_config, "ray_init.num_cpus")
            if (
                isinstance(ray_num_cpus, bool)
                or not isinstance(ray_num_cpus, int)
                or ray_num_cpus < self.backend.n_gpus
            ):
                raise ValueError(
                    "verl_config.ray_init.num_cpus must be an explicit integer "
                    "at least backend.n_gpus; null may eagerly spawn one worker "
                    "per host CPU"
                )
            object_store_memory = _select(
                self.verl_config, "ray_init.object_store_memory"
            )
            if (
                isinstance(object_store_memory, bool)
                or not isinstance(object_store_memory, int)
                or object_store_memory < 1_073_741_824
            ):
                raise ValueError(
                    "verl_config.ray_init.object_store_memory must be an explicit "
                    "integer of at least 1 GiB"
                )
        if self.run.schema_version != "concrete-problem-map-v1":
            raise ValueError("unsupported run.schema_version")
        if self.run.max_iterations < 1 or self.training.total_training_steps < 1:
            raise ValueError("iteration and training-step limits must be positive")
        if self.run.max_iterations <= self.training.total_training_steps:
            raise ValueError(
                "run.max_iterations must exceed total_training_steps to allow "
                "the discovery-only warmup cycle"
            )
        if isinstance(self.novelty.near_duplicate_threshold, bool) or not (
            0 < self.novelty.near_duplicate_threshold <= 1
        ):
            raise ValueError("near_duplicate_threshold must be in (0, 1]")
        if isinstance(self.novelty.parent_similarity_ceiling, bool) or not (
            0 < self.novelty.parent_similarity_ceiling <= 1
        ):
            raise ValueError("parent_similarity_ceiling must be in (0, 1]")
        if isinstance(
            self.novelty.parent_shingle_containment_ceiling, bool
        ) or not (0 < self.novelty.parent_shingle_containment_ceiling <= 1):
            raise ValueError(
                "parent_shingle_containment_ceiling must be in (0, 1]"
            )
        if (
            isinstance(self.novelty.parent_containment_min_shared_shingles, bool)
            or not isinstance(
                self.novelty.parent_containment_min_shared_shingles, int
            )
            or self.novelty.parent_containment_min_shared_shingles < 1
        ):
            raise ValueError(
                "parent_containment_min_shared_shingles must be a positive integer"
            )
        if isinstance(self.novelty.sibling_similarity_ceiling, bool) or not (
            0 < self.novelty.sibling_similarity_ceiling <= 1
        ):
            raise ValueError("sibling_similarity_ceiling must be in (0, 1]")
        if self.novelty.min_question_chars < 1 or (
            self.novelty.max_question_chars < self.novelty.min_question_chars
        ):
            raise ValueError("invalid novelty question length bounds")
        if self.training.resume_mode not in {"disable", "auto"}:
            raise ValueError("training.resume_mode must be disable or auto")


T = TypeVar("T")


def _construct(cls: type[T], value: dict[str, Any] | None) -> T:
    payload = dict(value or {})
    allowed = {item.name for item in fields(cls)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} field(s): {sorted(unknown)}")
    return cls(**payload)


def _select(value: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _load_with_extends(path: Path, seen: set[Path] | None = None) -> DictConfig:
    resolved = path.expanduser().resolve()
    seen = set() if seen is None else seen
    if resolved in seen:
        raise ValueError(f"cyclic config extends: {resolved}")
    seen.add(resolved)
    current = OmegaConf.load(resolved)
    parent_name = current.get("extends") if isinstance(current, DictConfig) else None
    if not parent_name:
        return current
    parent_path = Path(str(parent_name))
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    parent = _load_with_extends(parent_path, seen)
    current = OmegaConf.create(OmegaConf.to_container(current, resolve=False))
    current.pop("extends", None)
    return OmegaConf.merge(parent, current)


def load_raw_config(path: str | Path) -> DictConfig:
    return _load_with_extends(Path(path))


def load_config(path: str | Path) -> AppConfig:
    raw = load_raw_config(path)
    data = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    known = {
        "run",
        "generation",
        "labeling",
        "scoring",
        "domain",
        "novelty",
        "archive",
        "frontier",
        "training",
        "backend",
        "verl_config",
    }
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config section(s): {sorted(unknown)}")
    cfg = AppConfig(
        run=_construct(RunConfig, data.get("run")),
        generation=_construct(GenerationConfig, data.get("generation")),
        labeling=_construct(LabelingConfig, data.get("labeling")),
        scoring=_construct(ScoringConfig, data.get("scoring")),
        novelty=_construct(NoveltyConfig, data.get("novelty")),
        archive=_construct(ArchiveConfig, data.get("archive")),
        frontier=_construct(FrontierConfig, data.get("frontier")),
        training=_construct(TrainingConfig, data.get("training")),
        backend=_construct(BackendConfig, data.get("backend")),
        verl_config=dict(data.get("verl_config") or {}),
    )
    cfg.validate()
    return cfg


def as_dict(config: AppConfig) -> dict[str, Any]:
    if not is_dataclass(config):
        raise TypeError("config must be a dataclass")
    return OmegaConf.to_container(OmegaConf.structured(config), resolve=True)  # type: ignore[return-value]
