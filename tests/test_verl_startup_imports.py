from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rq_evolve_v02 import verl_backend


class _AttrDict(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - diagnostic parity
            raise AttributeError(name) from exc


def _trainer_config() -> _AttrDict:
    return _AttrDict(
        actor_rollout_ref=_AttrDict(
            actor=_AttrDict(strategy="fsdp", use_kl_loss=False)
        ),
        algorithm=_AttrDict(use_kl_in_reward=False),
        reward_model=_AttrDict(enable=False),
        trainer=_AttrDict(n_gpus_per_node=4, nnodes=1, device="cuda"),
    )


def test_driver_environment_is_complete_and_pythonpath_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project_src = str(project / "src")
    inherited = os.pathsep.join(["/existing", project_src, "/other"])
    monkeypatch.setenv("PYTHONPATH", inherited)
    monkeypatch.setenv("VLLM_USE_V1", "0")

    first = verl_backend._configure_driver_environment(project)
    second = verl_backend._configure_driver_environment(project)

    assert first == second
    assert first == {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_USE_V1": "1",
        "VLLM_LOGGING_LEVEL": "WARN",
        "PYTHONPATH": os.pathsep.join([project_src, "/existing", "/other"]),
    }
    for key, value in first.items():
        assert os.environ[key] == value


def test_ray_ppo_dependencies_are_resolved_without_starting_ray(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    fake_ray = SimpleNamespace()
    actor = object()
    async_actor = object()
    critic = object()
    reward = object()
    workers = SimpleNamespace(
        ActorRolloutRefWorker=actor,
        AsyncActorRolloutRefWorker=async_actor,
        CriticWorker=critic,
        RewardModelWorker=reward,
    )
    symbols: dict[str, object] = {
        "RayPPOTrainer": object(),
        "Role": object(),
        "ResourcePoolManager": object(),
        "RayWorkerGroup": object(),
        "collate_fn": object(),
    }

    def fake_import_module(name: str) -> object:
        # The native-library settings must already be visible at every import.
        assert os.environ["VLLM_USE_V1"] == "1"
        assert os.environ["TOKENIZERS_PARALLELISM"] == "true"
        events.append(f"module:{name}")
        if name == "ray":
            return fake_ray
        if name == "verl.workers.fsdp_workers":
            return workers
        raise AssertionError(f"unexpected module import: {name}")

    def fake_import_attr(candidates: Any) -> object:
        attribute = candidates[0][1]
        events.append(f"attribute:{attribute}")
        return symbols[attribute]

    monkeypatch.setattr(verl_backend.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(verl_backend, "_import_attr", fake_import_attr)
    verl_backend._configure_driver_environment(tmp_path)

    components = verl_backend._resolve_ray_ppo_components(_trainer_config())

    assert events[0] == "module:ray"
    assert events[-1] == "module:verl.workers.fsdp_workers"
    assert "attribute:RayPPOTrainer" in events
    assert components.ray is fake_ray
    assert components.trainer_cls is symbols["RayPPOTrainer"]
    assert components.actor_cls is async_actor
    assert components.critic_cls is critic
    assert components.reward_cls is reward


def test_trainer_builder_uses_only_pre_resolved_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_calls: list[object] = []

    class FakeRay:
        @staticmethod
        def remote(worker: object) -> tuple[str, object]:
            remote_calls.append(worker)
            return ("remote", worker)

    class Role:
        ActorRollout = "actor"
        Critic = "critic"
        RefPolicy = "reference"
        RewardModel = "reward"

    class ResourcePoolManager:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Trainer:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    actor = object()
    critic = object()
    components = verl_backend._RayPPOComponents(
        ray=FakeRay(),
        trainer_cls=Trainer,
        role=Role,
        resource_pool_manager_cls=ResourcePoolManager,
        worker_group_cls=object(),
        collate_fn=object(),
        actor_cls=actor,
        critic_cls=critic,
        reward_cls=None,
    )
    monkeypatch.setattr(
        verl_backend.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(
            AssertionError(f"late import during trainer construction: {name}")
        ),
    )

    trainer = verl_backend._build_ray_ppo_trainer(
        _trainer_config(),
        components=components,
        tokenizer="tokenizer",
        processor=None,
        train_dataset="train",
        val_dataset="val",
        train_sampler="sampler",
    )

    assert remote_calls == [actor, critic]
    assert trainer.kwargs["role_worker_mapping"] == {
        "actor": ("remote", actor),
        "critic": ("remote", critic),
    }
    assert trainer.kwargs["ray_worker_group_cls"] is components.worker_group_cls
    assert trainer.kwargs["collate_fn"] is components.collate_fn
    assert trainer.kwargs["device_name"] == "cuda"
