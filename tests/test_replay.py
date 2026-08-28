from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from rq_evolve_v02.backends import PolicyIdentity
from rq_evolve_v02.replay import (
    FailClosedReplayHook,
    ReplayBatchUnavailable,
    ReplayContractError,
    ScoreReplayBuffer,
    ScoreReplayGroup,
    payload_row_count,
    slice_score_payloads,
)
from rq_evolve_v02.training import (
    ExactReplayDataset,
    LaggedTrainingCandidate,
    ReplayTrainingBatch,
    ResidentReplayDataset,
)
from rq_evolve_v02.verl_backend import ResidentVerlTrainingBackend


class FakeTensorBatch:
    def __init__(self, size: int) -> None:
        self.batch_size = (size,)

    def __len__(self) -> int:
        return self.batch_size[0]


class FakeRewardRow:
    def __init__(self, value: float) -> None:
        self.value = value

    def sum(self):
        return self

    def item(self) -> float:
        return self.value


class FakeRewardBatch(FakeTensorBatch):
    def __init__(self, rewards) -> None:
        super().__init__(len(rewards))
        self.rewards = [FakeRewardRow(value) for value in rewards]

    def get(self, key):
        return self.rewards if key == "rm_scores" else None


class FakeProto:
    def __init__(self, rows, *, non_tensor_batch=None, meta_info=None) -> None:
        self.rows = list(rows)
        self.batch = FakeTensorBatch(len(self.rows))
        self.non_tensor_batch = dict(non_tensor_batch or {})
        self.meta_info = dict(meta_info or {})

    def slice(self, start: int, stop: int):
        return type(self)(self.rows[start:stop])

    @classmethod
    def concat(cls, payloads):
        rows = []
        for payload in payloads:
            rows.extend(payload.rows)
        return cls(rows)


def identity(version: int = 3) -> PolicyIdentity:
    return PolicyIdentity(
        run_uuid="run-a",
        policy_version=version,
        adapter_version=version,
        global_step=version,
        source_checkpoint=f"global_step_{version}",
    )


def replay_group(
    index: int,
    *,
    policy: PolicyIdentity | None = None,
    iteration: int = 4,
    group_size: int = 2,
    purpose: str = "score",
) -> ScoreReplayGroup:
    messages = [
        {"role": "system", "content": "Solve."},
        {"role": "user", "content": f"Question {index}?"},
    ]
    return ScoreReplayGroup.capture(
        score_observation_id=f"obs-{index}",
        problem_id=f"problem-{index}",
        score_iteration=iteration,
        policy=policy or identity(),
        messages=messages,
        answer=str(index),
        verifier={"mode": "expression"},
        payload=FakeProto([f"{index}-a", f"{index}-b"][:group_size]),
        group_size=group_size,
        purpose=purpose,
    )


def candidate(index: int, *, current_s_hat: float = 0.5) -> LaggedTrainingCandidate:
    return LaggedTrainingCandidate(
        problem_id=f"problem-{index}",
        score_observation_id=f"obs-{index}",
        domain="Algebra" if index % 2 == 0 else "Geometry",
        problem_type="function",
        current_s_hat=current_s_hat,
        current_rq_score=0.01 * index,
        score_iteration=4,
        selection_rq_score=1.0 - 0.01 * index,
        selection_score_iteration=3,
        selection_s_hat=0.5,
    )


def test_score_payload_slicing_is_exact_and_instance_major():
    output = FakeProto(range(6))
    groups = slice_score_payloads(output, num_groups=3, group_size=2)
    assert [group.rows for group in groups] == [[0, 1], [2, 3], [4, 5]]
    with pytest.raises(ReplayContractError, match="discard and rescore"):
        slice_score_payloads(FakeProto(range(5)), num_groups=3, group_size=2)


def test_label_payload_can_never_enter_replay():
    with pytest.raises(ReplayContractError, match="only score rollout"):
        replay_group(0, purpose="label")


def test_native_reward_scores_must_match_score_evidence():
    payload = FakeProto([0, 1])
    payload.batch = FakeRewardBatch([1.0, 0.0])
    generated = SimpleNamespace(
        request_id="request-1",
        samples=[
            SimpleNamespace(status="accepted"),
            SimpleNamespace(status="accepted"),
        ],
        payload=payload,
    )
    problem = SimpleNamespace(
        problem_id="problem-1",
        pseudo_gold="2",
        verifier={"mode": "expression"},
    )
    score = SimpleNamespace(
        observation_id="obs-1",
        request_id="request-1",
        iteration=4,
        num_rollouts=2,
        policy_version=3,
        rollouts=[
            SimpleNamespace(correct=True),
            SimpleNamespace(correct=True),
        ],
    )
    with pytest.raises(ReplayContractError, match="disagree"):
        ScoreReplayGroup.from_generated(
            problem=problem,
            score=score,
            generated_group=generated,
            policy=identity(),
            messages=[{"role": "user", "content": "What is 1+1?"}],
            group_size=2,
        )


def test_buffer_rejects_stale_policy_and_short_or_duplicate_batch():
    buffer = ScoreReplayBuffer(expected_group_size=2, expected_training_groups=2)
    buffer.begin_cycle(iteration=4, policy=identity())
    buffer.store(replay_group(0))
    buffer.store(replay_group(1))
    with pytest.raises(ReplayBatchUnavailable, match="exactly 2"):
        buffer.exact_groups(["obs-0"], expected_count=2)
    with pytest.raises(ReplayBatchUnavailable, match="repeat"):
        buffer.exact_groups(["obs-0", "obs-0"], expected_count=2)
    with pytest.raises(ReplayContractError, match="policy"):
        buffer.store(replay_group(2, policy=identity(99)))


def test_lagged_frontier_not_current_score_selects_replay_batch():
    buffer = ScoreReplayBuffer(expected_group_size=2, expected_training_groups=2)
    buffer.begin_cycle(iteration=4, policy=identity())
    buffer.store(replay_group(0))
    buffer.store(replay_group(1))
    # Fresh s_hat values have moved to both extremes. Selection must still use
    # t-1 selection_s_hat/R_Q and consume these valid current-policy payloads.
    batch = ReplayTrainingBatch.build(
        buffer=buffer,
        candidates=[
            candidate(0, current_s_hat=0.0),
            candidate(1, current_s_hat=1.0),
        ],
        training_groups=2,
        selection_lag=1,
        frontier_s_hat_low=0.1,
        frontier_s_hat_high=0.9,
        max_per_cell=1,
    )
    assert set(batch.score_observation_ids) == {"obs-0", "obs-1"}


def test_exact_dataset_has_no_modulo_padding():
    buffer = ScoreReplayBuffer(expected_group_size=2, expected_training_groups=2)
    buffer.begin_cycle(iteration=4, policy=identity())
    buffer.store(replay_group(0))
    buffer.store(replay_group(1))
    batch = ReplayTrainingBatch.build(
        buffer=buffer,
        candidates=[candidate(0), candidate(1)],
        training_groups=2,
        max_per_cell=1,
    )
    dataset = ExactReplayDataset()
    assert len(dataset) == 0
    dataset.stage(batch)
    assert len(dataset) == 2
    assert dataset[0]["extra_info"]["score_observation_id"] in {"obs-0", "obs-1"}
    with pytest.raises(IndexError):
        _ = dataset[2]


class FakeManager:
    def __init__(self) -> None:
        self.original_calls = 0

    def generate_sequences(self, batch):
        self.original_calls += 1
        return "generated"


def marked_training_call(batch: ReplayTrainingBatch) -> FakeProto:
    extras = []
    prompts = []
    rewards = []
    rows = []
    for selected in batch.groups:
        group = selected.group
        for rollout_index in range(group.key.group_size):
            rows.append((group.key.problem_id, rollout_index))
            extra = group.replay_metadata()
            extra["batch_id"] = batch.batch_id
            extras.append(extra)
            prompts.append([dict(value) for value in group.messages])
            rewards.append(
                {
                    "ground_truth": group.answer,
                    "verifier": dict(group.verifier),
                }
            )
    return FakeProto(
        rows,
        non_tensor_batch={
            "extra_info": extras,
            "raw_prompt": prompts,
            "reward_model": rewards,
        },
        meta_info={"temperature": 1.0},
    )


def test_hook_serves_exact_payload_and_never_calls_generator():
    buffer = ScoreReplayBuffer(expected_group_size=2, expected_training_groups=2)
    buffer.begin_cycle(iteration=4, policy=identity())
    buffer.store(replay_group(0))
    buffer.store(replay_group(1))
    batch = ReplayTrainingBatch.build(
        buffer=buffer,
        candidates=[candidate(0), candidate(1)],
        training_groups=2,
        max_per_cell=1,
    )
    manager = FakeManager()
    hook = FailClosedReplayHook(
        buffer,
        group_size=2,
        training_groups=2,
        concat_fn=FakeProto.concat,
    )
    hook.install(manager)
    served = manager.generate_sequences(marked_training_call(batch))
    assert payload_row_count(served) == 4
    assert manager.original_calls == 0
    assert buffer.stats.served_batches == 1


def test_hook_prompt_mismatch_fails_closed_instead_of_resampling():
    buffer = ScoreReplayBuffer(expected_group_size=2, expected_training_groups=2)
    buffer.begin_cycle(iteration=4, policy=identity())
    buffer.store(replay_group(0))
    buffer.store(replay_group(1))
    batch = ReplayTrainingBatch.build(
        buffer=buffer,
        candidates=[candidate(0), candidate(1)],
        training_groups=2,
        max_per_cell=1,
    )
    call = marked_training_call(batch)
    call.non_tensor_batch["raw_prompt"][0][-1]["content"] = "swapped"
    call.non_tensor_batch["raw_prompt"][1][-1]["content"] = "swapped"
    manager = FakeManager()
    hook = FailClosedReplayHook(
        buffer,
        group_size=2,
        training_groups=2,
        concat_fn=FakeProto.concat,
    )
    hook.install(manager)
    with pytest.raises(ReplayContractError, match="prompt differs"):
        manager.generate_sequences(call)
    assert manager.original_calls == 0
    assert buffer.stats.failed_calls == 1


def test_unmarked_generation_call_passes_through():
    buffer = ScoreReplayBuffer(expected_group_size=2, expected_training_groups=2)
    buffer.begin_cycle(iteration=4, policy=identity())
    manager = FakeManager()
    hook = FailClosedReplayHook(
        buffer,
        group_size=2,
        training_groups=2,
        concat_fn=FakeProto.concat,
    )
    hook.install(manager)
    unmarked = FakeProto(
        [0], non_tensor_batch={"extra_info": [{"generation_purpose": "label"}]}
    )
    assert manager.generate_sequences(unmarked) == "generated"
    assert manager.original_calls == 1
    assert buffer.stats.passthrough_calls == 1


class FakePolicyBackend:
    def __init__(self) -> None:
        self._identity = identity(0)

    @property
    def policy_identity(self):
        return self._identity

    def set_policy_identity(self, value):
        self._identity = value


class FakeResidentTrainer:
    def __init__(self, dataset: ResidentReplayDataset) -> None:
        self.dataset = dataset
        self.async_rollout_manager = FakeManager()
        self.checkpoint_manager = SimpleNamespace(update_weights=lambda *_: None)
        self.global_steps = 0
        self.saved_at = None

    def fit(self):
        # Matches RayPPOTrainer's convention: step 1 is pending while the first
        # dataloader batch is requested; after one update global_steps is 2.
        self.global_steps = 1
        rows = [self.dataset[index] for index in range(len(self.dataset))]
        repeated_extras = []
        repeated_prompts = []
        repeated_rewards = []
        payload_rows = []
        for row in rows:
            for rollout_index in range(2):
                payload_rows.append((row["extra_info"]["problem_id"], rollout_index))
                repeated_extras.append(dict(row["extra_info"]))
                repeated_prompts.append(list(row["raw_prompt"]))
                repeated_rewards.append(dict(row["reward_model"]))
        call = FakeProto(
            payload_rows,
            non_tensor_batch={
                "extra_info": repeated_extras,
                "raw_prompt": repeated_prompts,
                "reward_model": repeated_rewards,
            },
        )
        self.async_rollout_manager.generate_sequences(call)
        self.checkpoint_manager.update_weights(self.global_steps)
        self.global_steps = 2
        self.dataset.on_batch_end(batch=rows)

    def _save_checkpoint(self):
        self.saved_at = self.global_steps
        return f"fake://global_step_{self.global_steps}"


def test_resident_fit_applies_one_submitted_batch_without_restarting():
    buffer = ScoreReplayBuffer(expected_group_size=2, expected_training_groups=2)
    buffer.begin_cycle(iteration=4, policy=identity(0))
    buffer.store(replay_group(0, policy=identity(0)))
    buffer.store(replay_group(1, policy=identity(0)))
    batch = ReplayTrainingBatch.build(
        buffer=buffer,
        candidates=[candidate(0), candidate(1)],
        training_groups=2,
        max_per_cell=1,
    )
    dataset = ResidentReplayDataset(training_groups=2)
    trainer = FakeResidentTrainer(dataset)
    policy = FakePolicyBackend()
    backend = ResidentVerlTrainingBackend(
        trainer,
        replay_buffer=buffer,
        dataset=dataset,
        policy_backend=policy,
        group_size=2,
        training_groups=2,
    )
    backend.hook.concat_fn = FakeProto.concat
    metrics = backend.apply_replay_batch(batch)
    assert metrics["replay_served"] is True
    assert buffer.consumed is True
    assert buffer.stats.served_batches == 1
    assert trainer.async_rollout_manager.original_calls == 0
    assert policy.policy_identity.policy_version == 1
    assert policy.policy_identity.global_step == 1
    assert backend._fit_thread is not None and backend._fit_thread.is_alive()
    checkpoint = backend.save_checkpoint(
        global_step=1,
        pipeline_state={"active_training_batch_id": batch.batch_id},
    )
    assert checkpoint == "fake://global_step_1"
    assert trainer.saved_at == 1
    backend._fit_thread.join(timeout=2)
    assert not backend._fit_thread.is_alive()
    backend.close()


def test_resident_dataset_rejects_next_fetch_before_weight_sync_barrier():
    buffer = ScoreReplayBuffer(expected_group_size=2, expected_training_groups=2)
    buffer.begin_cycle(iteration=4, policy=identity())
    buffer.store(replay_group(0))
    buffer.store(replay_group(1))
    batch = ReplayTrainingBatch.build(
        buffer=buffer,
        candidates=[candidate(0), candidate(1)],
        training_groups=2,
        max_per_cell=1,
    )
    dataset = ResidentReplayDataset(training_groups=2)
    dataset.stage(batch)
    _ = dataset[0]
    _ = dataset[1]
    with pytest.raises(ReplayContractError, match="weight-sync checkpoint barrier"):
        _ = dataset[0]
    dataset.close()
