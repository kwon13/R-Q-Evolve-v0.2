"""VERL bridge for concrete-problem rollout generation and exact replay.

Imports of VERL, torch, and ray-adjacent modules are intentionally lazy so the
archive, parser, and replay contract remain testable in a CPU-only environment.
This module never imports either sibling project.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import math
import os
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np

from .backends import (
    GeneratedGroup,
    GeneratedSample,
    PolicyIdentity,
    SamplingSpec,
)
from .replay import (
    FailClosedReplayHook,
    ReplayContractError,
    ScoreReplayBuffer,
    prompt_hash,
    slice_score_payloads,
)
from .training import (
    ExactReplayDataset,
    ReplayTrainingBatch,
    ReplayUpdateGuard,
    ResidentReplayDataset,
)
from .verifier import normalize_verifier


def _startup_log(message: str) -> None:
    """Emit an unbuffered marker around native/Ray startup boundaries."""

    print(f"[v0.2-startup] {message}", flush=True)


def _require_verl() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import verl.utils.torch_functional as verl_F
        from verl import DataProto
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        from verl.utils.model import compute_position_id_with_mask
    except ImportError as exc:  # pragma: no cover - exercised on GPU hosts
        raise RuntimeError(
            "VERL backend requested but verl is not installed; install the "
            "project's train extra in the training environment"
        ) from exc
    return (
        DataProto,
        verl_F,
        pad_dataproto_to_divisor,
        unpad_dataproto,
        compute_position_id_with_mask,
    )


def _require_padding_helpers() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return VERL's padding pair without duplicating its five-item unpack."""

    _, _, pad_to_divisor, unpad, _ = _require_verl()
    return pad_to_divisor, unpad


class VerlPolicyBackend:
    """Use one resident VERL trainer's actor/rollout workers for all purposes."""

    def __init__(
        self,
        trainer: Any,
        *,
        policy_identity: PolicyIdentity,
        tokenizer: Any | None = None,
        max_prompt_length: int | None = None,
        truncation: str = "error",
        request_chunk_size: int = 8,
        domain_request_chunk_size: int = 56,
    ) -> None:
        self.trainer = trainer
        self.tokenizer = tokenizer or getattr(trainer, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("VERL trainer/tokenizer is required")
        self._policy_identity = policy_identity
        self.request_chunk_size = max(1, int(request_chunk_size))
        self.domain_request_chunk_size = max(1, int(domain_request_chunk_size))
        self._resident_sync_policy: tuple[str, int, int, int] | None = None
        config = getattr(trainer, "config", None)
        data_cfg = getattr(config, "data", None)
        configured_length = getattr(data_cfg, "max_prompt_length", None)
        self.max_prompt_length = int(max_prompt_length or configured_length or 4096)
        # Silent prompt truncation would invalidate replay's exact prompt hash.
        if truncation != "error":
            raise ValueError("v0.2 VERL backend supports only truncation='error'")
        rollout_cfg = getattr(
            getattr(config, "actor_rollout_ref", None), "rollout", None
        )
        self._sleep_enabled = bool(
            getattr(rollout_cfg, "free_cache_engine", True)
            and getattr(rollout_cfg, "enable_sleep_mode", True)
        )
        self._logprobs_mode = str(
            getattr(rollout_cfg, "logprobs_mode", "processed_logprobs")
        )

    @property
    def policy_identity(self) -> PolicyIdentity:
        return self._policy_identity

    def set_policy_identity(self, identity: PolicyIdentity) -> None:
        """Advance the stamp only after the corresponding optimizer update."""

        self._policy_identity = identity

    def _manager(self) -> Any:
        manager = getattr(self.trainer, "async_rollout_manager", None)
        if manager is None or not hasattr(manager, "generate_sequences"):
            raise RuntimeError(
                "VERL trainer exposes no async_rollout_manager.generate_sequences"
            )
        return manager

    def _wake_and_sync(self) -> None:
        stamp = (
            self._policy_identity.run_uuid,
            self._policy_identity.policy_version,
            self._policy_identity.adapter_version,
            self._policy_identity.global_step,
        )
        if not self._sleep_enabled and self._resident_sync_policy == stamp:
            return
        manager = getattr(self.trainer, "checkpoint_manager", None)
        if manager is not None and hasattr(manager, "update_weights"):
            manager.update_weights(int(self._policy_identity.global_step))
            if not self._sleep_enabled:
                self._resident_sync_policy = stamp

    def _sleep(self) -> None:
        if not self._sleep_enabled:
            return
        manager = getattr(self.trainer, "checkpoint_manager", None)
        if manager is not None and hasattr(manager, "sleep_replicas"):
            manager.sleep_replicas()

    def generate(
        self,
        messages: Sequence[list[dict[str, str]]],
        *,
        request_ids: Sequence[str],
        sampling: SamplingSpec,
        purpose: str,
        ground_truths: Sequence[str] | None = None,
        verifiers: Sequence[dict] | None = None,
    ) -> list[GeneratedGroup]:
        """Generate instance-major groups, retaining native payload only for score.

        Score calls must carry live pseudo-gold and verifier contracts.  VERL's
        agent loop writes rewards into the generated output at generation time;
        leaving the target blank would replay a zero-reward batch later.
        """

        if len(messages) != len(request_ids):
            raise ValueError("messages and request_ids must have equal length")
        if len(messages) > self.request_chunk_size:
            collected: list[GeneratedGroup] = []
            for start in range(0, len(messages), self.request_chunk_size):
                end = start + self.request_chunk_size
                collected.extend(
                    self.generate(
                        messages[start:end],
                        request_ids=request_ids[start:end],
                        sampling=sampling,
                        purpose=purpose,
                        ground_truths=(
                            ground_truths[start:end]
                            if ground_truths is not None
                            else None
                        ),
                        verifiers=(
                            verifiers[start:end] if verifiers is not None else None
                        ),
                    )
                )
            return collected
        if len(set(str(value) for value in request_ids)) != len(request_ids):
            raise ValueError("request_ids must be unique within one generate call")
        n = int(sampling.n)
        if n < 1:
            raise ValueError("sampling.n must be positive")
        purpose = str(purpose)
        retain_payload = purpose == "score"
        if retain_payload:
            if ground_truths is None or verifiers is None:
                raise ReplayContractError(
                    "score generation requires ground_truths and verifiers for "
                    "the replayed rm_scores"
                )
            if len(ground_truths) != len(messages) or len(verifiers) != len(messages):
                raise ValueError("score target arrays must align with messages")
            if any(not str(value).strip() for value in ground_truths):
                raise ReplayContractError("score ground truth must be non-empty")
        elif purpose == "label":
            # Label rollouts decide pseudo-gold. They cannot already have a gold
            # reward and are never legal optimizer payloads.
            if ground_truths is not None or verifiers is not None:
                raise ReplayContractError(
                    "label generation must be disjoint from score targets/payloads"
                )

        prompt_batch = self._make_prompt_batch(
            messages,
            ground_truths=ground_truths,
            verifiers=verifiers,
            purpose=purpose,
        )
        gen_batch = prompt_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[
                "raw_prompt_ids",
                "raw_prompt",
                "data_source",
                "reward_model",
                "extra_info",
            ],
        )
        prompt_batch.non_tensor_batch["uid"] = np.array(
            [str(value) for value in request_ids], dtype=object
        )
        if n > 1:
            gen_batch = gen_batch.repeat(repeat_times=n, interleave=True)
        gen_batch.meta_info.update(
            {
                "temperature": max(0.0, float(sampling.temperature)),
                "top_p": float(sampling.top_p),
                "max_tokens": int(sampling.max_tokens),
            }
        )
        if sampling.top_k is not None:
            gen_batch.meta_info["top_k"] = int(sampling.top_k)

        pad_to_divisor, unpad = _require_padding_helpers()
        manager = self._manager()
        workers = getattr(manager, "agent_loop_workers", None)
        divisor = max(1, len(workers) if workers is not None else 1)
        self._wake_and_sync()
        try:
            padded, pad_size = pad_to_divisor(gen_batch, divisor)
            output = unpad(manager.generate_sequences(padded), pad_size=pad_size)
        finally:
            # Entropy is an actor forward; under hybrid sleep mode vLLM must be
            # offloaded first.  With sleep mode disabled this is a no-op.
            self._sleep()

        expected_rows = len(messages) * n
        payloads = slice_score_payloads(output, num_groups=len(messages), group_size=n)
        # The same strict slicing check applies to every purpose, but non-score
        # payloads are discarded immediately below.
        responses = output.batch.get("responses")
        if responses is None or len(responses) != expected_rows:
            raise ReplayContractError(
                f"generation returned no complete response tensor ({purpose})"
            )
        response_mask = output.batch.get("response_mask")
        if response_mask is None or len(response_mask) != expected_rows:
            raise ReplayContractError(
                f"generation returned no aligned response mask ({purpose})"
            )
        response_lengths = [int(row.sum().item()) for row in response_mask]
        max_tokens_reached = [
            length >= int(sampling.max_tokens) for length in response_lengths
        ]
        if retain_payload and output.batch.get("rm_scores") is None:
            raise ReplayContractError(
                "score generation returned no rm_scores; check the custom reward "
                "function before attempting replay training"
            )
        decoded = [
            self.tokenizer.decode(row.tolist(), skip_special_tokens=True)
            for row in responses
        ]

        entropies: list[float | None]
        if retain_payload:
            full_batch = prompt_batch.repeat(repeat_times=n, interleave=True).union(
                output
            )
            entropies = self._normalized_response_entropies(full_batch)
            if len(entropies) != expected_rows:
                raise ReplayContractError(
                    "entropy output does not align with score rows"
                )
        else:
            entropies = [None] * expected_rows

        result: list[GeneratedGroup] = []
        for index, request_id in enumerate(request_ids):
            start = index * n
            samples = [
                GeneratedSample(
                    text=decoded[row],
                    entropy=entropies[row],
                    status=("rejected" if max_tokens_reached[row] else "accepted"),
                    reject_reason=(
                        "max_tokens_reached" if max_tokens_reached[row] else None
                    ),
                )
                for row in range(start, start + n)
            ]
            result.append(
                GeneratedGroup(
                    request_id=str(request_id),
                    samples=samples,
                    payload=payloads[index] if retain_payload else None,
                    prompt_fingerprint=prompt_hash(messages[index]),
                )
            )
        return result

    def binary_token_probabilities(
        self,
        messages: Sequence[list[dict[str, str]]],
        *,
        request_ids: Sequence[str],
        purpose: str,
    ) -> list[dict[str, float]]:
        """Restricted YES/NO first-token probabilities for domain arms."""

        if len(messages) != len(request_ids):
            raise ValueError("messages and request_ids must have equal length")
        if len(messages) > self.domain_request_chunk_size:
            collected: list[dict[str, float]] = []
            for start in range(0, len(messages), self.domain_request_chunk_size):
                end = start + self.domain_request_chunk_size
                collected.extend(
                    self.binary_token_probabilities(
                        messages[start:end],
                        request_ids=request_ids[start:end],
                        purpose=purpose,
                    )
                )
            return collected
        if self._logprobs_mode != "processed_logprobs":
            raise ReplayContractError(
                "binary token probabilities require processed_logprobs; raw "
                "logprobs are not normalized after the allowed-token mask"
            )
        yes_ids = self.tokenizer.encode("YES", add_special_tokens=False)
        no_ids = self.tokenizer.encode("NO", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1 or yes_ids[0] == no_ids[0]:
            raise ReplayContractError(
                "the active tokenizer must encode YES and NO as distinct single tokens"
            )
        prompt_batch = self._make_prompt_batch(
            messages, ground_truths=None, verifiers=None, purpose=purpose
        )
        gen_batch = prompt_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[
                "raw_prompt_ids",
                "raw_prompt",
                "data_source",
                "reward_model",
                "extra_info",
            ],
        )
        gen_batch.meta_info.update(
            {
                "temperature": 0.0,
                "max_tokens": 1,
                "logprobs": 1,
                "allowed_token_ids": [int(yes_ids[0]), int(no_ids[0])],
            }
        )
        pad_to_divisor, unpad = _require_padding_helpers()
        manager = self._manager()
        workers = getattr(manager, "agent_loop_workers", None)
        divisor = max(1, len(workers) if workers is not None else 1)
        self._wake_and_sync()
        try:
            padded, pad_size = pad_to_divisor(gen_batch, divisor)
            output = unpad(manager.generate_sequences(padded), pad_size=pad_size)
        finally:
            self._sleep()
        responses = output.batch.get("responses")
        logps = output.batch.get("rollout_log_probs")
        if responses is None or logps is None or len(responses) != len(messages):
            raise ReplayContractError(
                "VERL did not return restricted first-token log probabilities"
            )
        result: list[dict[str, float]] = []
        for tokens, row_logps in zip(responses, logps):
            chosen = int(tokens[0])
            probability = min(1.0, max(0.0, math.exp(float(row_logps[0]))))
            if chosen == int(yes_ids[0]):
                result.append({"YES": probability, "NO": 1.0 - probability})
            elif chosen == int(no_ids[0]):
                result.append({"YES": 1.0 - probability, "NO": probability})
            else:
                raise ReplayContractError("restricted generation emitted another token")
        return result

    def _make_prompt_batch(
        self,
        messages: Sequence[list[dict[str, str]]],
        *,
        ground_truths: Sequence[str] | None,
        verifiers: Sequence[dict] | None,
        purpose: str,
    ) -> Any:
        DataProto, verl_F, _, _, position_ids_with_mask = _require_verl()
        rendered: list[str] = []
        raw_ids: list[list[int]] = []
        for index, conversation in enumerate(messages):
            if not conversation:
                raise ValueError(f"empty chat conversation at index {index}")
            text = self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) > self.max_prompt_length:
                raise ReplayContractError(
                    f"prompt {index} has {len(token_ids)} tokens, above exact "
                    f"replay limit {self.max_prompt_length}; reject it instead "
                    "of silently changing the prompt"
                )
            rendered.append(text)
            raw_ids.append(token_ids)
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_length=self.max_prompt_length,
            pad_token_id=pad_id,
            left_pad=True,
            truncation="error",
        )
        position_ids = position_ids_with_mask(attention_mask)
        raw_prompt = np.empty(len(messages), dtype=object)
        raw_prompt_ids = np.empty(len(messages), dtype=object)
        reward_model = np.empty(len(messages), dtype=object)
        extra_info = np.empty(len(messages), dtype=object)
        for index, conversation in enumerate(messages):
            raw_prompt[index] = [dict(value) for value in conversation]
            raw_prompt_ids[index] = raw_ids[index]
            truth = str(ground_truths[index]) if ground_truths is not None else ""
            verifier = normalize_verifier(
                verifiers[index] if verifiers is not None else None,
                answer=truth if truth else None,
            )
            reward_model[index] = {
                "ground_truth": truth,
                "verifier": verifier,
            }
            extra_info[index] = {
                "verifier": verifier,
                "generation_purpose": str(purpose),
            }
        return DataProto.from_single_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "raw_prompt_ids": raw_prompt_ids,
                "raw_prompt": raw_prompt,
                "data_source": np.array(
                    [f"rq_evolve_v02_{purpose}"] * len(messages), dtype=object
                ),
                "reward_model": reward_model,
                "extra_info": extra_info,
            }
        )

    def _normalized_response_entropies(self, batch: Any) -> list[float]:
        import torch

        responses = batch.batch["responses"]
        response_length = responses.size(1)
        batch.batch["response_mask"] = batch.batch["attention_mask"][
            :, -response_length:
        ]
        batch.meta_info["global_token_num"] = torch.sum(
            batch.batch["attention_mask"], dim=-1
        ).tolist()
        pad_to_divisor, unpad = _require_padding_helpers()
        world_size = max(
            1, int(getattr(self.trainer.actor_rollout_wg, "world_size", 1))
        )
        padded, pad_size = pad_to_divisor(batch, world_size)
        worker = self.trainer.actor_rollout_wg
        if hasattr(worker, "compute_log_prob"):
            actor_output = worker.compute_log_prob(padded)
        elif hasattr(worker, "compute_log_probs"):
            try:
                actor_output = worker.compute_log_probs(padded, calculate_entropy=True)
            except TypeError:
                actor_output = worker.compute_log_probs(padded)
        else:
            raise RuntimeError("VERL actor exposes no log-probability method")
        actor_output = unpad(actor_output, pad_size=pad_size)
        entropy = actor_output.batch.get("entropys")
        if entropy is None:
            entropy = actor_output.batch.get("entropies")
        if entropy is None:
            raise ReplayContractError("VERL actor did not return token entropy")
        response_mask = batch.batch["response_mask"]
        try:
            vocab_size = int(len(self.tokenizer))
        except TypeError:
            vocab_size = int(getattr(self.tokenizer, "vocab_size", 0) or 0)
        if vocab_size < 2:
            raise ReplayContractError("cannot normalize entropy without vocab size")
        maximum = math.log(vocab_size)
        values: list[float] = []
        for row, mask in zip(entropy, response_mask):
            valid = row[mask.bool()]
            raw = float(valid.mean().item()) if valid.numel() else 0.0
            normalized = raw / maximum
            if (
                not math.isfinite(normalized)
                or normalized < -1e-6
                or normalized > 1.000001
            ):
                raise ReplayContractError(
                    f"actor returned invalid normalized entropy {normalized}"
                )
            values.append(min(1.0, max(0.0, normalized)))
        return values


class ResidentVerlTrainingBackend:
    """Apply exact replay batches without rebuilding the VERL trainer.

    Production starts ``trainer.fit`` exactly once in a resident driver thread.
    The dataloader blocks between submitted batches, and the live weight-sync
    call provides an exact update/checkpoint barrier. ``optimizer_step`` remains
    only as a small compatibility seam for CPU tests or another VERL release.
    No call recreates a trainer or Ray worker group.
    """

    def __init__(
        self,
        trainer: Any,
        *,
        replay_buffer: ScoreReplayBuffer,
        dataset: ExactReplayDataset,
        optimizer_step: Callable[[], dict[str, Any] | None] | None = None,
        policy_backend: VerlPolicyBackend | None = None,
        group_size: int,
        training_groups: int = 32,
    ) -> None:
        self.trainer = trainer
        self.replay_buffer = replay_buffer
        self.dataset = dataset
        self.optimizer_step = optimizer_step
        self.policy_backend = policy_backend
        self._fit_thread: threading.Thread | None = None
        self._fit_error: BaseException | None = None
        self._checkpoint_manager: Any = None
        self._original_update_weights: Callable[..., Any] | None = None
        self.hook = FailClosedReplayHook(
            replay_buffer,
            group_size=group_size,
            training_groups=training_groups,
        )
        manager = getattr(trainer, "async_rollout_manager", None)
        self.hook.install(manager)
        if optimizer_step is None:
            self._install_weight_sync_barrier()

    def _install_weight_sync_barrier(self) -> None:
        if not isinstance(self.dataset, ResidentReplayDataset):
            raise ReplayContractError(
                "resident weight-sync barrier requires ResidentReplayDataset"
            )
        manager = getattr(self.trainer, "checkpoint_manager", None)
        original = (
            getattr(manager, "update_weights", None) if manager is not None else None
        )
        if original is None:
            raise ReplayContractError(
                "resident VERL replay requires checkpoint_manager.update_weights "
                "to establish an exact post-update checkpoint boundary"
            )
        self._checkpoint_manager = manager
        self._original_update_weights = original

        def update_weights(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            # Initial/rescoring syncs have no issued training batch and return
            # immediately. The optimizer's sync blocks here until the engine
            # durably checkpoints policy K + DataLoader K.
            self.dataset.complete_after_weight_sync()
            return result

        manager.update_weights = update_weights

    def apply_replay_batch(self, batch: ReplayTrainingBatch) -> dict[str, Any]:
        if batch.policy != self.replay_buffer.policy:
            raise ReplayContractError(
                "training batch policy differs from resident replay cycle"
            )
        if self.optimizer_step is not None:
            with ReplayUpdateGuard(self.replay_buffer, self.dataset, batch):
                metrics = self.optimizer_step() or {}
        else:
            metrics = self._apply_through_resident_fit(batch)
        identity_payload: dict[str, Any] = {}
        if self.policy_backend is not None:
            old = self.policy_backend.policy_identity
            # RayPPOTrainer increments global_steps after each update. It starts
            # the first pending step at 1, so current weights correspond to
            # global_steps - 1 while the dataloader is blocked.
            policy_step = (
                max(0, int(getattr(self.trainer, "global_steps", 0)))
                if self.optimizer_step is None
                else max(old.global_step + 1, old.policy_version + 1)
            )
            new_identity = PolicyIdentity(
                run_uuid=old.run_uuid,
                policy_version=max(old.policy_version + 1, policy_step),
                adapter_version=max(old.adapter_version + 1, policy_step),
                global_step=policy_step,
                source_checkpoint=f"in-memory:global_step_{policy_step}",
            )
            self.policy_backend.set_policy_identity(new_identity)
            identity_payload = {
                "policy_version": new_identity.policy_version,
                "adapter_version": new_identity.adapter_version,
                "global_step": new_identity.global_step,
                "source_checkpoint": new_identity.source_checkpoint,
            }
        return {
            **dict(metrics),
            "replay_batch_id": batch.batch_id,
            "replay_groups": batch.training_groups,
            "replay_rows": batch.training_groups * batch.group_size,
            "replay_served": True,
            **identity_payload,
        }

    def start(self, *, timeout_s: float = 900.0) -> None:
        """Start ``trainer.fit`` once and wait until its dataloader is idle."""

        if self.optimizer_step is not None:
            return
        if not isinstance(self.dataset, ResidentReplayDataset):
            raise ReplayContractError(
                "background resident fit requires ResidentReplayDataset"
            )
        if self._fit_thread is None:

            def run() -> None:
                try:
                    self.trainer.fit()
                except BaseException as exc:  # propagated on the pipeline thread
                    self._fit_error = exc

            self._fit_thread = threading.Thread(
                target=run,
                name="rq-v02-verl-fit",
                daemon=True,
            )
            self._fit_thread.start()
        self.dataset.wait_until_blocked(timeout_s=timeout_s)
        self._raise_fit_error()
        if self.policy_backend is not None:
            old = self.policy_backend.policy_identity
            policy_step = max(0, int(getattr(self.trainer, "global_steps", 1)) - 1)
            self.policy_backend.set_policy_identity(
                PolicyIdentity(
                    run_uuid=old.run_uuid,
                    policy_version=policy_step,
                    adapter_version=policy_step,
                    global_step=policy_step,
                    source_checkpoint=(
                        f"loaded:global_step_{policy_step}"
                        if policy_step
                        else old.source_checkpoint
                    ),
                )
            )

    def _raise_fit_error(self) -> None:
        if self._fit_error is not None:
            raise RuntimeError("resident VERL fit loop failed") from self._fit_error

    def _apply_through_resident_fit(
        self, batch: ReplayTrainingBatch, *, timeout_s: float = 7200.0
    ) -> dict[str, Any]:
        assert isinstance(self.dataset, ResidentReplayDataset)
        self.start()
        self._raise_fit_error()
        served_before = self.replay_buffer.stats.served_batches
        self.dataset.stage(batch)
        deadline = time.monotonic() + float(timeout_s)
        while True:
            self._raise_fit_error()
            if self.dataset.completed(batch.batch_id):
                break
            thread = self._fit_thread
            if thread is not None and not thread.is_alive():
                # RayPPOTrainer returns before dataset.on_batch_end on its final
                # configured step. A clean exit after serving this payload is
                # the final batch's completion signal.
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"VERL optimizer step timed out for replay batch {batch.batch_id}"
                )
            time.sleep(0.1)
        served = self.replay_buffer.stats.served_batches - served_before
        self.dataset.clear()
        if served != 1:
            raise ReplayContractError(
                f"resident optimizer step completed after serving {served} replay "
                "batches; expected exactly one"
            )
        self.replay_buffer.mark_consumed()
        return {
            "training_global_steps_driver": int(
                getattr(self.trainer, "global_steps", 0) or 0
            )
        }

    def save_checkpoint(
        self, *, global_step: int, pipeline_state: dict[str, Any]
    ) -> str:
        """Save through the resident trainer; pipeline state is returned to caller.

        The engine persists ``pipeline_state`` atomically in its own run store.
        Payload tensors are intentionally absent from checkpoints and must be
        rescored after resume.
        """

        saver = getattr(self.trainer, "_save_checkpoint", None)
        if saver is None:
            saver = getattr(self.trainer, "save_checkpoint", None)
        if saver is None:
            raise RuntimeError("resident VERL trainer exposes no checkpoint method")
        requested_step = int(global_step)
        # The fit loop starts its pending step at K, updates theta_K, then calls
        # checkpoint_manager.update_weights(K).  Our wrapper blocks inside that
        # call, before logging and the later counter increment.  Therefore the
        # trainer counter, policy identity, model weights, and requested
        # checkpoint high-water mark must all be exactly K here.
        counter_name = (
            "global_steps"
            if hasattr(self.trainer, "global_steps")
            else "global_step" if hasattr(self.trainer, "global_step") else None
        )
        prior_counter = (
            int(getattr(self.trainer, counter_name)) if counter_name else None
        )
        if prior_counter is not None and prior_counter != requested_step:
            raise ReplayContractError(
                "checkpoint boundary mismatch: trainer is at "
                f"{prior_counter}, pipeline requested {requested_step}"
            )
        try:
            value = saver()
        except TypeError:
            value = saver(requested_step)
        if value is not None:
            checkpoint = str(value)
        else:
            config = getattr(self.trainer, "config", None)
            trainer_cfg = getattr(config, "trainer", None)
            root = getattr(trainer_cfg, "default_local_dir", "")
            checkpoint = (
                str(Path(root) / f"global_step_{requested_step}") if root else ""
            )
        if self.policy_backend is not None and checkpoint:
            current = self.policy_backend.policy_identity
            self.policy_backend.set_policy_identity(
                PolicyIdentity(
                    run_uuid=current.run_uuid,
                    policy_version=current.policy_version,
                    adapter_version=current.adapter_version,
                    global_step=current.global_step,
                    source_checkpoint=checkpoint,
                )
            )
        if isinstance(self.dataset, ResidentReplayDataset):
            batch_id = str(pipeline_state.get("active_training_batch_id") or "")
            if not batch_id:
                raise ReplayContractError(
                    "pipeline state must name the replay batch at checkpoint"
                )
            self.dataset.release_after_checkpoint(batch_id)
            # Once released, let the fit thread finish logging/incrementing and
            # block at the next clean dataloader boundary. On the final step it
            # exits instead, which is equally quiescent.
            deadline = time.monotonic() + 300.0
            while True:
                self._raise_fit_error()
                thread = self._fit_thread
                if thread is None or not thread.is_alive():
                    break
                try:
                    self.dataset.wait_until_blocked(timeout_s=0.25)
                    break
                except TimeoutError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "resident VERL trainer did not settle after checkpoint"
                        )
        return checkpoint

    def close(self) -> None:
        self.hook.uninstall()
        if (
            self._checkpoint_manager is not None
            and self._original_update_weights is not None
        ):
            self._checkpoint_manager.update_weights = self._original_update_weights
        self._checkpoint_manager = None
        self._original_update_weights = None
        if isinstance(self.dataset, ResidentReplayDataset):
            self.dataset.close()


class _StaticValidationDataset:
    """One harmless, unmarked row because VERL requires a non-empty val set."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        if int(index) != 0:
            raise IndexError(index)
        try:
            import torch

            dummy = torch.tensor([0], dtype=torch.uint8)
        except ImportError:  # pragma: no cover - VERL implies torch
            dummy = 0
        return {
            "raw_prompt": [
                {
                    "role": "system",
                    "content": "Solve the problem and put the final answer in \\boxed{}.",
                },
                {"role": "user", "content": "What is 1+1?"},
            ],
            "dummy_tensor": dummy,
            "data_source": "rq_evolve_v02_validation",
            "reward_model": {
                "ground_truth": "2",
                "verifier": {"mode": "expression"},
            },
            "extra_info": {
                "verifier": {"mode": "expression"},
                "generation_purpose": "validation",
            },
            "index": 0,
        }


@dataclass(slots=True)
class VerlRuntime:
    """All long-lived objects created once for one v0.2 run."""

    trainer: Any
    policy_backend: VerlPolicyBackend
    training_backend: ResidentVerlTrainingBackend
    replay_buffer: ScoreReplayBuffer
    train_dataset: ResidentReplayDataset
    tokenizer: Any
    verl_config: Any

    def close(self) -> None:
        self.training_backend.close()


@dataclass(frozen=True, slots=True)
class _RayPPOComponents:
    """VERL/Ray symbols resolved on the driver before Ray starts threads."""

    ray: Any
    trainer_cls: Any
    role: Any
    resource_pool_manager_cls: Any
    worker_group_cls: Any
    collate_fn: Any
    actor_cls: Any
    critic_cls: Any | None
    reward_cls: Any | None


def _configure_driver_environment(project: Path) -> dict[str, str]:
    """Set the native-runtime environment before importing Ray/VERL/vLLM.

    The same mapping is passed to Ray's worker runtime.  Setting it on the
    driver first is important: importing VERL's FSDP worker module transitively
    imports vLLM and native CUDA libraries, so applying these variables only in
    ``ray.init(runtime_env=...)`` is too late for the driver process.
    """

    project_src = str(project / "src")
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [
        part for part in inherited_pythonpath.split(os.pathsep) if part
    ]
    if project_src in pythonpath_parts:
        pythonpath_parts.remove(project_src)
    pythonpath = os.pathsep.join([project_src, *pythonpath_parts])
    env_vars = {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_USE_V1": "1",
        "VLLM_LOGGING_LEVEL": "WARN",
        "PYTHONPATH": pythonpath,
    }
    os.environ.update(env_vars)
    return env_vars


def _resolve_ray_ppo_components(config: Any) -> _RayPPOComponents:
    """Import everything the trainer builder needs before ``ray.init``."""

    strategy = str(config.actor_rollout_ref.actor.get("strategy", "fsdp"))
    if strategy not in {"fsdp", "fsdp2"}:
        raise NotImplementedError(
            f"v0.2 runtime currently supports fsdp/fsdp2, got {strategy!r}"
        )

    _startup_log("importing Ray driver module")
    ray = importlib.import_module("ray")
    _startup_log("Ray driver module ready; importing VERL trainer API")
    trainer_cls = _import_attr(
        [
            ("verl.trainer.ppo.ray_trainer", "RayPPOTrainer"),
            ("verl.trainer.ray_trainer", "RayPPOTrainer"),
        ]
    )
    role = _import_attr(
        [
            ("verl.trainer.ppo.ray_trainer", "Role"),
            ("verl.trainer.ppo.utils", "Role"),
            ("verl.trainer.ray_trainer", "Role"),
        ]
    )
    resource_pool_manager_cls = _import_attr(
        [
            ("verl.trainer.ppo.ray_trainer", "ResourcePoolManager"),
            ("verl.single_controller.ray", "ResourcePoolManager"),
            ("verl.trainer.ray_trainer", "ResourcePoolManager"),
        ]
    )
    worker_group_cls = _import_attr(
        [("verl.single_controller.ray", "RayWorkerGroup")]
    )
    collate_fn = _import_attr(
        [
            ("verl.utils.dataset.rl_dataset", "collate_fn"),
            ("verl.utils.dataset", "collate_fn"),
        ]
    )
    _startup_log("VERL trainer API ready; importing FSDP worker module")
    workers = importlib.import_module("verl.workers.fsdp_workers")
    actor_cls = getattr(workers, "ActorRolloutRefWorker")
    actor_cls = getattr(workers, "AsyncActorRolloutRefWorker", actor_cls)
    components = _RayPPOComponents(
        ray=ray,
        trainer_cls=trainer_cls,
        role=role,
        resource_pool_manager_cls=resource_pool_manager_cls,
        worker_group_cls=worker_group_cls,
        collate_fn=collate_fn,
        actor_cls=actor_cls,
        critic_cls=getattr(workers, "CriticWorker", None),
        reward_cls=getattr(workers, "RewardModelWorker", None),
    )
    _startup_log("VERL FSDP worker module ready")
    return components


def build_verl_runtime(
    app_config: Any,
    *,
    project_root: str | Path,
    run_uuid: str,
    reward_function: str = "src/rq_evolve_v02/reward.py:compute_score",
    start_fit_loop: bool = True,
) -> VerlRuntime:
    """Create workers, trainer, generation backend, and resident update loop.

    This is the production factory used by the CLI.  The trainer and Ray worker
    group are initialized exactly once.  ``trainer.fit`` is then started once
    and blocks on :class:`ResidentReplayDataset` between engine submissions.
    """

    project = Path(project_root).expanduser().resolve()
    driver_env = _configure_driver_environment(project)
    _startup_log("driver environment configured before Ray/VERL imports")
    if importlib.util.find_spec("verl") is None:
        raise RuntimeError("verl is not installed in this Python environment")
    verl_config = _load_and_patch_verl_config(
        app_config,
        project_root=project,
        reward_function=reward_function,
    )
    if int(verl_config.data.get("dataloader_num_workers", 0) or 0) != 0:
        raise ValueError(
            "exact resident replay requires verl_config.data."
            "dataloader_num_workers=0"
        )
    training_groups = int(app_config.frontier.training_batch_size)
    group_size = int(app_config.scoring.num_rollouts)
    if int(verl_config.data.train_batch_size) != training_groups:
        raise ValueError("VERL train_batch_size differs from frontier batch size")
    if int(verl_config.actor_rollout_ref.rollout.n) != group_size:
        raise ValueError("VERL rollout.n differs from score/replay group size")

    _startup_log("resolving RayPPOTrainer dependencies before Ray initialization")
    components = _resolve_ray_ppo_components(verl_config)
    _startup_log("RayPPOTrainer dependencies resolved")
    ray = components.ray
    if not ray.is_initialized():
        ray_cfg = verl_config.get("ray_init", {}) or {}
        ray_tmp = Path(
            os.environ.get(
                "RQ_V02_RAY_TMPDIR",
                str(project.parent / f".ray-rqv02-{str(run_uuid)[:8]}"),
            )
        )
        ray_tmp.mkdir(parents=True, exist_ok=True)
        ray_num_cpus = int(ray_cfg["num_cpus"])
        object_store_memory = int(ray_cfg["object_store_memory"])
        _startup_log(
            "initializing Ray "
            f"(num_cpus={ray_num_cpus}, "
            f"object_store_memory={object_store_memory}, temp={ray_tmp})"
        )
        ray.init(
            runtime_env={"env_vars": driver_env},
            num_cpus=ray_num_cpus,
            object_store_memory=object_store_memory,
            _temp_dir=str(ray_tmp),
        )
        _startup_log("Ray initialized")

    _startup_log("loading tokenizer and processor")
    tokenizer, processor = _build_tokenizer_and_processor(verl_config)
    _startup_log("tokenizer and processor ready")
    train_dataset = ResidentReplayDataset(training_groups=training_groups)
    val_dataset = _StaticValidationDataset()
    try:
        from torch.utils.data import SequentialSampler
    except ImportError as exc:  # pragma: no cover - VERL requires torch
        raise RuntimeError("torch is required for VERL training") from exc
    _startup_log("constructing RayPPOTrainer")
    trainer = _build_ray_ppo_trainer(
        verl_config,
        components=components,
        tokenizer=tokenizer,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_sampler=SequentialSampler(train_dataset),
    )
    _startup_log("RayPPOTrainer constructed; initializing workers")
    trainer.init_workers()
    _startup_log("VERL workers initialized")
    initial_identity = PolicyIdentity(
        run_uuid=str(run_uuid),
        policy_version=0,
        adapter_version=0,
        global_step=0,
        source_checkpoint=str(app_config.backend.model_path),
    )
    replay_buffer = ScoreReplayBuffer(
        expected_group_size=group_size,
        expected_training_groups=training_groups,
    )
    policy_backend = VerlPolicyBackend(
        trainer,
        policy_identity=initial_identity,
        tokenizer=tokenizer,
        max_prompt_length=int(verl_config.data.max_prompt_length),
        request_chunk_size=int(app_config.backend.request_chunk_size),
        domain_request_chunk_size=int(app_config.backend.domain_request_chunk_size),
    )
    training_backend = ResidentVerlTrainingBackend(
        trainer,
        replay_buffer=replay_buffer,
        dataset=train_dataset,
        policy_backend=policy_backend,
        group_size=group_size,
        training_groups=training_groups,
    )
    runtime = VerlRuntime(
        trainer=trainer,
        policy_backend=policy_backend,
        training_backend=training_backend,
        replay_buffer=replay_buffer,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        verl_config=verl_config,
    )
    if start_fit_loop:
        _startup_log("starting resident VERL fit loop")
        training_backend.start()
        _startup_log("resident VERL fit loop ready")
    return runtime


def _load_and_patch_verl_config(
    app_config: Any,
    *,
    project_root: Path,
    reward_function: str,
) -> Any:
    from omegaconf import OmegaConf, open_dict

    spec = importlib.util.find_spec("verl")
    if spec is None:
        raise RuntimeError("verl is not installed")
    if spec.submodule_search_locations:
        package_root = Path(next(iter(spec.submodule_search_locations)))
    elif spec.origin:
        package_root = Path(spec.origin).parent
    else:
        raise RuntimeError("cannot locate the installed verl package")
    base_path = package_root / "trainer" / "config" / "_generated_ppo_trainer.yaml"
    base = OmegaConf.load(base_path) if base_path.exists() else OmegaConf.create({})
    override = OmegaConf.create(dict(app_config.verl_config or {}))
    config = OmegaConf.merge(base, override)

    path_text, function_name = (
        reward_function.rsplit(":", 1)
        if ":" in reward_function
        else (reward_function, "compute_score")
    )
    reward_path = Path(path_text)
    if not reward_path.is_absolute():
        reward_path = project_root / reward_path
    if not reward_path.exists():
        raise FileNotFoundError(
            f"custom VERL reward function file is missing: {reward_path}"
        )
    with open_dict(config):
        config.actor_rollout_ref.model.path = str(app_config.backend.model_path)
        config.actor_rollout_ref.rollout.gpu_memory_utilization = float(
            app_config.backend.gpu_memory_utilization
        )
        config.actor_rollout_ref.rollout.logprobs_mode = "processed_logprobs"
        config.trainer.n_gpus_per_node = int(app_config.backend.n_gpus)
        config.trainer.total_training_steps = int(
            app_config.training.total_training_steps
        )
        # The engine checkpoints every successful update while the fit thread
        # is held at the post-weight-sync barrier. VERL's internal save occurs
        # *before* that sync and has no pipeline-state transaction, so disable
        # it to avoid a competing/half-described global_step_K checkpoint.
        config.trainer.save_freq = 0
        config.trainer.resume_mode = str(app_config.training.resume_mode)
        if (
            "custom_reward_function" not in config
            or config.custom_reward_function is None
        ):
            config.custom_reward_function = {}
        config.custom_reward_function.path = str(reward_path)
        config.custom_reward_function.name = str(function_name)
        if "reward_model" not in config or config.reward_model is None:
            config.reward_model = {}
        config.reward_model.enable = False
        if config.reward_model.get("reward_manager") is None:
            config.reward_model.reward_manager = "naive"
        config.data.reward_fn_key = "data_source"
        # Newer VERL nests the same reward configuration under `reward`.
        if "reward" in config and config.reward is not None:
            if (
                "custom_reward_function" not in config.reward
                or config.reward.custom_reward_function is None
            ):
                config.reward.custom_reward_function = {}
            config.reward.custom_reward_function.path = str(reward_path)
            config.reward.custom_reward_function.name = str(function_name)
    OmegaConf.resolve(config)
    return config


def _build_tokenizer_and_processor(config: Any) -> tuple[Any, Any]:
    copy_to_local = _optional_import_attr("verl.utils.fs", "copy_to_local")
    tokenizer_factory = _optional_import_attr("verl.utils", "hf_tokenizer")
    processor_factory = _optional_import_attr("verl.utils", "hf_processor")
    if tokenizer_factory is None:
        tokenizer_factory = _import_attr([("verl.utils.tokenizer", "get_tokenizer")])
    if processor_factory is None:
        processor_factory = _optional_import_attr(
            "verl.utils.tokenizer", "get_processor"
        )
    model_path = config.actor_rollout_ref.model.path
    local_path = (
        copy_to_local(
            model_path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        if copy_to_local is not None
        else model_path
    )
    trust = bool(
        config.data.get("trust_remote_code", False)
        or config.actor_rollout_ref.model.get("trust_remote_code", False)
    )
    tokenizer = tokenizer_factory(local_path, trust_remote_code=trust)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = (
        processor_factory(local_path, trust_remote_code=trust, use_fast=True)
        if processor_factory is not None
        else None
    )
    return tokenizer, processor


def _build_ray_ppo_trainer(
    config: Any,
    *,
    components: _RayPPOComponents,
    tokenizer: Any,
    processor: Any,
    train_dataset: Any,
    val_dataset: Any,
    train_sampler: Any,
) -> Any:
    ray = components.ray
    RayPPOTrainer = components.trainer_cls
    Role = components.role
    ResourcePoolManager = components.resource_pool_manager_cls
    RayWorkerGroup = components.worker_group_cls
    collate_fn = components.collate_fn
    strategy = str(config.actor_rollout_ref.actor.get("strategy", "fsdp"))
    if strategy not in {"fsdp", "fsdp2"}:
        raise NotImplementedError(
            f"v0.2 runtime currently supports fsdp/fsdp2, got {strategy!r}"
        )
    actor_cls = components.actor_cls
    critic_cls = components.critic_cls
    reward_cls = components.reward_cls
    actor_role = getattr(Role, "ActorRollout", getattr(Role, "ActorRolloutRef", None))
    if actor_role is None:
        raise RuntimeError("installed VERL exposes no actor-rollout Role")
    pool_id = "global_pool"
    role_workers: dict[Any, Any] = {actor_role: ray.remote(actor_cls)}
    mapping: dict[Any, str] = {actor_role: pool_id}
    critic_role = getattr(Role, "Critic", None)
    if critic_role is not None and critic_cls is not None:
        role_workers[critic_role] = ray.remote(critic_cls)
        mapping[critic_role] = pool_id
    ref_role = getattr(Role, "RefPolicy", None)
    needs_ref = bool(
        config.algorithm.get("use_kl_in_reward", False)
        or config.actor_rollout_ref.actor.get("use_kl_loss", False)
    )
    if ref_role is not None and needs_ref:
        role_workers[ref_role] = ray.remote(actor_cls)
        mapping[ref_role] = pool_id
    reward_role = getattr(Role, "RewardModel", None)
    if (
        bool(config.reward_model.get("enable", False))
        and reward_role is not None
        and reward_cls is not None
    ):
        role_workers[reward_role] = ray.remote(reward_cls)
        mapping[reward_role] = pool_id
    pool = ResourcePoolManager(
        resource_pool_spec={
            pool_id: [int(config.trainer.n_gpus_per_node)]
            * max(1, int(config.trainer.get("nnodes", 1)))
        },
        mapping=mapping,
    )
    kwargs = {
        "config": config,
        "tokenizer": tokenizer,
        "processor": processor,
        "role_worker_mapping": role_workers,
        "resource_pool_manager": pool,
        "ray_worker_group_cls": RayWorkerGroup,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "collate_fn": collate_fn,
        "train_sampler": train_sampler,
    }
    if _supports_kwarg(RayPPOTrainer.__init__, "device_name"):
        kwargs["device_name"] = config.trainer.get("device", "cuda")
    return RayPPOTrainer(**kwargs)


def _import_attr(candidates: Sequence[tuple[str, str]]) -> Any:
    errors: list[str] = []
    for module_name, attr_name in candidates:
        try:
            return getattr(importlib.import_module(module_name), attr_name)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}.{attr_name}: {exc}")
    raise ImportError("could not import any VERL candidate:\n" + "\n".join(errors))


def _optional_import_attr(module_name: str, attr_name: str) -> Any | None:
    try:
        return _import_attr([(module_name, attr_name)])
    except ImportError:
        return None


def _supports_kwarg(callable_obj: Any, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return name in signature.parameters or any(
        value.kind == inspect.Parameter.VAR_KEYWORD
        for value in signature.parameters.values()
    )
