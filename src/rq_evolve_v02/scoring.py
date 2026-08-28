"""Concrete-problem R_Q scoring under a fixed current policy."""

from __future__ import annotations

import math

from .backends import GeneratedGroup, PolicyIdentity
from .grading import GraderClient
from .models import RolloutSample, ScoreEvidence
from .output_parser import extract_last_boxed


def score_group(
    group: GeneratedGroup,
    *,
    requested_rollouts: int,
    gold: str,
    verifier: dict,
    grader: GraderClient,
    identity: PolicyIdentity,
    iteration: int,
    require_entropy: bool,
    sampling_contract: dict | None = None,
) -> tuple[ScoreEvidence | None, str | None]:
    if len(group.samples) != requested_rollouts:
        return None, "incomplete_score_group"
    if any(sample.status != "accepted" for sample in group.samples):
        # Infrastructure/truncation rejection creates a partial distribution.
        # Do not shrink the denominator or replay a payload with different rows.
        return None, "score_group_contains_rejected_sample"
    if require_entropy and any(sample.entropy is None for sample in group.samples):
        return None, "missing_score_entropy"
    if any(
        sample.entropy is not None
        and (
            not math.isfinite(float(sample.entropy))
            or not 0.0 <= float(sample.entropy) <= 1.0
        )
        for sample in group.samples
    ):
        return None, "invalid_normalized_entropy"
    rollouts: list[RolloutSample] = []
    flags: list[bool] = []
    entropies: list[float] = []
    for index, sample in enumerate(group.samples):
        predicted = extract_last_boxed(sample.text)
        correct = bool(
            predicted is not None and grader.grade(predicted, gold, verifier)
        )
        flags.append(correct)
        entropies.append(float(sample.entropy or 0.0))
        rollouts.append(
            RolloutSample(
                response=sample.text,
                predicted_answer=predicted,
                correct=correct,
                entropy=float(sample.entropy or 0.0),
                policy_version=identity.policy_version,
                adapter_version=identity.adapter_version,
                global_step=identity.global_step,
                source_checkpoint=identity.source_checkpoint,
                sample_index=index,
            )
        )
    m = requested_rollouts
    s_hat = sum(flags) / m
    learnability = (m / (m - 1)) * s_hat * (1 - s_hat) if m >= 2 else 0.0
    u_score = sum(entropies) / m
    return (
        ScoreEvidence(
            iteration=iteration,
            policy_version=identity.policy_version,
            s_hat=s_hat,
            learnability=learnability,
            u_score=u_score,
            rq_score=learnability * u_score,
            num_rollouts=m,
            num_correct=sum(flags),
            policy_run_uuid=identity.run_uuid,
            adapter_version=identity.adapter_version,
            global_step=identity.global_step,
            source_checkpoint=identity.source_checkpoint,
            sampling_contract=dict(sampling_contract or {}),
            rollouts=rollouts,
            request_id=group.request_id,
        ),
        None,
    )


def in_frontier(score: ScoreEvidence, *, low: float, high: float) -> bool:
    return low < score.s_hat < high
