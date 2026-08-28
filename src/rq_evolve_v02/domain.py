"""Seven binary-arm, label-blind Omni top-level domain assignment."""

from __future__ import annotations

import math

from .backends import PolicyBackend
from .concepts import DOMAINS
from .models import DomainEvidence
from .prompts import PromptBook
from .utils import stable_id


def _logit(probability: float) -> float:
    p = min(max(float(probability), 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def label_domains(
    questions: list[str],
    *,
    backend: PolicyBackend,
    prompts: PromptBook,
    min_probability: float,
    min_logit_margin: float,
    iteration: int,
    request_namespace: str = "",
) -> list[DomainEvidence]:
    messages: list[list[dict[str, str]]] = []
    request_ids: list[str] = []
    for question_index, question in enumerate(questions):
        for domain in DOMAINS:
            messages.append(prompts.domain_messages(question, domain))
            request_ids.append(
                stable_id(
                    "domain",
                    iteration,
                    request_namespace,
                    question_index,
                    question,
                    domain,
                )
            )
    outputs = backend.binary_token_probabilities(
        messages, request_ids=request_ids, purpose="domain_label"
    )
    expected = len(questions) * len(DOMAINS)
    if len(outputs) != expected:
        raise RuntimeError(
            f"domain backend returned {len(outputs)} arms; expected {expected}"
        )
    result: list[DomainEvidence] = []
    for question_index in range(len(questions)):
        scores: dict[str, float] = {}
        for domain_index, domain in enumerate(DOMAINS):
            row = outputs[question_index * len(DOMAINS) + domain_index]
            yes = float(row.get("YES", 0.0))
            no = float(row.get("NO", 0.0))
            denominator = yes + no
            scores[domain] = yes / denominator if denominator > 0 else 0.0
        ranked = sorted(
            scores.items(), key=lambda item: (-item[1], DOMAINS.index(item[0]))
        )
        top_domain, top_probability = ranked[0]
        runner_probability = ranked[1][1]
        margin = _logit(top_probability) - _logit(runner_probability)
        high_confidence = [
            domain
            for domain, probability in scores.items()
            if probability >= min_probability
        ]
        accepted = len(high_confidence) == 1 and margin >= min_logit_margin
        reason = None
        if not high_confidence:
            reason = "no_high_confidence_domain"
        elif len(high_confidence) > 1:
            reason = "multiple_high_confidence_domains"
        elif margin < min_logit_margin:
            reason = "low_margin"
        result.append(
            DomainEvidence(
                probabilities=scores,
                domain=top_domain if accepted else None,
                top_probability=top_probability,
                logit_margin=margin,
                accepted=accepted,
                reason=reason,
                labeler_run_uuid=backend.policy_identity.run_uuid,
                labeler_policy_version=backend.policy_identity.policy_version,
                labeler_adapter_version=backend.policy_identity.adapter_version,
                labeler_global_step=backend.policy_identity.global_step,
                source_checkpoint=backend.policy_identity.source_checkpoint,
                prompt_fingerprint=stable_id(
                    "domain-prompt",
                    [
                        prompts.domain_messages(questions[question_index], name)
                        for name in DOMAINS
                    ],
                    length=64,
                ),
                min_probability=float(min_probability),
                min_logit_margin=float(min_logit_margin),
            )
        )
    return result
