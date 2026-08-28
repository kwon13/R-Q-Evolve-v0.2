"""Fixed-denominator pseudo-labeling with typed, order-independent clustering."""

from __future__ import annotations

from dataclasses import dataclass

from .backends import GeneratedGroup, PolicyIdentity
from .grading import GraderClient
from .models import LabelEvidence, RolloutSample
from .output_parser import extract_last_boxed


def _components(edges: list[set[int]]) -> list[list[int]]:
    seen: set[int] = set()
    result: list[list[int]] = []
    for start in range(len(edges)):
        if start in seen:
            continue
        stack = [start]
        component: list[int] = []
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for other in edges[node]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        result.append(sorted(component))
    return result


def build_pseudo_label(
    group: GeneratedGroup,
    *,
    requested_rollouts: int,
    proposed_answer: str,
    verifier: dict,
    grader: GraderClient,
    identity: PolicyIdentity | None = None,
) -> LabelEvidence:
    identity_fields = {
        "request_id": group.request_id,
        "policy_run_uuid": identity.run_uuid if identity else "",
        "policy_version": identity.policy_version if identity else -1,
        "adapter_version": identity.adapter_version if identity else -1,
        "global_step": identity.global_step if identity else -1,
        "source_checkpoint": identity.source_checkpoint if identity else "",
    }
    if len(group.samples) != requested_rollouts:
        return LabelEvidence(
            None,
            [],
            0.0,
            False,
            False,
            "incomplete_label_group",
            [],
            **identity_fields,
        )
    if any(sample.status != "accepted" for sample in group.samples):
        return LabelEvidence(
            None,
            [],
            0.0,
            False,
            False,
            "label_group_contains_rejected_sample",
            [],
            **identity_fields,
        )
    rollouts: list[RolloutSample] = []
    valid_answers: list[str] = []
    valid_indices: list[int] = []
    for index, sample in enumerate(group.samples):
        answer = (
            extract_last_boxed(sample.text) if sample.status == "accepted" else None
        )
        rollouts.append(
            RolloutSample(
                response=sample.text,
                predicted_answer=answer,
                status=sample.status,
                reject_reason=sample.reject_reason,
                sample_index=index,
            )
        )
        if answer is not None:
            valid_answers.append(answer)
            valid_indices.append(index)
    if not valid_answers:
        return LabelEvidence(
            None,
            [],
            0.0,
            False,
            False,
            "no_valid_label_answers",
            rollouts,
            **identity_fields,
        )

    edges = [set([i]) for i in range(len(valid_answers))]
    for left in range(len(valid_answers)):
        for right in range(left):
            if grader.equivalent(valid_answers[left], valid_answers[right], verifier):
                edges[left].add(right)
                edges[right].add(left)
    components = _components(edges)
    # A connected component must be a clique. Otherwise the comparator induced
    # a non-transitive cluster and there is no safe pseudo-label equivalence class.
    for component in components:
        if any(other not in edges[node] for node in component for other in component):
            return LabelEvidence(
                None,
                sorted((len(x) for x in components), reverse=True),
                0.0,
                False,
                False,
                "non_transitive_answer_equivalence",
                rollouts,
                **identity_fields,
            )
    components.sort(key=lambda values: (-len(values), valid_answers[values[0]]))
    sizes = [len(values) for values in components]
    if len(components) > 1 and len(components[0]) == len(components[1]):
        return LabelEvidence(
            None,
            sizes,
            sizes[0] / requested_rollouts,
            False,
            False,
            "label_tie",
            rollouts,
            **identity_fields,
        )
    winner = components[0]
    pseudo_gold = valid_answers[winner[0]]
    agreement = len(winner) / requested_rollouts
    proposed_matches = grader.grade(proposed_answer, pseudo_gold, verifier)
    return LabelEvidence(
        pseudo_gold=pseudo_gold,
        cluster_sizes=sizes,
        agreement=agreement,
        proposed_matches=proposed_matches,
        accepted=True,
        reason=None,
        rollouts=rollouts,
        **identity_fields,
    )
