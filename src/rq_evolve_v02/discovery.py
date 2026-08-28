"""Concrete child generation, pseudo-labeling, descriptors, and admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .archive import ConcreteMapArchive
from .backends import GeneratedGroup, PolicyBackend, PolicyIdentity, SamplingSpec
from .config import AppConfig
from .domain import label_domains
from .grading import GraderClient
from .labeling import build_pseudo_label
from .models import Candidate, CandidateEvent, LabelEvidence, ParentPair, ProblemRecord
from .novelty import NoveltyDecision, NoveltyIndex
from .output_parser import parse_problem_response
from .problem_type import annotate_problem_type, verifier_for_problem_type
from .prompts import PromptBook
from .storage import RunStore
from .utils import canonical_text, stable_id
from .verifier import verifier_for_answer


@dataclass(slots=True)
class CandidateDraft:
    candidate: Candidate
    problem_type: str
    preliminary_verifier: dict[str, Any]
    preliminary_novelty: NoveltyDecision


@dataclass(slots=True)
class DiscoveryWaveResult:
    generated: int
    parsed: int
    label_accepted: int
    domain_accepted: int
    archived: list[ProblemRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated": self.generated,
            "parsed": self.parsed,
            "label_accepted": self.label_accepted,
            "domain_accepted": self.domain_accepted,
            "archived_problem_ids": [record.problem_id for record in self.archived],
        }


class DiscoveryRunner:
    def __init__(
        self,
        *,
        config: AppConfig,
        backend: PolicyBackend,
        prompts: PromptBook,
        grader: GraderClient,
        store: RunStore,
        archive: ConcreteMapArchive,
        novelty: NoveltyIndex,
    ) -> None:
        self.config = config
        self.backend = backend
        self.prompts = prompts
        self.grader = grader
        self.store = store
        self.archive = archive
        self.novelty = novelty

    def _event(
        self,
        *,
        iteration: int,
        wave_index: int,
        phase: str,
        status: str,
        reason: str | None,
        candidate_id: str | None,
        identity: tuple[Any, ...],
        details: dict[str, Any] | None = None,
    ) -> None:
        self.store.append_event(
            CandidateEvent(
                event_id=stable_id(
                    "event", iteration, wave_index, phase, status, identity
                ),
                iteration=iteration,
                candidate_id=candidate_id,
                phase=phase,
                status=status,
                reason=reason,
                details=details or {},
            )
        )

    def _assert_policy(self, frozen: PolicyIdentity) -> None:
        if self.backend.policy_identity != frozen:
            raise RuntimeError(
                "policy identity changed during the frozen discovery cycle"
            )

    def _generate_drafts(
        self,
        *,
        iteration: int,
        wave_index: int,
        pairs: list[ParentPair],
        frozen: PolicyIdentity,
    ) -> tuple[int, list[CandidateDraft]]:
        cfg = self.config
        messages: list[list[dict[str, str]]] = []
        request_ids: list[str] = []
        pair_by_request: dict[str, ParentPair] = {}
        for pair in pairs:
            left, right = (
                self.archive.records[pair.left_id],
                self.archive.records[pair.right_id],
            )
            request_id = stable_id(
                "crossover",
                frozen.run_uuid,
                frozen.policy_version,
                iteration,
                wave_index,
                pair.pair_id,
            )
            request_ids.append(request_id)
            pair_by_request[request_id] = pair
            messages.append(self.prompts.crossover_messages(left, right))
        groups = self.backend.generate(
            messages,
            request_ids=request_ids,
            sampling=SamplingSpec(
                n=cfg.generation.children_per_pair,
                temperature=cfg.generation.temperature,
                top_p=cfg.generation.top_p,
                max_tokens=cfg.generation.max_tokens,
            ),
            purpose="crossover",
        )
        self._assert_policy(frozen)
        output_by_id = {group.request_id: group for group in groups}
        if len(output_by_id) != len(groups):
            raise RuntimeError("crossover backend returned duplicate request IDs")
        generated = 0
        drafts: list[CandidateDraft] = []
        for request_id in request_ids:
            pair = pair_by_request[request_id]
            group = output_by_id.get(request_id)
            if group is None:
                self._event(
                    iteration=iteration,
                    wave_index=wave_index,
                    phase="generation",
                    status="rejected",
                    reason="missing_generation_group",
                    candidate_id=None,
                    identity=(pair.pair_id,),
                    details={"pair": pair.to_dict(), "request_id": request_id},
                )
                continue
            if len(group.samples) != cfg.generation.children_per_pair:
                self._event(
                    iteration=iteration,
                    wave_index=wave_index,
                    phase="generation",
                    status="rejected",
                    reason="incomplete_generation_group",
                    candidate_id=None,
                    identity=(pair.pair_id,),
                    details={
                        "pair": pair.to_dict(),
                        "request_id": request_id,
                        "received": len(group.samples),
                    },
                )
            for child_index, sample in enumerate(group.samples):
                generated += 1
                attempt_id = stable_id(
                    "candidate",
                    frozen.run_uuid,
                    frozen.policy_version,
                    iteration,
                    wave_index,
                    pair.pair_id,
                    child_index,
                )
                identity = (pair.pair_id, child_index)
                if sample.status != "accepted":
                    self._event(
                        iteration=iteration,
                        wave_index=wave_index,
                        phase="parse",
                        status="rejected",
                        reason=sample.reject_reason or "generation_rejected",
                        candidate_id=attempt_id,
                        identity=identity,
                        details={"raw_response": sample.text},
                    )
                    continue
                parsed, parse_error = parse_problem_response(sample.text)
                if parsed is None:
                    self._event(
                        iteration=iteration,
                        wave_index=wave_index,
                        phase="parse",
                        status="rejected",
                        reason=parse_error,
                        candidate_id=attempt_id,
                        identity=identity,
                        details={"raw_response": sample.text},
                    )
                    continue
                question = canonical_text(parsed.question)
                if (
                    not cfg.novelty.min_question_chars
                    <= len(question)
                    <= cfg.novelty.max_question_chars
                ):
                    self._event(
                        iteration=iteration,
                        wave_index=wave_index,
                        phase="parse",
                        status="rejected",
                        reason="question_length_out_of_bounds",
                        candidate_id=attempt_id,
                        identity=identity,
                        details={
                            "question_length": len(question),
                            "raw_response": sample.text,
                        },
                    )
                    continue
                annotation = annotate_problem_type(question)
                if annotation.problem_type is None or annotation.confidence != "high":
                    self._event(
                        iteration=iteration,
                        wave_index=wave_index,
                        phase="problem_type",
                        status="rejected",
                        reason=annotation.review_reason or "ambiguous_problem_type",
                        candidate_id=attempt_id,
                        identity=identity,
                        details={
                            "annotation": (
                                annotation.__dict__
                                if hasattr(annotation, "__dict__")
                                else {
                                    "problem_type": annotation.problem_type,
                                    "confidence": annotation.confidence,
                                    "evidence": annotation.evidence,
                                    "request_window": annotation.request_window,
                                    "review_reason": annotation.review_reason,
                                }
                            )
                        },
                    )
                    continue
                left = self.archive.records[pair.left_id]
                right = self.archive.records[pair.right_id]
                novelty = self.novelty.check(
                    question,
                    parent_questions=[left.question, right.question],
                    near_threshold=cfg.novelty.near_duplicate_threshold,
                    parent_ceiling=cfg.novelty.parent_similarity_ceiling,
                )
                if not novelty.accepted:
                    self._event(
                        iteration=iteration,
                        wave_index=wave_index,
                        phase="preliminary_novelty",
                        status="rejected",
                        reason=novelty.reason,
                        candidate_id=attempt_id,
                        identity=identity,
                        details={"novelty": novelty.to_dict(), "question": question},
                    )
                    continue
                candidate = Candidate(
                    candidate_id=attempt_id,
                    question=question,
                    proposed_answer=parsed.answer,
                    parent_ids=(pair.left_id, pair.right_id),
                    pair_id=pair.pair_id,
                    iteration=iteration,
                    child_index=child_index,
                    raw_response=sample.text,
                )
                drafts.append(
                    CandidateDraft(
                        candidate=candidate,
                        problem_type=annotation.problem_type,
                        preliminary_verifier=verifier_for_problem_type(
                            annotation.problem_type
                        ),
                        preliminary_novelty=novelty,
                    )
                )
                self._event(
                    iteration=iteration,
                    wave_index=wave_index,
                    phase="parse",
                    status="accepted",
                    reason=None,
                    candidate_id=attempt_id,
                    identity=identity,
                    details={
                        "candidate": candidate.to_dict(),
                        "problem_type": annotation.problem_type,
                    },
                )
        return generated, drafts

    def _label_drafts(
        self,
        drafts: list[CandidateDraft],
        *,
        iteration: int,
        wave_index: int,
        frozen: PolicyIdentity,
    ) -> list[tuple[CandidateDraft, LabelEvidence]]:
        cfg = self.config.labeling
        pending = list(drafts)
        accepted: list[tuple[CandidateDraft, LabelEvidence]] = []
        for attempt in range(cfg.max_infrastructure_retries + 1):
            if not pending:
                break
            request_ids = [
                stable_id(
                    "label",
                    frozen.run_uuid,
                    frozen.policy_version,
                    iteration,
                    wave_index,
                    draft.candidate.candidate_id,
                    attempt,
                )
                for draft in pending
            ]
            groups = self.backend.generate(
                [
                    self.prompts.solver_messages(draft.candidate.question)
                    for draft in pending
                ],
                request_ids=request_ids,
                sampling=SamplingSpec(
                    n=cfg.num_rollouts,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    max_tokens=cfg.max_tokens,
                ),
                purpose="label",
                ground_truths=None,
                verifiers=None,
            )
            self._assert_policy(frozen)
            by_id = {group.request_id: group for group in groups}
            retry: list[CandidateDraft] = []
            for draft, request_id in zip(pending, request_ids, strict=True):
                group = by_id.get(
                    request_id, GeneratedGroup(request_id=request_id, samples=[])
                )
                evidence = build_pseudo_label(
                    group,
                    requested_rollouts=cfg.num_rollouts,
                    proposed_answer=draft.candidate.proposed_answer,
                    verifier=draft.preliminary_verifier,
                    grader=self.grader,
                    min_agreement=cfg.min_agreement,
                    require_proposed_match=cfg.require_proposed_answer_match,
                    identity=frozen,
                )
                infrastructure = evidence.reason in {
                    "incomplete_label_group",
                    "label_group_contains_rejected_sample",
                }
                if infrastructure and attempt < cfg.max_infrastructure_retries:
                    self._event(
                        iteration=iteration,
                        wave_index=wave_index,
                        phase="pseudo_label",
                        status="retrying",
                        reason=evidence.reason,
                        candidate_id=draft.candidate.candidate_id,
                        identity=(draft.candidate.candidate_id, "retry", attempt),
                        details={
                            "attempt": attempt,
                            "request_id": group.request_id,
                            "received": len(group.samples),
                            "sample_statuses": [
                                sample.status for sample in group.samples
                            ],
                        },
                    )
                    retry.append(draft)
                    continue
                label_observation_id = stable_id(
                    "label_observation",
                    frozen.run_uuid,
                    frozen.policy_version,
                    iteration,
                    wave_index,
                    draft.candidate.candidate_id,
                )
                self.store.append_label_observation(
                    observation_id=label_observation_id,
                    candidate_id=draft.candidate.candidate_id,
                    iteration=iteration,
                    evidence=evidence,
                )
                status = "accepted" if evidence.accepted else "rejected"
                self._event(
                    iteration=iteration,
                    wave_index=wave_index,
                    phase="pseudo_label",
                    status=status,
                    reason=evidence.reason,
                    candidate_id=draft.candidate.candidate_id,
                    identity=(draft.candidate.candidate_id,),
                    details={
                        "label_observation_id": label_observation_id,
                        "pseudo_gold": evidence.pseudo_gold,
                        "cluster_sizes": evidence.cluster_sizes,
                        "agreement": evidence.agreement,
                        "proposed_matches": evidence.proposed_matches,
                        "attempt": attempt,
                    },
                )
                if evidence.accepted:
                    accepted.append((draft, evidence))
            pending = retry
        return accepted

    def run_wave(
        self,
        *,
        iteration: int,
        wave_index: int,
        pairs: list[ParentPair],
        frozen: PolicyIdentity,
    ) -> DiscoveryWaveResult:
        generated, drafts = self._generate_drafts(
            iteration=iteration,
            wave_index=wave_index,
            pairs=pairs,
            frozen=frozen,
        )
        labeled = self._label_drafts(
            drafts,
            iteration=iteration,
            wave_index=wave_index,
            frozen=frozen,
        )
        if not labeled:
            return DiscoveryWaveResult(generated, len(drafts), 0, 0, [])
        evidence = label_domains(
            [draft.candidate.question for draft, _ in labeled],
            backend=self.backend,
            prompts=self.prompts,
            min_probability=self.config.domain.min_probability,
            min_logit_margin=self.config.domain.min_logit_margin,
            iteration=iteration,
            request_namespace=f"wave-{wave_index}",
        )
        self._assert_policy(frozen)
        archived: list[ProblemRecord] = []
        domain_accepted = 0
        for (draft, label), domain in zip(labeled, evidence, strict=True):
            candidate = draft.candidate
            if not domain.accepted or domain.domain is None:
                self._event(
                    iteration=iteration,
                    wave_index=wave_index,
                    phase="domain",
                    status="rejected",
                    reason=domain.reason or "ambiguous_domain",
                    candidate_id=candidate.candidate_id,
                    identity=(candidate.candidate_id,),
                    details={"domain_evidence": domain.to_dict()},
                )
                continue
            domain_accepted += 1
            assert label.pseudo_gold is not None
            try:
                verifier = verifier_for_answer(
                    draft.preliminary_verifier["mode"], label.pseudo_gold
                )
            except ValueError as exc:
                self._event(
                    iteration=iteration,
                    wave_index=wave_index,
                    phase="verifier",
                    status="rejected",
                    reason="invalid_final_verifier",
                    candidate_id=candidate.candidate_id,
                    identity=(candidate.candidate_id,),
                    details={"error": str(exc), "pseudo_gold": label.pseudo_gold},
                )
                continue
            left, right = (
                self.archive.records[candidate.parent_ids[0]],
                self.archive.records[candidate.parent_ids[1]],
            )
            final_novelty = self.novelty.check(
                candidate.question,
                parent_questions=[left.question, right.question],
                near_threshold=self.config.novelty.near_duplicate_threshold,
                parent_ceiling=self.config.novelty.parent_similarity_ceiling,
            )
            if not final_novelty.accepted:
                self._event(
                    iteration=iteration,
                    wave_index=wave_index,
                    phase="final_novelty",
                    status="rejected",
                    reason=final_novelty.reason,
                    candidate_id=candidate.candidate_id,
                    identity=(candidate.candidate_id,),
                    details={"novelty": final_novelty.to_dict()},
                )
                continue
            problem_id = stable_id(
                "problem", canonical_text(candidate.question).lower()
            )
            roots = tuple(
                sorted(
                    set(left.lineage_root_ids or (left.problem_id,))
                    | set(right.lineage_root_ids or (right.problem_id,))
                )
            )
            record = ProblemRecord(
                problem_id=problem_id,
                question=candidate.question,
                proposed_answer=candidate.proposed_answer,
                pseudo_gold=label.pseudo_gold,
                verifier=verifier,
                domain=domain.domain,
                problem_type=draft.problem_type,
                parent_ids=candidate.parent_ids,
                lineage_root_ids=roots,
                generation=max(left.generation, right.generation) + 1,
                created_iteration=iteration,
                created_policy_version=frozen.policy_version,
                label_evidence={
                    "label_observation_id": stable_id(
                        "label_observation",
                        frozen.run_uuid,
                        frozen.policy_version,
                        iteration,
                        wave_index,
                        candidate.candidate_id,
                    ),
                    "pseudo_gold": label.pseudo_gold,
                    "cluster_sizes": label.cluster_sizes,
                    "agreement": label.agreement,
                    "proposed_matches": label.proposed_matches,
                    "policy_run_uuid": label.policy_run_uuid,
                    "policy_version": label.policy_version,
                    "adapter_version": label.adapter_version,
                    "global_step": label.global_step,
                    "source_checkpoint": label.source_checkpoint,
                },
                domain_evidence=domain.to_dict(),
                novelty=final_novelty.to_dict(),
            )
            # Single-writer transaction order: append the accepted source row,
            # then expose it through the in-memory/index projections.
            if not self.store.append_problem(record):
                self._event(
                    iteration=iteration,
                    wave_index=wave_index,
                    phase="archive",
                    status="rejected",
                    reason="duplicate_problem_id",
                    candidate_id=candidate.candidate_id,
                    identity=(candidate.candidate_id,),
                    details={"problem_id": problem_id},
                )
                continue
            if not self.archive.add(record):
                raise RuntimeError(
                    f"freshly persisted problem already in MAP: {problem_id}"
                )
            self.novelty.add(record.problem_id, record.question)
            archived.append(record)
            self._event(
                iteration=iteration,
                wave_index=wave_index,
                phase="archive",
                status="accepted",
                reason=None,
                candidate_id=candidate.candidate_id,
                identity=(candidate.candidate_id,),
                details={"problem_id": problem_id, "cell": record.cell},
            )
        return DiscoveryWaveResult(
            generated=generated,
            parsed=len(drafts),
            label_accepted=len(labeled),
            domain_accepted=domain_accepted,
            archived=archived,
        )
