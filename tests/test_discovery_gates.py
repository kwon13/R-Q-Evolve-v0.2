from __future__ import annotations

from pathlib import Path

from rq_evolve_v02.archive import ConcreteMapArchive
from rq_evolve_v02.backends import GeneratedSample
from rq_evolve_v02.config import AppConfig
from rq_evolve_v02.discovery import DiscoveryRunner
from rq_evolve_v02.grading import GraderClient
from rq_evolve_v02.mock_backend import DeterministicMockBackend
from rq_evolve_v02.models import ParentPair
from rq_evolve_v02.novelty import NoveltyIndex
from rq_evolve_v02.prompts import PromptBook
from rq_evolve_v02.storage import RunStore, read_jsonl
from rq_evolve_v02.utils import stable_id

from .helpers import make_record


class TracingMockBackend(DeterministicMockBackend):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.call_order: list[str] = []

    def generate(self, *args, purpose: str, **kwargs):
        self.call_order.append(purpose)
        return super().generate(*args, purpose=purpose, **kwargs)

class CountingPseudoGoldBackend(TracingMockBackend):
    def generate(self, *args, purpose: str, **kwargs):
        groups = super().generate(*args, purpose=purpose, **kwargs)
        if purpose == "label":
            for group in groups:
                for sample in group.samples:
                    sample.text = r"Independent reasoning. \boxed{1+1}"
        return groups


class AlwaysEquivalentGrader:
    def equivalent(self, left: str, right: str, verifier: dict) -> bool:
        return True

    def grade(self, pred: str, gold: str, verifier: dict) -> bool:
        return True


def test_sibling_gate_runs_before_domain_and_pseudo_label(tmp_path: Path) -> None:
    left = make_record("left", "What is 7+8?")
    right = make_record(
        "right",
        "Find the area of a square with side length 9.",
        domain="geometry",
    )
    archive = ConcreteMapArchive([left, right])
    pair = ParentPair("pair", left.problem_id, right.problem_id, 0)
    identity = TracingMockBackend().policy_identity
    request_id = stable_id(
        "crossover",
        identity.run_uuid,
        identity.policy_version,
        0,
        0,
        pair.pair_id,
    )
    first = (
        "A box contains 17 red balls and 23 blue balls. What is the total "
        "number of balls in the box?"
    )
    second = (
        "A box has 19 red balls and 29 blue balls. What is the total number "
        "of balls in the box?"
    )
    backend = TracingMockBackend(
        identity=identity,
        scripts={
            ("crossover", request_id): [
                f"<question>{first}</question><domain>applied_mathematics</domain>\\boxed{{40}}",
                f"<question>{second}</question><domain>applied_mathematics</domain>\\boxed{{48}}",
            ]
        },
        answers={first: "40", second: "48"},
    )
    config = AppConfig()
    store = RunStore(tmp_path / "run", fsync_jsonl=False)
    grader = GraderClient(timeout_s=1.0)
    runner = DiscoveryRunner(
        config=config,
        backend=backend,
        prompts=PromptBook(),
        grader=grader,
        store=store,
        archive=archive,
        novelty=NoveltyIndex([left, right]),
    )
    try:
        result = runner.run_wave(
            iteration=0,
            wave_index=0,
            pairs=[pair],
            frozen=identity,
        )
    finally:
        grader.close()

    assert result.generated == 2
    assert result.parsed == 2
    assert result.preflight_accepted == 2
    assert result.domain_accepted == 2
    assert result.wave_novelty_accepted == 1
    assert result.label_accepted == 1
    assert len(result.archived) == 1
    assert backend.call_order[:2] == ["crossover", "label"]
    assert result.archived[0].domain == "applied_mathematics"
    assert result.archived[0].domain_evidence == {
        "accepted": True,
        "domain": "applied_mathematics",
        "source": "crossover_self_report",
        "independently_verified": False,
    }

    events = list(read_jsonl(store.events_path))
    sibling_rejections = [
        event
        for event in events
        if event["phase"] == "wave_novelty"
        and event["status"] == "rejected"
    ]
    assert len(sibling_rejections) == 1
    assert sibling_rejections[0]["reason"] == "sibling_near_duplicate"
    assert sibling_rejections[0]["details"]["wave_novelty"]["maximum"] >= 0.82


def test_invalid_declared_domain_never_reaches_pseudo_label(tmp_path: Path) -> None:
    left = make_record("left", "What is 7+8?")
    right = make_record(
        "right",
        "Find the area of a square with side length 9.",
        domain="geometry",
    )
    archive = ConcreteMapArchive([left, right])
    pair = ParentPair("pair", left.problem_id, right.problem_id, 0)
    identity = DeterministicMockBackend().policy_identity
    request_id = stable_id(
        "crossover",
        identity.run_uuid,
        identity.policy_version,
        0,
        0,
        pair.pair_id,
    )
    first = (
        "A box contains 17 red balls and 23 blue balls. What is the total "
        "number of balls in the box?"
    )
    second = "How many diagonals does a convex decagon have?"
    backend = TracingMockBackend(
        identity=identity,
        scripts={
            ("crossover", request_id): [
                f"<question>{first}</question><domain>Algebra</domain>\\boxed{{40}}",
                f"<question>{second}</question><domain>algebra,geometry</domain>\\boxed{{35}}",
            ]
        },
        answers={first: "40", second: "35"},
    )
    store = RunStore(tmp_path / "run", fsync_jsonl=False)
    grader = GraderClient(timeout_s=1.0)
    runner = DiscoveryRunner(
        config=AppConfig(),
        backend=backend,
        prompts=PromptBook(),
        grader=grader,
        store=store,
        archive=archive,
        novelty=NoveltyIndex([left, right]),
    )
    try:
        result = runner.run_wave(
            iteration=0,
            wave_index=0,
            pairs=[pair],
            frozen=identity,
        )
    finally:
        grader.close()

    assert result.generated == 2
    assert result.parsed == 0
    assert result.preflight_accepted == 0
    assert result.domain_accepted == 0
    assert result.label_accepted == 0
    label_calls = [
        call for call in backend.generation_calls if call["purpose"] == "label"
    ]
    assert not label_calls
    assert backend.call_order == ["crossover"]
    events = list(read_jsonl(store.events_path))
    assert [event["reason"] for event in events if event["phase"] == "parse"] == [
        "invalid_domain",
        "invalid_domain",
    ]


def test_final_pseudo_gold_must_preserve_problem_type_shape(tmp_path: Path) -> None:
    left = make_record("left", "What is 7+8?")
    right = make_record(
        "right",
        "Find the area of a square with side length 9.",
        domain="geometry",
    )
    archive = ConcreteMapArchive([left, right])
    pair = ParentPair("pair", left.problem_id, right.problem_id, 0)
    identity = DeterministicMockBackend().policy_identity
    request_id = stable_id(
        "crossover",
        identity.run_uuid,
        identity.policy_version,
        0,
        0,
        pair.pair_id,
    )
    question = "How many elements are in the set containing a and b?"
    backend = CountingPseudoGoldBackend(
        identity=identity,
        scripts={
            ("crossover", request_id): [
                f"<question>{question}</question><domain>discrete_mathematics</domain>\\boxed{{2}}",
                GeneratedSample(
                    text="",
                    status="rejected",
                    reject_reason="test_generation_rejection",
                ),
            ]
        },
    )
    store = RunStore(tmp_path / "run", fsync_jsonl=False)
    runner = DiscoveryRunner(
        config=AppConfig(),
        backend=backend,
        prompts=PromptBook(),
        grader=AlwaysEquivalentGrader(),  # type: ignore[arg-type]
        store=store,
        archive=archive,
        novelty=NoveltyIndex([left, right]),
    )
    result = runner.run_wave(
        iteration=0,
        wave_index=0,
        pairs=[pair],
        frozen=identity,
    )

    assert result.preflight_accepted == 1
    assert result.domain_accepted == 1
    assert result.label_accepted == 1
    assert not result.archived
    events = list(read_jsonl(store.events_path))
    final_contract = [
        event
        for event in events
        if event["phase"] == "answer_contract"
        and event["reason"] == "pseudo_gold_counting_answer_not_nonnegative_integer"
    ]
    assert len(final_contract) == 1
