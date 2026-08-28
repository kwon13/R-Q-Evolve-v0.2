from __future__ import annotations

from rq_evolve_v02.domain import label_domains
from rq_evolve_v02.backends import PolicyIdentity
from rq_evolve_v02.mock_backend import DeterministicMockBackend
from rq_evolve_v02.problem_type import annotate_problem_type, verifier_for_problem_type
from rq_evolve_v02.prompts import PromptBook


def test_problem_type_examples_and_proof_abstention() -> None:
    cases = {
        "Determine whether there exists an integer x with x^2=3.": "decision",
        "Find all integers x satisfying x^2-5x+6=0.": "search",
        "How many integers from 1 to 100 are divisible by 3?": "counting",
        "Find the maximum value of xy when x+y=20.": "optimization",
        "Find the remainder when 2^100 is divided by 7.": "function",
    }
    for question, expected in cases.items():
        annotation = annotate_problem_type(question)
        assert annotation.problem_type == expected
        assert annotation.confidence == "high"
    proof = annotate_problem_type("Prove that there are infinitely many primes.")
    assert proof.problem_type is None
    assert proof.review_reason == "proof_or_justification"
    assert verifier_for_problem_type("decision") == {"mode": "boolean"}
    assert verifier_for_problem_type("search") == {"mode": "set", "elements": []}


def test_seven_arm_domain_labeling_with_mock_policy() -> None:
    question = "Find all real roots of x^2-5x+6=0."
    backend = DeterministicMockBackend(domain_labels={question: "algebra"})
    evidence = label_domains(
        [question],
        backend=backend,
        prompts=PromptBook(),
        min_probability=0.55,
        min_logit_margin=0.5,
        iteration=2,
    )
    assert len(evidence) == 1
    assert evidence[0].accepted
    assert evidence[0].domain == "algebra"
    assert evidence[0].probabilities["algebra"] == 0.97
    assert len(backend.binary_calls[0]["request_ids"]) == 7


class FlatDomainBackend:
    policy_identity = PolicyIdentity("flat", 0, 0, 0, "mock://flat")

    def binary_token_probabilities(self, messages, *, request_ids, purpose):
        return [{"YES": 0.5, "NO": 0.5} for _ in request_ids]


def test_domain_labeling_abstains_on_tied_arms() -> None:
    evidence = label_domains(
        ["What is 2+2?"],
        backend=FlatDomainBackend(),  # type: ignore[arg-type]
        prompts=PromptBook(),
        min_probability=0.55,
        min_logit_margin=0.5,
        iteration=0,
    )[0]
    assert not evidence.accepted
    assert evidence.domain is None
    assert evidence.reason == "no_high_confidence_domain"


class TwoPositiveDomainBackend:
    policy_identity = PolicyIdentity("multi", 0, 0, 0, "mock://multi")

    def binary_token_probabilities(self, messages, *, request_ids, purpose):
        rows = [{"YES": 0.1, "NO": 0.9} for _ in request_ids]
        rows[0] = {"YES": 0.90, "NO": 0.10}
        rows[1] = {"YES": 0.60, "NO": 0.40}
        return rows


def test_domain_labeling_requires_exactly_one_high_confidence_arm() -> None:
    evidence = label_domains(
        ["A deliberately ambiguous problem."],
        backend=TwoPositiveDomainBackend(),  # type: ignore[arg-type]
        prompts=PromptBook(),
        min_probability=0.55,
        min_logit_margin=0.5,
        iteration=0,
    )[0]
    assert not evidence.accepted
    assert evidence.domain is None
    assert evidence.reason == "multiple_high_confidence_domains"
