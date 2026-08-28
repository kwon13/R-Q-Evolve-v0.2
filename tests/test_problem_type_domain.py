from __future__ import annotations

from rq_evolve_v02.domain import label_domains
from rq_evolve_v02.backends import PolicyIdentity
from rq_evolve_v02.mock_backend import DeterministicMockBackend
from rq_evolve_v02.problem_type import (
    annotate_problem_type,
    answer_contract_error,
    verifier_for_problem_type,
)
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


def test_problem_type_rejects_observed_compound_crossover_requests() -> None:
    cases = {
        (
            "A data set of 10 measurements has a mean of 41. One measurement "
            "was incorrectly recorded as 41, but its correct value is 11. "
            "Additionally, evaluate the definite integral integral from 0 to 2 "
            "of 20x^4. What is the corrected mean and the value of the integral?"
        ): "multiple_output_requests",
        (
            "A data set was reported to have mean 41. What is the corrected "
            "mean? Evaluate the integral from 0 to 2 of 20x^4. What is the sum "
            "of the corrected mean and the integral?"
        ): "multiple_output_requests",
        (
            "Find all positive integer lengths x that form the requested "
            "triangle and also determine the maximum number of edges in a "
            "triangle-free graph."
        ): "mixed_output_contract",
    }
    for question, reason in cases.items():
        annotation = annotate_problem_type(question)
        assert annotation.problem_type is None
        assert annotation.review_reason == reason

    # The second sentence only fixes the representation of one decision output.
    decision = annotate_problem_type("Does an integer solution exist? Answer Yes or No.")
    assert decision.problem_type == "decision"
    assert decision.confidence == "high"
    formatted = annotate_problem_type(
        "How many such integers are there? Give your answer as an integer."
    )
    assert formatted.problem_type == "counting"
    assert formatted.confidence == "high"
    simplified = annotate_problem_type(
        "Evaluate 1/2 + 1/3, and give your answer in simplest form."
    )
    assert simplified.problem_type == "function"
    assert simplified.confidence == "high"


def test_problem_type_answer_shape_contracts() -> None:
    assert answer_contract_error("decision", "No") is None
    assert answer_contract_error("decision", "maybe") == "decision_answer_not_boolean"
    assert answer_contract_error("search", r"\{2,3\}") is None
    assert answer_contract_error("search", "2, 3") is None
    assert answer_contract_error("search", "2") is None
    assert answer_contract_error("search", "2,,3") == "search_answer_not_finite_set"
    assert answer_contract_error("counting", "47") is None
    assert answer_contract_error("counting", "12,345") is None
    assert (
        answer_contract_error("counting", "-1")
        == "counting_answer_not_nonnegative_integer"
    )
    assert answer_contract_error("function", r"\frac{1}{2}") is None
    assert answer_contract_error("function", "0") is None
    assert answer_contract_error("optimization", "1") is None
    assert answer_contract_error("function", r"\gcd(12,18)") is None
    assert answer_contract_error("function", "(1,2)") is None
    assert answer_contract_error("function", "(1,2]") == "malformed_scalar_answer"
    assert answer_contract_error("function", "1,000") is None
    assert answer_contract_error("function", "38, 128") == "compound_scalar_answer"
    assert (
        answer_contract_error("optimization", r"\{9,16\}, 6")
        == "compound_scalar_answer"
    )


def test_problem_type_abstains_on_unverifiable_justification_and_witnesses() -> None:
    for question, reason in (
        ("Explain why 2 is prime.", "proof_or_justification"),
        ("Does a solution exist? Justify your answer.", "proof_or_justification"),
        (
            "Determine whether a solution exists and justify your answer.",
            "proof_or_justification",
        ),
        ("Can you prove that 2 is prime?", "proof_or_justification"),
        ("Construct an integer solution.", "unsupported_witness_request"),
        ("Give an example of a prime.", "unsupported_witness_request"),
    ):
        annotation = annotate_problem_type(question)
        assert annotation.problem_type is None
        assert annotation.review_reason == reason
    assert (
        answer_contract_error("function", r"38 \text{ and } 128")
        == "compound_scalar_answer"
    )


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
