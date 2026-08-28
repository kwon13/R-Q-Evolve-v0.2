from rq_evolve_v02.output_parser import (
    candidate_id,
    extract_last_boxed,
    parse_problem_response,
)


def test_parse_problem_response_accepts_think_prefix_and_nested_box() -> None:
    parsed, error = parse_problem_response(
        "<think>private work</think>\n"
        "<question>  Find   the value. </question>\n"
        "<domain>algebra</domain>\n"
        r"\boxed{\frac{1}{2}}"
    )
    assert error is None
    assert parsed is not None
    assert parsed.question == "Find the value."
    assert parsed.domain == "algebra"
    assert parsed.answer == r"\frac{1}{2}"


def test_parse_problem_response_accepts_box_before_domain() -> None:
    parsed, error = parse_problem_response(
        "<question>Find the value.</question>"
        r"\boxed{\frac{1}{2}}"
        "<domain>algebra</domain>"
    )
    assert error is None
    assert parsed is not None
    assert parsed.domain == "algebra"
    assert parsed.answer == r"\frac{1}{2}"


def test_parse_problem_response_fails_closed_on_public_noise_or_copies() -> None:
    cases = {
        "preamble<question>Q?</question><domain>algebra</domain>\\boxed{1}": "missing_question_open",
        "<question>Q?</question>\\boxed{1}": "missing_domain_open",
        "<question>Q?</question><domain>algebra\\boxed{1}": "missing_domain_close",
        "<question>Q?</question><domain>Algebra</domain>\\boxed{1}": "invalid_domain",
        "<question>Q?</question><domain>algebra,geometry</domain>\\boxed{1}": "invalid_domain",
        "<question>Q?</question><domain>algebra</domain><domain>geometry</domain>\\boxed{1}": "multiple_domain_blocks",
        "<question>Q?</question><domain>algebra</domain>\\boxed{1} trailing": "trailing_output",
        "<question>Q?</question><question>Q2?</question><domain>algebra</domain>\\boxed{1}": "multiple_question_blocks",
        "<question>Q?</question><domain>algebra</domain>1": "missing_boxed_answer",
        "<question>Q?</question><domain>algebra</domain>\\boxed{": "unclosed_boxed_answer",
    }
    for text, expected in cases.items():
        parsed, error = parse_problem_response(text)
        assert parsed is None
        assert error == expected


def test_last_boxed_and_candidate_id_are_stable() -> None:
    assert (
        extract_last_boxed(r"work \boxed{1}; correction \boxed{\sqrt{4}}")
        == r"\sqrt{4}"
    )
    assert extract_last_boxed("no final answer") is None
    assert candidate_id("What   is 2+2?", "4") == candidate_id("What is 2+2?", "4")
