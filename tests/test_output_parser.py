from rq_evolve_v02.output_parser import (
    candidate_id,
    extract_last_boxed,
    parse_problem_response,
)


def test_parse_problem_response_accepts_think_prefix_and_nested_box() -> None:
    parsed, error = parse_problem_response(
        "<think>private work</think>\n"
        "<question>  Find   the value. </question>\n"
        r"\boxed{\frac{1}{2}}"
    )
    assert error is None
    assert parsed is not None
    assert parsed.question == "Find the value."
    assert parsed.answer == r"\frac{1}{2}"


def test_parse_problem_response_fails_closed_on_public_noise_or_copies() -> None:
    cases = {
        "preamble<question>Q?</question>\\boxed{1}": "missing_question_open",
        "<question>Q?</question>\\boxed{1} trailing": "trailing_output",
        "<question>Q?</question><question>Q2?</question>\\boxed{1}": "multiple_question_blocks",
        "<question>Q?</question>1": "missing_boxed_answer",
        "<question>Q?</question>\\boxed{": "unclosed_boxed_answer",
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
