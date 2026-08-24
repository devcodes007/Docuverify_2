from app.generation.llm import MockLLMProvider


def test_mock_llm_focuses_answer_on_question_terms():
    prompt = """Question: give me summary of problem 1.3

Evidence:
[1] (source: uploaded-pdf://Problems.pdf, section: Problems > Page 1)
Problem 1.1 asks about linear equations. Problem 1.3 discusses probability distributions.

[2] (source: uploaded-pdf://Problems.pdf, section: Problems > Page 2)
Problem 2.1 covers optimization and gradients."""

    answer = MockLLMProvider().generate("", prompt)

    assert "Problem 1.3 discusses probability distributions" in answer
    assert "Problem 2.1 covers optimization" not in answer
