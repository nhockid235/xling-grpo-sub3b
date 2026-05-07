"""Pytest cho R1 correctness reward."""

from __future__ import annotations

import pytest


def test_r1_returns_list_of_floats():
    from src.rewards.correctness import r1_correctness

    out = r1_correctness(
        prompts=["p1", "p2"],
        completions=["c1", "c2"],
        answer=["1", "2"],
    )
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(x, float) for x in out)


def test_r1_correct_answer_returns_one():
    from src.rewards.correctness import r1_correctness

    out = r1_correctness(
        prompts=["p"],
        completions=["<answer>\\boxed{4}</answer>"],
        answer=["4"],
    )
    assert out == [1.0]


def test_r1_wrong_answer_returns_zero():
    from src.rewards.correctness import r1_correctness

    out = r1_correctness(
        prompts=["p"],
        completions=["<answer>\\boxed{5}</answer>"],
        answer=["4"],
    )
    assert out == [0.0]


def test_r1_sympy_equivalence():
    """Math-Verify nên match equivalent forms: 1/2 == 0.5 == 0.50."""
    from src.rewards.correctness import r1_correctness

    out = r1_correctness(
        prompts=["p", "p", "p"],
        completions=[
            "<answer>\\boxed{1/2}</answer>",
            "<answer>\\boxed{0.5}</answer>",
            "<answer>\\boxed{0.50}</answer>",
        ],
        answer=["0.5", "0.5", "0.5"],
    )
    assert out == [1.0, 1.0, 1.0]
