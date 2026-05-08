"""Pytest cho R2 format reward — Open-RS verbatim format.

Phase 8.2 (2026-05-06): R2 require BOTH `<think>...</think>` AND
`<answer>...</answer>` blocks (matches Open-RS training system_prompt spec)."""

from __future__ import annotations

import pytest


def test_r2_both_tags_returns_one():
    """"""
    from src.rewards.format import r2_format
    out = r2_format(
        prompts=["p"] * 3,
        completions=[
            "<think>2+2=4</think><answer>\\boxed{4}</answer>",
            "<think>5*7=35</think>\n<answer>35</answer>",
            "Some text <think>x=12/3=4</think> more <answer>4</answer>",
        ],
    )
    assert out == [1.0, 1.0, 1.0]


def test_r2_no_tags_returns_zero():
    from src.rewards.format import r2_format
    out = r2_format(
        prompts=["p"] * 3,
        completions=[
            "Just 4",
            "Without any thinking, 35",
            "Answer: 4",
        ],
    )
    assert out == [0.0, 0.0, 0.0]


def test_r2_only_think_returns_zero():
    """"""
    from src.rewards.format import r2_format
    out = r2_format(
        prompts=["p"],
        completions=["<think>reasoning</think>The answer is 4"],
    )
    assert out == [0.0]


def test_r2_only_answer_returns_zero():
    """"""
    from src.rewards.format import r2_format
    out = r2_format(
        prompts=["p"],
        completions=["<answer>4</answer>"],
    )
    assert out == [0.0]


def test_r2_wrong_order_returns_zero():
    """"""
    from src.rewards.format import r2_format
    out = r2_format(
        prompts=["p"],
        completions=["<answer>4</answer><think>reasoning</think>"],
    )
    assert out == [0.0]


def test_r2_length_mismatch_raises():
    from src.rewards.format import r2_format
    with pytest.raises(ValueError, match="Length mismatch"):
        r2_format(prompts=["p1", "p2"], completions=["c1"])
