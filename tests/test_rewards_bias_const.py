"""Pytest cho bias_const reward — Phase 9.3 mechanism ablation."""

from __future__ import annotations

import pytest


def test_bias_const_returns_one_always():
    from src.rewards.bias_const import bias_const
    out = bias_const(
        prompts=["p1", "p2", "p3"],
        completions=["c1", "c2", "c3"],
    )
    assert out == [1.0, 1.0, 1.0]


def test_bias_const_ignores_content():
    """Reward must be 1.0 regardless of completion quality / format / language."""
    from src.rewards.bias_const import bias_const
    out = bias_const(
        prompts=["What is 2+2?", "Tính 5*7?"],
        completions=[
            "<answer>WRONG</answer>",          # wrong
            "Random gibberish 42 hoa quả lalala",  # nonsense
        ],
    )
    assert out == [1.0, 1.0]


def test_bias_const_handles_kwargs():
    """Must accept arbitrary dataset columns via **kwargs (TRL passes them)."""
    from src.rewards.bias_const import bias_const
    out = bias_const(
        prompts=["p"],
        completions=["c"],
        answer=["42"],
        extra_column="ignored",
    )
    assert out == [1.0]


def test_bias_const_length_mismatch_raises():
    from src.rewards.bias_const import bias_const
    with pytest.raises(ValueError, match="Length mismatch"):
        bias_const(prompts=["p1", "p2"], completions=["c1"])


def test_bias_const_registered():
    from src.rewards import REWARD_REGISTRY, get_reward
    assert "bias_const" in REWARD_REGISTRY
    fn = get_reward("bias_const")
    assert fn(prompts=["p"], completions=["c"]) == [1.0]


def test_bias_const_empty_input():
    from src.rewards.bias_const import bias_const
    assert bias_const(prompts=[], completions=[]) == []
