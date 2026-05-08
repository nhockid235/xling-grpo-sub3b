"""Tests for the reward registry integration in the GRPO trainer."""

from __future__ import annotations

import pytest


def test_registry_contains_5_rewards():
    from src.rewards import REWARD_REGISTRY

    expected = {"correctness", "format", "length", "tag", "lang"}
    assert expected.issubset(set(REWARD_REGISTRY.keys()))


def test_get_reward_returns_callable():
    from src.rewards import get_reward

    fn = get_reward("format")
    assert callable(fn)


def test_get_reward_unknown_raises():
    from src.rewards import get_reward

    with pytest.raises(KeyError, match="unknown"):
        get_reward("unknown")


def test_format_reward_call_signature():
    """Format reward returns a list[float] with one entry per prompt."""
    from src.rewards import get_reward

    fn = get_reward("format")
    out = fn(
        prompts=["p1", "p2"],
        completions=[
            "<think>x</think><answer>1</answer>",
            "no tags",
        ],
    )
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(x, float) for x in out)


def test_correctness_reward_handles_kwargs():
    """Reward must accept extra dataset columns via **kwargs."""
    from src.rewards import get_reward

    fn = get_reward("correctness")
    out = fn(
        prompts=["p"],
        completions=["<answer>4</answer>"],
        answer=["4"],
        extra_unused_column="should not crash",
    )
    assert isinstance(out, list)
    assert len(out) == 1


def test_partial_lang_reward():
    """Lang reward can be partial-bound with config args (mirrors GRPO trainer setup)."""
    from functools import partial
    from src.rewards import get_reward

    base_fn = get_reward("lang")
    bound = partial(
        base_fn,
        fasttext_model="nonexistent.bin",
        no_penalty_for_en=True,
        min_response_tokens=10,
    )
    out = bound(prompts=["p"], completions=["short"])
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0] == 0.0
