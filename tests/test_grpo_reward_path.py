"""Pytest cho reward registry integration trong GRPO trainer.

Verify reward functions register-able và có thể call như TRL sẽ call.
"""

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
    """Verify format reward chấp nhận call signature TRL sẽ dùng."""
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
    """Reward must accept extra dataset columns qua **kwargs."""
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
    """R5 có thể partial-bind config-time args (như grpo.py làm)."""
    from functools import partial
    from src.rewards import get_reward

    base_fn = get_reward("lang")
    bound = partial(
        base_fn,
        fasttext_model="nonexistent.bin",
        no_penalty_for_en=True,
        min_response_tokens=10,
    )
    # Partial phải callable với same TRL signature
    out = bound(prompts=["p"], completions=["short"])
    assert isinstance(out, list)
    assert len(out) == 1
    # Dù path không tồn tại, lang.py phải gracefully return 0.0 không crash
    assert out[0] == 0.0
