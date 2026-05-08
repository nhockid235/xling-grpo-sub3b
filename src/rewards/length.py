"""R3 -- Length cosine reward (DAPO-style).

Cosine schedule on response length. Encourages medium-length CoT,
penalizes too-short and too-long. Used in all conditions.

Reference: DAPO arXiv:2503.14476 (token-level normalization)."""

from __future__ import annotations

import math
from typing import Any

from src.rewards import register


def _token_count(text: str) -> int:
    """Whitespace-token approximation."""
    return len(text.split())


@register("length")
def r3_length_cosine(
    prompts: list[str],
    completions: list[str],
    min_length: int = 32,
    max_length: int = 3584,
    **kwargs: Any,
) -> list[float]:
    """Cosine reward over response length.

    Defines mid = (min_length + max_length) / 2 and span = (max_length - min_length) / 2.
    When length == mid, cos(0) = 1 -> reward = 1.0. Outside [min_length, max_length]
    the reward is 0.0.

    Returns:
        list[float] of length len(prompts), values in [0.0, 1.0].
    """
    if max_length <= min_length:
        raise ValueError(
            f"max_length ({max_length}) must be > min_length ({min_length})"
        )
    if len(completions) != len(prompts):
        raise ValueError(
            f"Length mismatch: prompts={len(prompts)} completions={len(completions)}"
        )

    mid = (min_length + max_length) / 2.0
    span = (max_length - min_length) / 2.0

    rewards: list[float] = []
    for c in completions:
        length = _token_count(c)
        if length < min_length or length > max_length:
            rewards.append(0.0)
            continue
        x = math.pi * (length - mid) / span
        val = 0.5 * (1.0 + math.cos(x))
        rewards.append(float(max(0.0, min(1.0, val))))
    return rewards
