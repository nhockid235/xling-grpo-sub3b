"""Constant-bias reward — Phase 9.3 mechanism ablation.

Returns 1.0 for every (prompt, completion) pair, regardless of content.
Used to test whether A3's R5 lang-consistency reward acts via:
    (a) reward magnitude → PPO clipping geometry effect (this reward replicates)
    (b) reward content → R5 specifically signals language match (this reward fails)

If A4 (R1 + R2 + bias_const) ≈ A3 (R1 + R2 + R5) on AIME-2024,
hypothesis (a) is supported. Otherwise (b) is supported.

This reward intentionally provides ZERO content signal — its only role is to
shift the mean reward by a constant 1.0, matching A3's reward magnitude on
English data (where R5 fires uniformly at 1.0).
"""

from __future__ import annotations

from typing import Any

from src.rewards import register


@register("bias_const")
def bias_const(
    prompts: list[str],
    completions: list[str],
    **kwargs: Any,
) -> list[float]:
    """Return 1.0 for every input pair.

    Phase 9 Tier 2 mechanism ablation. Total reward in A4:
        R_total = R1_correctness + R2_format + 1.0
    matches the magnitude of A3:
        R_total = R1_correctness + R2_format + R5_lang_consistency
    when R5 fires uniformly at 1.0 (which it does on EN data).

    Returns:
        list[float] of length len(prompts), all entries = 1.0.
    """
    if len(completions) != len(prompts):
        raise ValueError(
            f"Length mismatch: prompts={len(prompts)} completions={len(completions)}"
        )
    return [1.0] * len(prompts)
