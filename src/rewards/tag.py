"""R4 — Tag count reward.

Counts of `<think>`, `</think>`, `<answer>`, `</answer>` must each == 1.
Sub-reward of format check, but discrete to give partial credit.
Used in: all conditions."""

from __future__ import annotations

from typing import Any

from src.rewards import register

_EXPECTED_TAGS: tuple[str, ...] = ("<think>", "</think>", "<answer>", "</answer>")


def _score_one(completion: str) -> float:
    """"""
    score = 0.0
    for tag in _EXPECTED_TAGS:
        if completion.count(tag) == 1:
            score += 0.25
    return score


@register("tag")
def r4_tag_count(
    prompts: list[str],
    completions: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward = fraction of 4 expected tags ({open,close} × {think,answer}) matched exactly.

    Returns:
    """
    if len(completions) != len(prompts):
        raise ValueError(
            f"Length mismatch: prompts={len(prompts)} completions={len(completions)}"
        )
    return [_score_one(c) for c in completions]
