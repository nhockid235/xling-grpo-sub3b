"""R2 -- Format reward (Open-RS native).

Returns 1.0 if the completion contains both `<think>...</think>` and
`<answer>...</answer>` blocks (DeepSeek-R1 native format). Used in all conditions.
"""

from __future__ import annotations

import re
from typing import Any

from src.rewards import register

# Open-RS verbatim format reward: require BOTH <think>...</think> AND <answer>...</answer>
_FORMAT_PATTERN = re.compile(
    r"<think>.+?</think>.*<answer>.+?</answer>",
    re.DOTALL,
)


def _has_required_tags(completion: str) -> bool:
    return bool(_FORMAT_PATTERN.search(completion))


@register("format")
def r2_format(
    prompts: list[str],
    completions: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward 1.0 if completion contains both `<think>...</think>` and `<answer>...</answer>` blocks.

    Matches Open-RS `format_reward`.

    Returns:
        list[float] of length len(prompts), values in {0.0, 1.0}.
    """
    if len(completions) != len(prompts):
        raise ValueError(
            f"Length mismatch: prompts={len(prompts)} completions={len(completions)}"
        )
    return [1.0 if _has_required_tags(c) else 0.0 for c in completions]
