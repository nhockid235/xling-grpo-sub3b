"""R2 — Format reward (Open-RS native).

Returns 1.0 nếu completion chứa `<think>...</think>` block (DeepSeek-R1 native format).
Used in: all conditions.

Phase 8 update (2026-05-06): Đổi từ `<think>...<answer>...</answer>` (paper original)
sang Open-RS native pattern để tương thích DeepSeek-R1-Distill base — model này được
distilled với format `<think>...</think>{...\\boxed{N}}` (không có `<answer>` tag).
"""

from __future__ import annotations

import re
from typing import Any

from src.rewards import register

# Open-RS verbatim format reward: require BOTH <think>...</think> AND
# <answer>...</answer> blocks. Verified từ Open-RS recipes/grpo.yaml + lighteval task.
# Training system prompt yêu cầu cả 2 tags → reward kiểm tra cả 2.
# DeepSeek chat template auto-prepend `<think>\n` không ảnh hưởng vì model output
# vẫn cần đóng `</think>` và mở/đóng `<answer>...</answer>`.
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
    """Reward 1.0 if completion contains a non-empty `<think>...</think>` block.

    Match Open-RS reward function (their `format_reward`).

    Returns:
        list[float] of length len(prompts), values in {0.0, 1.0}.
    """
    if len(completions) != len(prompts):
        raise ValueError(
            f"Length mismatch: prompts={len(prompts)} completions={len(completions)}"
        )
    return [1.0 if _has_required_tags(c) else 0.0 for c in completions]
