"""Answer extraction parsers — regex + Math-Verify wrappers."""

from __future__ import annotations

import re

# Pattern phổ biến cho extracting answer khỏi completion
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
GSM8K_GOLD_PATTERN = re.compile(r"####\s*(.+)")
LAST_NUMBER_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")


def extract_answer_tag(text: str) -> str | None:
    """Extract content from <answer>...</answer> tag, nếu có."""
    m = ANSWER_TAG_PATTERN.search(text)
    return m.group(1).strip() if m else None


def extract_boxed(text: str) -> str | None:
    """Extract content from `\\boxed{...}`."""
    m = BOXED_PATTERN.search(text)
    return m.group(1).strip() if m else None


def extract_gsm8k_gold(text: str) -> str | None:
    """GSM8K gold format: '#### <answer>'."""
    m = GSM8K_GOLD_PATTERN.search(text)
    return m.group(1).strip() if m else None


def extract_last_number(text: str) -> str | None:
    """Fallback: lấy số cuối trong text (cho GSM8K predictions)."""
    matches = LAST_NUMBER_PATTERN.findall(text)
    return matches[-1] if matches else None
