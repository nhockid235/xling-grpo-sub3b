"""Answer extraction parsers -- regex helpers (Math-Verify wrappers live in eval._common)."""

from __future__ import annotations

import re

BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
GSM8K_GOLD_PATTERN = re.compile(r"####\s*(.+)")
LAST_NUMBER_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")


def extract_answer_tag(text: str) -> str | None:
    """Extract content from `<answer>...</answer>`."""
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
    """Return the last numeric substring in ``text`` (signed, decimal), or None."""
    matches = LAST_NUMBER_PATTERN.findall(text)
    return matches[-1] if matches else None
