"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_prompts() -> list[str]:
    return [
        "What is 2 + 2?",
        "Tính 5 × 7?",
        "Solve x in 3x = 12.",
    ]


@pytest.fixture
def sample_completions_correct() -> list[str]:
    return [
        "<think>2 plus 2 equals 4.</think><answer>\\boxed{4}</answer>",
        "<think>5 nhân 7 = 35.</think><answer>\\boxed{35}</answer>",
        "<think>x = 12/3 = 4.</think><answer>\\boxed{4}</answer>",
    ]


@pytest.fixture
def sample_completions_malformed() -> list[str]:
    return [
        "Just 4",
        "<think>Tính: 5*7=35</think>35",
        "Answer: 4",
    ]
