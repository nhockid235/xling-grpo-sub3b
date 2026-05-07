"""Pytest cho seed_everything."""

from __future__ import annotations

import random

from src.utils.seed import seed_everything


def test_seed_deterministic_python_random():
    seed_everything(42)
    a = [random.random() for _ in range(5)]

    seed_everything(42)
    b = [random.random() for _ in range(5)]

    assert a == b


def test_seed_different_seeds_differ():
    seed_everything(42)
    a = random.random()

    seed_everything(123)
    b = random.random()

    assert a != b
