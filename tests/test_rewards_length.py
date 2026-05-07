"""Pytest cho R3 length cosine reward."""

from __future__ import annotations

import pytest


def test_r3_too_short_returns_zero():
    from src.rewards.length import r3_length_cosine
    out = r3_length_cosine(prompts=["p"], completions=["short"], min_length=32, max_length=3584)
    assert out == [0.0]


def test_r3_too_long_returns_zero():
    from src.rewards.length import r3_length_cosine
    very_long = "word " * 5000
    out = r3_length_cosine(prompts=["p"], completions=[very_long], min_length=32, max_length=3584)
    assert out == [0.0]


def test_r3_medium_length_positive():
    from src.rewards.length import r3_length_cosine
    medium = "word " * 500
    out = r3_length_cosine(prompts=["p"], completions=[medium], min_length=32, max_length=3584)
    assert 0.0 < out[0] <= 1.0
