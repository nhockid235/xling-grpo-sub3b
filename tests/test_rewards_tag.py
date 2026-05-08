"""Pytest cho R4 tag count reward."""

from __future__ import annotations

import pytest


def test_r4_all_4_tags_returns_one(sample_completions_correct):
    from src.rewards.tag import r4_tag_count
    out = r4_tag_count(prompts=["p"] * 3, completions=sample_completions_correct)
    assert out == [1.0, 1.0, 1.0]


def test_r4_no_tags_returns_zero():
    from src.rewards.tag import r4_tag_count
    out = r4_tag_count(prompts=["p"], completions=["just text"])
    assert out == [0.0]


def test_r4_partial_credit():
    """"""
    from src.rewards.tag import r4_tag_count
    out = r4_tag_count(prompts=["p"], completions=["<think>x</think>"])
    assert out == [0.5]


def test_r4_duplicate_tags_returns_less():
    """"""
    from src.rewards.tag import r4_tag_count
    out = r4_tag_count(
        prompts=["p"],
        completions=["<think>x</think><think>y</think><answer>1</answer>"],
    )
    assert out[0] < 1.0
