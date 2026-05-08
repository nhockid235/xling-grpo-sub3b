"""Pytest cho R5 lang-consistency reward."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="needs lid.176.bin")
def test_r5_en_prompt_returns_zero_no_penalty():
    """"""
    from src.rewards.lang import r5_lang_consistency
    out = r5_lang_consistency(
        prompts=["What is 2+2?"],
        completions=["This is a long English answer explaining the math."],
        no_penalty_for_en=True,
    )
    assert out == [0.0]


@pytest.mark.skip(reason="needs lid.176.bin")
def test_r5_vi_response_to_vi_prompt_returns_one():
    from src.rewards.lang import r5_lang_consistency
    out = r5_lang_consistency(
        prompts=["Tính tổng của hai và hai là bao nhiêu?"],
        completions=["Hai cộng hai bằng bốn. Đây là phép cộng cơ bản trong toán học."],
    )
    assert out == [1.0]


@pytest.mark.skip(reason="needs lid.176.bin")
def test_r5_en_response_to_vi_prompt_returns_zero():
    from src.rewards.lang import r5_lang_consistency
    out = r5_lang_consistency(
        prompts=["Tính tổng của hai và hai là bao nhiêu?"],
        completions=["Two plus two equals four. This is basic arithmetic."],
    )
    assert out == [0.0]


@pytest.mark.skip(reason="needs lid.176.bin")
def test_r5_short_response_returns_zero():
    """Response < min_response_tokens → fastText unreliable, return 0."""
    from src.rewards.lang import r5_lang_consistency
    out = r5_lang_consistency(
        prompts=["Tính 2+2?"],
        completions=["4"],
        min_response_tokens=10,
    )
    assert out == [0.0]
