"""Pytest cho answer extraction parsers."""

from __future__ import annotations

from src.utils.parsing import (
    extract_answer_tag,
    extract_boxed,
    extract_gsm8k_gold,
    extract_last_number,
)


def test_extract_answer_tag_basic():
    assert extract_answer_tag("<answer>42</answer>") == "42"


def test_extract_answer_tag_multiline():
    text = "<answer>\n  42\n</answer>"
    assert extract_answer_tag(text) == "42"


def test_extract_answer_tag_missing():
    assert extract_answer_tag("no tag here") is None


def test_extract_boxed_basic():
    assert extract_boxed("\\boxed{42}") == "42"


def test_extract_boxed_with_fraction():
    assert extract_boxed("answer is \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_extract_gsm8k_gold():
    text = "Step 1: ... Step 2: ...\n#### 42"
    assert extract_gsm8k_gold(text) == "42"


def test_extract_last_number_basic():
    assert extract_last_number("Step 1: 5+5=10. Final: 42.") == "42"


def test_extract_last_number_negative():
    assert extract_last_number("answer: -7") == "-7"


def test_extract_last_number_decimal():
    assert extract_last_number("result is 3.14") == "3.14"


def test_extract_last_number_none():
    assert extract_last_number("no numbers") is None
