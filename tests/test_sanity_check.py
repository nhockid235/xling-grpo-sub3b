"""Pytest cho sanity_check.py — W3 mid-run gating logic."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import sanity_check  # noqa: E402


def _make_df(*rows) -> pd.DataFrame:
    cols = [
        "run_id", "model", "condition", "seed", "stage", "step",
        "benchmark", "language", "pass_at_1", "maj_at_8",
        "lang_consistency", "avg_tokens", "n_samples",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_no_data():
    df = _make_df()
    out = sanity_check.evaluate_gates(df)
    assert out["verdict"] == "no_data"


def test_all_gates_pass():
    df = _make_df(
        ("qwen15b_en_42", "qwen15b", "en", 42, "grpo", 500, "gsm8k", "", 0.55, "", "", 300, 1300),
        ("qwen15b_en_42", "qwen15b", "en", 42, "grpo", 500, "mgsm", "vi", 0.40, "", 0.85, 300, 250),
        ("qwen15b_vi_42", "qwen15b", "vi", 42, "grpo", 500, "mgsm", "vi", 0.50, "", 0.90, 300, 250),
        ("qwen15b_enlang_42", "qwen15b", "enlang", 42, "grpo", 500, "mgsm", "vi", 0.42, "", 0.88, 300, 250),
    )
    out = sanity_check.evaluate_gates(df)
    assert out["verdict"] == "pass"
    assert out["gates"]["gsm8k_en_min30"]["pass"]
    assert out["gates"]["mgsm_vi_en_vs_vi_gap_min1pp"]["pass"]
    assert out["gates"]["enlang_lang_consist_vi_min50pct"]["pass"]


def test_gsm8k_too_low():
    df = _make_df(
        ("qwen15b_en_42", "qwen15b", "en", 42, "grpo", 500, "gsm8k", "", 0.20, "", "", 300, 1300),
        ("qwen15b_en_42", "qwen15b", "en", 42, "grpo", 500, "mgsm", "vi", 0.40, "", 0.85, 300, 250),
        ("qwen15b_vi_42", "qwen15b", "vi", 42, "grpo", 500, "mgsm", "vi", 0.50, "", 0.90, 300, 250),
        ("qwen15b_enlang_42", "qwen15b", "enlang", 42, "grpo", 500, "mgsm", "vi", 0.42, "", 0.88, 300, 250),
    )
    out = sanity_check.evaluate_gates(df)
    assert out["verdict"] == "fail"
    assert not out["gates"]["gsm8k_en_min30"]["pass"]


def test_mgsm_gap_too_small():
    """"""
    df = _make_df(
        ("qwen15b_en_42", "qwen15b", "en", 42, "grpo", 500, "gsm8k", "", 0.55, "", "", 300, 1300),
        ("qwen15b_en_42", "qwen15b", "en", 42, "grpo", 500, "mgsm", "vi", 0.500, "", 0.85, 300, 250),
        ("qwen15b_vi_42", "qwen15b", "vi", 42, "grpo", 500, "mgsm", "vi", 0.505, "", 0.90, 300, 250),
        ("qwen15b_enlang_42", "qwen15b", "enlang", 42, "grpo", 500, "mgsm", "vi", 0.42, "", 0.88, 300, 250),
    )
    out = sanity_check.evaluate_gates(df)
    assert out["verdict"] == "fail"
    assert not out["gates"]["mgsm_vi_en_vs_vi_gap_min1pp"]["pass"]


def test_lang_consistency_too_low():
    df = _make_df(
        ("qwen15b_en_42", "qwen15b", "en", 42, "grpo", 500, "gsm8k", "", 0.55, "", "", 300, 1300),
        ("qwen15b_en_42", "qwen15b", "en", 42, "grpo", 500, "mgsm", "vi", 0.40, "", 0.85, 300, 250),
        ("qwen15b_vi_42", "qwen15b", "vi", 42, "grpo", 500, "mgsm", "vi", 0.50, "", 0.90, 300, 250),
        ("qwen15b_enlang_42", "qwen15b", "enlang", 42, "grpo", 500, "mgsm", "vi", 0.42, "", 0.30, 300, 250),
    )
    out = sanity_check.evaluate_gates(df)
    assert out["verdict"] == "fail"
    assert not out["gates"]["enlang_lang_consist_vi_min50pct"]["pass"]
