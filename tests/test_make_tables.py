"""Tests for scripts/make_tables.py -- LaTeX table rendering."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import make_tables  # noqa: E402


@pytest.fixture
def synthetic_master() -> pd.DataFrame:
    """3 cells x 6 langs MGSM + 3 cells x 3 EN benchmarks (dummy)."""
    rows = []
    for cond in ["en", "vi", "enlang"]:
        # MGSM 6 langs
        for lang in ["en", "vi", "zh", "fr", "th", "sw"]:
            rows.append({
                "run_id": f"qwen15b_{cond}_42",
                "model": "qwen15b",
                "condition": cond,
                "seed": 42,
                "stage": "grpo",
                "step": 500,
                "benchmark": "mgsm",
                "language": lang,
                "pass_at_1": 0.5 + 0.01 * (hash(cond + lang) % 20),
                "maj_at_8": "",
                "lang_consistency": 0.7 + 0.01 * (hash(cond + lang) % 25),
                "avg_tokens": 300,
                "n_samples": 250,
            })
        # EN sanity
        for bm in ["gsm8k", "math500", "aime2024"]:
            rows.append({
                "run_id": f"qwen15b_{cond}_42",
                "model": "qwen15b",
                "condition": cond,
                "seed": 42,
                "stage": "grpo",
                "step": 500,
                "benchmark": bm,
                "language": "",
                "pass_at_1": 0.6 + 0.01 * (hash(cond + bm) % 15),
                "maj_at_8": "",
                "lang_consistency": "",
                "avg_tokens": 300,
                "n_samples": 100,
            })
    return pd.DataFrame(rows)


def test_table1_xling_transfer_compiles(synthetic_master):
    latex = make_tables.table1_xling_transfer(synthetic_master)
    assert "\\begin{table}" in latex
    assert "\\toprule" in latex
    assert "\\midrule" in latex
    assert "\\bottomrule" in latex
    assert "\\end{table}" in latex
    assert "\\caption" in latex
    assert "tab:xling_transfer" in latex


def test_table2_en_sanity_compiles(synthetic_master):
    latex = make_tables.table2_en_sanity(synthetic_master)
    assert "tab:en_sanity" in latex
    assert "gsm8k" in latex.lower() or "math500" in latex.lower()


def test_table3_lang_consistency_compiles(synthetic_master):
    latex = make_tables.table3_lang_consistency(synthetic_master)
    assert "tab:lang_consistency" in latex
    assert "\\toprule" in latex


def test_tables_bold_max_per_row(synthetic_master):
    """Max value per row should be \\textbf-wrapped."""
    latex = make_tables.table1_xling_transfer(synthetic_master)
    assert "\\textbf{" in latex


def test_no_vertical_rules(synthetic_master):
    """Booktabs convention: no vertical bars."""
    for fn in (
        make_tables.table1_xling_transfer,
        make_tables.table2_en_sanity,
        make_tables.table3_lang_consistency,
    ):
        latex = fn(synthetic_master)
        assert "|c" not in latex
        assert "c|" not in latex
