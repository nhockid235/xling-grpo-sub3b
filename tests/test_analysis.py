"""Pytest cho analysis tools — bootstrap CI, aggregate run-id parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.aggregate import (
    aggregate,
    eval_json_to_row,
    parse_run_id,
    parse_stage,
    parse_step,
)
from src.analysis.bootstrap import bootstrap_ci, format_ci


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_ci_all_ones():
    """Mọi value = 1.0 → mean=1.0, CI cũng [1.0, 1.0]."""
    mean, lo, hi = bootstrap_ci([1.0] * 100, n_bootstrap=100, seed=42)
    assert mean == pytest.approx(1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_ci_all_zeros():
    mean, lo, hi = bootstrap_ci([0.0] * 100, n_bootstrap=100, seed=42)
    assert mean == pytest.approx(0.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)


def test_bootstrap_ci_half_correct():
    """Mean ≈ 0.5; CI bao quanh 0.5."""
    values = [1.0] * 50 + [0.0] * 50
    mean, lo, hi = bootstrap_ci(values, n_bootstrap=1000, seed=42)
    assert mean == pytest.approx(0.5)
    assert lo < mean < hi
    # CI khá rộng cho n=100, nhưng nằm trong [0.3, 0.7]
    assert 0.3 < lo < 0.5
    assert 0.5 < hi < 0.7


def test_bootstrap_deterministic_with_seed():
    values = [1.0, 0.0, 1.0, 1.0, 0.0]
    a = bootstrap_ci(values, n_bootstrap=100, seed=42)
    b = bootstrap_ci(values, n_bootstrap=100, seed=42)
    assert a == b


def test_bootstrap_empty_returns_nan():
    import math

    mean, lo, hi = bootstrap_ci([], n_bootstrap=100, seed=42)
    assert math.isnan(mean)
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_format_ci_percent():
    out = format_ci(0.456, 0.421, 0.490, as_percent=True)
    assert out == "45.6% [42.1, 49.0]"


def test_format_ci_raw():
    out = format_ci(0.5, 0.4, 0.6, as_percent=False)
    assert out == "0.5 [0.4, 0.6]"


# ---------------------------------------------------------------------------
# Aggregate parsing
# ---------------------------------------------------------------------------


def test_parse_run_id_standard():
    out = parse_run_id("qwen15b_en_42")
    assert out == {"model": "qwen15b", "condition": "en", "seed": "42"}


def test_parse_run_id_enlang():
    out = parse_run_id("llama3b_enlang_123")
    assert out == {"model": "llama3b", "condition": "enlang", "seed": "123"}


def test_parse_run_id_reproduce():
    out = parse_run_id("reproduce_openrs_rs2_42")
    assert out["model"] == "deepseek_r1_distill_15b"
    assert out["condition"] == "baseline_rs2"
    assert out["seed"] == "42"


def test_parse_run_id_unknown():
    out = parse_run_id("garbage_id")
    assert out == {"model": "unknown", "condition": "unknown", "seed": "unknown"}


def test_parse_step_numeric():
    assert parse_step("results/grpo/qwen15b_en_42/checkpoint-50") == "50"


def test_parse_step_final():
    assert parse_step("results/sft/qwen15b_en_42/checkpoint-final") == "final"


def test_parse_step_missing():
    assert parse_step("") == "unknown"
    assert parse_step(None) == "unknown"


def test_parse_stage_grpo():
    assert parse_stage("results/grpo/qwen15b_en_42/checkpoint-100") == "grpo"


def test_parse_stage_sft():
    assert parse_stage("results/sft/qwen15b_en_42/checkpoint-final") == "sft"


def test_parse_stage_unknown():
    assert parse_stage("/somewhere/else/checkpoint-5") == "unknown"


# ---------------------------------------------------------------------------
# Aggregate end-to-end
# ---------------------------------------------------------------------------


def _write_eval_json(path: Path, **overrides) -> None:
    data = {
        "run_id": "qwen15b_en_42",
        "benchmark": "mgsm",
        "language": "vi",
        "n_samples": 250,
        "pass_at_1": 0.456,
        "maj_at_8": None,
        "lang_consistency_rate": 0.83,
        "avg_response_tokens": 312.4,
        "responses": [],
        "metadata": {
            "model_path": "results/grpo/qwen15b_en_42/checkpoint-500",
            "eval_date": "2026-05-15",
            "vllm_version": "0.7.2",
            "max_tokens": 2048,
            "temperature": 0.0,
        },
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_eval_json_to_row_basic(tmp_path: Path):
    p = tmp_path / "eval.json"
    _write_eval_json(p)
    row = eval_json_to_row(p)
    assert row["run_id"] == "qwen15b_en_42"
    assert row["model"] == "qwen15b"
    assert row["condition"] == "en"
    assert row["seed"] == "42"
    assert row["stage"] == "grpo"
    assert row["step"] == "500"
    assert row["benchmark"] == "mgsm"
    assert row["language"] == "vi"
    assert row["pass_at_1"] == 0.456
    assert row["lang_consistency"] == 0.83


def test_aggregate_writes_csv(tmp_path: Path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    _write_eval_json(eval_dir / "a.json")
    _write_eval_json(
        eval_dir / "b.json",
        run_id="qwen15b_vi_42",
        language="vi",
        pass_at_1=0.512,
    )

    out_csv = tmp_path / "master.csv"
    n = aggregate(eval_dir, out_csv)
    assert n == 2

    text = out_csv.read_text(encoding="utf-8")
    assert "run_id,model,condition,seed" in text
    assert "qwen15b_en_42" in text
    assert "qwen15b_vi_42" in text


def test_aggregate_skips_invalid_json(tmp_path: Path, capsys):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    _write_eval_json(eval_dir / "good.json")
    (eval_dir / "bad.json").write_text("not valid json {", encoding="utf-8")

    out_csv = tmp_path / "master.csv"
    n = aggregate(eval_dir, out_csv)
    assert n == 1   # bad.json bị skip, good.json vẫn ghi
