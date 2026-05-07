"""Pytest cho 8-gram decontamination.

Synthetic fixtures only — no HF download. Test sets are local JSONL files
materialised inside `tmp_path` and consumed via `--tests <dir>`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.decontaminate import normalize, ngrams, run
from src.utils.io import read_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# Helpers + fixtures
# ---------------------------------------------------------------------------


# A made-up GSM8K-style test problem (long enough to yield many 8-grams).
TEST_PROBLEM_GSM = (
    "Janet has 16 ducks and they each lay 4 eggs every morning. "
    "She uses three eggs to bake a cake and sells the rest at the farmers "
    "market for two dollars per egg. How much money does she make daily?"
)

TEST_PROBLEM_MATH500 = (
    "Find the sum of all positive integers less than 200 that are divisible "
    "by either 3 or 5 but not by 15, and prove your answer is correct."
)

UNRELATED_TRAIN_PROBLEM = (
    "Compute the limit as x approaches infinity of the function f(x) = "
    "(2x squared plus 3x minus 1) divided by (x squared plus 7)."
)


def _write_test_dir(tmp_path: Path) -> Path:
    """Materialize fake test sets as JSONL into tmp_path/tests/."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    write_jsonl(
        [{"problem": TEST_PROBLEM_GSM}],
        tests_dir / "gsm8k.jsonl",
    )
    write_jsonl(
        [{"problem": TEST_PROBLEM_MATH500}],
        tests_dir / "math500.jsonl",
    )
    return tests_dir


# ---------------------------------------------------------------------------
# Sanity checks for the helpers (cheap, fast)
# ---------------------------------------------------------------------------


def test_normalize_strips_punct_and_lowercases():
    out = normalize("Hello, World! 2+2=4.")
    assert out == ["hello", "world", "2", "2", "4"]


def test_ngrams_basic():
    toks = ["a", "b", "c", "d", "e"]
    assert ngrams(toks, 3) == ["a b c", "b c d", "c d e"]


def test_ngrams_short_input_falls_back_to_full_join():
    """Records shorter than n still produce a single shingle so they index/match."""
    toks = ["only", "five", "tokens", "here", "ok"]
    assert ngrams(toks, 8) == ["only five tokens here ok"]


# ---------------------------------------------------------------------------
# Required Phase 2 tests
# ---------------------------------------------------------------------------


def test_decontam_removes_exact_match(tmp_path: Path):
    """Training sample chứa nguyên text test → must be removed."""
    tests_dir = _write_test_dir(tmp_path)
    train_path = tmp_path / "train.jsonl"
    out_path = tmp_path / "clean.jsonl"
    stats_path = tmp_path / "stats.json"

    write_jsonl(
        [
            {"problem": TEST_PROBLEM_GSM, "solution": "answer ...", "lang": "en"},
            {"problem": UNRELATED_TRAIN_PROBLEM, "solution": "limit = 2", "lang": "en"},
        ],
        train_path,
    )

    stats = run(
        train_path=train_path,
        tests_arg=str(tests_dir),
        output_path=out_path,
        stats_path=stats_path,
        n=8,
        threshold=0.5,
    )

    kept = read_jsonl(out_path)
    assert len(kept) == 1
    assert kept[0]["problem"] == UNRELATED_TRAIN_PROBLEM
    assert stats["removed"] == 1
    assert stats["by_test_set"]["gsm8k"] == 1


def test_decontam_keeps_unrelated(tmp_path: Path):
    """Training sample không match test 8-gram → kept."""
    tests_dir = _write_test_dir(tmp_path)
    train_path = tmp_path / "train.jsonl"
    out_path = tmp_path / "clean.jsonl"

    train_records = [
        {"problem": UNRELATED_TRAIN_PROBLEM, "solution": "limit = 2", "lang": "en"},
        {
            "problem": "Solve for x: 4x - 7 = 21. Show your steps.",
            "solution": "x = 7",
            "lang": "en",
        },
        {
            "problem": "What is the integral of sin(x) from 0 to pi?",
            "solution": "2",
            "lang": "en",
        },
    ]
    write_jsonl(train_records, train_path)

    stats = run(
        train_path=train_path,
        tests_arg=str(tests_dir),
        output_path=out_path,
        stats_path=None,
        n=8,
        threshold=0.5,
    )

    kept = read_jsonl(out_path)
    assert len(kept) == 3
    assert stats["removed"] == 0
    assert all(v == 0 for v in stats["by_test_set"].values())


def test_decontam_asymmetric(tmp_path: Path):
    """Test files must remain byte-for-byte identical after decontamination.

    CLAUDE.md pitfall #12: never filter test data.
    """
    tests_dir = _write_test_dir(tmp_path)
    gsm_path = tests_dir / "gsm8k.jsonl"
    math_path = tests_dir / "math500.jsonl"
    before_gsm = gsm_path.read_bytes()
    before_math = math_path.read_bytes()

    train_path = tmp_path / "train.jsonl"
    out_path = tmp_path / "clean.jsonl"
    write_jsonl(
        [
            {"problem": TEST_PROBLEM_GSM},  # Will be removed
            {"problem": TEST_PROBLEM_MATH500},  # Will be removed
            {"problem": UNRELATED_TRAIN_PROBLEM},  # Kept
        ],
        train_path,
    )

    run(
        train_path=train_path,
        tests_arg=str(tests_dir),
        output_path=out_path,
        stats_path=None,
        n=8,
        threshold=0.5,
    )

    # Test files must be untouched.
    assert gsm_path.read_bytes() == before_gsm
    assert math_path.read_bytes() == before_math
    # Train output reflects the removals.
    kept = read_jsonl(out_path)
    assert len(kept) == 1
    assert kept[0]["problem"] == UNRELATED_TRAIN_PROBLEM


def test_decontam_stats_recorded(tmp_path: Path):
    """Stats JSON must contain required keys with correct counts per test set."""
    tests_dir = _write_test_dir(tmp_path)
    train_path = tmp_path / "train.jsonl"
    out_path = tmp_path / "clean.jsonl"
    stats_path = tmp_path / "stats.json"

    write_jsonl(
        [
            {"problem": TEST_PROBLEM_GSM},
            {"problem": TEST_PROBLEM_MATH500},
            {"problem": UNRELATED_TRAIN_PROBLEM},
        ],
        train_path,
    )

    run(
        train_path=train_path,
        tests_arg=str(tests_dir),
        output_path=out_path,
        stats_path=stats_path,
        n=8,
        threshold=0.5,
    )

    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    # Required keys
    for key in ("total_train", "kept", "removed", "by_test_set", "threshold", "ngram"):
        assert key in stats, f"missing key: {key}"
    assert stats["total_train"] == 3
    assert stats["kept"] == 1
    assert stats["removed"] == 2
    assert stats["by_test_set"]["gsm8k"] == 1
    assert stats["by_test_set"]["math500"] == 1
