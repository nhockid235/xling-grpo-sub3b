"""Pytest cho math500_vi_hand eval adapter — Phase 9.1.6."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.eval import math500_vi_hand


class _MockLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses

    def generate(self, prompts, sampling_params=None):
        outs = []
        for i, _ in enumerate(prompts):
            mock = MagicMock()
            mock.outputs = [MagicMock(text=self._responses[i % len(self._responses)])]
            outs.append(mock)
        return outs


@pytest.fixture
def mock_records() -> list[dict]:
    return [
        {"problem": "Tính tổng 2 + 3 = ?", "answer": "5", "level": "1", "subject": "Algebra"},
        {"problem": "Tính 4 × 7 = ?", "answer": "28", "level": "1", "subject": "Algebra"},
        {"problem": "Diện tích hình vuông cạnh 6 = ?", "answer": "36", "level": "2", "subject": "Geometry"},
    ]


def test_math500_vi_returns_schema(mock_records):
    correct = ["<answer>5</answer>", "<answer>28</answer>", "<answer>36</answer>"]
    model = _MockLLM(correct)

    result = math500_vi_hand.evaluate(
        model=model,
        n_samples=3,
        seed=42,
        run_id="test_run",
        model_path="/fake/ckpt",
        dataset=mock_records,
    )

    for key in ("run_id", "benchmark", "language", "n_samples", "pass_at_1",
                "lang_consistency_rate", "avg_response_tokens", "responses",
                "metadata"):
        assert key in result, f"missing key: {key}"

    assert result["benchmark"] == "math500_vi_hand"
    assert result["language"] == "vi"
    assert result["n_samples"] == 3
    assert result["pass_at_1"] == 1.0
    assert len(result["responses"]) == 3


def test_math500_vi_partial_correct(mock_records):
    responses = ["<answer>5</answer>", "<answer>WRONG</answer>", "<answer>WRONG</answer>"]
    model = _MockLLM(responses)
    result = math500_vi_hand.evaluate(
        model=model,
        n_samples=3,
        dataset=mock_records,
    )
    assert result["pass_at_1"] == pytest.approx(1.0 / 3.0, abs=0.01)


def test_math500_vi_missing_jsonl_returns_empty():
    """Nếu JSONL chưa tồn tại (user chưa dịch), return empty result schema-compliant."""
    model = _MockLLM(["foo"])
    result = math500_vi_hand.evaluate(
        model=model,
        n_samples=10,
        run_id="test",
        model_path="/fake",
        jsonl_path="data/processed/nonexistent.jsonl",
    )
    assert result["benchmark"] == "math500_vi_hand"
    assert result["n_samples"] == 0
    assert result["language"] == "vi"
    assert "error" in result["metadata"]


def test_math500_vi_loads_jsonl_from_disk(tmp_path: Path):
    """Test loader đọc JSONL từ disk khi không inject dataset."""
    jsonl = tmp_path / "vi.jsonl"
    records = [
        {"problem": "p1", "answer": "1"},
        {"problem": "p2", "answer": "2"},
    ]
    with jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    model = _MockLLM(["<answer>1</answer>", "<answer>2</answer>"])
    result = math500_vi_hand.evaluate(
        model=model,
        n_samples=2,
        run_id="test",
        model_path="/fake",
        jsonl_path=jsonl,
    )
    assert result["n_samples"] == 2
    assert result["pass_at_1"] == 1.0


def test_math500_vi_runner_dispatch():
    """Smoke check: runner registers math500_vi_hand."""
    from src.eval.runner import _BENCHMARK_MODULES
    assert "math500_vi_hand" in _BENCHMARK_MODULES
