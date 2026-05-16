"""Tests for the AMC23 eval adapter -- schema and dispatch verification."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.eval import amc23


class _MockLLM:
    """Mock vLLM LLM returning fixed completions in prompt order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses

    def generate(self, prompts: list[str], sampling_params: Any = None) -> list:
        outs = []
        for i, _ in enumerate(prompts):
            mock = MagicMock()
            mock.outputs = [MagicMock(text=self._responses[i % len(self._responses)])]
            outs.append(mock)
        return outs


@pytest.fixture
def mock_records() -> list[dict]:
    """3-row mock matching the knoveleng/AMC-23 schema."""
    return [
        {"id": "1", "problem": "What is 2+2?", "question": "What is 2+2?",
         "answer": "4", "url": "x"},
        {"id": "2", "problem": "Compute 5*7.", "question": "Compute 5*7.",
         "answer": "35", "url": "x"},
        {"id": "3", "problem": "Sum of 1..10.", "question": "Sum of 1..10.",
         "answer": "55", "url": "x"},
    ]


def test_amc23_evaluate_returns_schema(mock_records):
    """Greedy responses match all 3 -> pass@1 = 1.0; all schema keys present."""
    correct = ["<answer>4</answer>", "<answer>35</answer>", "<answer>55</answer>"]
    model = _MockLLM(correct)

    result = amc23.evaluate(
        model=model,
        n_samples=3,
        seed=42,
        n_seeds_for_maj=1,
        run_id="test_run",
        model_path="/fake/checkpoint-50",
        dataset=mock_records,   # injectable, skip HF load
    )

    for key in ("run_id", "benchmark", "language", "n_samples", "pass_at_1",
                "lang_consistency_rate", "avg_response_tokens", "responses",
                "metadata"):
        assert key in result, f"missing key: {key}"

    assert result["benchmark"] == "amc23"
    assert result["language"] == "en"
    assert result["n_samples"] == 3
    assert result["pass_at_1"] == 1.0
    assert len(result["responses"]) == 3


def test_amc23_partial_correct(mock_records):
    """1/3 correct -> pass@1 ~= 0.333."""
    responses = ["<answer>4</answer>", "<answer>WRONG</answer>", "<answer>WRONG</answer>"]
    model = _MockLLM(responses)

    result = amc23.evaluate(
        model=model,
        n_samples=3,
        seed=42,
        n_seeds_for_maj=1,
        run_id="test",
        model_path="/fake/checkpoint-50",
        dataset=mock_records,
    )
    assert result["pass_at_1"] == pytest.approx(1.0 / 3.0, abs=0.01)


def test_amc23_n_samples_limit(mock_records):
    """n_samples=2 should score only 2 records even if the dataset has 3."""
    responses = ["<answer>4</answer>", "<answer>35</answer>"]
    model = _MockLLM(responses)

    result = amc23.evaluate(
        model=model,
        n_samples=2,
        seed=42,
        n_seeds_for_maj=1,
        run_id="test",
        model_path="/fake/checkpoint-50",
        dataset=mock_records,
    )
    assert result["n_samples"] == 2


def test_amc23_metadata_includes_hf_dataset(mock_records):
    """Metadata must reference the correct HF dataset id."""
    model = _MockLLM(["<answer>4</answer>"])
    result = amc23.evaluate(
        model=model,
        n_samples=1,
        seed=42,
        n_seeds_for_maj=1,
        run_id="test",
        model_path="/fake/checkpoint-50",
        dataset=mock_records,
    )
    assert result["metadata"].get("hf_dataset") == "knoveleng/AMC-23"
    assert result["metadata"].get("split") == "train"


def test_amc23_maj_at_4_majority_vote(mock_records):
    """Rollouts vote 'WRONG'/'4'/'4'/'4' -> majority='4'; gold='4' -> correct."""
    rotation = ["<answer>WRONG</answer>", "<answer>4</answer>",
                "<answer>4</answer>", "<answer>4</answer>"]

    class _RotatingLLM:
        def __init__(self, rot: list[str]) -> None:
            self._rot = rot
            self._call = 0

        def generate(self, prompts, sampling_params=None):
            text = self._rot[self._call % len(self._rot)]
            self._call += 1
            outs = []
            for _ in prompts:
                mock = MagicMock()
                mock.outputs = [MagicMock(text=text)]
                outs.append(mock)
            return outs

    model = _RotatingLLM(rotation)

    # 1 problem from mock_records, 4 maj seeds → 5 generate calls (1 greedy + 4 maj)
    result = amc23.evaluate(
        model=model,
        n_samples=1,
        seed=42,
        n_seeds_for_maj=4,
        run_id="test",
        model_path="/fake/ckpt",
        dataset=mock_records[:1],   # answer = "4"
    )
    # Greedy first call → "WRONG"; 4 maj calls → "4","4","4" with one "WRONG" leftover
    # rotation[1..4]: "4","4","4","4" — wait rotation has 4 elements; after greedy
    # call uses [0]="WRONG", maj uses [1..4]: "4","4","4","WRONG" (wraps).
    # Wait len 4: idx 0,1,2,3 → after greedy (call 0 → idx 0), maj calls 1..4 → idx 1,2,3,0
    # = "4","4","4","WRONG". Majority of {4,4,4,WRONG} = "4". Match gold "4" → maj_at_4 = 1.0
    assert result["pass_at_1"] == 0.0  # greedy was WRONG
    assert result["maj_at_4"] == 1.0    # majority correct


def test_amc23_runner_dispatch_includes_amc23():
    """Smoke check: runner._BENCHMARK_MODULES has an entry for 'amc23'."""
    from src.eval.runner import _BENCHMARK_MODULES

    assert "amc23" in _BENCHMARK_MODULES
    assert _BENCHMARK_MODULES["amc23"] == "src.eval.amc23"


def test_runner_passes_sampling_params_maj_to_amc23():
    """Phase 9.1 fix: runner phải pass sampling_params_maj cho amc23 dispatch.
    Bug trước đây: maj@4 luôn 0 vì sampling_params_maj=None → all 4 samples identical.
    """
    import inspect
    from src.eval import runner as runner_module
    src = inspect.getsource(runner_module.run_benchmark)
    # Verify dispatch path cho amc23 mention sampling_params_maj
    amc23_block = src[src.find('elif benchmark == "amc23"'):src.find("else:", src.find('elif benchmark == "amc23"'))]
    assert "sampling_params_maj=sampling_params_maj" in amc23_block, (
        "runner.py amc23 dispatch must pass sampling_params_maj to fix maj@N bug"
    )


def test_amc23_empty_dataset_returns_empty_result(mock_records):
    """Empty dataset -> schema-compliant empty result."""
    model = _MockLLM([])
    result = amc23.evaluate(
        model=model,
        n_samples=0,
        seed=42,
        run_id="test",
        model_path="/fake",
        dataset=[],
    )
    assert result["n_samples"] == 0
    assert result["benchmark"] == "amc23"
