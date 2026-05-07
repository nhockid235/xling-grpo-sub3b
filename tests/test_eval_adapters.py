"""Unit tests for evaluation adapters — no GPU, no real vLLM, no HF download.

Strategy:
  - Mock the LLM with a stub class returning fixed completions per prompt index.
  - Provide in-memory list[dict] datasets to bypass HF `load_dataset`.
  - Verify schema (CLAUDE.md § Logging schema) keys & types, plus correctness
    of language validation and AIME field handling.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.eval import aime, gsm8k, math500, mgsm, msvamp
from src.eval._common import majority_vote, normalize_number_string, numeric_match
from src.eval.lang_consistency import (
    _normalize_fasttext_label,
    _parse_filename,
    process_files,
    write_csv,
)
from src.eval.prompts import DEFAULT_SYSTEM_PROMPT, build_prompt
from src.utils.parsing import extract_gsm8k_gold


# ----------------------------- mock LLM -----------------------------


class MockOutput:
    def __init__(self, text: str) -> None:
        self.text = text


class MockRequestOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [MockOutput(text)]


class MockLLM:
    """Returns the next completion from a queue, cycling per prompt."""

    def __init__(self, completions: list[str]) -> None:
        self._completions = list(completions)

    def generate(self, prompts: list[str], sampling_params: Any = None):
        outs = []
        for i, _ in enumerate(prompts):
            text = self._completions[i % len(self._completions)] if self._completions else ""
            outs.append(MockRequestOutput(text))
        return outs


# ------------------------- schema-compliance ------------------------

REQUIRED_KEYS = {
    "run_id",
    "benchmark",
    "language",
    "n_samples",
    "pass_at_1",
    "maj_at_8",
    "lang_consistency_rate",
    "avg_response_tokens",
    "responses",
    "metadata",
}


def _assert_schema(result: dict, benchmark: str, language: str | None = None) -> None:
    assert REQUIRED_KEYS.issubset(result.keys()), (
        f"Missing keys: {REQUIRED_KEYS - set(result.keys())}"
    )
    assert result["benchmark"] == benchmark
    assert result["language"] == language
    assert isinstance(result["n_samples"], int)
    assert isinstance(result["responses"], list)
    assert isinstance(result["avg_response_tokens"], float)
    md = result["metadata"]
    for k in ("model_path", "eval_date", "vllm_version", "max_tokens", "temperature", "seed"):
        assert k in md, f"metadata missing {k}"


# ----------------------------- gsm8k --------------------------------


def test_gsm8k_extracts_gold_from_hash_format():
    """#### 42 → '42'"""
    assert extract_gsm8k_gold("Step1...\n#### 42") == "42"
    assert extract_gsm8k_gold("Step1...\n#### 1,234") == "1,234"


def test_gsm8k_evaluate_basic_schema_and_correctness():
    fake_ds = [
        {"question": "What is 2+2?", "answer": "Two plus two...\n#### 4"},
        {"question": "What is 5*7?", "answer": "Five times seven...\n#### 35"},
    ]
    completions = [
        "<think>simple</think><answer>\\boxed{4}</answer>",
        "<think>simple</think><answer>\\boxed{35}</answer>",
    ]
    result = gsm8k.evaluate(
        model=MockLLM(completions),
        dataset=fake_ds,
        run_id="qwen15b_en_42",
        model_path="results/grpo/qwen15b_en_42/checkpoint-500",
    )
    _assert_schema(result, "gsm8k", language="en")
    assert result["n_samples"] == 2
    assert result["pass_at_1"] == 1.0
    assert len(result["responses"]) == 2
    assert result["maj_at_8"] is None


def test_gsm8k_handles_comma_thousands_in_gold():
    """Gold '1,234' must match prediction '1234'."""
    fake_ds = [{"question": "Q", "answer": "...\n#### 1,234"}]
    completions = ["<answer>1234</answer>"]
    result = gsm8k.evaluate(model=MockLLM(completions), dataset=fake_ds)
    assert result["pass_at_1"] == 1.0


# ----------------------------- math500 ------------------------------


def test_math500_evaluate_schema():
    fake_ds = [
        {"problem": "Find x", "answer": "42", "solution": "...", "level": "Level 1"},
    ]
    completions = ["<answer>\\boxed{42}</answer>"]
    result = math500.evaluate(
        model=MockLLM(completions),
        dataset=fake_ds,
        run_id="qwen15b_en_42",
    )
    _assert_schema(result, "math500", language="en")
    assert result["n_samples"] == 1
    # numeric_match fallback covers '42' → '42' even if math-verify missing
    assert result["pass_at_1"] == 1.0


# ------------------------------ aime --------------------------------


def test_aime_uses_capitalized_fields():
    """AIME schema is CAPITALIZED — lowercase access would KeyError."""
    fake_ds = [
        {"ID": "I-1", "Problem": "P1", "Solution": "S1", "Answer": 100},
        {"ID": "I-2", "Problem": "P2", "Solution": "S2", "Answer": 50},
    ]
    completions = [
        "<answer>100</answer>",
        "<answer>50</answer>",
    ]
    # n_seeds_for_maj8=0 to skip stochastic loop in unit test
    result = aime.evaluate(
        model=MockLLM(completions),
        dataset=fake_ds,
        n_seeds_for_maj8=0,
        run_id="qwen15b_en_42",
    )
    _assert_schema(result, "aime2024", language="en")
    assert result["n_samples"] == 2
    assert result["pass_at_1"] == 1.0
    # maj_at_8 None when n_seeds_for_maj8=0
    assert result["maj_at_8"] is None


def test_aime_lowercase_fields_would_raise():
    """Sanity check that lowercase 'problem' field is NOT used (would KeyError)."""
    fake_ds = [{"problem": "P1", "answer": 100}]  # lowercase = wrong
    completions = ["<answer>100</answer>"]
    with pytest.raises(KeyError):
        aime.evaluate(
            model=MockLLM(completions), dataset=fake_ds, n_seeds_for_maj8=0
        )


def test_aime_maj_at_8_majority_vote():
    """When maj_at_8 enabled, K stochastic samples → majority vote."""
    fake_ds = [{"ID": "I-1", "Problem": "P1", "Solution": "S", "Answer": 42}]
    # MockLLM cycles same completion regardless of K — all 8 will vote 42.
    completions = ["<answer>42</answer>"]
    result = aime.evaluate(
        model=MockLLM(completions),
        dataset=fake_ds,
        n_seeds_for_maj8=8,
        run_id="qwen15b_en_42",
    )
    assert result["pass_at_1"] == 1.0
    assert result["maj_at_8"] == 1.0


# ------------------------------ mgsm --------------------------------


def test_mgsm_rejects_unknown_language():
    with pytest.raises(ValueError, match="not in MGSM"):
        mgsm.evaluate(model=MockLLM([]), language="xx")


def test_mgsm_accepts_each_guaranteed_language():
    fake_ds = [{"question": "Q", "answer": "A", "answer_number": 7, "equation_solution": "..."}]
    completions = ["<answer>7</answer>"]
    for lang in mgsm.MGSM_LANGS_GUARANTEED:
        result = mgsm.evaluate(
            model=MockLLM(completions),
            language=lang,
            dataset=fake_ds,
            run_id="qwen15b_en_42",
        )
        _assert_schema(result, "mgsm", language=lang)
        assert result["pass_at_1"] == 1.0


def test_mgsm_telugu_rejected():
    """Telugu listed in card but NOT in 10 guaranteed → reject."""
    with pytest.raises(ValueError):
        mgsm.evaluate(model=MockLLM([]), language="te")


# ------------------------------ msvamp ------------------------------


def test_msvamp_rejects_unknown_language():
    with pytest.raises(ValueError, match="not in MSVAMP"):
        msvamp.evaluate(model=MockLLM([]), language="xx")


def test_msvamp_accepts_each_listed_language():
    fake_ds = [{"id": 1, "question": "Q", "answer": "A", "answer_number": 12}]
    completions = ["<answer>12</answer>"]
    for lang in msvamp.MSVAMP_LANGS:
        result = msvamp.evaluate(
            model=MockLLM(completions),
            language=lang,
            dataset=fake_ds,
            run_id="qwen15b_en_42",
        )
        _assert_schema(result, "msvamp", language=lang)
        assert result["pass_at_1"] == 1.0


# ------------------------------ prompts -----------------------------


def test_build_prompt_default_system():
    p = build_prompt("What is 2+2?")
    assert DEFAULT_SYSTEM_PROMPT in p
    assert "What is 2+2?" in p


def test_build_prompt_with_tokenizer_chat_template():
    class FakeTok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            return "<|im_start|>" + "\n".join(m["content"] for m in messages) + "<|im_end|>"

    p = build_prompt("hello", chat_template_tokenizer=FakeTok())
    assert p.startswith("<|im_start|>")
    assert "hello" in p


# ------------------------------ helpers -----------------------------


def test_normalize_number_string_strip_commas_and_dollar():
    assert normalize_number_string("$1,234") == "1234"
    assert normalize_number_string("+42") == "42"
    assert normalize_number_string("42.") == "42"
    assert normalize_number_string(None) is None


def test_numeric_match_float_equivalence():
    assert numeric_match("42", "42.0") is True
    assert numeric_match("1,234", "1234") is True
    assert numeric_match("42", "43") is False
    assert numeric_match(None, "42") is False


def test_majority_vote():
    assert majority_vote(["a", "b", "a", "c"]) == "a"
    assert majority_vote([None, None, "b", "b"]) == "b"
    assert majority_vote([None, None]) is None


# -------------------------- lang_consistency ------------------------


def test_normalize_fasttext_label():
    assert _normalize_fasttext_label("__label__zh") == "zh"
    assert _normalize_fasttext_label("__label__zh-cn") == "zh"
    assert _normalize_fasttext_label("__label__pt-br") == "pt"


def test_parse_filename_with_lang():
    out = _parse_filename("qwen15b_en_42_mgsm_vi.json")
    assert out["benchmark"] == "mgsm"
    assert out["language"] == "vi"
    assert out["run_id"] == "qwen15b_en_42"


def test_parse_filename_without_lang():
    out = _parse_filename("qwen15b_en_42_gsm8k.json")
    assert out["benchmark"] == "gsm8k"
    assert out["language"] is None
    assert out["run_id"] == "qwen15b_en_42"


class FakeFastText:
    """Stub fastText model for testing process_files without real model."""

    def __init__(self, predictions: dict[str, str]) -> None:
        # text → label without __label__ prefix
        self._preds = predictions

    def predict(self, text: str, k: int = 1):
        for needle, lang in self._preds.items():
            if needle in text:
                return ([f"__label__{lang}"], [0.99])
        return ([f"__label__en"], [0.5])


def test_process_files_with_mock_fasttext(tmp_path, monkeypatch):
    # Write a fake eval JSON
    eval_json = {
        "run_id": "qwen15b_en_42",
        "benchmark": "mgsm",
        "language": "vi",
        "n_samples": 3,
        "pass_at_1": 0.5,
        "maj_at_8": None,
        "lang_consistency_rate": None,
        "avg_response_tokens": 100.0,
        "responses": [
            "Tính tổng hai số rất là đơn giản và đây là kết quả cuối cùng",
            "The answer is forty-two and here is the reasoning behind it",
            "short",  # under min_response_tokens → skipped
        ],
        "metadata": {
            "model_path": "x",
            "eval_date": "2026-05-05T00:00:00Z",
            "vllm_version": "0.7.2",
            "max_tokens": 2048,
            "temperature": 0.0,
            "seed": 42,
        },
    }
    p = tmp_path / "qwen15b_en_42_mgsm_vi.json"
    import json as _json

    p.write_text(_json.dumps(eval_json, ensure_ascii=False))

    fake_model = FakeFastText({"Tính": "vi", "answer": "en"})

    # monkey-patch _load_fasttext to bypass real model loading
    from src.eval import lang_consistency as lc

    monkeypatch.setattr(lc, "_load_fasttext", lambda path: fake_model)

    rows = process_files([p], fasttext_model_path="dummy", min_response_tokens=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "qwen15b_en_42"
    assert row["benchmark"] == "mgsm"
    assert row["language"] == "vi"
    # 1 vi match, 1 en mismatch counted, 1 skipped → rate = 0.5
    assert row["n_counted"] == 2
    assert row["n_skipped"] == 1
    assert row["lang_consistency_rate"] == 0.5

    # CSV write smoke test
    csv_out = tmp_path / "out.csv"
    write_csv(rows, csv_out)
    text = csv_out.read_text()
    assert "lang_consistency_rate" in text
    assert "qwen15b_en_42" in text
