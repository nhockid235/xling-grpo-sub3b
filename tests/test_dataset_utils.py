"""Pytest cho dataset_utils — chat template rendering + column shaping."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.trainers.dataset_utils import (
    DEFAULT_SYSTEM_PROMPT_EN,
    DEFAULT_SYSTEM_PROMPT_VI,
    prepare_grpo_dataset,
    prepare_sft_dataset,
)
from src.utils.io import write_jsonl


class _MockTokenizer:
    """Minimal fake tokenizer cho unit test mà không cần HF download."""

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        # Format đơn giản: [ROLE]: content\n
        parts = []
        for m in messages:
            parts.append(f"[{m['role'].upper()}]: {m['content']}")
        text = "\n".join(parts)
        if add_generation_prompt:
            text += "\n[ASSISTANT]:"
        return text


@pytest.fixture
def mock_tokenizer() -> _MockTokenizer:
    return _MockTokenizer()


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    records = [
        {"problem": "What is 2+2?", "solution": "4", "answer": "4"},
        {"problem": "Tính 5*7?", "solution": "35", "answer": "35"},
    ]
    p = tmp_path / "sample.jsonl"
    write_jsonl(records, p)
    return p


def test_sft_dataset_has_text_column(mock_tokenizer, sample_jsonl):
    ds = prepare_sft_dataset(
        source=sample_jsonl,
        tokenizer=mock_tokenizer,
        system_prompt=DEFAULT_SYSTEM_PROMPT_EN,
    )
    assert ds.column_names == ["text"]
    assert len(ds) == 2
    # System prompt + user + assistant đều phải xuất hiện
    sample = ds[0]["text"]
    assert "[SYSTEM]:" in sample
    assert "[USER]:" in sample
    assert "[ASSISTANT]:" in sample
    assert "What is 2+2?" in sample
    assert "4" in sample


def test_sft_uses_vi_prompt(mock_tokenizer, sample_jsonl):
    ds = prepare_sft_dataset(
        source=sample_jsonl,
        tokenizer=mock_tokenizer,
        system_prompt=DEFAULT_SYSTEM_PROMPT_VI,
    )
    assert "tiếng Việt" in ds[0]["text"]


def test_grpo_dataset_has_prompt_and_answer(mock_tokenizer, sample_jsonl):
    ds = prepare_grpo_dataset(
        source=sample_jsonl,
        tokenizer=mock_tokenizer,
        system_prompt=DEFAULT_SYSTEM_PROMPT_EN,
    )
    assert set(ds.column_names) == {"prompt", "answer"}
    assert len(ds) == 2
    sample = ds[0]
    # Prompt phải có generation prompt cuối
    assert sample["prompt"].endswith("[ASSISTANT]:")
    assert sample["answer"] == "4"


def test_grpo_dataset_no_solution_column(mock_tokenizer, sample_jsonl):
    """GRPO không được giữ 'solution' (sẽ leak training signal vào reward kwargs)."""
    ds = prepare_grpo_dataset(
        source=sample_jsonl,
        tokenizer=mock_tokenizer,
        system_prompt=DEFAULT_SYSTEM_PROMPT_EN,
    )
    assert "solution" not in ds.column_names
    assert "problem" not in ds.column_names


def test_grpo_dataset_falls_back_to_solution_when_no_answer(mock_tokenizer, tmp_path: Path):
    """Nếu dataset không có 'answer' → fallback dùng 'solution' làm answer."""
    records = [{"problem": "Q1", "solution": "S1"}]  # no 'answer'
    p = tmp_path / "no_answer.jsonl"
    write_jsonl(records, p)

    ds = prepare_grpo_dataset(
        source=p,
        tokenizer=mock_tokenizer,
        system_prompt=DEFAULT_SYSTEM_PROMPT_EN,
    )
    assert ds[0]["answer"] == "S1"


def test_sft_raises_on_missing_problem(mock_tokenizer, tmp_path: Path):
    records = [{"foo": "bar"}]
    p = tmp_path / "bad.jsonl"
    write_jsonl(records, p)
    with pytest.raises(ValueError, match="problem"):
        prepare_sft_dataset(
            source=p,
            tokenizer=mock_tokenizer,
            system_prompt=DEFAULT_SYSTEM_PROMPT_EN,
        )


def test_grpo_raises_on_missing_problem(mock_tokenizer, tmp_path: Path):
    records = [{"foo": "bar"}]
    p = tmp_path / "bad.jsonl"
    write_jsonl(records, p)
    with pytest.raises(ValueError, match="problem"):
        prepare_grpo_dataset(
            source=p,
            tokenizer=mock_tokenizer,
            system_prompt=DEFAULT_SYSTEM_PROMPT_EN,
        )
