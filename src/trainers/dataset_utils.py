"""Dataset preparation for SFT and GRPO training (chat template rendering).

GRPO prompts are built with ``add_generation_prompt=True``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from src.utils.io import read_jsonl

# Open-RS verbatim training system prompt (recipes/grpo.yaml).
DEFAULT_SYSTEM_PROMPT_EN = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The assistant first thinks about the reasoning process in "
    "the mind and then provides the user with the answer, and put your final answer "
    "within \\boxed{{}} . The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think> <answer> answer here </answer>. "
    "Note that respond by English, NOT use other languages."
)

DEFAULT_SYSTEM_PROMPT_VI = (
    "Cuộc hội thoại giữa User và Assistant. User đặt câu hỏi và Assistant giải đáp. "
    "Assistant trước tiên suy nghĩ trong đầu rồi đưa ra đáp án, đặt đáp án cuối "
    "vào \\boxed{{}} . Quá trình suy luận và câu trả lời được đặt trong "
    "<think> </think> và <answer> </answer>, ví dụ: "
    "<think> quá trình suy luận </think> <answer> câu trả lời </answer>. "
    "Lưu ý: trả lời bằng tiếng Việt, KHÔNG dùng ngôn ngữ khác."
)


def _is_jsonl(source: str | Path) -> bool:
    return Path(str(source)).suffix.lower() == ".jsonl"


def _load_source(source: str | Path, split: str = "train") -> Dataset:
    """Load a HF Dataset from a local JSONL file or a HF dataset id."""
    if _is_jsonl(source):
        records = read_jsonl(Path(source))
        return Dataset.from_list(records)
    # HF dataset id
    return load_dataset(str(source), split=split)


def prepare_sft_dataset(
    source: str | Path,
    tokenizer: Any,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT_EN,
    split: str = "train",
) -> Dataset:
    """Render an SFT dataset to a single ``text`` column via the chat template.

    Args:
        source: JSONL path or HF dataset id with ``problem`` and ``solution`` columns.
        tokenizer: HF tokenizer.
        system_prompt: system message (EN or VI).
        split: HF split name (ignored for JSONL).

    Returns:
        Dataset with a single ``text`` column.
    """
    ds = _load_source(source, split=split)

    if "problem" not in ds.column_names or "solution" not in ds.column_names:
        raise ValueError(
            f"Dataset must have 'problem' and 'solution'; got {ds.column_names}"
        )

    def _render(example: dict[str, Any]) -> dict[str, str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["solution"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    ds = ds.map(_render, remove_columns=ds.column_names)
    return ds


def prepare_grpo_dataset(
    source: str | Path,
    tokenizer: Any,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT_EN,
    split: str = "train",
) -> Dataset:
    """Render a GRPO dataset with ``prompt`` and ``answer`` columns.

    Args:
        source: JSONL path or HF dataset id with at least a ``problem`` column.
        tokenizer: HF tokenizer.
        system_prompt: system message (EN or VI).
        split: HF split name (ignored for JSONL).

    Returns:
        Dataset with columns ``prompt`` and ``answer``.
    """
    ds = _load_source(source, split=split)

    if "problem" not in ds.column_names:
        raise ValueError(
            f"Dataset must have 'problem'; got columns: {ds.column_names}"
        )

    has_answer = "answer" in ds.column_names

    def _render(example: dict[str, Any]) -> dict[str, str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["problem"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if has_answer:
            answer = str(example.get("answer", ""))
        else:
            answer = str(example.get("solution", ""))
        return {"prompt": prompt, "answer": answer}

    ds = ds.map(_render, remove_columns=ds.column_names)
    return ds
