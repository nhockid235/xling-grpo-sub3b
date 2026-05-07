"""MATH-500 eval adapter.

HF: HuggingFaceH4/MATH-500, split=test (500 records).
Schema: {problem, solution, answer, subject, level, unique_id}. Math-Verify
(sympy equivalence) là chuẩn chấm — fallback numeric match khi parse fail.
"""

from __future__ import annotations

from typing import Any

from src.eval._common import (
    avg_response_tokens,
    empty_result,
    extract_prediction,
    make_metadata,
    math_verify_match,
    vllm_generate,
)
from src.eval.prompts import build_prompts


def evaluate(
    model: Any,
    n_samples: int | None = None,
    seed: int = 42,
    sampling_params: Any | None = None,
    *,
    run_id: str = "unknown",
    model_path: str = "unknown",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    chat_template_tokenizer: Any | None = None,
    system_prompt: str | None = None,
    dataset: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    metadata = make_metadata(model_path, max_tokens, temperature, seed)

    if dataset is None:
        from datasets import load_dataset

        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    records = list(dataset)
    if n_samples is not None and n_samples < len(records):
        records = records[:n_samples]

    if not records:
        return empty_result(run_id, "math500", None, metadata)

    problems = [r["problem"] for r in records]
    golds = [str(r["answer"]) for r in records]

    prompts = build_prompts(
        problems,
        system_prompt=system_prompt,
        chat_template_tokenizer=chat_template_tokenizer,
    )
    completions = vllm_generate(model, prompts, sampling_params=sampling_params)

    n_correct = 0
    for completion, gold in zip(completions, golds):
        pred = extract_prediction(completion)
        if math_verify_match(pred, gold):
            n_correct += 1

    n = len(records)
    return {
        "run_id": run_id,
        "benchmark": "math500",
        "language": "en",
        "n_samples": n,
        "pass_at_1": n_correct / n if n > 0 else 0.0,
        "maj_at_8": None,
        "lang_consistency_rate": None,
        "avg_response_tokens": avg_response_tokens(completions),
        "responses": completions,
        "metadata": metadata,
    }
