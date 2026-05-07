"""MGSM eval adapter — multilingual GSM8K.

HF: juletxara/mgsm, split=test, 250/lang.
⚠️ 10 langs guaranteed: en, es, fr, de, ru, zh, ja, th, sw, bn (NOT 11 — Telugu may be missing).
Schema: {question, answer, answer_number, equation_solution}.
Metrics: pass@1 + lang_consistency (compute từ responses[] post-hoc).
"""

from __future__ import annotations

from typing import Any

from src.eval._common import (
    avg_response_tokens,
    empty_result,
    extract_prediction,
    make_metadata,
    numeric_match,
    vllm_generate,
)
from src.eval.prompts import build_prompts

MGSM_LANGS_GUARANTEED = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn"]


def evaluate(
    model: Any,
    language: str,
    n_samples: int = 250,
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
    if language not in MGSM_LANGS_GUARANTEED:
        raise ValueError(
            f"language='{language}' not in MGSM 10 guaranteed langs. "
            f"Allowed: {MGSM_LANGS_GUARANTEED}"
        )

    metadata = make_metadata(model_path, max_tokens, temperature, seed)
    metadata["language"] = language

    if dataset is None:
        from datasets import load_dataset

        # juletxara/mgsm uses language code as config name
        dataset = load_dataset("juletxara/mgsm", language, split="test")

    records = list(dataset)
    if n_samples is not None and n_samples < len(records):
        records = records[:n_samples]

    if not records:
        return empty_result(run_id, "mgsm", language, metadata)

    problems = [r["question"] for r in records]
    # answer_number is the canonical numeric gold (avoids parsing local-format text)
    golds = [str(r["answer_number"]) for r in records]

    prompts = build_prompts(
        problems,
        system_prompt=system_prompt,
        chat_template_tokenizer=chat_template_tokenizer,
    )
    completions = vllm_generate(model, prompts, sampling_params=sampling_params)

    n_correct = 0
    for completion, gold in zip(completions, golds):
        pred = extract_prediction(completion)
        if numeric_match(pred, gold):
            n_correct += 1

    n = len(records)
    return {
        "run_id": run_id,
        "benchmark": "mgsm",
        "language": language,
        "n_samples": n,
        "pass_at_1": n_correct / n if n > 0 else 0.0,
        "maj_at_8": None,
        # lang_consistency computed post-hoc qua src/eval/lang_consistency.py
        "lang_consistency_rate": None,
        "avg_response_tokens": avg_response_tokens(completions),
        "responses": completions,
        "metadata": metadata,
    }
