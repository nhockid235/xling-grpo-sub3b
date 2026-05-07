"""MSVAMP eval adapter.

HF: Mathoctopus/MSVAMP, ~1K/lang × 10 langs.
Languages: bn, zh, en, fr, de, ja, ru, es, sw, th.
Schema: {id, question, answer, answer_number}.
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

MSVAMP_LANGS = ["bn", "zh", "en", "fr", "de", "ja", "ru", "es", "sw", "th"]

# Mathoctopus/MSVAMP layout: per-language config or per-language split — verify
# at runtime. Phần lớn release of Mathoctopus dùng config name = language.
_MSVAMP_HF_ID = "Mathoctopus/MSVAMP"


def evaluate(
    model: Any,
    language: str,
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
    if language not in MSVAMP_LANGS:
        raise ValueError(
            f"language='{language}' not in MSVAMP. Allowed: {MSVAMP_LANGS}"
        )

    metadata = make_metadata(model_path, max_tokens, temperature, seed)
    metadata["language"] = language

    if dataset is None:
        from datasets import load_dataset

        # Mathoctopus/MSVAMP: config name = uppercase lang code per HF card,
        # nhưng số release dùng lowercase. Thử cả hai.
        try:
            dataset = load_dataset(_MSVAMP_HF_ID, language, split="test")
        except Exception:
            dataset = load_dataset(_MSVAMP_HF_ID, language.upper(), split="test")

    records = list(dataset)
    if n_samples is not None and n_samples < len(records):
        records = records[:n_samples]

    if not records:
        return empty_result(run_id, "msvamp", language, metadata)

    problems = [r["question"] for r in records]
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
        "benchmark": "msvamp",
        "language": language,
        "n_samples": n,
        "pass_at_1": n_correct / n if n > 0 else 0.0,
        "maj_at_8": None,
        "lang_consistency_rate": None,
        "avg_response_tokens": avg_response_tokens(completions),
        "responses": completions,
        "metadata": metadata,
    }
