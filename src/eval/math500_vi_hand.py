"""Hand-translated MATH-500 Vietnamese subset eval adapter.

Phase 9.1.6: bypass MGSM/MSVAMP not covering Vietnamese by evaluating on
100 problems hand-translated from `HuggingFaceH4/MATH-500` by the paper author
(native Vietnamese speaker).

Source file: ``data/processed/math500_vi_100.jsonl``
Each line is a JSON object with fields:
    problem (str, VI), answer (str), level (str), subject (str), source (str)

Native translation gives a clean baseline for cross-lingual transfer claim
without translation-machine artifacts (avoids confounds in MGSM Google MT).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.eval._common import (
    avg_response_tokens,
    empty_result,
    extract_prediction,
    make_metadata,
    math_verify_match,
    numeric_match,
    vllm_generate,
)
from src.eval.prompts import build_prompts

DEFAULT_JSONL = "data/processed/math500_vi_100.jsonl"


def _load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    if numeric_match(pred, gold):
        return True
    return math_verify_match(pred, gold)


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
    dataset: list[dict] | None = None,
    jsonl_path: str | Path = DEFAULT_JSONL,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate model on hand-translated VI MATH-500 subset.

    Args:
        model: vLLM ``LLM`` (or mock with ``.generate(prompts, sampling_params)``).
        n_samples: limit cho dev. None = all.
        sampling_params: greedy params (temperature 0).
        run_id, model_path: metadata fields.
        chat_template_tokenizer: HF tokenizer cho chat template (optional).
        dataset: injectable list[dict] cho test mock; None = load from jsonl_path.
        jsonl_path: path tới hand-translated VI JSONL (default ``DEFAULT_JSONL``).

    Returns:
        Dict matching CLAUDE.md eval JSON schema, with `language="vi"`.
    """
    metadata = make_metadata(model_path, max_tokens, temperature, seed)
    metadata["dataset"] = "math500_vi_hand"
    metadata["jsonl_path"] = str(jsonl_path)
    metadata["language"] = "vi"

    if dataset is None:
        dataset = _load_jsonl(jsonl_path)

    if not dataset:
        # JSONL chưa tồn tại (user chưa hand-translate xong) — return empty result.
        metadata["error"] = f"JSONL file not found or empty at {jsonl_path}"
        return empty_result(run_id, "math500_vi_hand", "vi", metadata)

    records = list(dataset)
    if n_samples is not None and n_samples < len(records):
        records = records[:n_samples]

    problems = [r["problem"] for r in records]
    golds = [str(r["answer"]) for r in records]
    n = len(records)

    prompts = build_prompts(
        problems,
        system_prompt=system_prompt,
        chat_template_tokenizer=chat_template_tokenizer,
    )

    completions = vllm_generate(model, prompts, sampling_params=sampling_params)
    n_correct = sum(
        1 for c, g in zip(completions, golds)
        if _match(extract_prediction(c), g)
    )
    pass_at_1 = n_correct / n if n > 0 else 0.0

    return {
        "run_id": run_id,
        "benchmark": "math500_vi_hand",
        "language": "vi",
        "n_samples": n,
        "pass_at_1": pass_at_1,
        "maj_at_8": None,
        "lang_consistency_rate": None,   # post-hoc compute via lang_consistency.py
        "avg_response_tokens": avg_response_tokens(completions),
        "responses": completions,
        "metadata": metadata,
    }
