"""Internal helpers chia sẻ giữa các benchmark adapters.

Không export public — chỉ dùng nội bộ trong `src.eval`.
"""

from __future__ import annotations

import datetime as _dt
from collections import Counter
from typing import Any

from src.utils.parsing import (
    extract_answer_tag,
    extract_boxed,
    extract_last_number,
)


def extract_prediction(completion: str) -> str | None:
    """Ưu tiên: <answer>...</answer> → \\boxed{...} → last number.

    Trùng logic với `src.rewards.correctness._extract_prediction`. Tách ra
    helper riêng để tránh import vòng (rewards module có @register decorator).
    """
    inside = extract_answer_tag(completion)
    if inside is not None:
        boxed_inner = extract_boxed(inside)
        return boxed_inner if boxed_inner is not None else inside

    boxed = extract_boxed(completion)
    if boxed is not None:
        return boxed

    return extract_last_number(completion)


def normalize_number_string(s: str | None) -> str | None:
    """Normalize numeric string: strip commas, '+' prefix, trailing dots, $."""
    if s is None:
        return None
    s = s.strip()
    # Remove common dollar/percent/units
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    # Strip leading '+'
    if s.startswith("+"):
        s = s[1:]
    # Strip trailing punctuation
    s = s.rstrip(".")
    return s if s else None


def numeric_match(pred: str | None, gold: str | None) -> bool:
    """So sánh số: thử cast float trước, nếu fail thì exact-string sau normalize."""
    p = normalize_number_string(pred)
    g = normalize_number_string(gold)
    if p is None or g is None:
        return False
    try:
        return float(p) == float(g)
    except (ValueError, TypeError):
        return p == g


def math_verify_match(pred: str | None, gold: str | None) -> bool:
    """Math-Verify wrapper với fallback graceful."""
    if pred is None or gold is None:
        return False
    try:
        from math_verify import parse, verify  # type: ignore

        gold_parsed = parse(str(gold))
        pred_parsed = parse(str(pred))
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        # Sympy/Math-Verify thiếu hoặc parse fail → fallback numeric match
        return numeric_match(pred, gold)


def majority_vote(predictions: list[str | None]) -> str | None:
    """Majority vote, ignoring None. Returns most common prediction or None."""
    valid = [p for p in predictions if p is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def vllm_generate(
    model: Any,
    prompts: list[str],
    sampling_params: Any | None = None,
) -> list[str]:
    """Wrap vLLM `LLM.generate(...)` → list of completion strings.

    Hỗ trợ mock model: nếu `model` có method `generate` trả list-like với
    `outputs[0].text`, ta đọc ra.
    """
    if model is None:
        # Dry-run mode: trả empty completions với cùng length.
        return ["" for _ in prompts]

    # vLLM signature: model.generate(prompts, sampling_params)
    # SamplingParams optional khi mock.
    if sampling_params is not None:
        outs = model.generate(prompts, sampling_params)
    else:
        outs = model.generate(prompts)

    completions: list[str] = []
    for o in outs:
        # vLLM RequestOutput.outputs[0].text
        if hasattr(o, "outputs") and len(o.outputs) > 0:
            completions.append(o.outputs[0].text)
        elif isinstance(o, str):
            completions.append(o)
        else:
            completions.append(str(o))
    return completions


def avg_response_tokens(responses: list[str]) -> float:
    """Whitespace-token approximation (chính xác đủ cho avg)."""
    if not responses:
        return 0.0
    total = sum(len(r.split()) for r in responses)
    return total / len(responses)


def make_metadata(
    model_path: str,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    """Standard metadata block."""
    try:
        import vllm  # type: ignore

        vllm_version = vllm.__version__
    except Exception:
        vllm_version = "unknown"
    return {
        "model_path": str(model_path),
        "eval_date": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "vllm_version": vllm_version,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }


def empty_result(
    run_id: str,
    benchmark: str,
    language: str | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Skeleton dict đầy đủ keys theo CLAUDE.md schema, mọi metric None/empty."""
    return {
        "run_id": run_id,
        "benchmark": benchmark,
        "language": language,
        "n_samples": 0,
        "pass_at_1": None,
        "maj_at_8": None,
        "lang_consistency_rate": None,
        "avg_response_tokens": 0.0,
        "responses": [],
        "metadata": metadata,
    }
