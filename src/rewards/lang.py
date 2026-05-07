"""R5 — Language consistency reward (fastText langID).

Reward 1.0 nếu fastText predicts response language match prompt language.
ONLY active for non-EN prompts. EN prompts return 0.0 (no penalty).
Short responses (<10 tokens) return 0.0 (fastText unreliable).

Used in: Cond C (enlang) only.

Setup:
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P data/raw/
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from src.rewards import register

# Module-level cache cho fastText model: load once, reuse cho mọi call.
# Key = absolute path string; value = loaded model object hoặc None nếu file thiếu.
_FT_MODEL: dict[str, Any] = {}


def _load_fasttext(path: str | Path) -> Any | None:
    """Lazy load fastText model. Return None nếu file thiếu (graceful degrade)."""
    abs_path = str(Path(path).expanduser().resolve())
    if abs_path in _FT_MODEL:
        return _FT_MODEL[abs_path]

    if not Path(abs_path).exists():
        warnings.warn(
            f"fastText model not found at {abs_path}. "
            "R5 lang-consistency will return 0.0 for all samples. "
            "Run: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P data/raw/",
            RuntimeWarning,
            stacklevel=2,
        )
        _FT_MODEL[abs_path] = None
        return None

    try:
        import fasttext  # type: ignore

        # fastText prints C-level warnings on load; suppress nếu có thể
        model = fasttext.load_model(abs_path)
        _FT_MODEL[abs_path] = model
        return model
    except Exception as exc:
        warnings.warn(
            f"Failed to load fastText model at {abs_path}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        _FT_MODEL[abs_path] = None
        return None


def _detect_lang(model: Any, text: str) -> str | None:
    """Detect ISO-639-1 language code (e.g. 'en', 'vi', 'zh') from text.

    fastText label format: '__label__<iso>'. Input phải là single line.
    """
    # IMPORTANT: fastText input must be single-line — strip newlines/CR
    cleaned = text.replace("\n", " ").replace("\r", " ").strip()
    if not cleaned:
        return None
    try:
        labels, _probs = model.predict(cleaned, k=1)
        if not labels:
            return None
        # labels là tuple/list, mỗi item dạng '__label__en'
        label = labels[0]
        if label.startswith("__label__"):
            return label[len("__label__"):]
        return label
    except Exception:
        return None


@register("lang")
def r5_lang_consistency(
    prompts: list[str],
    completions: list[str],
    fasttext_model: str | Path = "data/raw/lid.176.bin",
    no_penalty_for_en: bool = True,
    min_response_tokens: int = 10,
    **kwargs: Any,
) -> list[float]:
    """fastText langID reward — match prompt language.

    Args:
        fasttext_model: path đến lid.176.bin.
        no_penalty_for_en: nếu True, prompts EN luôn return 0 (no penalty/bonus).
        min_response_tokens: dưới ngưỡng → return 0 (fastText unreliable on short text).

    Returns:
        list[float] of length len(prompts), values in {0.0, 1.0}.
    """
    if len(completions) != len(prompts):
        raise ValueError(
            f"Length mismatch: prompts={len(prompts)} completions={len(completions)}"
        )

    n = len(prompts)
    model = _load_fasttext(fasttext_model)
    if model is None:
        # Graceful: file thiếu → return tất cả 0.0 thay vì crash
        return [0.0] * n

    rewards: list[float] = []
    for prompt, completion in zip(prompts, completions):
        # Edge: completion quá ngắn (whitespace tokens) → fastText không reliable
        if len(completion.split()) < min_response_tokens:
            rewards.append(0.0)
            continue

        expected_lang = _detect_lang(model, prompt)
        if expected_lang is None:
            rewards.append(0.0)
            continue

        # No-penalty rule: prompt EN → 0.0 bất kể completion lang
        if no_penalty_for_en and expected_lang == "en":
            rewards.append(0.0)
            continue

        actual_lang = _detect_lang(model, completion)
        if actual_lang is None:
            rewards.append(0.0)
            continue

        rewards.append(1.0 if actual_lang == expected_lang else 0.0)

    return rewards
