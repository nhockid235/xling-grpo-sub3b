"""R5 — Language consistency reward (fastText langID).

ONLY active for non-EN prompts. EN prompts return 0.0 (no penalty).
Short responses (<10 tokens) return 0.0 (fastText unreliable).

Used in: Cond C (enlang) only.

Setup:
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P data/raw/"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from src.rewards import register

_FT_MODEL: dict[str, Any] = {}


def _load_fasttext(path: str | Path) -> Any | None:
    """Load fastText langID model with caching; warns and returns None on failure."""
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
    """Detect ISO-639-1 language code (e.g. 'en', 'vi', 'zh') from text."""
    # IMPORTANT: fastText input must be single-line -- strip newlines/CR
    cleaned = text.replace("\n", " ").replace("\r", " ").strip()
    if not cleaned:
        return None
    try:
        labels, _probs = model.predict(cleaned, k=1)
        if not labels:
            return None
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
    """fastText langID reward -- 1.0 if completion language matches the prompt language.

    EN prompts return 0.0 (no penalty) when ``no_penalty_for_en`` is True.
    Completions shorter than ``min_response_tokens`` return 0.0.

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
        return [0.0] * n

    rewards: list[float] = []
    for prompt, completion in zip(prompts, completions):
        if len(completion.split()) < min_response_tokens:
            rewards.append(0.0)
            continue

        expected_lang = _detect_lang(model, prompt)
        if expected_lang is None:
            rewards.append(0.0)
            continue

        if no_penalty_for_en and expected_lang == "en":
            rewards.append(0.0)
            continue

        actual_lang = _detect_lang(model, completion)
        if actual_lang is None:
            rewards.append(0.0)
            continue

        rewards.append(1.0 if actual_lang == expected_lang else 0.0)

    return rewards
