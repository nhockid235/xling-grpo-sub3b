"""R1 — Correctness reward via Math-Verify.

Returns 1.0 nếu predicted answer match gold answer (sympy equivalence), 0.0 otherwise.
Used in: all conditions (en, vi, enlang).
"""

from __future__ import annotations

from typing import Any

from src.rewards import register
from src.utils.parsing import (
    extract_answer_tag,
    extract_boxed,
    extract_last_number,
)


def _extract_prediction(completion: str) -> str | None:
    """Ưu tiên: <answer>...</answer> → \\boxed{...} → last number.

    Math-Verify parse hoạt động tốt nhất khi input là một biểu thức gọn,
    nên ta cố gắng trích đoạn đáp án trước khi pass cho parse().
    """
    # Step 1: thử lấy nội dung trong <answer> tag
    inside = extract_answer_tag(completion)
    if inside is not None:
        # Trong <answer> có thể vẫn bọc \boxed{...} → unwrap thêm một lớp
        boxed_inner = extract_boxed(inside)
        return boxed_inner if boxed_inner is not None else inside

    # Step 2: \boxed{...} ở bất kỳ đâu trong completion
    boxed = extract_boxed(completion)
    if boxed is not None:
        return boxed

    # Step 3: fallback — số cuối cùng (GSM8K-style)
    return extract_last_number(completion)


def _math_verify_match(pred: str | None, gold: str | None) -> float:
    """Wrap Math-Verify với try/except. Return 1.0 nếu match else 0.0."""
    if pred is None or gold is None:
        return 0.0
    try:
        # Lazy import để tránh fail nếu math-verify không có lúc import module
        from math_verify import parse, verify  # type: ignore

        gold_parsed = parse(str(gold))
        pred_parsed = parse(str(pred))
        # Math-Verify trả về True/False (or có thể truthy object)
        return 1.0 if verify(gold_parsed, pred_parsed) else 0.0
    except Exception:
        # Parse fail (LaTeX errors, sympy timeout, ...) → 0.0
        return 0.0


@register("correctness")
def r1_correctness(
    prompts: list[str],
    completions: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward 1.0 if Math-Verify(extract_answer(completion), gold) else 0.0.

    Args:
        prompts: list of prompt strings (length N).
        completions: list of model completions (length N).
        **kwargs: dataset columns passed by TRL. Expects ``answer`` or ``gold``
            column (list of N strings).

    Returns:
        list[float] of length N, values in {0.0, 1.0}.
    """
    n = len(prompts)
    # Cho phép nhận key 'answer' (mặc định) hoặc 'gold' (alias)
    golds = kwargs.get("answer", kwargs.get("gold"))
    if golds is None:
        # Không có gold → không thể chấm → tất cả 0.0
        return [0.0] * n
    if len(golds) != n:
        raise ValueError(
            f"Length mismatch: prompts={n} but answer/gold={len(golds)}"
        )

    rewards: list[float] = []
    for completion, gold in zip(completions, golds):
        pred = _extract_prediction(completion)
        rewards.append(_math_verify_match(pred, gold))
    return rewards
