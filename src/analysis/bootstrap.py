"""Bootstrap 95% CI cho headline numbers.

Percentile method, 1000 samples mặc định.
Pure numpy — không phụ thuộc scipy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) at given confidence.

    Args:
        values: per-sample correctness/scores (e.g., 0/1 cho pass@1).
        n_bootstrap: số bootstrap samples.
        confidence: 0.95 → return 2.5th/97.5th percentiles.
        seed: random seed cho reproducibility.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    n = arr.size

    # Vectorized: sample N×B indices once, gather, reduce
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    samples_means = arr[idx].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(samples_means, 100.0 * alpha))
    upper = float(np.percentile(samples_means, 100.0 * (1.0 - alpha)))
    mean = float(arr.mean())
    return (mean, lower, upper)


def format_ci(mean: float, lower: float, upper: float, as_percent: bool = True) -> str:
    """Render '45.6% [42.1, 49.0]' kiểu paper."""
    scale = 100.0 if as_percent else 1.0
    suffix = "%" if as_percent else ""
    return f"{mean * scale:.1f}{suffix} [{lower * scale:.1f}, {upper * scale:.1f}]"
