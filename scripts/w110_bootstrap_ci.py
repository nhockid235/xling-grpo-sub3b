"""W1.10 — Bootstrap 95% CI for headline Δ-vs-base claims (CPU local).

Reads results/eval/*_step50/*aime2024*.json and results/eval/w17_*/*aime2024*.json,
extracts per-problem maj@8 correctness, computes seed-level means + bootstrap CI
for each arm − base comparison.

Bootstrap method: subject bootstrap over seed-level means (n=3 seeds per arm),
10,000 resamples. Per rebuttal_prep §5.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_run_id(rid: str) -> tuple[str, int | None]:
    rid = rid.replace("_step50", "").replace("_v2", "")
    m = re.match(r"^(.+?)_(\d+)$", rid)
    if m:
        return m.group(1), int(m.group(2))
    return rid, None


def load_aime(eval_root: Path) -> dict[str, dict[int, dict]]:
    """{arm: {seed: {pass_at_1, maj_at_8, problem_correct}}}."""
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for fp in sorted(eval_root.glob("**/*aime2024*.json")):
        if "responses_only" in fp.name or "tmp_" in fp.name:
            continue
        d = json.loads(fp.read_text())
        rid = d.get("run_id", fp.stem)
        arm, seed = parse_run_id(rid)
        if seed is None:
            # Single-seed entry (e.g. base). Bucket under seed=42 by convention.
            if arm.startswith("base") or arm.startswith("ckpt50") or arm.startswith("openrs2"):
                seed = 42
            else:
                continue
        # Get per-problem maj@8 correctness if available
        per_problem = d.get("per_problem_maj_at_8", d.get("per_problem_pass_at_1"))
        out[arm][seed] = {
            "pass_at_1": d.get("pass_at_1"),
            "maj_at_8": d.get("maj_at_8"),
            "per_problem": per_problem,
            "file": str(fp),
        }
    return out


def bootstrap_delta(arm_seed_vals: list[float], base_val: float,
                    n_bootstrap: int = 10000, seed: int = 42) -> tuple[float, float, float, float]:
    """Subject bootstrap over seed-level means.

    Returns (mean_delta, ci_low, ci_high, p_value_one_sided).
    """
    rng = np.random.default_rng(seed)
    arr = np.array(arm_seed_vals, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    # Resample seeds with replacement, take mean, subtract base
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    sample_means = arr[idx].mean(axis=1)
    deltas = sample_means - base_val
    obs = arr.mean() - base_val
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    # One-sided p: fraction of resamples that crossed 0 in the opposite direction
    if obs >= 0:
        p = float((deltas <= 0).mean())
    else:
        p = float((deltas >= 0).mean())
    return float(obs), lo, hi, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, default=Path("results/eval"))
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    args = ap.parse_args()

    data = load_aime(args.eval_root)
    print(f"\n=== Eval JSONs found ===")
    for arm in sorted(data):
        seeds = sorted(data[arm].keys())
        print(f"  {arm:<30} seeds={seeds}")

    # Determine base — prefer post-fix base in this priority order:
    base_priority = ["base_distill_v2", "base_v3", "base_deepseek_r1_distill_15b"]
    base_key = next((k for k in base_priority if k in data), None)
    if base_key is None:
        base_key = next((k for k in data if k.startswith("base")), None)
    if base_key is None:
        print("ERROR: no base eval found")
        return
    # Exclude all other "base*" / "ckpt50*" / "openrs*" entries from arm comparison
    skip_arms = {k for k in data if k.startswith("base") or k.startswith("ckpt50") or k.startswith("openrs")}
    skip_arms.discard(base_key)
    base_seeds = data[base_key]
    base_m8 = np.mean([d["maj_at_8"] for d in base_seeds.values() if d.get("maj_at_8") is not None])
    base_p1 = np.mean([d["pass_at_1"] for d in base_seeds.values() if d.get("pass_at_1") is not None])
    print(f"\nBase ({base_key}) mean: p@1={base_p1:.4f} m@8={base_m8:.4f}")
    if skip_arms:
        print(f"Skipping non-canonical base entries: {sorted(skip_arms)}")

    print(f"\n=== Bootstrap 95% CI (n={args.n_bootstrap} resamples) ===")
    print(f"{'Comparison':<35} {'Δ p@1 (pp)':<28} {'Δ m@8 (pp)':<28}")
    print("-" * 95)
    for arm in sorted(data):
        if arm == base_key or arm in skip_arms:
            continue
        seeds = sorted(data[arm].keys())
        p1_vals = [data[arm][s].get("pass_at_1") for s in seeds if data[arm][s].get("pass_at_1") is not None]
        m8_vals = [data[arm][s].get("maj_at_8") for s in seeds if data[arm][s].get("maj_at_8") is not None]

        d1, l1, h1, _ = bootstrap_delta(p1_vals, base_p1, args.n_bootstrap)
        d8, l8, h8, p8 = bootstrap_delta(m8_vals, base_m8, args.n_bootstrap)

        s1 = f"{d1*100:+.1f} [{l1*100:+.1f}, {h1*100:+.1f}]"
        s8 = f"{d8*100:+.1f} [{l8*100:+.1f}, {h8*100:+.1f}]"
        sig_m8 = "*" if (l8 > 0 or h8 < 0) else " "
        print(f"{arm:<35} {s1:<28} {s8:<28} {sig_m8}")

    print("\n* = 95% CI excludes 0 → statistically distinguishable at 5% level.")
    print("\nNote: n=3 seeds per arm. Bootstrap on seed-level means is conservative;")
    print("with so few seeds, CIs are wide. Interpret * as 'directional evidence,' not strict NHST.")


if __name__ == "__main__":
    main()
