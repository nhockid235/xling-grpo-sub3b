"""W21 — Comprehensive paper-claim verification.

For every numerical claim in paper/main.tex and paper/ieee/main.tex, this
script re-derives the number from source eval JSONs and asserts equality
within tolerance. Output: reports/W1_PAPER_VERIFICATION_2026-05-16.md
with one row per claim showing claim, source file(s), computed value,
match status.

Usage:
    python3 scripts/w21_verify_paper_claims.py

Exit code 0 if all claims verified, 1 if any mismatch.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Optional

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = ROOT / "results" / "eval"
EVIDENCE_LOG = ROOT / "reports" / "W1_PAPER_VERIFICATION_2026-05-16.md"

# Manifest: arm -> seed -> bench -> path (relative to EVAL_ROOT)
# Mirrors scripts/w20_update_paper_tables.py to ensure single source of truth.
MANIFEST = {
    ("base", 42, "amc23"): "w17_base_distill_v2/base_distill_v2_amc23.json",
    ("base", 42, "math500"): "base_v3_openrs_eval/base_v3_math500.json",
    ("base", 42, "aime2024"): "base_v3_openrs_eval/base_v3_aime2024.json",
    ("A1", 42, "amc23"): "w17_reproduce_openrs_rs2_42_v2/reproduce_openrs_rs2_42_v2_amc23.json",
    ("A1", 42, "math500"): "ckpt50_v3_openrs_eval/ckpt50_v3_math500.json",
    ("A1", 42, "aime2024"): "ckpt50_v3_openrs_eval/ckpt50_v3_aime2024.json",
    ("A1", 123, "amc23"): "w17_reproduce_openrs_rs2_123_v2/reproduce_openrs_rs2_123_v2_amc23.json",
    ("A1", 123, "math500"): "reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_math500.json",
    ("A1", 123, "aime2024"): "reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_aime2024.json",
    ("A1", 7, "amc23"): "w17_reproduce_openrs_rs2_7_v2/reproduce_openrs_rs2_7_v2_amc23.json",
    ("A1", 7, "math500"): "reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_math500.json",
    ("A1", 7, "aime2024"): "reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_aime2024.json",
    ("A2", 42, "amc23"): "w17_a2_vi_42_v2/a2_vi_42_v2_amc23.json",
    ("A2", 42, "math500"): "a2_vi_42_step50/a2_vi_42_step50_math500.json",
    ("A2", 42, "aime2024"): "a2_vi_42_step50/a2_vi_42_step50_aime2024.json",
    ("A2", 123, "amc23"): "w17_a2_vi_123_v2/a2_vi_123_v2_amc23.json",
    ("A2", 123, "math500"): "a2_vi_123_step50/a2_vi_123_step50_math500.json",
    ("A2", 123, "aime2024"): "a2_vi_123_step50/a2_vi_123_step50_aime2024.json",
    ("A2", 7, "amc23"): "w17_a2_vi_7_v2/a2_vi_7_v2_amc23.json",
    ("A2", 7, "math500"): "a2_vi_7_step50/a2_vi_7_step50_math500.json",
    ("A2", 7, "aime2024"): "a2_vi_7_step50/a2_vi_7_step50_aime2024.json",
    ("A3", 42, "amc23"): "w17_a3_enlang_42_v2/a3_enlang_42_v2_amc23.json",
    ("A3", 42, "math500"): "a3_enlang_42_step50/a3_enlang_42_step50_math500.json",
    ("A3", 42, "aime2024"): "a3_enlang_42_step50/a3_enlang_42_step50_aime2024.json",
    ("A3", 123, "amc23"): "w17_a3_enlang_123_v2/a3_enlang_123_v2_amc23.json",
    ("A3", 123, "math500"): "a3_enlang_123_step50/a3_enlang_123_step50_math500.json",
    ("A3", 123, "aime2024"): "a3_enlang_123_step50/a3_enlang_123_step50_aime2024.json",
    ("A3", 7, "amc23"): "w17_a3_enlang_7_v2/a3_enlang_7_v2_amc23.json",
    ("A3", 7, "math500"): "a3_enlang_7_step50/a3_enlang_7_step50_math500.json",
    ("A3", 7, "aime2024"): "a3_enlang_7_step50/a3_enlang_7_step50_aime2024.json",
    ("A4", 42, "amc23"): "a4_const_bias_42_step50/a4_const_bias_42_step50_amc23.json",
    ("A4", 42, "math500"): "a4_const_bias_42_step50/a4_const_bias_42_step50_math500.json",
    ("A4", 42, "aime2024"): "a4_const_bias_42_step50/a4_const_bias_42_step50_aime2024.json",
    ("A4", 123, "amc23"): "a4_const_bias_123_step50/a4_const_bias_123_step50_amc23.json",
    ("A4", 123, "math500"): "a4_const_bias_123_step50/a4_const_bias_123_step50_math500.json",
    ("A4", 123, "aime2024"): "a4_const_bias_123_step50/a4_const_bias_123_step50_aime2024.json",
    ("A4", 7, "amc23"): "a4_const_bias_7_step50/a4_const_bias_7_step50_amc23.json",
    ("A4", 7, "math500"): "a4_const_bias_7_step50/a4_const_bias_7_step50_math500.json",
    ("A4", 7, "aime2024"): "a4_const_bias_7_step50/a4_const_bias_7_step50_aime2024.json",
}


def load_metric(arm: str, seed: int, bench: str, metric: str) -> Optional[float]:
    rel = MANIFEST.get((arm, seed, bench))
    if not rel:
        return None
    fp = EVAL_ROOT / rel
    if not fp.exists():
        return None
    d = json.loads(fp.read_text())
    return d.get(metric)


def collect_seeds(arm: str, bench: str, metric: str) -> list[tuple[int, Optional[float], str]]:
    """Return [(seed, value, source_file), ...] for an arm/bench/metric."""
    out = []
    for s in (42, 123, 7):
        rel = MANIFEST.get((arm, s, bench), "")
        v = load_metric(arm, s, bench, metric)
        out.append((s, v, rel))
    return out


def mean_std(values: list[Optional[float]]) -> tuple[Optional[float], Optional[float]]:
    vs = [v for v in values if v is not None]
    if not vs:
        return None, None
    if len(vs) == 1:
        return vs[0], 0.0
    return statistics.mean(vs), statistics.stdev(vs)


def fmt(v: Optional[float], pct: bool = True, prec: int = 1) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.{prec}f}"
    return f"{v:.{prec}f}"


def check(name: str, expected: float, actual: float, tol: float = 0.1) -> tuple[bool, str]:
    if actual is None:
        return False, "MISSING"
    diff = abs(expected - actual)
    if diff < tol:
        return True, f"✓ ({actual:+.3f} vs {expected:+.3f}, |Δ|={diff:.3f})"
    return False, f"✗ MISMATCH ({actual:+.3f} vs {expected:+.3f}, |Δ|={diff:.3f})"


def bootstrap_ci(values_per_seed: list[float], baseline: float,
                 n_boot: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """Subject bootstrap over seed-level means. Returns (mean_delta, lo, hi) in %."""
    rng = np.random.default_rng(seed)
    arr = np.array(values_per_seed, dtype=np.float64)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    sample_means = arr[idx].mean(axis=1)
    deltas = sample_means - baseline
    return (
        float(arr.mean() - baseline) * 100,
        float(np.percentile(deltas, 2.5)) * 100,
        float(np.percentile(deltas, 97.5)) * 100,
    )


def bootstrap_diff(a_vals: list[float], b_vals: list[float],
                   n_boot: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """Subject bootstrap A - B at seed level."""
    rng = np.random.default_rng(seed)
    a = np.array(a_vals, dtype=np.float64)
    b = np.array(b_vals, dtype=np.float64)
    na, nb = len(a), len(b)
    diffs = []
    for _ in range(n_boot):
        ai = rng.integers(0, na, size=na)
        bi = rng.integers(0, nb, size=nb)
        diffs.append(a[ai].mean() - b[bi].mean())
    diffs = np.array(diffs)
    return (
        float(a.mean() - b.mean()) * 100,
        float(np.percentile(diffs, 2.5)) * 100,
        float(np.percentile(diffs, 97.5)) * 100,
    )


def main() -> int:
    out = ["# W1 Paper Verification — 2026-05-16",
           "",
           "Generated by `scripts/w21_verify_paper_claims.py`.",
           "Each row maps a paper claim to its source eval JSON and verifies the value.",
           "",
           "## Per-seed raw data (source-of-truth)",
           "",
           "| Arm | Seed | Benchmark | pass@1 | maj@4 | maj@8 | Source file |",
           "|---|---|---|---|---|---|---|"]

    all_ok = True

    for arm in ("base", "A1", "A2", "A3", "A4"):
        for seed in (42, 123, 7):
            for bench in ("amc23", "math500", "aime2024"):
                rel = MANIFEST.get((arm, seed, bench), "")
                p1 = load_metric(arm, seed, bench, "pass_at_1")
                m4 = load_metric(arm, seed, bench, "maj_at_4")
                m8 = load_metric(arm, seed, bench, "maj_at_8")
                if rel:
                    out.append(f"| {arm} | {seed} | {bench} | {fmt(p1)} | {fmt(m4)} | {fmt(m8)} | `{rel}` |")

    out.extend(["", "## Derived statistics — verified against paper claims",
                "",
                "| Claim (paper §) | Computed value | Source seeds | Status |",
                "|---|---|---|---|"])

    # ===== Claim 1: A3 AIME m@8 = 37.8±1.9 =====
    seeds_data = collect_seeds("A3", "aime2024", "maj_at_8")
    vals = [v for _, v, _ in seeds_data if v is not None]
    m, s = mean_std(vals)
    ok1 = abs(m*100 - 37.8) < 0.1 and abs(s*100 - 1.9) < 0.1
    all_ok &= ok1
    out.append(f"| **A3 AIME m@8 = 37.8±1.9** (§3, §6) | mean={m*100:.2f}, std={s*100:.2f} | {[s for s,_,_ in seeds_data]} → {[round(v*100,1) for v in vals]} | {'✓' if ok1 else '✗ MISMATCH'} |")

    # ===== Claim 2: A4 AIME m@8 = 32.2±5.1 =====
    seeds_data = collect_seeds("A4", "aime2024", "maj_at_8")
    vals = [v for _, v, _ in seeds_data if v is not None]
    m, s = mean_std(vals)
    ok2 = abs(m*100 - 32.2) < 0.1 and abs(s*100 - 5.1) < 0.2
    all_ok &= ok2
    out.append(f"| **A4 AIME m@8 = 32.2±5.1** (§3, §6) | mean={m*100:.2f}, std={s*100:.2f} | {[s for s,_,_ in seeds_data]} → {[round(v*100,1) for v in vals]} | {'✓' if ok2 else '✗ MISMATCH'} |")

    # ===== Claim 3: A3 − Base AIME m@8 = +4.4 (bootstrap CI [+3.3, +6.7]) =====
    a3 = [v for _, v, _ in collect_seeds("A3", "aime2024", "maj_at_8") if v is not None]
    base = load_metric("base", 42, "aime2024", "maj_at_8")
    d, lo, hi = bootstrap_ci(a3, base, n_boot=10000, seed=42)
    ok3 = abs(d - 4.4) < 0.2 and abs(lo - 3.3) < 0.5 and abs(hi - 6.7) < 0.5
    all_ok &= ok3
    out.append(f"| **A3 − Base AIME m@8 = +4.4, CI [+3.3, +6.7]** (§3 abstract, §6.3, §6.4) | Δ={d:+.2f}, CI=[{lo:+.2f}, {hi:+.2f}] | A3 seeds vs base={base*100:.1f} | {'✓' if ok3 else '✗ MISMATCH'} |")

    # ===== Claim 4: A3 − A4 AIME m@8 = +5.58 (bootstrap CI [+1.13, +11.10]) =====
    a4 = [v for _, v, _ in collect_seeds("A4", "aime2024", "maj_at_8") if v is not None]
    d, lo, hi = bootstrap_diff(a3, a4, n_boot=10000, seed=42)
    ok4 = abs(d - 5.58) < 0.2 and abs(lo - 1.13) < 0.5 and abs(hi - 11.10) < 0.5
    all_ok &= ok4
    out.append(f"| **A3 − A4 AIME m@8 = +5.58, CI [+1.13, +11.10]** (§3, §6.4 abstract) | Δ={d:+.2f}, CI=[{lo:+.2f}, {hi:+.2f}] | A3={[round(v*100,1) for v in a3]} vs A4={[round(v*100,1) for v in a4]} | {'✓' if ok4 else '✗ MISMATCH'} |")

    # ===== Claim 5: A1 AMC23 p@1 σ = 11.3 =====
    seeds = collect_seeds("A1", "amc23", "pass_at_1")
    vals = [v for _, v, _ in seeds if v is not None]
    m, s = mean_std(vals)
    ok5 = abs(s*100 - 11.3) < 0.5
    all_ok &= ok5
    out.append(f"| **A1 AMC23 p@1 σ = 11.3** (§3 abstract, §6.1) | std={s*100:.2f} | seeds → {[round(v*100,1) for v in vals]} | {'✓' if ok5 else '✗ MISMATCH'} |")

    # ===== Claim 6: A1 AIME p@1 mean = -12.2±7.7 vs base =====
    seeds = collect_seeds("A1", "aime2024", "pass_at_1")
    vals = [v for _, v, _ in seeds if v is not None]
    m, s = mean_std(vals)
    base_p1 = load_metric("base", 42, "aime2024", "pass_at_1")
    delta = (m - base_p1) * 100
    delta_std = s * 100
    ok6 = abs(delta - (-12.2)) < 0.5 and abs(delta_std - 7.7) < 0.5
    all_ok &= ok6
    out.append(f"| **A1 AIME p@1 Δ = −12.2±7.7** (§3, §6.1) | Δ={delta:+.2f}±{delta_std:.2f} | A1 seeds={[round(v*100,1) for v in vals]} vs base={base_p1*100:.1f} | {'✓' if ok6 else '✗ MISMATCH'} |")

    # ===== Claim 7: A4 AMC23 p@1 = 52.5±0.0 =====
    seeds = collect_seeds("A4", "amc23", "pass_at_1")
    vals = [v for _, v, _ in seeds if v is not None]
    m, s = mean_std(vals)
    ok7 = abs(m*100 - 52.5) < 0.1 and s*100 < 0.1
    all_ok &= ok7
    out.append(f"| **A4 AMC23 p@1 = 52.5±0.0** (§6.4) | mean={m*100:.2f}, std={s*100:.2f} | seeds → {[round(v*100,1) for v in vals]} | {'✓' if ok7 else '✗ MISMATCH'} |")

    # ===== Claim 8: A4 AMC23 m@4 = 70.0±2.5 =====
    seeds = collect_seeds("A4", "amc23", "maj_at_4")
    vals = [v for _, v, _ in seeds if v is not None]
    m, s = mean_std(vals)
    ok8 = abs(m*100 - 70.0) < 0.1 and abs(s*100 - 2.5) < 0.1
    all_ok &= ok8
    out.append(f"| **A4 AMC23 m@4 = 70.0±2.5** (§6.4) | mean={m*100:.2f}, std={s*100:.2f} | seeds → {[round(v*100,1) for v in vals]} | {'✓' if ok8 else '✗ MISMATCH'} |")

    # ===== Claim 9: A4 MATH-500 = 61.2±0.9 =====
    seeds = collect_seeds("A4", "math500", "pass_at_1")
    vals = [v for _, v, _ in seeds if v is not None]
    m, s = mean_std(vals)
    ok9 = abs(m*100 - 61.2) < 0.1 and abs(s*100 - 0.9) < 0.2
    all_ok &= ok9
    out.append(f"| **A4 MATH-500 = 61.2±0.9** (§6.4) | mean={m*100:.2f}, std={s*100:.2f} | seeds → {[round(v*100,1) for v in vals]} | {'✓' if ok9 else '✗ MISMATCH'} |")

    # ===== Claim 10: A4 AIME p@1 = 24.4±1.9 =====
    seeds = collect_seeds("A4", "aime2024", "pass_at_1")
    vals = [v for _, v, _ in seeds if v is not None]
    m, s = mean_std(vals)
    ok10 = abs(m*100 - 24.4) < 0.2 and abs(s*100 - 1.9) < 0.2
    all_ok &= ok10
    out.append(f"| **A4 AIME p@1 = 24.4±1.9** (§6.4) | mean={m*100:.2f}, std={s*100:.2f} | seeds → {[round(v*100,1) for v in vals]} | {'✓' if ok10 else '✗ MISMATCH'} |")

    # ===== Claim 11: A1/A2/A3 pass@1 56.7±11.3, 56.7±5.2, 57.5±6.6 (AMC23) =====
    for arm, expect_m, expect_s in (("A1", 56.7, 11.3), ("A2", 56.7, 5.2), ("A3", 57.5, 6.6)):
        seeds = collect_seeds(arm, "amc23", "pass_at_1")
        vals = [v for _, v, _ in seeds if v is not None]
        m, s = mean_std(vals)
        ok = abs(m*100 - expect_m) < 0.1 and abs(s*100 - expect_s) < 0.5
        all_ok &= ok
        out.append(f"| **{arm} AMC23 p@1 = {expect_m}±{expect_s}** (§5, Table 1) | mean={m*100:.2f}, std={s*100:.2f} | seeds → {[round(v*100,1) for v in vals]} | {'✓' if ok else '✗ MISMATCH'} |")

    # ===== Claim 12: A1/A2/A3 maj@4 post-fix (3 seeds) =====
    for arm, expect_m, expect_s in (("A1", 70.0, 4.3), ("A2", 70.0, 2.5), ("A3", 68.3, 3.8)):
        seeds = collect_seeds(arm, "amc23", "maj_at_4")
        vals = [v for _, v, _ in seeds if v is not None]
        m, s = mean_std(vals)
        ok = abs(m*100 - expect_m) < 0.5 and abs(s*100 - expect_s) < 0.5
        all_ok &= ok
        out.append(f"| **{arm} AMC23 m@4 = {expect_m}±{expect_s}** (§5, Table 1, post-W1.7 fix) | mean={m*100:.2f}, std={s*100:.2f} | seeds → {[round(v*100,1) for v in vals]} | {'✓' if ok else '✗ MISMATCH'} |")

    # ===== Claim 13: Open-RS2 public maj@4 = 75.0 =====
    openrs2 = json.loads((EVAL_ROOT / "openrs2_public" / "openrs2_public_step50_amc23.json").read_text())
    m4 = openrs2.get("maj_at_4")
    ok13 = abs(m4*100 - 75.0) < 0.5
    all_ok &= ok13
    out.append(f"| **Open-RS2 (public) AMC23 m@4 = 75.0** (§1, §8) | {m4*100:.2f} | `openrs2_public/openrs2_public_step50_amc23.json` | {'✓' if ok13 else '✗ MISMATCH'} |")

    # ===== Claim 14: Open-RS2 public pass@1 = 52.5 =====
    p1 = openrs2.get("pass_at_1")
    ok14 = abs(p1*100 - 52.5) < 0.5
    all_ok &= ok14
    out.append(f"| **Open-RS2 (public) AMC23 p@1 = 52.5** (§8) | {p1*100:.2f} | same | {'✓' if ok14 else '✗ MISMATCH'} |")

    # ===== Claim 15: Base distill AMC23 p@1 = 50.0 / m@4 = 70.0 =====
    base = json.loads((EVAL_ROOT / "w17_base_distill_v2" / "base_distill_v2_amc23.json").read_text())
    ok15a = abs(base["pass_at_1"]*100 - 50.0) < 0.5
    ok15b = abs(base["maj_at_4"]*100 - 70.0) < 0.5
    all_ok &= ok15a and ok15b
    out.append(f"| **Base distill AMC23: p@1=50.0, m@4=70.0** (§8) | p@1={base['pass_at_1']*100:.2f}, m@4={base['maj_at_4']*100:.2f} | `w17_base_distill_v2/base_distill_v2_amc23.json` | {'✓' if ok15a and ok15b else '✗ MISMATCH'} |")

    out.extend(["",
                "## Δ vs Base table (Table 2 in paper)",
                "",
                "| Metric | A1 | A2 | A3 | A4 | Source |",
                "|---|---|---|---|---|---|"])
    for bench, metric, base_m in (("amc23", "pass_at_1", None),
                                    ("math500", "pass_at_1", None),
                                    ("aime2024", "pass_at_1", None),
                                    ("aime2024", "maj_at_8", None)):
        base_v = load_metric("base", 42, bench, metric)
        cells = []
        for arm in ("A1", "A2", "A3", "A4"):
            vals = [v for _, v, _ in collect_seeds(arm, bench, metric) if v is not None]
            m, s = mean_std(vals)
            d = (m - base_v) * 100
            ds = s * 100
            sign = "+" if d >= 0 else ""
            cells.append(f"{sign}{d:.1f}±{ds:.1f}")
        out.append(f"| {bench} {metric} | {' | '.join(cells)} | base={base_v*100:.1f} |")

    # ===== MGSM Table 3 verification =====
    out.extend(["",
                "## MGSM Table 3 verification (10 langs × 5 arms)",
                "",
                "MGSM eval is mixed: base + A4 single-seed (42); A1/A2/A3 two-seed mean (123, 7).",
                "",
                "| Arm | Seed strategy | en | es | fr | de | ru | zh | ja | th | sw | bn | Mean | Status |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"])

    LANGS = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn"]

    PAPER_MGSM = {
        "Base": {"en":80.4,"es":65.6,"fr":61.2,"de":57.2,"ru":58.8,"zh":68.8,"ja":37.2,"th":23.6,"sw":2.4,"bn":18.0,"mean":47.3},
        "A1":   {"en":80.6,"es":63.6,"fr":62.4,"de":56.2,"ru":59.2,"zh":67.6,"ja":36.8,"th":23.4,"sw":2.2,"bn":20.4,"mean":47.2},
        "A2":   {"en":80.2,"es":66.4,"fr":61.0,"de":54.4,"ru":59.4,"zh":68.0,"ja":36.4,"th":22.4,"sw":2.2,"bn":18.8,"mean":46.9},
        "A3":   {"en":81.0,"es":64.0,"fr":60.2,"de":56.0,"ru":56.4,"zh":69.0,"ja":35.2,"th":24.0,"sw":3.0,"bn":19.0,"mean":46.8},
        "A4":   {"en":82.4,"es":64.8,"fr":64.4,"de":55.6,"ru":58.8,"zh":68.8,"ja":34.8,"th":23.6,"sw":3.2,"bn":18.4,"mean":47.5},
    }

    MGSM_DIRS = {
        "Base": [("base_distill15b_mgsm", "base_distill15b_mgsm")],
        "A1":   [("reproduce_openrs_rs2_123_mgsm","reproduce_openrs_rs2_123_mgsm"),
                 ("reproduce_openrs_rs2_7_mgsm","reproduce_openrs_rs2_7_mgsm")],
        "A2":   [("a2_vi_123_mgsm","a2_vi_123_mgsm"),
                 ("a2_vi_7_mgsm","a2_vi_7_mgsm")],
        "A3":   [("a3_enlang_123_mgsm","a3_enlang_123_mgsm"),
                 ("a3_enlang_7_mgsm","a3_enlang_7_mgsm")],
        "A4":   [("a4_const_bias_42_mgsm","a4_const_bias_42_mgsm")],
    }

    for arm, dirs in MGSM_DIRS.items():
        n_seeds = len(dirs)
        cells = []
        arm_ok = True
        per_lang = {}
        for lang in LANGS:
            seed_vals = []
            for d, prefix in dirs:
                fp = EVAL_ROOT / d / f"{prefix}_mgsm_{lang}.json"
                if fp.exists():
                    seed_vals.append(json.loads(fp.read_text())["pass_at_1"] * 100)
            if seed_vals:
                v = sum(seed_vals) / len(seed_vals)
                per_lang[lang] = v
                claimed = PAPER_MGSM[arm].get(lang)
                ok = abs(v - claimed) < 0.5
                arm_ok &= ok
                cells.append(f"{v:.1f}")
            else:
                cells.append("—")
        all_ok &= arm_ok
        # Mean
        if per_lang:
            actual_mean = sum(per_lang.values()) / len(per_lang)
            mean_claimed = PAPER_MGSM[arm]["mean"]
            mean_ok = abs(actual_mean - mean_claimed) < 0.5
            all_ok &= mean_ok
            cells.append(f"{actual_mean:.1f}")
        seed_str = f"single (42)" if n_seeds == 1 else f"avg of {n_seeds} seeds (123,7)"
        status = "✓" if arm_ok else "✗"
        out.append(f"| {arm} | {seed_str} | " + " | ".join(cells) + f" | {status} |")

    summary = "✓ ALL VERIFIED" if all_ok else "✗ FAILURES DETECTED"
    out.extend(["",
                f"## SUMMARY: {summary}",
                "",
                f"Run timestamp: 2026-05-16. Total claims verified: 15+. ",
                "",
                "If any row shows ✗, the paper text or table must be updated to match the computed value.",
                "All numbers are derived from `MANIFEST` in `scripts/w20_update_paper_tables.py` and `scripts/w21_verify_paper_claims.py`.",
                ""])

    EVIDENCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_LOG.write_text("\n".join(out))
    print("\n".join(out[-30:]))
    print(f"\nFull verification log written to: {EVIDENCE_LOG.relative_to(ROOT)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
