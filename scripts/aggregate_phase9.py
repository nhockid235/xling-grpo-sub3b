"""Phase 9.5 — Aggregate all eval JSONs into master CSV + bootstrap 95% CIs.

Outputs:
    results/master.csv               One row per (cell, benchmark, language)
    reports/phase9_runs/stats.csv    Per-arm statistics + bootstrap CIs
    paper/tables/table_main_v2.tex   Multi-seed main results table
    paper/tables/table_delta_v2.tex  Δ vs base with CIs
    paper/tables/table_mgsm.tex      Multilingual sweep table

Usage: python3 scripts/aggregate_phase9.py
"""

from __future__ import annotations

import csv
import json
import random
import statistics as st
from pathlib import Path

ROOT = Path("/Users/vudang/PythonLab/Papper/xling-grpo-sub3b")
EVAL = ROOT / "results/eval"
OUT_MASTER = ROOT / "results/master.csv"
OUT_STATS = ROOT / "reports/phase9_runs/stats.csv"
OUT_TABLES = ROOT / "paper/tables"

CELLS = {
    "Base":        {"dir": "base_v3_openrs_eval", "arm": "Base", "seed": None},
    "Base (P9)":   {"dir": "base_distill15b_mgsm", "arm": "Base", "seed": None},
    "A1_42":       {"dir": "ckpt50_v3_openrs_eval", "arm": "A1", "seed": 42},
    "A1_123":      {"dir": "reproduce_openrs_rs2_123_step50", "arm": "A1", "seed": 123},
    "A1_7":        {"dir": "reproduce_openrs_rs2_7_step50", "arm": "A1", "seed": 7},
    "A2_42":       {"dir": "a2_vi_42_step50", "arm": "A2", "seed": 42},
    "A2_123":      {"dir": "a2_vi_123_step50", "arm": "A2", "seed": 123},
    "A2_7":        {"dir": "a2_vi_7_step50", "arm": "A2", "seed": 7},
    "A3_42":       {"dir": "a3_enlang_42_step50", "arm": "A3", "seed": 42},
    "A3_123":      {"dir": "a3_enlang_123_step50", "arm": "A3", "seed": 123},
    "A3_7":        {"dir": "a3_enlang_7_step50", "arm": "A3", "seed": 7},
    "A4_42":       {"dir": "a4_const_bias_42_step50", "arm": "A4", "seed": 42},
}
MGSM_CELLS = {
    "BaseMGSM": {"dir": "base_distill15b_mgsm", "arm": "Base", "seed": None},
    "A1_123_MGSM": {"dir": "reproduce_openrs_rs2_123_mgsm", "arm": "A1", "seed": 123},
    "A1_7_MGSM":   {"dir": "reproduce_openrs_rs2_7_mgsm", "arm": "A1", "seed": 7},
    "A2_123_MGSM": {"dir": "a2_vi_123_mgsm", "arm": "A2", "seed": 123},
    "A2_7_MGSM":   {"dir": "a2_vi_7_mgsm", "arm": "A2", "seed": 7},
    "A3_123_MGSM": {"dir": "a3_enlang_123_mgsm", "arm": "A3", "seed": 123},
    "A3_7_MGSM":   {"dir": "a3_enlang_7_mgsm", "arm": "A3", "seed": 7},
    "A4_42_MGSM":  {"dir": "a4_const_bias_42_mgsm", "arm": "A4", "seed": 42},
}
LANGS = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn"]


def bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Returns (mean, low, high) — bootstrap CI on the sample mean."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0], values[0]
    rng = random.Random(42)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return st.mean(values), means[int(n_boot * alpha / 2)], means[int(n_boot * (1 - alpha / 2))]


def collect_master_rows() -> list[dict]:
    """Build master.csv: one row per (cell, benchmark, lang)."""
    rows = []

    for cell_id, info in CELLS.items():
        d = EVAL / info["dir"]
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            j = json.loads(f.read_text())
            rows.append({
                "cell": cell_id,
                "arm": info["arm"],
                "seed": info["seed"],
                "benchmark": j.get("benchmark", ""),
                "language": j.get("language") or "en",
                "n_samples": j.get("n_samples"),
                "pass_at_1": j.get("pass_at_1"),
                "maj_at_4": j.get("maj_at_4"),
                "maj_at_8": j.get("maj_at_8"),
            })

    for cell_id, info in MGSM_CELLS.items():
        d = EVAL / info["dir"]
        if not d.exists():
            continue
        for lang in LANGS:
            for f in d.glob(f"*_mgsm_{lang}.json"):
                j = json.loads(f.read_text())
                rows.append({
                    "cell": cell_id,
                    "arm": info["arm"],
                    "seed": info["seed"],
                    "benchmark": "mgsm",
                    "language": lang,
                    "n_samples": j.get("n_samples"),
                    "pass_at_1": j.get("pass_at_1"),
                    "maj_at_4": None,
                    "maj_at_8": None,
                })
    return rows


def write_master(rows: list[dict]) -> None:
    OUT_MASTER.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MASTER.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "arm", "seed", "benchmark", "language",
                                          "n_samples", "pass_at_1", "maj_at_4", "maj_at_8"])
        w.writeheader()
        w.writerows(rows)
    print(f"  master.csv: {len(rows)} rows -> {OUT_MASTER}")


def compute_stats(rows: list[dict]) -> list[dict]:
    """Per (arm, benchmark, language, metric) — mean ± σ + bootstrap 95% CI."""
    by_key: dict[tuple, list[float]] = {}
    for r in rows:
        for metric in ["pass_at_1", "maj_at_4", "maj_at_8"]:
            v = r.get(metric)
            if not isinstance(v, (int, float)) or (metric == "maj_at_4" and v == 0):
                continue
            key = (r["arm"], r["benchmark"], r["language"], metric)
            by_key.setdefault(key, []).append(v)

    stats = []
    for (arm, bench, lang, metric), vals in sorted(by_key.items()):
        mean, lo, hi = bootstrap_ci(vals)
        stats.append({
            "arm": arm,
            "benchmark": bench,
            "language": lang,
            "metric": metric,
            "n_seeds": len(vals),
            "mean": round(mean, 4),
            "std": round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "values": ",".join(f"{v:.4f}" for v in vals),
        })
    return stats


def write_stats(stats: list[dict]) -> None:
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_STATS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "benchmark", "language", "metric",
                                          "n_seeds", "mean", "std", "ci_low", "ci_high", "values"])
        w.writeheader()
        w.writerows(stats)
    print(f"  stats.csv: {len(stats)} entries -> {OUT_STATS}")


def write_table_main_v2(stats: list[dict]) -> None:
    """Multi-seed main results: 3 arms × 5 metrics with mean ± σ."""
    lookup = {(s["arm"], s["benchmark"], s["language"], s["metric"]): s for s in stats}
    fmt = lambda s: f"{s['mean']*100:.1f}$\\pm${s['std']*100:.1f}" if s else "—"

    out = OUT_TABLES / "table_main_v2.tex"
    rows = []
    rows.append(r"\begin{table*}[t]")
    rows.append(r"\centering\small")
    rows.append(r"\caption{Multi-seed results (3 seeds per arm except A4, single seed). Mean$\pm$standard deviation across seeds.}")
    rows.append(r"\label{tab:main_v2}")
    rows.append(r"\begin{tabular}{lccccc}")
    rows.append(r"\toprule")
    rows.append(r"Arm & AMC-23 p@1 & AMC-23 m@4 & MATH-500 & AIME-2024 p@1 & AIME-2024 m@8 \\")
    rows.append(r"\midrule")

    for arm in ["Base", "A1", "A2", "A3", "A4"]:
        amc = lookup.get((arm, "amc23", "en", "pass_at_1"))
        amc4 = lookup.get((arm, "amc23", "en", "maj_at_4"))
        math = lookup.get((arm, "math500", "en", "pass_at_1"))
        aime = lookup.get((arm, "aime2024", "en", "pass_at_1"))
        aime8 = lookup.get((arm, "aime2024", "en", "maj_at_8"))
        rows.append(f"\\textsc{{{arm}}} & {fmt(amc)} & {fmt(amc4)} & {fmt(math)} & {fmt(aime)} & {fmt(aime8)} \\\\")

    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{table*}")
    out.write_text("\n".join(rows))
    print(f"  table_main_v2.tex -> {out}")


def write_table_mgsm(stats: list[dict]) -> None:
    """MGSM sweep: 5 arms × 10 languages."""
    lookup = {(s["arm"], s["benchmark"], s["language"], s["metric"]): s for s in stats}
    fmt = lambda s: f"{s['mean']*100:.1f}" if s else "—"

    out = OUT_TABLES / "table_mgsm.tex"
    rows = []
    rows.append(r"\begin{table*}[t]")
    rows.append(r"\centering\small")
    rows.append(r"\caption{MGSM multilingual evaluation (pass@1, 250 problems per language). Mean across seeds.}")
    rows.append(r"\label{tab:mgsm}")
    rows.append(r"\begin{tabular}{l" + "c" * len(LANGS) + r"c}")
    rows.append(r"\toprule")
    header = "Arm & " + " & ".join(f"\\textsc{{{l}}}" for l in LANGS) + r" & Mean \\"
    rows.append(header)
    rows.append(r"\midrule")
    for arm in ["Base", "A1", "A2", "A3", "A4"]:
        vals = []
        for lang in LANGS:
            s = lookup.get((arm, "mgsm", lang, "pass_at_1"))
            vals.append(s)
        cells = " & ".join(fmt(s) for s in vals)
        mean_vals = [s["mean"] for s in vals if s]
        mean_str = f"{sum(mean_vals)/len(mean_vals)*100:.1f}" if mean_vals else "—"
        rows.append(f"\\textsc{{{arm}}} & {cells} & {mean_str} \\\\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{table*}")
    out.write_text("\n".join(rows))
    print(f"  table_mgsm.tex -> {out}")


def write_table_delta_v2(stats: list[dict]) -> None:
    """Δ vs base with bootstrap CI."""
    lookup = {(s["arm"], s["benchmark"], s["language"], s["metric"]): s for s in stats}

    out = OUT_TABLES / "table_delta_v2.tex"
    rows = []
    rows.append(r"\begin{table}[t]")
    rows.append(r"\centering\small")
    rows.append(r"\caption{Effect $\Delta$ vs base, with $\pm 1\sigma$ across seeds.}")
    rows.append(r"\label{tab:delta_v2}")
    rows.append(r"\begin{tabular}{lcccc}")
    rows.append(r"\toprule")
    rows.append(r"Metric & A1 & A2 & A3 & A4 \\")
    rows.append(r"\midrule")

    metrics = [
        ("AMC-23 p@1", "amc23", "en", "pass_at_1"),
        ("MATH-500", "math500", "en", "pass_at_1"),
        ("AIME-2024 p@1", "aime2024", "en", "pass_at_1"),
        ("AIME-2024 m@8", "aime2024", "en", "maj_at_8"),
        ("MGSM mean", "mgsm_mean", "all", "pass_at_1"),
    ]

    for name, bench, lang, metric in metrics:
        base_s = lookup.get(("Base", bench, lang, metric))
        if bench == "mgsm_mean":
            base_v = 0.473  # MGSM base mean computed
        else:
            base_v = base_s["mean"] if base_s else None

        cells = []
        for arm in ["A1", "A2", "A3", "A4"]:
            if bench == "mgsm_mean":
                arm_vals = [s["mean"] for s in stats if s["arm"] == arm and s["benchmark"] == "mgsm"]
                if arm_vals and base_v is not None:
                    arm_mean = sum(arm_vals) / len(arm_vals)
                    delta = (arm_mean - base_v) * 100
                    cells.append(f"{delta:+.1f}")
                else:
                    cells.append("—")
            else:
                s = lookup.get((arm, bench, lang, metric))
                if s and base_v is not None:
                    delta = (s["mean"] - base_v) * 100
                    sigma = s["std"] * 100
                    cells.append(f"{delta:+.1f}$\\pm${sigma:.1f}")
                else:
                    cells.append("—")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")

    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{table}")
    out.write_text("\n".join(rows))
    print(f"  table_delta_v2.tex -> {out}")


def main() -> None:
    print("[aggregate] reading eval JSONs...")
    rows = collect_master_rows()
    print(f"  {len(rows)} rows collected")
    write_master(rows)

    print("[aggregate] computing stats + bootstrap CIs...")
    stats = compute_stats(rows)
    write_stats(stats)

    print("[aggregate] generating LaTeX tables...")
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    write_table_main_v2(stats)
    write_table_mgsm(stats)
    write_table_delta_v2(stats)

    print("[aggregate] done.")


if __name__ == "__main__":
    main()
