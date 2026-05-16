"""W20 — Regenerate paper tables (table_main_v2.tex, table_delta_v2.tex)
from W1.7 (post-fix AMC23) + existing MATH-500/AIME data + W1.9 (A4 multi-seed).

Uses an EXPLICIT manifest mapping (arm, seed, benchmark) → JSON path to avoid
ambiguity when multiple eval-file versions exist for the same checkpoint.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Optional


# Explicit manifest. None means "not yet produced; check after W1.7/W1.9 completes."
# Path is relative to --eval-root (default: results/eval/).
MANIFEST: dict[tuple[str, int, str], Optional[str]] = {
    # ---------- BASE ----------
    ("base", 42, "amc23"): "w17_base_distill_v2/base_distill_v2_amc23.json",
    ("base", 42, "math500"): "base_v3_openrs_eval/base_v3_math500.json",
    ("base", 42, "aime2024"): "base_v3_openrs_eval/base_v3_aime2024.json",
    # ---------- A1 (reproduce_openrs_rs2) ----------
    ("A1", 42, "amc23"): "w17_reproduce_openrs_rs2_42_v2/reproduce_openrs_rs2_42_v2_amc23.json",
    ("A1", 42, "math500"): "ckpt50_v3_openrs_eval/ckpt50_v3_math500.json",
    ("A1", 42, "aime2024"): "ckpt50_v3_openrs_eval/ckpt50_v3_aime2024.json",
    ("A1", 123, "amc23"): "w17_reproduce_openrs_rs2_123_v2/reproduce_openrs_rs2_123_v2_amc23.json",
    ("A1", 123, "math500"): "reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_math500.json",
    ("A1", 123, "aime2024"): "reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_aime2024.json",
    ("A1", 7, "amc23"): "w17_reproduce_openrs_rs2_7_v2/reproduce_openrs_rs2_7_v2_amc23.json",
    ("A1", 7, "math500"): "reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_math500.json",
    ("A1", 7, "aime2024"): "reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_aime2024.json",
    # ---------- A2 (a2_vi) ----------
    ("A2", 42, "amc23"): "w17_a2_vi_42_v2/a2_vi_42_v2_amc23.json",
    ("A2", 42, "math500"): "a2_vi_42_step50/a2_vi_42_step50_math500.json",
    ("A2", 42, "aime2024"): "a2_vi_42_step50/a2_vi_42_step50_aime2024.json",
    ("A2", 123, "amc23"): "w17_a2_vi_123_v2/a2_vi_123_v2_amc23.json",
    ("A2", 123, "math500"): "a2_vi_123_step50/a2_vi_123_step50_math500.json",
    ("A2", 123, "aime2024"): "a2_vi_123_step50/a2_vi_123_step50_aime2024.json",
    ("A2", 7, "amc23"): "w17_a2_vi_7_v2/a2_vi_7_v2_amc23.json",
    ("A2", 7, "math500"): "a2_vi_7_step50/a2_vi_7_step50_math500.json",
    ("A2", 7, "aime2024"): "a2_vi_7_step50/a2_vi_7_step50_aime2024.json",
    # ---------- A3 (a3_enlang) ----------
    ("A3", 42, "amc23"): "w17_a3_enlang_42_v2/a3_enlang_42_v2_amc23.json",
    ("A3", 42, "math500"): "a3_enlang_42_step50/a3_enlang_42_step50_math500.json",
    ("A3", 42, "aime2024"): "a3_enlang_42_step50/a3_enlang_42_step50_aime2024.json",
    ("A3", 123, "amc23"): "w17_a3_enlang_123_v2/a3_enlang_123_v2_amc23.json",
    ("A3", 123, "math500"): "a3_enlang_123_step50/a3_enlang_123_step50_math500.json",
    ("A3", 123, "aime2024"): "a3_enlang_123_step50/a3_enlang_123_step50_aime2024.json",
    ("A3", 7, "amc23"): "w17_a3_enlang_7_v2/a3_enlang_7_v2_amc23.json",
    ("A3", 7, "math500"): "a3_enlang_7_step50/a3_enlang_7_step50_math500.json",
    ("A3", 7, "aime2024"): "a3_enlang_7_step50/a3_enlang_7_step50_aime2024.json",
    # ---------- A4 (a4_const_bias) ----------
    # Note: a4 seed 42 LoRA missing on disk → using existing pre-fix-but-post-bugfix eval JSON (m@4=0.675 already valid).
    ("A4", 42, "amc23"): "a4_const_bias_42_step50/a4_const_bias_42_step50_amc23.json",
    ("A4", 42, "math500"): "a4_const_bias_42_step50/a4_const_bias_42_step50_math500.json",
    ("A4", 42, "aime2024"): "a4_const_bias_42_step50/a4_const_bias_42_step50_aime2024.json",
    # W1.9 will produce A4 seeds 123 + 7:
    ("A4", 123, "amc23"): "a4_const_bias_123_step50/a4_const_bias_123_step50_amc23.json",
    ("A4", 123, "math500"): "a4_const_bias_123_step50/a4_const_bias_123_step50_math500.json",
    ("A4", 123, "aime2024"): "a4_const_bias_123_step50/a4_const_bias_123_step50_aime2024.json",
    ("A4", 7, "amc23"): "a4_const_bias_7_step50/a4_const_bias_7_step50_amc23.json",
    ("A4", 7, "math500"): "a4_const_bias_7_step50/a4_const_bias_7_step50_math500.json",
    ("A4", 7, "aime2024"): "a4_const_bias_7_step50/a4_const_bias_7_step50_aime2024.json",
}


def load_metric(eval_root: Path, arm: str, seed: int, bench: str, metric: str) -> Optional[float]:
    rel = MANIFEST.get((arm, seed, bench))
    if rel is None:
        return None
    fp = eval_root / rel
    if not fp.exists():
        return None
    d = json.loads(fp.read_text())
    return d.get(metric)


def mean_std(values: list[Optional[float]]) -> tuple[Optional[float], Optional[float]]:
    vs = [v for v in values if v is not None]
    if not vs:
        return None, None
    if len(vs) == 1:
        return vs[0], 0.0
    return statistics.mean(vs), statistics.stdev(vs)


def fmt_pct(m: Optional[float], s: Optional[float]) -> str:
    if m is None:
        return "—"
    pct = m * 100
    if s is None or s == 0:
        return f"{pct:.1f}"
    return f"{pct:.1f}$\\pm${s * 100:.1f}"


def fmt_delta(m: Optional[float], s: Optional[float], base: Optional[float]) -> str:
    if m is None or base is None:
        return "—"
    d = (m - base) * 100
    sign = "+" if d >= 0 else ""
    if s is None or s == 0:
        return f"{sign}{d:.1f}"
    return f"{sign}{d:.1f}$\\pm${s * 100:.1f}"


SEEDS = [42, 123, 7]


def gather(eval_root: Path) -> dict:
    """Return {arm: {bench: {metric: (mean, std)}}}."""
    arms = ["base", "A1", "A2", "A3", "A4"]
    benches = {
        ("amc23", "pass_at_1"),
        ("amc23", "maj_at_4"),
        ("math500", "pass_at_1"),
        ("aime2024", "pass_at_1"),
        ("aime2024", "maj_at_8"),
    }
    out: dict = {}
    for arm in arms:
        out[arm] = {}
        for bench, metric in benches:
            vals = [load_metric(eval_root, arm, s, bench, metric) for s in SEEDS]
            n_found = sum(1 for v in vals if v is not None)
            m, s = mean_std(vals)
            out[arm][(bench, metric)] = (m, s, n_found)
    return out


def gen_table_main(data: dict) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{Multi-seed results (3 seeds per arm; A4 seeds 123 and 7 from W1.9 retrain). "
        r"Mean$\pm$std across seeds. AMC-23 \texttt{maj@4} numbers are post-bugfix (W1.7).}",
        r"\label{tab:main_v2}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Arm & AMC-23 p@1 & AMC-23 m@4 & MATH-500 & AIME-2024 p@1 & AIME-2024 m@8 \\",
        r"\midrule",
    ]
    label_map = {"base": r"\textsc{Base}", "A1": r"\textsc{A1}",
                 "A2": r"\textsc{A2}", "A3": r"\textsc{A3}", "A4": r"\textsc{A4}"}
    for arm in ["base", "A1", "A2", "A3", "A4"]:
        d = data[arm]
        cells = [
            label_map[arm],
            fmt_pct(*d[("amc23", "pass_at_1")][:2]),
            fmt_pct(*d[("amc23", "maj_at_4")][:2]),
            fmt_pct(*d[("math500", "pass_at_1")][:2]),
            fmt_pct(*d[("aime2024", "pass_at_1")][:2]),
            fmt_pct(*d[("aime2024", "maj_at_8")][:2]),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def gen_table_delta(data: dict) -> str:
    base_p1_amc = data["base"][("amc23", "pass_at_1")][0]
    base_math = data["base"][("math500", "pass_at_1")][0]
    base_aime_p1 = data["base"][("aime2024", "pass_at_1")][0]
    base_aime_m8 = data["base"][("aime2024", "maj_at_8")][0]

    rows = [
        ("AMC-23 p@1", "amc23", "pass_at_1", base_p1_amc),
        ("MATH-500", "math500", "pass_at_1", base_math),
        ("AIME-2024 p@1", "aime2024", "pass_at_1", base_aime_p1),
        ("AIME-2024 m@8", "aime2024", "maj_at_8", base_aime_m8),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{Effect $\Delta$ vs base, with $\pm 1\sigma$ across seeds. Post-fix maj@4 numbers from W1.7; A4 from W1.9 multi-seed.}",
        r"\label{tab:delta_v2}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Metric & A1 & A2 & A3 & A4 \\",
        r"\midrule",
    ]
    for label, bench, metric, base_v in rows:
        cells = [label]
        for arm in ["A1", "A2", "A3", "A4"]:
            m, s, _ = data[arm][(bench, metric)]
            cells.append(fmt_delta(m, s, base_v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, default=Path("results/eval"))
    ap.add_argument("--out-tables", type=Path, default=Path("paper/tables"))
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    data = gather(args.eval_root)

    print("\n========== Aggregated data (mean, std, n_found) ==========")
    for arm in ["base", "A1", "A2", "A3", "A4"]:
        print(f"\n[{arm}]")
        for (bench, metric), (m, s, n) in data[arm].items():
            ms = f"{m*100:.2f}" if m is not None else "—"
            ss = f"{s*100:.2f}" if s is not None else "—"
            print(f"  {bench:<10} {metric:<12} mean={ms:<8} std={ss:<8} n_seeds_found={n}")

    if args.print_only:
        return

    args.out_tables.mkdir(parents=True, exist_ok=True)
    (args.out_tables / "table_main_v2.tex").write_text(gen_table_main(data))
    (args.out_tables / "table_delta_v2.tex").write_text(gen_table_delta(data))
    print(f"\nWrote {args.out_tables / 'table_main_v2.tex'}")
    print(f"Wrote {args.out_tables / 'table_delta_v2.tex'}")


if __name__ == "__main__":
    main()
