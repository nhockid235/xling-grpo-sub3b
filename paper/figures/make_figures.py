"""Generate paper figures từ master.csv + training logs.

Outputs (300 DPI, embed fonts cho arXiv):
    fig1_arm_comparison.pdf — bar chart 4 cells × 4 metrics
    fig2_training_curves.pdf — reward signal R1/R2/R5 + total per arm over 50 steps
    fig3_effect_sizes.pdf — Δ vs base summary

Usage:
    cd xling-grpo-sub3b
    python paper/figures/make_figures.py \\
        --master /path/to/master.csv \\
        --training_logs /path/to/training/ \\
        --output paper/figures/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ACL-style typography
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ARMS = ["base", "en", "vi", "enlang"]
ARM_LABELS = {"base": "Base", "en": "A1 (EN)", "vi": "A2 (VI)", "enlang": "A3 (EN+R5)"}
ARM_COLORS = {"base": "#888888", "en": "#1f77b4", "vi": "#ff7f0e", "enlang": "#2ca02c"}
BENCHMARKS = ["amc23", "math500", "aime2024"]
BENCH_LABELS = {"amc23": "AMC23", "math500": "MATH-500", "aime2024": "AIME-2024 (p@1)"}
OPENRS_REPORTED = {"amc23": 80.0, "math500": 85.4, "aime2024": 30.0}


def fig1_arm_comparison(df: pd.DataFrame, output: Path) -> Path:
    """Bar chart: 4 arms × 3 benchmarks (pass@1) + 4 arms × AIME-2024 maj@8."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6), sharey=False)

    # First 3: pass@1 on AMC23, MATH-500, AIME
    for ax, bench in zip(axes[:3], BENCHMARKS):
        sub = df[(df["benchmark"] == bench) & df["condition"].isin(ARMS)]
        if sub.empty:
            ax.text(0.5, 0.5, f"no {bench} data", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        # Pivot: arm → pass@1
        vals = []
        labels = []
        colors = []
        for arm in ARMS:
            row = sub[sub["condition"] == arm]
            if not row.empty:
                vals.append(float(row["pass_at_1"].iloc[0]) * 100.0)
                labels.append(ARM_LABELS[arm])
                colors.append(ARM_COLORS[arm])
        bars = ax.bar(range(len(vals)), vals, color=colors, width=0.65)
        # Reference line at Open-RS reported
        if bench in OPENRS_REPORTED:
            ax.axhline(OPENRS_REPORTED[bench], color="red", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.text(len(vals) - 0.3, OPENRS_REPORTED[bench] + 1.5, "Open-RS RS2",
                    color="red", fontsize=7, ha="right")
        # Annotate bar values
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.5)
        ax.set_title(BENCH_LABELS[bench])
        ax.set_ylim(0, max(vals + [OPENRS_REPORTED.get(bench, 0)]) * 1.18)
        if bench == "amc23":
            ax.set_ylabel("pass@1 (%)")

    # 4th: AIME maj@8
    ax = axes[3]
    sub = df[(df["benchmark"] == "aime2024") & df["condition"].isin(ARMS)]
    sub = sub.dropna(subset=["maj_at_8"])
    if not sub.empty:
        vals, labels, colors = [], [], []
        for arm in ARMS:
            row = sub[sub["condition"] == arm]
            if not row.empty:
                vals.append(float(row["maj_at_8"].iloc[0]) * 100.0)
                labels.append(ARM_LABELS[arm])
                colors.append(ARM_COLORS[arm])
        bars = ax.bar(range(len(vals)), vals, color=colors, width=0.65)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.5)
        ax.set_title("AIME-2024 (maj@8)")
        ax.set_ylim(0, max(vals) * 1.18)

    fig.suptitle("Three-arm GRPO comparison on three math benchmarks (single seed)", y=1.02, fontsize=10)
    out = output / "fig1_arm_comparison.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _parse_metrics_from_csv(metrics_csv: Path) -> pd.DataFrame:
    """Read training metrics CSV (extract_metrics.py output)."""
    if not metrics_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(metrics_csv)
    return df


import ast
import json

DICT_RE = re.compile(r"(\{[^{}]*['\"]loss['\"][^{}]*\})")


def _parse_metrics_from_log(log_path: Path) -> pd.DataFrame:
    """Extract per-step metric dicts từ training.log (matches extract_metrics.py)."""
    if not log_path.exists():
        return pd.DataFrame()
    text = log_path.read_text(errors="ignore")
    # Split on \r and \n to expand progress-bar overwrites
    fragments = re.split(r"[\r\n]+", text)
    rows = []
    for frag in fragments:
        for m in DICT_RE.finditer(frag):
            s = m.group(1)
            d = None
            try:
                d = json.loads(s)
            except Exception:
                try:
                    d = ast.literal_eval(s)
                except Exception:
                    continue
            if d and "loss" in d:
                rows.append(d)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fig2_training_curves(training_dir: Path, output: Path) -> Path:
    """3-panel: R1/R2 + R5 (where present) + total reward over 50 steps for each arm."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharex=True)

    # Try CSVs first, fallback to parsing training logs.
    arms_dirs = {
        "en": training_dir / "metrics_a1.csv",
        "vi": training_dir / "metrics_a2.csv",
        "enlang": training_dir / "metrics_a3.csv",
    }
    arm_dfs = {}
    for arm, csv_path in arms_dirs.items():
        df = _parse_metrics_from_csv(csv_path)
        if df.empty:
            # Fallback: parse from training_a*.log
            fallback_log = training_dir / f"training_{arm if arm == 'enlang' else arm}.log"
            log_map = {"en": "training.log", "vi": "training_a2.log", "enlang": "training_a3.log"}
            log_path = training_dir / log_map[arm]
            df = _parse_metrics_from_log(log_path)
        arm_dfs[arm] = df

    # Override arms_dirs with parsed dataframes via closure trick
    arms_dirs = {arm: None for arm in arm_dfs}

    # Panel 1: R1 over steps (all 3 arms)
    ax = axes[0]
    for arm in ["en", "vi", "enlang"]:
        df = arm_dfs.get(arm, pd.DataFrame())
        if df.empty or "rewards/r1_correctness" not in df.columns:
            continue
        # Step number = (index + 1) * logging_steps (5)
        steps = np.arange(1, len(df) + 1) * 5
        ax.plot(steps, df["rewards/r1_correctness"], marker="o", markersize=3,
                label=ARM_LABELS[arm], color=ARM_COLORS[arm])
    ax.set_xlabel("training step")
    ax.set_ylabel("R1 (correctness)")
    ax.set_title("R1: Correctness reward")
    ax.legend(frameon=False)
    ax.set_ylim(0, 0.4)

    # Panel 2: R2 over steps
    ax = axes[1]
    for arm in ["en", "vi", "enlang"]:
        df = arm_dfs.get(arm, pd.DataFrame())
        if df.empty or "rewards/r2_format" not in df.columns:
            continue
        steps = np.arange(1, len(df) + 1) * 5
        ax.plot(steps, df["rewards/r2_format"], marker="s", markersize=3,
                label=ARM_LABELS[arm], color=ARM_COLORS[arm])
    ax.set_xlabel("training step")
    ax.set_ylabel("R2 (format)")
    ax.set_title("R2: Format reward")
    ax.legend(frameon=False)
    ax.set_ylim(-0.05, 1.05)

    # Panel 3: R5 (only A3 has it)
    ax = axes[2]
    df = arm_dfs.get("enlang", pd.DataFrame())
    if not df.empty and "rewards/r5_lang_consistency" in df.columns:
        steps = np.arange(1, len(df) + 1) * 5
        ax.plot(steps, df["rewards/r5_lang_consistency"], marker="^", markersize=3,
                label=ARM_LABELS["enlang"], color=ARM_COLORS["enlang"])
    ax.set_xlabel("training step")
    ax.set_ylabel("R5 (lang-consistency)")
    ax.set_title("R5: Language consistency (A3 only)")
    ax.legend(frameon=False)
    ax.set_ylim(0.0, 1.05)

    out = output / "fig2_training_curves.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig3_effect_sizes(df: pd.DataFrame, output: Path) -> Path:
    """Diverging bar chart: Δ vs base for each (arm, benchmark)."""
    fig, ax = plt.subplots(figsize=(8, 3))

    # Get base values per benchmark
    base = df[df["condition"] == "base"].set_index("benchmark")
    arms_to_plot = ["en", "vi", "enlang"]
    metrics = [("amc23", "pass_at_1", "AMC23"),
               ("math500", "pass_at_1", "MATH-500"),
               ("aime2024", "pass_at_1", "AIME-2024 p@1"),
               ("aime2024", "maj_at_8", "AIME-2024 m@8")]

    n_metrics = len(metrics)
    width = 0.24
    x = np.arange(n_metrics)

    for i, arm in enumerate(arms_to_plot):
        deltas = []
        for bench, metric, _ in metrics:
            base_val = base.loc[bench, metric] if bench in base.index else None
            arm_row = df[(df["condition"] == arm) & (df["benchmark"] == bench)]
            if arm_row.empty or base_val is None or pd.isna(arm_row[metric].iloc[0]):
                deltas.append(0.0)
                continue
            delta_pp = (float(arm_row[metric].iloc[0]) - float(base_val)) * 100.0
            deltas.append(delta_pp)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, deltas, width=width, label=ARM_LABELS[arm], color=ARM_COLORS[arm])
        for b, v in zip(bars, deltas):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + (0.4 if v >= 0 else -1.2),
                    f"{v:+.1f}", ha="center", fontsize=7,
                    color="black" if v >= 0 else "darkred")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m[2] for m in metrics])
    ax.set_ylabel("Δ vs base (pp)")
    ax.set_title("Effect of each post-training arm relative to untrained base")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.set_ylim(min(-20, ax.get_ylim()[0]), max(15, ax.get_ylim()[1]))

    out = output / "fig3_effect_sizes.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path,
                        default=Path("../../results/training/master.csv"))
    parser.add_argument("--training_logs", type=Path,
                        default=Path("../../results/training/"))
    parser.add_argument("--output", type=Path,
                        default=Path("./"))
    args = parser.parse_args()

    df = pd.read_csv(args.master)
    df["pass_at_1"] = pd.to_numeric(df["pass_at_1"], errors="coerce")
    df["maj_at_8"] = pd.to_numeric(df["maj_at_8"], errors="coerce")

    # Filter to clean runs (drop old v1/v2 evals)
    keep_runs = df["run_id"].str.contains(
        "a2_vi_42_step50|a3_enlang_42_step50|ckpt50_v3|base_v3", na=False
    )
    df_clean = df[keep_runs].copy()

    args.output.mkdir(parents=True, exist_ok=True)
    p1 = fig1_arm_comparison(df_clean, args.output)
    p2 = fig2_training_curves(args.training_logs, args.output)
    p3 = fig3_effect_sizes(df_clean, args.output)
    print(f"[make_figures] saved:\n  {p1}\n  {p2}\n  {p3}")


if __name__ == "__main__":
    main()
