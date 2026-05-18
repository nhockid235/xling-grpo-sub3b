"""Multi-seed figures for the paper.

Reads all eval JSONs in results/eval/ and emits 6 publication-grade PDFs
(300 DPI, embedded fonts, ACL-friendly typography) under paper/figures/.

Outputs:
    fig1_arm_means_with_ci.pdf       Bar chart of mean ± σ across 3 seeds, 3 arms × 5 metrics
    fig2_seed_scatter.pdf            Scatter of per-seed values to show variance directly
    fig3_variance_ratios.pdf         σ ratio (A1 vs A2 vs A3) per metric — the variance story
    fig4_effect_vs_base.pdf          Δ from base (mean ± σ) per arm per metric
    fig5_training_curves_multiseed.pdf  Reward / KL / length over 100 steps, mean ± σ across 3 seeds
    fig6_aime_focus.pdf              Single high-impact panel: AIME pass@1 + maj@8 across all 9 cells

Usage:
    python paper/figures/make_figures_v2.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/vudang/PythonLab/Papper/xling-grpo-sub3b")
EVAL_DIR = ROOT / "results/eval"
GRPO_DIR = ROOT / "results/grpo"
OUT_DIR = ROOT / "paper/figures"

CELLS = {
    "Base": "base_v3_openrs_eval",
    "A1 seed=42": "ckpt50_v3_openrs_eval",
    "A1 seed=123": "reproduce_openrs_rs2_123_step50",
    "A1 seed=7": "reproduce_openrs_rs2_7_step50",
    "A2 seed=42": "a2_vi_42_step50",
    "A2 seed=123": "a2_vi_123_step50",
    "A2 seed=7": "a2_vi_7_step50",
    "A3 seed=42": "a3_enlang_42_step50",
    "A3 seed=123": "a3_enlang_123_step50",
    "A3 seed=7": "a3_enlang_7_step50",
}

CELL_DIR_MAP = {
    "A1 seed=123": "reproduce_openrs_rs2_123",
    "A1 seed=7": "reproduce_openrs_rs2_7",
    "A2 seed=123": "a2_vi_123",
    "A2 seed=7": "a2_vi_7",
    "A3 seed=123": "a3_enlang_123",
    "A3 seed=7": "a3_enlang_7",
    "A4 seed=42": "a4_const_bias_42",
}

ARMS = ["A1", "A2", "A3"]
ARM_COLORS = {"A1": "#d62728", "A2": "#2ca02c", "A3": "#1f77b4", "A4": "#9467bd", "Base": "#7f7f7f"}
METRICS = ["AMC", "AMC@4", "MATH", "AIME", "AIME@8"]
METRIC_LABELS = {
    "AMC": "AMC-23 pass@1",
    "AMC@4": "AMC-23 maj@4",
    "MATH": "MATH-500 pass@1",
    "AIME": "AIME-2024 pass@1",
    "AIME@8": "AIME-2024 maj@8",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,  # embed Type 42 (TrueType) — ACL/IEEE/arXiv friendly
})


def load_data() -> dict[str, dict[str, float | None]]:
    """Read every eval JSON and return {cell: {metric: pct (0-1) or None}}."""
    data: dict[str, dict[str, float | None]] = {}
    for label, dirn in CELLS.items():
        d = EVAL_DIR / dirn
        if not d.exists():
            continue
        row: dict[str, float | None] = {m: None for m in METRICS}
        for f in d.glob("*.json"):
            j = json.loads(f.read_text())
            bench = j.get("benchmark", "")
            p1 = j.get("pass_at_1")
            m4 = j.get("maj_at_4")
            m8 = j.get("maj_at_8")
            if "amc23" in bench:
                row["AMC"] = p1
                if isinstance(m4, (int, float)) and m4 > 0:
                    row["AMC@4"] = m4
            elif "math500" in bench:
                row["MATH"] = p1
            elif "aime" in bench:
                row["AIME"] = p1
                row["AIME@8"] = m8
        data[label] = row
    return data


def arm_stats(data: dict, arm: str, metric: str) -> tuple[list[float], float | None, float | None]:
    """Return (per-seed values, mean, std) for arm × metric in percent (0-100)."""
    vals = []
    for label, row in data.items():
        if label.startswith(arm) and isinstance(row.get(metric), (int, float)):
            vals.append(row[metric] * 100)
    if len(vals) < 2:
        return vals, (vals[0] if vals else None), None
    return vals, st.mean(vals), st.stdev(vals)


def fig1_arm_means_with_ci(data: dict) -> None:
    """Bar chart of mean ± σ across 3 seeds, 3 arms × 5 metrics."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    base_row = data.get("Base", {})
    n_arms = len(ARMS)
    n_metrics = len(METRICS)
    bar_width = 0.22
    x = np.arange(n_metrics)

    for i, arm in enumerate(ARMS):
        means = []
        errs = []
        for metric in METRICS:
            _, m, s = arm_stats(data, arm, metric)
            means.append(m if m is not None else 0)
            errs.append(s if s is not None else 0)
        offset = (i - (n_arms - 1) / 2) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=errs, capsize=3,
               color=ARM_COLORS[arm], edgecolor="black", linewidth=0.5,
               label=f"{arm} (n=3 seeds)")

    base_means = [base_row.get(m, 0) * 100 if isinstance(base_row.get(m), (int, float)) else None for m in METRICS]
    for j, bm in enumerate(base_means):
        if bm is not None:
            ax.hlines(bm, x[j] - 0.45, x[j] + 0.45, colors="black", linestyles="--", linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=15, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title("Multi-seed mean ± σ across three GRPO arms (n=3 seeds each).\n"
                 "Black dashed: untrained base.")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_arm_means_with_ci.pdf", bbox_inches="tight")
    plt.close()
    print("  fig1_arm_means_with_ci.pdf")


def fig2_seed_scatter(data: dict) -> None:
    """Scatter of per-seed values to show variance directly."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(13, 3.5), sharey=False)
    base_row = data.get("Base", {})
    for ax, metric in zip(axes, METRICS):
        for i, arm in enumerate(ARMS):
            vals, m, s = arm_stats(data, arm, metric)
            x_pos = i + 1
            ax.scatter([x_pos] * len(vals), vals, s=80, color=ARM_COLORS[arm],
                       edgecolor="black", linewidth=0.5, zorder=3)
            if m is not None:
                ax.hlines(m, x_pos - 0.25, x_pos + 0.25, colors=ARM_COLORS[arm], linewidth=2.5)
            if s is not None:
                ax.vlines(x_pos, m - s, m + s, colors=ARM_COLORS[arm], linewidth=1.5, alpha=0.6)
        bm = base_row.get(metric)
        if isinstance(bm, (int, float)):
            ax.axhline(bm * 100, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="Base")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(ARMS)
        ax.set_title(METRIC_LABELS[metric], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(0.5, 3.5)
    axes[0].set_ylabel("Score (%)")
    fig.suptitle("Per-seed values: each dot is one of three random seeds.\n"
                 "Bars: mean. Vertical line: ±1 σ. Dashed: untrained base.", fontsize=10, y=1.05)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_seed_scatter.pdf", bbox_inches="tight")
    plt.close()
    print("  fig2_seed_scatter.pdf")


def fig3_variance_ratios(data: dict) -> None:
    """σ ratio (A1 vs A2 vs A3) per metric — variance story."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sigmas = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for metric in METRICS:
            _, _, s = arm_stats(data, arm, metric)
            sigmas[arm].append(s if s is not None else 0)

    n_metrics = len(METRICS)
    bar_width = 0.27
    x = np.arange(n_metrics)
    for i, arm in enumerate(ARMS):
        offset = (i - 1) * bar_width
        ax.bar(x + offset, sigmas[arm], bar_width,
               color=ARM_COLORS[arm], edgecolor="black", linewidth=0.5,
               label=arm)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=15, ha="right")
    ax.set_ylabel("σ across 3 seeds (pp)")
    ax.set_title("Seed variance per arm: lower bars = more reproducible.\n"
                 "A1 (vanilla EN GRPO) consistently exhibits the highest variance.")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig3_variance_ratios.pdf", bbox_inches="tight")
    plt.close()
    print("  fig3_variance_ratios.pdf")


def fig4_effect_vs_base(data: dict) -> None:
    """Δ from base (mean ± σ) per arm per metric."""
    base_row = data.get("Base", {})
    fig, ax = plt.subplots(figsize=(8.5, 4))
    n_arms = len(ARMS)
    bar_width = 0.27
    x = np.arange(len(METRICS))

    for i, arm in enumerate(ARMS):
        deltas = []
        errs = []
        for metric in METRICS:
            _, m, s = arm_stats(data, arm, metric)
            base = base_row.get(metric)
            if isinstance(base, (int, float)) and m is not None:
                deltas.append(m - base * 100)
                errs.append(s if s is not None else 0)
            else:
                deltas.append(0)
                errs.append(0)
        offset = (i - 1) * bar_width
        ax.bar(x + offset, deltas, bar_width, yerr=errs, capsize=3,
               color=ARM_COLORS[arm], edgecolor="black", linewidth=0.5,
               label=arm)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=15, ha="right")
    ax.set_ylabel("Δ from base (pp)")
    ax.set_title("Effect size relative to untrained base, mean ± σ across 3 seeds.\n"
                 "Positive = improvement; negative = degradation.")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig4_effect_vs_base.pdf", bbox_inches="tight")
    plt.close()
    print("  fig4_effect_vs_base.pdf")


def load_trainer_history(cell_name: str, cell_dir: str) -> list[dict] | None:
    """Read trainer_state.json log_history for a cell."""
    p = GRPO_DIR / cell_dir / "checkpoint-100" / "trainer_state.json"
    if not p.exists():
        p = GRPO_DIR / cell_dir / "checkpoint-50" / "trainer_state.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("log_history", [])
    except Exception:
        return None


def fig5_training_curves(data: dict) -> None:
    """Reward / KL / length over 100 steps, mean ± σ across 3 seeds per arm."""
    arm_to_dirs = {
        "A1": ["reproduce_openrs_rs2_123", "reproduce_openrs_rs2_7"],
        "A2": ["a2_vi_123", "a2_vi_7"],
        "A3": ["a3_enlang_123", "a3_enlang_7"],
    }

    panels = ["reward", "kl", "completion_length"]
    panel_titles = {
        "reward": "Total reward",
        "kl": "KL divergence",
        "completion_length": "Completion length (tokens)",
    }
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.5))

    for ax, panel in zip(axes, panels):
        for arm, dirs in arm_to_dirs.items():
            histories = []
            for d in dirs:
                h = load_trainer_history(arm, d)
                if h:
                    histories.append(h)
            if not histories:
                continue
            steps = sorted(set(e["step"] for h in histories for e in h))
            ys = []
            for s in steps:
                vs = [e[panel] for h in histories for e in h if e.get("step") == s and panel in e]
                if vs:
                    ys.append((s, vs))
            if not ys:
                continue
            xs = [y[0] for y in ys]
            means = [st.mean(y[1]) for y in ys]
            stds = [st.stdev(y[1]) if len(y[1]) > 1 else 0 for y in ys]
            ax.plot(xs, means, color=ARM_COLORS[arm], label=arm, linewidth=1.7)
            ax.fill_between(xs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=ARM_COLORS[arm], alpha=0.18)
        ax.set_title(panel_titles[panel])
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Value")
    axes[0].legend(loc="best")
    fig.suptitle("Training dynamics across arms (mean ± σ across 2 seeds, 100 steps)",
                 fontsize=10, y=1.05)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig5_training_curves_multiseed.pdf", bbox_inches="tight")
    plt.close()
    print("  fig5_training_curves_multiseed.pdf")


def fig6_aime_focus(data: dict) -> None:
    """Single high-impact panel: AIME pass@1 + maj@8 across all 9 cells."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["AIME", "AIME@8"]):
        base = data.get("Base", {}).get(metric)
        if isinstance(base, (int, float)):
            ax.axhline(base * 100, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
                       label=f"Base = {base * 100:.1f}%")
        for i, arm in enumerate(ARMS):
            vals, m, s = arm_stats(data, arm, metric)
            x_pos = i + 1
            ax.scatter([x_pos] * len(vals), vals, s=110, color=ARM_COLORS[arm],
                       edgecolor="black", linewidth=0.6, zorder=3, alpha=0.85)
            if m is not None:
                ax.hlines(m, x_pos - 0.3, x_pos + 0.3, colors=ARM_COLORS[arm], linewidth=3)
                ax.text(x_pos + 0.32, m, f"{m:.1f}", color=ARM_COLORS[arm], fontsize=9,
                        va="center", fontweight="bold")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(ARMS)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylabel("Score (%)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(0.5, 3.7)
        ax.legend(loc="upper left")
    fig.suptitle("AIME-2024: hardest benchmark, where mechanism shows clearly.\n"
                 "A3 achieves highest mean and lowest variance on maj@8.",
                 fontsize=10, y=1.04)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig6_aime_focus.pdf", bbox_inches="tight")
    plt.close()
    print("  fig6_aime_focus.pdf")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[figures] reading from {EVAL_DIR}")
    data = load_data()
    print(f"[figures] loaded {len(data)} cells")
    for cell, row in data.items():
        non_null = sum(1 for v in row.values() if v is not None)
        print(f"  {cell}: {non_null}/5 metrics")

    print("[figures] generating PDFs:")
    fig1_arm_means_with_ci(data)
    fig2_seed_scatter(data)
    fig3_variance_ratios(data)
    fig4_effect_vs_base(data)
    fig5_training_curves(data)
    fig6_aime_focus(data)
    print(f"[figures] done. wrote 6 PDFs to {OUT_DIR}")


if __name__ == "__main__":
    main()
