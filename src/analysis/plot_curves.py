"""Plot 3 headline figures cho paper.

Fig 1: per-language transfer gap (pre / SFT / GRPO).
Fig 2: lang-consistency curve over training steps.
Fig 3: Pareto frontier (avg tokens vs MGSM-vi pass@1).

ACL template requirements:
    - serif font matching template
    - 300 DPI
    - embed fonts (pdf.fonttype=42)
    - color-blind friendly palette (tab10)

Usage:
    python src/analysis/plot_curves.py --master results/master.csv --output paper/figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Match ACL template typography
_RC_PARAMS = {
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
    "pdf.fonttype": 42,    # embed fonts
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Default 6 target languages cho Fig 1/2 (subset of MGSM 10)
HEADLINE_LANGS = ["en", "vi", "zh", "fr", "th", "sw"]
HEADLINE_CONDITIONS = ["en", "vi", "enlang"]


def _setup_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_RC_PARAMS)


def _load_master(master_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(master_csv)
    # Cast numeric columns; '' → NaN
    for col in ("pass_at_1", "maj_at_8", "lang_consistency", "avg_tokens", "n_samples", "step"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fig1_transfer_gap(df: pd.DataFrame, output: Path) -> Path:
    """Fig 1: per-language MGSM gap (EN minus target lang) by training stage."""
    import matplotlib.pyplot as plt

    sub = df[(df["benchmark"] == "mgsm") & (df["condition"] == "en")].copy()
    if sub.empty:
        # Stub khi chưa có data thật — vẽ placeholder rồi bỏ qua
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        ax.text(0.5, 0.5, "Fig 1 — no MGSM data yet", ha="center", va="center")
        ax.axis("off")
    else:
        # Pivot: rows=lang, cols=stage; value=mean pass@1
        pivot = sub.pivot_table(
            index="language",
            columns="stage",
            values="pass_at_1",
            aggfunc="mean",
        )
        # Compute gap: en_score - lang_score per stage
        en_score = pivot.loc["en"] if "en" in pivot.index else pivot.iloc[0]
        gap = (en_score - pivot.T).T  # broadcast subtract
        gap = gap.reindex(HEADLINE_LANGS).dropna(how="all")

        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        gap.plot(ax=ax, marker="o", colormap="tab10")
        ax.set_xlabel("Target language")
        ax.set_ylabel("EN − target pass@1 gap")
        ax.set_title("Cross-lingual transfer gap by stage")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.legend(title="Stage", loc="upper right", frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    out_path = output / "fig1_transfer_gap.pdf"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig2_lang_consistency_curve(df: pd.DataFrame, output: Path) -> Path:
    """Fig 2: lang-consistency rate vs GRPO step, per condition.

    Note: cần multiple checkpoints per run để vẽ curve. Khi chỉ có ckpt-final
    → degenerates thành scatter.
    """
    import matplotlib.pyplot as plt

    sub = df[(df["benchmark"] == "mgsm") & (df["language"] == "vi")].copy()
    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    if sub.empty or sub["lang_consistency"].isna().all():
        ax.text(0.5, 0.5, "Fig 2 — no lang_consistency data yet", ha="center", va="center")
        ax.axis("off")
    else:
        for cond in HEADLINE_CONDITIONS:
            cond_sub = sub[sub["condition"] == cond].sort_values("step")
            if cond_sub.empty:
                continue
            ax.plot(
                cond_sub["step"],
                cond_sub["lang_consistency"],
                marker="o",
                label=f"Cond {cond}",
            )
        ax.set_xlabel("GRPO step")
        ax.set_ylabel("Lang-consistency rate (vi)")
        ax.set_title("Cross-lingual collapse over training")
        ax.set_ylim(0.0, 1.05)
        ax.legend(frameon=False)

    out_path = output / "fig2_lang_consistency.pdf"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig3_pareto(df: pd.DataFrame, output: Path) -> Path:
    """Fig 3: Pareto frontier — avg response tokens vs MGSM-vi pass@1."""
    import matplotlib.pyplot as plt

    sub = df[(df["benchmark"] == "mgsm") & (df["language"] == "vi")].copy()
    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    if sub.empty:
        ax.text(0.5, 0.5, "Fig 3 — no MGSM-vi data yet", ha="center", va="center")
        ax.axis("off")
    else:
        markers = {"en": "o", "vi": "s", "enlang": "^"}
        for cond in HEADLINE_CONDITIONS:
            cond_sub = sub[sub["condition"] == cond]
            if cond_sub.empty:
                continue
            ax.scatter(
                cond_sub["avg_tokens"],
                cond_sub["pass_at_1"],
                marker=markers.get(cond, "x"),
                label=f"Cond {cond}",
                s=40,
                alpha=0.8,
            )
        ax.set_xlabel("Avg response tokens")
        ax.set_ylabel("MGSM-vi pass@1")
        ax.set_title("Accuracy / token-budget Pareto frontier")
        ax.legend(frameon=False)

    out_path = output / "fig3_pareto.pdf"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, default=Path("results/master.csv"))
    parser.add_argument("--output", type=Path, default=Path("paper/figures/"))
    args = parser.parse_args()

    _setup_style()
    df = _load_master(args.master)
    args.output.mkdir(parents=True, exist_ok=True)

    p1 = fig1_transfer_gap(df, args.output)
    p2 = fig2_lang_consistency_curve(df, args.output)
    p3 = fig3_pareto(df, args.output)
    print(f"[plot_curves] saved: {p1}, {p2}, {p3}")


if __name__ == "__main__":
    main()
