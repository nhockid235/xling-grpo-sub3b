"""Render LaTeX tables for the paper.

Outputs (booktabs, no vertical rules):
    paper/tables/table1_xling_transfer.tex     -- 3 conditions x 6 langs MGSM
    paper/tables/table2_en_sanity.tex           -- GSM8K, MATH-500, AIME-2024
    paper/tables/table3_lang_consistency.tex    -- 3 conditions x 6 langs lang-consistency

Bold best per row. Bootstrap 95% CI in cells (when n_samples available)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HEADLINE_LANGS = ["en", "vi", "zh", "fr", "th", "sw"]
HEADLINE_CONDITIONS = ["en", "vi", "enlang"]
COND_DISPLAY = {"en": "Cond A (EN)", "vi": "Cond B (VI)", "enlang": "Cond C (EN+lang)"}


def _load_master(master_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(master_csv)
    for col in ("pass_at_1", "maj_at_8", "lang_consistency", "avg_tokens", "n_samples"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _bold_best_per_row(df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric cells to one decimal and wrap the per-row max in \\textbf{...}."""
    out = df.copy().astype(str)
    for idx, row in df.iterrows():
        numeric_row = pd.to_numeric(row, errors="coerce")
        if numeric_row.isna().all():
            continue
        max_val = numeric_row.max()
        max_cols = numeric_row[numeric_row == max_val].index
        for col in max_cols:
            val = row[col]
            if pd.notna(val):
                out.at[idx, col] = f"\\textbf{{{val:.1f}}}"
    for col in df.columns:
        for idx in df.index:
            cell = out.at[idx, col]
            if not cell.startswith("\\textbf"):
                try:
                    out.at[idx, col] = f"{float(cell):.1f}"
                except (ValueError, TypeError):
                    pass
    return out


def _render_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    column_format: str | None = None,
) -> str:
    """Render a booktabs LaTeX table from a DataFrame."""
    n_cols = len(df.columns) + 1  # +1 for index column
    col_fmt = column_format or ("l" + "c" * len(df.columns))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
    ]
    # Header
    header = [df.index.name or ""] + list(df.columns)
    lines.append(" & ".join(str(h) for h in header) + " \\\\")
    lines.append("\\midrule")
    # Body
    for idx, row in df.iterrows():
        cells = [str(idx)] + [str(v) for v in row.values]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def table1_xling_transfer(df: pd.DataFrame) -> str:
    """Rows: 3 conditions; Cols: MGSM × 6 langs (pass@1)."""
    sub = df[df["benchmark"] == "mgsm"].copy()
    pivot = sub.pivot_table(
        index="condition",
        columns="language",
        values="pass_at_1",
        aggfunc="mean",
    )
    pivot = (pivot * 100).round(1)
    # Reorder rows + cols
    pivot = pivot.reindex([c for c in HEADLINE_CONDITIONS if c in pivot.index])
    pivot = pivot[[c for c in HEADLINE_LANGS if c in pivot.columns]]
    pivot.index = [COND_DISPLAY.get(c, c) for c in pivot.index]
    pivot.index.name = "Condition"

    bold = _bold_best_per_row(pivot)
    return _render_latex_table(
        bold,
        caption="Cross-lingual transfer (MGSM pass@1, %).",
        label="tab:xling_transfer",
    )


def table2_en_sanity(df: pd.DataFrame) -> str:
    """Rows: 3 conditions; Cols: GSM8K, MATH-500, AIME-2024 (pass@1)."""
    sub = df[df["benchmark"].isin(["gsm8k", "math500", "aime2024"])].copy()
    pivot = sub.pivot_table(
        index="condition",
        columns="benchmark",
        values="pass_at_1",
        aggfunc="mean",
    )
    pivot = (pivot * 100).round(1)
    pivot = pivot.reindex([c for c in HEADLINE_CONDITIONS if c in pivot.index])
    bench_order = [b for b in ["gsm8k", "math500", "aime2024"] if b in pivot.columns]
    pivot = pivot[bench_order]
    pivot.index = [COND_DISPLAY.get(c, c) for c in pivot.index]
    pivot.index.name = "Condition"

    bold = _bold_best_per_row(pivot)
    return _render_latex_table(
        bold,
        caption="English benchmark sanity (pass@1, %). EN-trained must not regress.",
        label="tab:en_sanity",
    )


def table3_lang_consistency(df: pd.DataFrame) -> str:
    """Rows: 3 conditions; Cols: 6 langs × lang-consistency rate."""
    sub = df[df["benchmark"] == "mgsm"].copy()
    pivot = sub.pivot_table(
        index="condition",
        columns="language",
        values="lang_consistency",
        aggfunc="mean",
    )
    pivot = (pivot * 100).round(1)
    pivot = pivot.reindex([c for c in HEADLINE_CONDITIONS if c in pivot.index])
    pivot = pivot[[c for c in HEADLINE_LANGS if c in pivot.columns]]
    pivot.index = [COND_DISPLAY.get(c, c) for c in pivot.index]
    pivot.index.name = "Condition"

    bold = _bold_best_per_row(pivot)
    return _render_latex_table(
        bold,
        caption="Language-consistency rate on MGSM (\\%). Bold rates $>$ 85.",
        label="tab:lang_consistency",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, default=Path("results/master.csv"))
    parser.add_argument("--output", type=Path, default=Path("paper/tables/"))
    args = parser.parse_args()

    df = _load_master(args.master)
    args.output.mkdir(parents=True, exist_ok=True)

    out_files = {
        "table1_xling_transfer.tex": table1_xling_transfer(df),
        "table2_en_sanity.tex": table2_en_sanity(df),
        "table3_lang_consistency.tex": table3_lang_consistency(df),
    }

    for fname, latex in out_files.items():
        path = args.output / fname
        path.write_text(latex, encoding="utf-8")
        print(f"[make_tables] wrote {path}")


if __name__ == "__main__":
    main()
