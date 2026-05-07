"""W3 mid-run sanity gate.

Mục đích: sau khi train 3 cells đầu (qwen15b × 3 conditions), đánh giá xem
có signal cross-lingual transfer không. Nếu gap EN/VI < 1pp → pivot trước
khi spend $100+ cho 6 cells còn lại.

Decision rule:
    - GSM8K (en condition): tối thiểu pass@1 ≥ 0.30 (sanity, không broken)
    - MGSM-vi gap: |pass_at_1(cond=en) - pass_at_1(cond=vi)| ≥ 0.01 (1pp signal)
    - Lang-consistency rate Cond C MGSM-vi ≥ 0.50 (R5 reward đang work)

Usage:
    python scripts/sanity_check.py --output results/sanity_w3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Tránh đụng src/analysis import phức tạp — load CSV nhẹ
import pandas as pd

GATING_GSM8K_MIN = 0.30
GATING_MGSM_VI_GAP_MIN = 0.01   # 1pp
GATING_LANG_CONSIST_MIN = 0.50


def _read(master_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(master_csv)
    df["pass_at_1"] = pd.to_numeric(df["pass_at_1"], errors="coerce")
    df["lang_consistency"] = pd.to_numeric(df["lang_consistency"], errors="coerce")
    return df


def _row_value(df: pd.DataFrame, **filters: str) -> float | None:
    sub = df.copy()
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    if sub.empty:
        return None
    return float(sub["pass_at_1"].mean())


def evaluate_gates(df: pd.DataFrame) -> dict:
    qwen15b = df[df["model"] == "qwen15b"]
    if qwen15b.empty:
        return {
            "verdict": "no_data",
            "reason": "No qwen15b runs found in master.csv. Run W3 first 3 cells.",
            "gates": {},
        }

    # Gate 1: GSM8K Cond A pass@1 sanity
    g1 = _row_value(qwen15b, condition="en", benchmark="gsm8k")
    g1_pass = (g1 is not None) and (g1 >= GATING_GSM8K_MIN)

    # Gate 2: MGSM-vi gap between Cond A (EN-trained) vs Cond B (VI-trained)
    en_vi = _row_value(qwen15b, condition="en", benchmark="mgsm", language="vi")
    vi_vi = _row_value(qwen15b, condition="vi", benchmark="mgsm", language="vi")
    if en_vi is not None and vi_vi is not None:
        gap = abs(en_vi - vi_vi)
        g2_pass = gap >= GATING_MGSM_VI_GAP_MIN
    else:
        gap = None
        g2_pass = False

    # Gate 3: Cond C lang-consistency MGSM-vi
    enlang_consist_rows = qwen15b[
        (qwen15b["condition"] == "enlang")
        & (qwen15b["benchmark"] == "mgsm")
        & (qwen15b["language"] == "vi")
    ]
    if enlang_consist_rows.empty:
        g3_consist = None
        g3_pass = False
    else:
        g3_consist = float(enlang_consist_rows["lang_consistency"].mean())
        g3_pass = g3_consist >= GATING_LANG_CONSIST_MIN

    overall_pass = g1_pass and g2_pass and g3_pass

    return {
        "verdict": "pass" if overall_pass else "fail",
        "reason": (
            "All 3 W3 gates passed — proceed with full 9-cell sweep."
            if overall_pass
            else "Some gates failed — review before spending $100+ on 6 remaining cells."
        ),
        "gates": {
            "gsm8k_en_min30": {
                "value": g1,
                "threshold": GATING_GSM8K_MIN,
                "pass": g1_pass,
            },
            "mgsm_vi_en_vs_vi_gap_min1pp": {
                "en_score": en_vi,
                "vi_score": vi_vi,
                "gap": gap,
                "threshold": GATING_MGSM_VI_GAP_MIN,
                "pass": g2_pass,
            },
            "enlang_lang_consist_vi_min50pct": {
                "value": g3_consist,
                "threshold": GATING_LANG_CONSIST_MIN,
                "pass": g3_pass,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, default=Path("results/master.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/sanity_w3.json"))
    args = parser.parse_args()

    if not args.master.exists():
        result = {
            "verdict": "no_data",
            "reason": f"master.csv not found at {args.master}. Run aggregate.py first.",
            "gates": {},
        }
    else:
        df = _read(args.master)
        result = evaluate_gates(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[sanity_check] verdict={result['verdict']}")
    print(f"[sanity_check] reason: {result['reason']}")
    print(f"[sanity_check] full report → {args.output}")


if __name__ == "__main__":
    main()
