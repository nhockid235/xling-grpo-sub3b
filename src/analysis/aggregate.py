"""Aggregate eval JSONs into master.csv.

Schema:
    run_id, model, condition, seed, stage, step, benchmark, language,
    pass_at_1, maj_at_8, lang_consistency, avg_tokens, n_samples

Parses run_id pattern ``{model}_{condition}_{seed}``. Open-RS reproduction runs
use the special pattern ``reproduce_openrs_rs2_{seed}`` and are handled separately.

Usage:
    python src/analysis/aggregate.py --eval_dir results/eval/ --output results/master.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

# Run ID patterns. Tolerates suffix `_step{N}` for step-specific eval runs.
RUN_ID_PATTERN = re.compile(
    r"^(?P<model>qwen15b|qwen3b|llama3b|deepseek_r1_distill_15b)"
    r"_(?P<condition>en|vi|enlang|baseline)"
    r"_(?P<seed>\d+)(?:_step\d+)?$"
)
REPRODUCE_PATTERN = re.compile(
    r"^reproduce_openrs_(?P<exp>rs[123])_(?P<seed>\d+)(?:_step\d+)?$"
)
# A-series patterns: a1_en_42 / a2_vi_42 / a3_enlang_42
A_SERIES_PATTERN = re.compile(
    r"^(?P<arm>a[123])_(?P<condition>en|vi|enlang)_(?P<seed>\d+)(?:_step\d+)?$"
)
# Bare ckpt eval runs (e.g., ckpt50_v3)
CKPT_PATTERN = re.compile(r"^ckpt(?P<step>\d+)_v\d+$")
# Base model eval runs (untrained)
BASE_PATTERN = re.compile(r"^base(?:_v\d+)?(?:_(?P<model>.+))?$")
CKPT_STEP_PATTERN = re.compile(r"checkpoint-(?P<step>\d+|final)")


CSV_COLUMNS = [
    "run_id",
    "model",
    "condition",
    "seed",
    "stage",
    "step",
    "benchmark",
    "language",
    "pass_at_1",
    "maj_at_8",
    "lang_consistency",
    "avg_tokens",
    "n_samples",
]


def parse_run_id(run_id: str) -> dict[str, str]:
    """Parse run_id → {model, condition, seed}. Handles all naming patterns."""
    # Standard: model_condition_seed
    m = RUN_ID_PATTERN.match(run_id)
    if m:
        return {
            "model": m.group("model"),
            "condition": m.group("condition"),
            "seed": m.group("seed"),
        }
    # Reproduce Open-RS: reproduce_openrs_rsN_seed
    m = REPRODUCE_PATTERN.match(run_id)
    if m:
        return {
            "model": "deepseek_r1_distill_15b",
            "condition": f"baseline_{m.group('exp')}",
            "seed": m.group("seed"),
        }
    # A-series: a1_en_42 / a2_vi_42 / a3_enlang_42
    m = A_SERIES_PATTERN.match(run_id)
    if m:
        return {
            "model": "deepseek_r1_distill_15b",
            "condition": m.group("condition"),
            "seed": m.group("seed"),
        }
    # Ckpt-eval style: ckpt50_v3 (= reproduce_openrs ckpt-50, eval v3)
    m = CKPT_PATTERN.match(run_id)
    if m:
        return {
            "model": "deepseek_r1_distill_15b",
            "condition": "en",   # A1 = ckpt-50 v3
            "seed": "42",
        }
    # Base model eval (untrained baseline)
    m = BASE_PATTERN.match(run_id)
    if m:
        return {
            "model": "deepseek_r1_distill_15b",
            "condition": "base",
            "seed": "0",
        }
    return {"model": "unknown", "condition": "unknown", "seed": "unknown"}


def parse_step(model_path: str | None) -> str:
    """Extract step from a model_path string. Falls back to 'unknown'."""
    if not model_path:
        return "unknown"
    m = CKPT_STEP_PATTERN.search(model_path)
    return m.group("step") if m else "unknown"


def parse_stage(model_path: str | None) -> str:
    """Infer stage (sft/grpo) from a path. Falls back to 'unknown'."""
    if not model_path:
        return "unknown"
    if "/grpo/" in model_path or "\\grpo\\" in model_path:
        return "grpo"
    if "/sft/" in model_path or "\\sft\\" in model_path:
        return "sft"
    return "unknown"


def eval_json_to_row(json_path: Path) -> dict[str, Any]:
    """Read one eval JSON and return a dict for one CSV row."""
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    run_id = data.get("run_id", "")
    parsed = parse_run_id(run_id)
    metadata = data.get("metadata") or {}
    model_path = metadata.get("model_path", "")

    return {
        "run_id": run_id,
        "model": parsed["model"],
        "condition": parsed["condition"],
        "seed": parsed["seed"],
        "stage": parse_stage(model_path),
        "step": parse_step(model_path),
        "benchmark": data.get("benchmark", ""),
        "language": data.get("language", "") or "",
        "pass_at_1": data.get("pass_at_1", ""),
        "maj_at_8": data.get("maj_at_8", "") if data.get("maj_at_8") is not None else "",
        "lang_consistency": data.get("lang_consistency_rate", "")
            if data.get("lang_consistency_rate") is not None else "",
        "avg_tokens": data.get("avg_response_tokens", ""),
        "n_samples": data.get("n_samples", ""),
    }


def aggregate(eval_dir: Path, output: Path) -> int:
    """Walk `eval_dir/**/*.json`, parse each, write `output` CSV. Returns row count."""
    rows: list[dict[str, Any]] = []
    json_files = sorted(eval_dir.rglob("*.json"))
    for jf in json_files:
        try:
            rows.append(eval_json_to_row(jf))
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[aggregate] WARNING: skipping {jf}: {exc}")
            continue

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=Path, default=Path("results/eval/"))
    parser.add_argument("--output", type=Path, default=Path("results/master.csv"))
    args = parser.parse_args()

    n = aggregate(args.eval_dir, args.output)
    print(f"[aggregate] wrote {n} rows → {args.output}")


if __name__ == "__main__":
    main()
