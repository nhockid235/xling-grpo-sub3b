"""Prepare NuminaMath-CoT (EN) → JSONL.

HF: AI-MO/NuminaMath-CoT, ~860K. Schema: {problem, solution, source}.
Output: data/processed/en_train_full.jsonl (rồi qua decontaminate + filter_7k).

Usage:
    python data/prepare_numinamath.py --output data/processed/en_train_full.jsonl
    # Quick dev test:
    python data/prepare_numinamath.py --output /tmp/sample.jsonl --limit 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.io import write_jsonl  # noqa: E402


def prepare(
    *, output: Path, cache_dir: Path | None = None, limit: int | None = None
) -> int:
    """Load NuminaMath-CoT and dump JSONL with the unified schema.

    Returns number of records written (skipped records with missing fields are
    counted as drops, not errors).
    """
    from datasets import load_dataset

    cache_kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", **cache_kwargs)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    records: list[dict] = []
    for r in ds:
        problem = r.get("problem")
        solution = r.get("solution")
        source = r.get("source") or "numinamath-cot"
        if not problem or not solution:
            continue
        records.append(
            {
                "problem": str(problem),
                "solution": str(solution),
                "source": str(source),
                "lang": "en",
            }
        )

    write_jsonl(records, output)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, default=Path("data/raw/"))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of records (dev/testing).",
    )
    args = parser.parse_args()
    n = prepare(output=args.output, cache_dir=args.cache_dir, limit=args.limit)
    print(f"wrote {n} records -> {args.output}")


if __name__ == "__main__":
    main()
