"""Prepare Open-RS dataset (knoveleng/open-rs) cho W2 gating reproduction.

HF: knoveleng/open-rs, single 'train' split, 7K rows.
Schema: {problem, solution, answer, level} where level ∈ {Hard, Easy}.
NO Stage-1 split — single flat file.

Output: data/processed/open_rs_7k.jsonl
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
    """Load Open-RS 7K split and dump JSONL preserving level/answer."""
    from datasets import load_dataset

    cache_kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
    ds = load_dataset("knoveleng/open-rs", split="train", **cache_kwargs)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    records: list[dict] = []
    for r in ds:
        problem = r.get("problem")
        solution = r.get("solution")
        if not problem or not solution:
            continue
        records.append(
            {
                "problem": str(problem),
                "solution": str(solution),
                "answer": r.get("answer"),
                "level": r.get("level"),
                "source": "open-rs",
                "lang": "en",
            }
        )

    write_jsonl(records, output)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, default=Path("data/raw/"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    n = prepare(output=args.output, cache_dir=args.cache_dir, limit=args.limit)
    print(f"wrote {n} records -> {args.output}")


if __name__ == "__main__":
    main()
