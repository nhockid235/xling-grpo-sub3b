"""Prepare MetaMathQA-VI → JSONL.

HF: 5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated, 40K.
⚠️ License field unset on Hub — contact 5CD-AI before public release.
Schema upstream: {query_vi, response_vi, query (EN), response (EN), ...}.

Usage:
    python data/prepare_metamath_vi.py --output data/processed/vi_train_full.jsonl
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
) -> tuple[int, int]:
    """Load MetaMathQA-VI and emit (problem,solution,source,lang) JSONL.

    Returns (kept, skipped). Records with empty/None query_vi or response_vi
    are skipped.
    """
    from datasets import load_dataset

    cache_kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
    ds = load_dataset(
        "5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated",
        split="train",
        **cache_kwargs,
    )
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    kept: list[dict] = []
    skipped = 0
    for r in ds:
        q = r.get("query_vi")
        a = r.get("response_vi")
        if q is None or a is None:
            skipped += 1
            continue
        q_str = str(q).strip()
        a_str = str(a).strip()
        if not q_str or not a_str:
            skipped += 1
            continue
        kept.append(
            {
                "problem": q_str,
                "solution": a_str,
                "source": "metamathqa-vi-mt",
                "lang": "vi",
            }
        )

    write_jsonl(kept, output)
    return len(kept), skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, default=Path("data/raw/"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    kept, skipped = prepare(
        output=args.output, cache_dir=args.cache_dir, limit=args.limit
    )
    print(f"wrote {kept} records -> {args.output} (skipped {skipped} empty-VI rows)")


if __name__ == "__main__":
    main()
