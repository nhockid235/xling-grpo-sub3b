"""Filter decontaminated training pool xuống 7K samples (match Open-RS scale).

Strategy options:
    - random sample với fixed seed (default).
    - difficulty stratified sampling — preserve per-`level` proportions.

Usage:
    python data/filter_7k.py \
        --input data/processed/en_train_clean.jsonl \
        --output data/processed/en_train_7k.jsonl \
        --n 7000 --seed 42
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.io import read_jsonl, write_jsonl  # noqa: E402


def filter_random(records: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    return shuffled[:n]


def filter_difficulty(records: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified sample by `level` field. Preserves per-level proportions.

    If `level` field missing on a record, that record is bucketed under
    `__unknown__`. When n exceeds total records, returns the full shuffled
    set (still seeded).
    """
    rng = random.Random(seed)
    if not records:
        return []

    buckets: dict[str, list[dict]] = {}
    for r in records:
        level = r.get("level") or "__unknown__"
        buckets.setdefault(str(level), []).append(r)

    total = len(records)
    if n >= total:
        out = list(records)
        rng.shuffle(out)
        return out

    # Allocate quota per bucket using floor + remainder distribution to hit n
    # exactly without losing whole buckets to rounding.
    raw_quotas: list[tuple[str, float]] = [
        (lvl, n * len(rs) / total) for lvl, rs in buckets.items()
    ]
    quotas: dict[str, int] = {lvl: int(math.floor(q)) for lvl, q in raw_quotas}
    leftover = n - sum(quotas.values())
    # Distribute leftover by largest fractional remainder, deterministic ties
    # by bucket name.
    remainders = sorted(
        ((lvl, q - math.floor(q)) for lvl, q in raw_quotas),
        key=lambda kv: (-kv[1], kv[0]),
    )
    for i in range(leftover):
        lvl = remainders[i % len(remainders)][0]
        quotas[lvl] += 1

    out: list[dict] = []
    for lvl, rs in buckets.items():
        take = min(quotas.get(lvl, 0), len(rs))
        shuffled = list(rs)
        rng.shuffle(shuffled)
        out.extend(shuffled[:take])

    rng.shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy", choices=["random", "difficulty"], default="random"
    )
    args = parser.parse_args()

    records = read_jsonl(args.input)
    if args.strategy == "random":
        out = filter_random(records, args.n, args.seed)
    else:
        out = filter_difficulty(records, args.n, args.seed)

    write_jsonl(out, args.output)
    print(
        f"strategy={args.strategy} input={len(records)} "
        f"output={len(out)} -> {args.output}"
    )


if __name__ == "__main__":
    main()
