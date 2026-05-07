"""Decontamination — 8-gram match against test sets.

Test sets: GSM8K, MATH-500, AIME-2024, MGSM (10 langs), MSVAMP (10 langs).
Asymmetric: filter training data only, NEVER test data (CLAUDE.md pitfall #12).

Algorithm:
    - Tokenize problem text -> whitespace-split, lowercase, strip basic punctuation.
    - Extract overlapping 8-grams.
    - Build a MinHashLSH index over the union of test-set 8-gram shingle sets.
    - For each train sample, compute MinHash from its 8-grams; query LSH for any
      candidate test record with Jaccard >= threshold (default 0.5). If hit,
      drop the train sample and increment per-test-set counter (counter is
      attributed to the *first* match found).

Tradeoffs (see report phase2_02_data.md):
    - Exact 8-gram intersection (set / Bloom filter) catches verbatim copies but
      misses rephrased contamination. MinHashLSH approximates Jaccard between
      shingle sets, scaling sub-quadratically and tolerating minor edits.
    - Permutations num_perm=128 -> good Jaccard error (<0.05) at moderate cost.

Usage:
    python data/decontaminate.py \
        --train data/processed/en_train_full.jsonl \
        --tests hf \
        --output data/processed/en_train_clean.jsonl \
        --stats data/processed/decontam_stats.json
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

# Make `src` importable when running as a script (python data/decontaminate.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.io import read_jsonl, write_jsonl  # noqa: E402

# Punctuation table giữ chữ và số, bỏ dấu câu cơ bản (ASCII).
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> list[str]:
    """Lowercase, strip ASCII punctuation, whitespace-tokenize."""
    if not text:
        return []
    text = text.lower().translate(_PUNCT_TABLE)
    text = _WS_RE.sub(" ", text).strip()
    if not text:
        return []
    return text.split(" ")


def ngrams(tokens: list[str], n: int) -> list[str]:
    """Generate space-joined n-grams. Returns empty list when len(tokens) < n."""
    if len(tokens) < n:
        # Fallback: dùng toàn bộ chuỗi như 1 shingle để vẫn có thể match
        # (test record quá ngắn vẫn cần index).
        if not tokens:
            return []
        return [" ".join(tokens)]
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def make_minhash(shingles: Iterable[str], num_perm: int = 128):
    """Create a datasketch MinHash from string shingles."""
    from datasketch import MinHash

    mh = MinHash(num_perm=num_perm)
    for s in shingles:
        mh.update(s.encode("utf-8"))
    return mh


# --- Test set loaders -------------------------------------------------------


def _load_hf_test_sets(cache_dir: Path | None = None) -> dict[str, list[str]]:
    """Load all configured test sets from HuggingFace Hub.

    Returns mapping: test_set_name -> list of problem strings. test_set_name
    contains language suffix for MGSM/MSVAMP (e.g. "mgsm:vi").
    """
    from datasets import load_dataset

    cache_kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        cache_kwargs["cache_dir"] = str(cache_dir)

    out: dict[str, list[str]] = {}

    # GSM8K (canonical openai/gsm8k, config=main, split=test)
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test", **cache_kwargs)
        out["gsm8k"] = [str(r["question"]) for r in ds]
    except Exception as e:  # pragma: no cover - network dependent
        print(f"[warn] failed to load gsm8k: {e}", file=sys.stderr)

    # MATH-500
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test", **cache_kwargs)
        out["math500"] = [str(r["problem"]) for r in ds]
    except Exception as e:  # pragma: no cover
        print(f"[warn] failed to load math500: {e}", file=sys.stderr)

    # AIME-2024 (capitalized fields, split=train)
    try:
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train", **cache_kwargs)
        out["aime2024"] = [str(r["Problem"]) for r in ds]
    except Exception as e:  # pragma: no cover
        print(f"[warn] failed to load aime2024: {e}", file=sys.stderr)

    # MGSM 10 langs guaranteed
    mgsm_langs = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn"]
    for lang in mgsm_langs:
        try:
            ds = load_dataset("juletxara/mgsm", lang, split="test", **cache_kwargs)
            out[f"mgsm:{lang}"] = [str(r["question"]) for r in ds]
        except Exception as e:  # pragma: no cover
            print(f"[warn] failed to load mgsm:{lang}: {e}", file=sys.stderr)

    # MSVAMP 10 langs
    msvamp_langs = ["bn", "zh", "en", "fr", "de", "ja", "ru", "es", "sw", "th"]
    for lang in msvamp_langs:
        try:
            ds = load_dataset("Mathoctopus/MSVAMP", lang, split="test", **cache_kwargs)
            out[f"msvamp:{lang}"] = [str(r["question"]) for r in ds]
        except Exception as e:  # pragma: no cover
            print(f"[warn] failed to load msvamp:{lang}: {e}", file=sys.stderr)

    return out


def _load_local_test_sets(tests_dir: Path) -> dict[str, list[str]]:
    """Load test sets from local JSONL files inside `tests_dir`.

    Each *.jsonl file is treated as a separate test set named after its stem.
    Each record must have a `problem` (or `question`) field.
    """
    out: dict[str, list[str]] = {}
    if not tests_dir.exists():
        return out
    for path in sorted(tests_dir.glob("*.jsonl")):
        records = read_jsonl(path)
        problems: list[str] = []
        for r in records:
            txt = r.get("problem") or r.get("question") or r.get("Problem")
            if txt:
                problems.append(str(txt))
        out[path.stem] = problems
    return out


def load_test_sets(tests_arg: str, cache_dir: Path | None = None) -> dict[str, list[str]]:
    """Resolve --tests argument: 'hf' or path to a local directory."""
    if tests_arg == "hf":
        return _load_hf_test_sets(cache_dir=cache_dir)
    return _load_local_test_sets(Path(tests_arg))


# --- Index + query ----------------------------------------------------------


def build_index(
    test_sets: dict[str, list[str]],
    *,
    n: int,
    threshold: float,
    num_perm: int = 128,
) -> tuple[Any, dict[str, str]]:
    """Build MinHashLSH index over test problems.

    Returns:
        lsh: datasketch.MinHashLSH populated with one entry per test record.
        key_to_set: mapping LSH key -> originating test_set_name (so query hits
            can be attributed to a specific test set in stats).
    """
    from datasketch import MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    key_to_set: dict[str, str] = {}

    for set_name, problems in test_sets.items():
        for i, problem in enumerate(problems):
            tokens = normalize(problem)
            shingles = ngrams(tokens, n)
            if not shingles:
                continue
            mh = make_minhash(shingles, num_perm=num_perm)
            key = f"{set_name}::{i}"
            # MinHashLSH.insert raises if key already present; safe here since
            # set_name is unique and i strictly increments.
            lsh.insert(key, mh)
            key_to_set[key] = set_name

    return lsh, key_to_set


def query_contaminated(
    record: dict,
    lsh: Any,
    key_to_set: dict[str, str],
    *,
    n: int,
    num_perm: int = 128,
) -> str | None:
    """Return name of first test set with a Jaccard >= threshold hit, else None."""
    text = record.get("problem") or record.get("question") or ""
    tokens = normalize(str(text))
    shingles = ngrams(tokens, n)
    if not shingles:
        return None
    mh = make_minhash(shingles, num_perm=num_perm)
    hits = lsh.query(mh)
    if not hits:
        return None
    # Deterministic attribution: pick alphabetically first key (avoids set order
    # nondeterminism across runs).
    first_key = sorted(hits)[0]
    return key_to_set.get(first_key)


def iter_decontaminate(
    train_records: list[dict],
    lsh: Any,
    key_to_set: dict[str, str],
    *,
    n: int,
    num_perm: int = 128,
) -> Iterator[tuple[dict, str | None]]:
    """Yield (record, contam_source_or_None) for each train record."""
    for r in train_records:
        hit = query_contaminated(r, lsh, key_to_set, n=n, num_perm=num_perm)
        yield r, hit


# --- CLI --------------------------------------------------------------------


def run(
    *,
    train_path: Path,
    tests_arg: str,
    output_path: Path,
    stats_path: Path | None,
    n: int,
    threshold: float,
    num_perm: int = 128,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Programmatic entry point — used by tests."""
    train_records = read_jsonl(train_path)
    test_sets = load_test_sets(tests_arg, cache_dir=cache_dir)
    lsh, key_to_set = build_index(
        test_sets, n=n, threshold=threshold, num_perm=num_perm
    )

    kept: list[dict] = []
    by_test_set: dict[str, int] = {name: 0 for name in test_sets}
    removed = 0
    for record, hit in iter_decontaminate(
        train_records, lsh, key_to_set, n=n, num_perm=num_perm
    ):
        if hit is None:
            kept.append(record)
        else:
            removed += 1
            by_test_set[hit] = by_test_set.get(hit, 0) + 1

    write_jsonl(kept, output_path)

    stats = {
        "total_train": len(train_records),
        "kept": len(kept),
        "removed": removed,
        "threshold": threshold,
        "ngram": n,
        "num_perm": num_perm,
        "by_test_set": by_test_set,
    }
    if stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument(
        "--tests",
        type=str,
        default="hf",
        help="'hf' to fetch test sets from HuggingFace, or path to local dir of *.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--ngram", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_perm", type=int, default=128)
    parser.add_argument("--cache_dir", type=Path, default=None)
    args = parser.parse_args()

    stats = run(
        train_path=args.train,
        tests_arg=args.tests,
        output_path=args.output,
        stats_path=args.stats,
        n=args.ngram,
        threshold=args.threshold,
        num_perm=args.num_perm,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
