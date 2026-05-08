"""Language consistency post-hoc analysis.

Edge cases:
  - Empty responses array -> return 0.0 with a warning.
  - fastText `lid.176.bin` may emit 'zh-cn'/'zh-tw' codes; suffix is stripped.

Usage:
    python src/eval/lang_consistency.py \\
        --input "results/eval/*_mgsm_*.json" \\
        --output results/lang_consist.csv \\
        --fasttext_model data/raw/lid.176.bin"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any

# fastText's lid.176 returns __label__<lang> tokens. Strip prefix.
_LABEL_RE = re.compile(r"^__label__")

# Map fastText output normalization → expected_lang code (ISO 639-1).
# fastText sometimes emits e.g. "zh", "zh-cn"; we normalize to first 2 chars.
_FASTTEXT_MODEL_CACHE: dict[str, Any] = {}


def _load_fasttext(model_path: str | Path) -> Any:
    key = str(model_path)
    if key in _FASTTEXT_MODEL_CACHE:
        return _FASTTEXT_MODEL_CACHE[key]
    try:
        import fasttext  # type: ignore
    except ImportError as e:
        raise ImportError(
            "fasttext not installed. `pip install fasttext` and download lid.176.bin "
            "from https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        ) from e

    model = fasttext.load_model(key)
    _FASTTEXT_MODEL_CACHE[key] = model
    return model


def _normalize_fasttext_label(label: str) -> str:
    """'__label__zh' -> 'zh', '__label__zh-cn' -> 'zh'."""
    s = _LABEL_RE.sub("", label)
    # Truncate at first non-alpha to handle 'zh-cn', 'pt-br', etc.
    return re.split(r"[^a-zA-Z]", s)[0].lower()


def _clean_text_for_fasttext(text: str) -> str:
    """fastText predict() requires single-line input."""
    return text.replace("\n", " ").replace("\r", " ").strip()


def compute_consistency_rate(
    responses: list[str],
    expected_lang: str,
    fasttext_model: str | Path = "data/raw/lid.176.bin",
    min_response_tokens: int = 10,
) -> tuple[float, int, int]:
    """Compute language-consistency rate against an expected language.

    Args:
        responses: list of response strings.
        expected_lang: ISO 639-1 code (e.g., 'vi', 'zh').
        fasttext_model: path to lid.176.bin or pre-loaded fasttext model object.

    Returns:
        (consistency_rate in [0, 1], n_counted, n_skipped)
    """
    if not responses:
        return 0.0, 0, 0

    # Allow caller to pass a pre-loaded model (faster for batch processing).
    if isinstance(fasttext_model, (str, Path)):
        model = _load_fasttext(fasttext_model)
    else:
        model = fasttext_model

    expected = expected_lang.lower()
    n_match = 0
    n_counted = 0
    n_skipped = 0

    for resp in responses:
        cleaned = _clean_text_for_fasttext(resp or "")
        # Skip empty / too-short
        if len(cleaned.split()) < min_response_tokens:
            n_skipped += 1
            continue

        try:
            labels, _probs = model.predict(cleaned, k=1)
        except Exception:
            # fastText can throw on certain unicode edge cases.
            n_skipped += 1
            continue

        if not labels:
            n_skipped += 1
            continue

        pred_lang = _normalize_fasttext_label(labels[0])
        n_counted += 1
        if pred_lang == expected:
            n_match += 1

    rate = n_match / n_counted if n_counted > 0 else 0.0
    return rate, n_counted, n_skipped


_FNAME_LANG_RE = re.compile(r"^(?P<run_id>.+?)_(?P<benchmark>[a-z0-9]+)_(?P<lang>[a-z]{2})\.json$")
_FNAME_NOLANG_RE = re.compile(r"^(?P<run_id>.+?)_(?P<benchmark>[a-z0-9]+)\.json$")


def _parse_filename(name: str) -> dict[str, str | None]:
    """Best-effort parse from filename. Returns dict with keys
    {run_id, benchmark, language} (any may be None)."""
    m = _FNAME_LANG_RE.match(name)
    if m:
        return {"run_id": m["run_id"], "benchmark": m["benchmark"], "language": m["lang"]}
    m = _FNAME_NOLANG_RE.match(name)
    if m:
        return {"run_id": m["run_id"], "benchmark": m["benchmark"], "language": None}
    return {"run_id": None, "benchmark": None, "language": None}


def process_files(
    input_paths: list[Path],
    fasttext_model_path: str | Path,
    min_response_tokens: int = 10,
) -> list[dict[str, Any]]:
    """Compute lang consistency for each input JSON. Returns rows for CSV."""
    rows: list[dict[str, Any]] = []
    for p in input_paths:
        with p.open() as f:
            data = json.load(f)

        # Prefer JSON metadata over filename parsing.
        run_id = data.get("run_id") or _parse_filename(p.name).get("run_id")
        benchmark = data.get("benchmark") or _parse_filename(p.name).get("benchmark")
        language = data.get("language") or _parse_filename(p.name).get("language")

        responses = data.get("responses", [])
        if language is None:
            # Cannot compute without expected language.
            rows.append({
                "run_id": run_id,
                "benchmark": benchmark,
                "language": None,
                "n_samples": len(responses),
                "n_counted": 0,
                "n_skipped": len(responses),
                "lang_consistency_rate": None,
                "input_file": str(p),
            })
            continue

        rate, n_counted, n_skipped = compute_consistency_rate(
            responses,
            expected_lang=language,
            fasttext_model=fasttext_model_path,
            min_response_tokens=min_response_tokens,
        )

        rows.append({
            "run_id": run_id,
            "benchmark": benchmark,
            "language": language,
            "n_samples": len(responses),
            "n_counted": n_counted,
            "n_skipped": n_skipped,
            "lang_consistency_rate": rate,
            "input_file": str(p),
        })
    return rows


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "benchmark",
        "language",
        "n_samples",
        "n_counted",
        "n_skipped",
        "lang_consistency_rate",
        "input_file",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="eval JSON files (supports glob patterns)",
    )
    parser.add_argument("--output", type=Path, required=True, help="CSV output")
    parser.add_argument(
        "--fasttext_model",
        type=Path,
        default=Path("data/raw/lid.176.bin"),
    )
    parser.add_argument("--min_response_tokens", type=int, default=10)
    args = parser.parse_args()

    # Expand globs (argparse doesn't expand them on its own).
    paths: list[Path] = []
    for pattern in args.input:
        expanded = glob.glob(pattern)
        if expanded:
            paths.extend(Path(p) for p in expanded)
        else:
            paths.append(Path(pattern))

    rows = process_files(paths, args.fasttext_model, args.min_response_tokens)
    write_csv(rows, args.output)
    print(f"[lang_consistency] wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
