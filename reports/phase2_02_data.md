# Phase 2 — Data Pipeline Implementation

**Date:** 2026-05-05
**Agent:** implementation (Phase 2 / data)
**Scope:** Replace `NotImplementedError` stubs in `data/*.py` with working
loaders + an 8-gram MinHashLSH decontamination pipeline. Activate the four
skipped tests in `tests/test_decontaminate.py`.

---

## 1. Files modified

All paths are absolute.

| File | Status |
|------|--------|
| `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/data/decontaminate.py` | Implemented — MinHashLSH with `--tests {hf,<dir>}` switch, deterministic stats. |
| `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/data/prepare_numinamath.py` | Implemented — HF `AI-MO/NuminaMath-CoT` → JSONL, supports `--limit`. |
| `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/data/prepare_metamath_vi.py` | Implemented — `query_vi`/`response_vi` → unified schema, skips empty rows. |
| `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/data/prepare_open_rs.py` | Implemented — preserves `level` + `answer` for downstream stratification. |
| `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/data/filter_7k.py` | Implemented — `random` and stratified `difficulty` strategies. |
| `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/tests/test_decontaminate.py` | Un-skipped + 4 required tests + 3 helper-level tests. All passing. |

Dependencies added (installed locally; should be added to `pyproject.toml`
in a follow-up by the env owner):

- `datasketch==1.10.0`
- `xxhash==3.7.0` (already present transitively, kept explicit)

---

## 2. Decontamination algorithm — choice & tradeoffs

### Algorithm (chosen): MinHashLSH over 8-gram shingles

1. **Normalise** problem text: lowercase, strip ASCII punctuation
   (`string.punctuation` translated to whitespace), collapse whitespace,
   tokenize by `split(" ")`.
2. **Shingle** into overlapping 8-grams: `tokens[i:i+8]` joined with single
   spaces. For records shorter than 8 tokens we fall back to a single shingle
   containing the entire normalised string — this avoids silently dropping
   short test records (which mostly hit MGSM/MSVAMP for low-resource langs).
3. **Index** every test record into a `datasketch.MinHashLSH(threshold=0.5,
   num_perm=128)` keyed `f"{set_name}::{i}"`.
4. **Query** each train record: build its MinHash, ask LSH for any candidate
   above the Jaccard threshold. First hit (alphabetically first key, for
   determinism) determines attribution in `stats["by_test_set"]`.
5. **Persist** stats JSON and the cleaned JSONL.

### Tradeoffs considered

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Exact-match Python `set` of 8-grams | Trivial, deterministic | Misses paraphrases, OOM on 860K × ~3K test if you keep position info | rejected |
| Bloom filter of 8-grams | Compact, fast `O(1)` lookups | Only verbatim hits; no Jaccard semantics; false positives on long texts | rejected as primary |
| **MinHashLSH (datasketch)** | Sub-quadratic, tunable Jaccard threshold, tolerates rephrasing | Approximate: Jaccard error ~0.05 at `num_perm=128`; constant-factor slower than Bloom | **chosen** |
| `datasets`-native dedup (e.g. `text-dedup`) | Production-tested | Heavyweight, our config is more bespoke | rejected, overkill |

`num_perm=128` and `threshold=0.5` mirror the defaults in NeuMatch / DeepSeek's
released decontamination code; we expose both as CLI flags in case the
empirical contamination on NuminaMath needs tuning.

Performance estimate (CLAUDE.md pitfall #2):
- Indexing ~3,000 test records: O(3K · 128) hashes ≈ 380K hashes, <30s.
- Querying 860K train records: O(860K · 128) ≈ 110M hashes, ~5–10 min on a
  single CPU core. Fits inside the assumed budget; no parallelisation needed.

### Asymmetry guarantee (CLAUDE.md pitfall #12)

`run()` only writes `output_path` (cleaned train) and the optional
`stats_path`. It never writes back to the test-set source — whether HF cache
or local `--tests <dir>`. `test_decontam_asymmetric` asserts byte equality of
the test files before and after a run.

---

## 3. Schema mapping

### Output schema (canonical, used by SFT/GRPO trainers)

```json
{
  "problem": "<str — input math problem>",
  "solution": "<str — chain-of-thought solution>",
  "source": "<str — provenance tag>",
  "lang": "<en|vi>",
  "answer": "<optional, only when upstream provides>",
  "level": "<optional, only Open-RS provides>"
}
```

### Per-dataset mapping

| Dataset | Upstream column | Output column | Notes |
|---------|-----------------|----------------|-------|
| `AI-MO/NuminaMath-CoT` | `problem` | `problem` | identity copy |
| | `solution` | `solution` | identity copy |
| | `source` | `source` | fallback `"numinamath-cot"` if missing |
| | — | `lang` | hardcoded `"en"` |
| `5CD-AI/...MetaMathQA-40K-gg-translated` | `query_vi` | `problem` | `.strip()`; skip if empty/None |
| | `response_vi` | `solution` | `.strip()`; skip if empty/None |
| | — | `source` | hardcoded `"metamathqa-vi-mt"` |
| | — | `lang` | hardcoded `"vi"` |
| `knoveleng/open-rs` | `problem` | `problem` | |
| | `solution` | `solution` | |
| | `answer` | `answer` | preserved verbatim for verifiable reward |
| | `level` | `level` | `Hard`/`Easy`; powers `filter_7k --strategy difficulty` |
| | — | `source` | hardcoded `"open-rs"` |
| | — | `lang` | hardcoded `"en"` |

Decontamination (`decontaminate.py`) reads `problem` (or `question` /
`Problem` fallbacks) so mismatched upstream test schemas are absorbed without
re-mapping.

---

## 4. Synthetic test fixture design

The test file uses three hand-crafted long-form English problem strings as
fixtures (long enough to yield ≥10 8-grams each):

- `TEST_PROBLEM_GSM` — Janet's-ducks-style word problem, 40+ tokens.
- `TEST_PROBLEM_MATH500` — number-theory sum problem, 30+ tokens.
- `UNRELATED_TRAIN_PROBLEM` — calculus limit, no token overlap with the test
  problems.

`_write_test_dir(tmp_path)` materialises these as `gsm8k.jsonl` and
`math500.jsonl` inside `tmp_path/tests/`. Decontaminate runs with
`--tests <tmp_path/tests>` so the loader takes the local-directory branch
(`_load_local_test_sets`) — **no HuggingFace network call**. Train data is
written via `write_jsonl` and the run is triggered programmatically through
`data.decontaminate.run(...)` so we don't have to spawn a subprocess.

This keeps the suite self-contained, deterministic, and fast (<300 ms) while
exercising the same `build_index → query_contaminated → write_jsonl + stats`
code path the production CLI runs.

---

## 5. Test output (verbatim)

```
$ /Users/vudang/miniconda3/bin/python -m pytest tests/test_decontaminate.py -v
============================= test session starts ==============================
platform darwin -- Python 3.13.11, pytest-9.0.3, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: /Users/vudang/PythonLab/Papper/xling-grpo-sub3b
configfile: pyproject.toml
plugins: anyio-4.12.1
collecting ... collected 7 items

tests/test_decontaminate.py::test_normalize_strips_punct_and_lowercases PASSED [ 14%]
tests/test_decontaminate.py::test_ngrams_basic PASSED                    [ 28%]
tests/test_decontaminate.py::test_ngrams_short_input_falls_back_to_full_join PASSED [ 42%]
tests/test_decontaminate.py::test_decontam_removes_exact_match PASSED    [ 57%]
tests/test_decontaminate.py::test_decontam_keeps_unrelated PASSED        [ 71%]
tests/test_decontaminate.py::test_decontam_asymmetric PASSED             [ 85%]
tests/test_decontaminate.py::test_decontam_stats_recorded PASSED         [100%]

============================== 7 passed in 0.21s ===============================
```

Adjacent suites (`tests/test_io.py`, `tests/test_seed.py`) re-ran clean to
confirm no regressions touching shared `src/utils/io.py`.

A manual smoke test for `filter_7k` with 20 records (7 `Hard`, 13 `Easy`)
asking for `n=5` returned a stratified split of 2 Hard / 3 Easy as expected
from the floor+remainder allocator.

---

## 6. Risks — open items for downstream phases

1. **`5CD-AI/...VI` license still unset on Hub.** Loader will work, but
   redistributing derived JSONL is blocked until the upstream license is
   declared or the team pivots to GPT-4o-translated MetaMathQA (CLAUDE.md §
   datasets footnote, Phase 0 risk #1). This is a release-time blocker, not a
   training blocker.
2. **MGSM / MSVAMP per-language load latency.** `_load_hf_test_sets` fires 20
   `load_dataset` calls (10 langs × 2 benchmarks). On a cold HF cache that's
   ~3–5 min before indexing even starts. Each call is wrapped in a
   `try/except` so a single language outage doesn't fail the whole job — the
   warning surfaces on stderr and stats simply omit that test set.
3. **Decontamination performance unverified at full scale.** Benchmarks are
   estimates (Section 2). When NuminaMath actually loads, profile the first
   5K records to extrapolate. If wall-clock > 30 min, drop `num_perm` to 64
   (linear speedup, modest accuracy loss) before adding multiprocessing.
4. **MGSM-Pro (47-lang) does not exist.** Phase 0 verified only the
   McGill-NLP partial 9-lang variant exists. `_load_hf_test_sets` does not
   query it — paper relies on MGSM + MSVAMP + manual VI MATH-500 subset for
   eval breadth (CLAUDE.md § Datasets, MGSM-Pro fallback).
5. **Missing-field guards.** All three prepare scripts skip records with
   empty `problem`/`solution`. NuminaMath has no documented null rate but
   MetaMathQA-VI has reported MT-failure rows. The skipped count is printed
   on stdout; consider piping it into the eventual `master.csv` to keep
   provenance intact for paper appendix.
6. **AIME-2024 capitalised fields** (`Problem`, `Solution`, `Answer`, `ID`):
   handled in `decontaminate._load_hf_test_sets` via `r["Problem"]`. Any
   other consumers (eval scripts) must remember Phase 0 pitfall #18.
7. **GSM8K canonical id.** Decontamination uses `openai/gsm8k` (Phase 0
   recommendation), not the bare `gsm8k` alias which is now a script-loader.
8. **datasketch + xxhash not yet pinned in `pyproject.toml`.** Currently
   pip-installed in the working venv. The infra agent should add them to
   `dependencies` before CI runs the data step.

---

## 7. Follow-on work (out of Phase 2 scope)

- Wire prepare → decontaminate → filter_7k as a one-shot orchestration in
  `scripts/prepare_data.sh` so reviewers can reproduce the cleaned corpus
  with a single command.
- Emit a hash digest (e.g. `sha256(clean.jsonl)`) into `decontam_stats.json`
  for paper-appendix reproducibility traceability.
- Add an end-to-end smoke test that mocks `datasets.load_dataset` (e.g. via
  `monkeypatch`) so the prepare scripts gain coverage too.
