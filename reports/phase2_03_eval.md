# Phase 2 — Implementation Report: Evaluation Pipeline

**Date:** 2026-05-05
**Agent:** Phase 2 implementation agent (eval)
**Scope:** `src/eval/{runner,gsm8k,math500,aime,mgsm,msvamp,lang_consistency,prompts,_common}.py` + `tests/test_eval_adapters.py`

---

## 1. Files modified / created (absolute paths)

Created:

- `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/src/eval/prompts.py` — shared prompt builder + DEFAULT_SYSTEM_PROMPT.
- `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/src/eval/_common.py` — internal helpers (extract_prediction, numeric_match, majority_vote, vllm_generate, make_metadata, etc.).
- `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/tests/test_eval_adapters.py` — 21 unit tests, no GPU/HF needed (mock LLM + in-memory datasets + monkey-patched fastText).

Replaced (NotImplementedError → real impl):

- `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/src/eval/runner.py` — vLLM dispatcher, multilingual loop, JSON output writer.
- `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/src/eval/gsm8k.py`
- `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/src/eval/math500.py`
- `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/src/eval/aime.py`
- `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/src/eval/mgsm.py`
- `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/src/eval/msvamp.py`
- `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/src/eval/lang_consistency.py`

Untouched (other agents): `src/rewards/*`, `src/trainers/*`, `data/*`, configs, scripts.

---

## 2. Per-benchmark schema mapping (HuggingFace → internal)

| Benchmark | HF id | Split | Problem field (HF) | Gold field (HF) | Gold extraction | Match strategy |
|---|---|---|---|---|---|---|
| gsm8k | `openai/gsm8k` (config `main`) | test | `question` | `answer` | `extract_gsm8k_gold` (#### regex on full answer string) | `numeric_match` (string compare after stripping commas/`$`/`+`) |
| math500 | `HuggingFaceH4/MATH-500` | test | `problem` | `answer` | direct `str()` of field | `math_verify_match` (sympy parse + verify; falls back to `numeric_match` when math-verify missing or fails) |
| aime2024 | `Maxwell-Jia/AIME_2024` | **train** (only split) | **`Problem`** (CAPS) | **`Answer`** (CAPS) | `normalize_number_string` on capital `Answer` field | `_aime_match` = numeric-first then math-verify |
| mgsm | `juletxara/mgsm` (config = lang) | test | `question` | `answer_number` | direct `str()` (numeric, locale-safe) | `numeric_match` |
| msvamp | `Mathoctopus/MSVAMP` (config = lang, falls back to upper-case) | test | `question` | `answer_number` | direct `str()` | `numeric_match` |

All adapters expose:

```python
def evaluate(model, n_samples=None, seed=42, sampling_params=None, *,
             run_id, model_path, max_tokens, temperature,
             chat_template_tokenizer=None, system_prompt=None,
             dataset=None, **kwargs) -> dict
```

`dataset` lets unit tests inject a `list[dict]` instead of touching HF Hub. Each evaluator returns the full CLAUDE.md schema (10 keys: `run_id, benchmark, language, n_samples, pass_at_1, maj_at_8, lang_consistency_rate, avg_response_tokens, responses, metadata`).

---

## 3. vLLM SamplingParams chosen + rationale

Two sets of params are built in `runner._build_sampling_params{,_maj}`, parameterized by `configs/eval.yaml :: generation`:

**Greedy (`pass_at_1`) — used by every benchmark:**
- `temperature=0.0`, `top_p=1.0` → deterministic per checkpoint (with `seed=args.seed`).
- `max_tokens=2048` (config) — matches CLAUDE.md GRPO `max_new_tokens`.
- `stop=["<|im_end|>", "</answer>"]` — Qwen2.5 chat template terminator + answer-tag close to keep responses tight (helps `avg_response_tokens` not blow up on math-rich inputs).

**Stochastic (`maj@8`, AIME only):**
- `temperature=0.7`, `top_p=0.95` (config keys `temperature_maj`, `top_p_maj` w/ defaults). Standard self-consistency setting; not too low (no diversity) nor too high (collapses on AIME).
- Per-sample seed = `base_seed + k` for k in 0..7 → reproducible voting set.

Why split: greedy is the headline metric and must be deterministic; maj@8 needs diversity. Mixing them in one SamplingParams would either kill maj diversity (T=0) or noise pass@1 (T>0).

---

## 4. Prompt construction strategy + system prompt rationale

`src/eval/prompts.py :: build_prompt`:

1. Default system prompt is **English** for ALL languages, including mgsm/msvamp non-EN.
2. If `chat_template_tokenizer` provided, calls `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` so the prompt has the model-specific markers (e.g. `<|im_start|>system\n...<|im_end|>` for Qwen).
3. Fallback string format `"System: ... \n\nUser: ... \n\nAssistant:"` is used when no tokenizer (testing path).

**Why English system prompt for all languages:** The thesis tests *transfer* of EN-trained reasoning. Switching the system prompt to the eval language confounds two variables (training-data language vs. instruction-language). Open-RS, M-Thinker, and Cross-lingual Collapse all hold the system prompt fixed, so this choice keeps comparisons apples-to-apples.

The system prompt requires `<think>...</think><answer>...</answer>`, matching the format reward (R2) used during GRPO so eval-time and train-time formatting agree.

---

## 5. lang_consistency edge cases handled

`src/eval/lang_consistency.py`:

- **Short responses** (`len(text.split()) < min_response_tokens`, default 10): excluded from denominator, counted in `n_skipped`. fastText is documented to be unreliable on <10-token snippets.
- **Empty responses array**: returns `(0.0, 0, 0)` without calling fastText.
- **Newlines confuse fastText `predict()`**: `_clean_text_for_fasttext` replaces `\n`/`\r` with spaces before predict.
- **fastText label suffix**: `__label__zh-cn` and `__label__pt-br` normalized to ISO 639-1 (`zh`, `pt`) via regex split on first non-alpha.
- **fastText predict throws** (rare unicode edge case): wrapped in try/except, increment `n_skipped`.
- **CSV row when language is unknown**: emits row with `lang_consistency_rate=None` so master.csv aggregator sees the file but doesn't try to score it.
- **Filename pattern fallback**: `{run_id}_{benchmark}_{lang}.json` parsed via regex when JSON metadata missing (`_FNAME_LANG_RE`/`_FNAME_NOLANG_RE`).
- **Model caching**: `_FASTTEXT_MODEL_CACHE` keyed by path so batch processing N files only loads the 131MB model once.

CSV output columns: `run_id, benchmark, language, n_samples, n_counted, n_skipped, lang_consistency_rate, input_file`. Slightly richer than the spec (added `n_counted, n_skipped, input_file`) since they cost nothing and help debugging.

---

## 6. Test output (verbatim)

```
============================= test session starts ==============================
platform darwin -- Python 3.13.11, pytest-9.0.3, pluggy-1.6.0 -- /Users/vudang/miniconda3/bin/python
cachedir: .pytest_cache
rootdir: /Users/vudang/PythonLab/Papper/xling-grpo-sub3b
configfile: pyproject.toml
plugins: anyio-4.12.1
collecting ... collected 21 items

tests/test_eval_adapters.py::test_gsm8k_extracts_gold_from_hash_format PASSED [  4%]
tests/test_eval_adapters.py::test_gsm8k_evaluate_basic_schema_and_correctness PASSED [  9%]
tests/test_eval_adapters.py::test_gsm8k_handles_comma_thousands_in_gold PASSED [ 14%]
tests/test_eval_adapters.py::test_math500_evaluate_schema PASSED         [ 19%]
tests/test_eval_adapters.py::test_aime_uses_capitalized_fields PASSED    [ 23%]
tests/test_eval_adapters.py::test_aime_lowercase_fields_would_raise PASSED [ 28%]
tests/test_eval_adapters.py::test_aime_maj_at_8_majority_vote PASSED     [ 33%]
tests/test_eval_adapters.py::test_mgsm_rejects_unknown_language PASSED   [ 38%]
tests/test_eval_adapters.py::test_mgsm_accepts_each_guaranteed_language PASSED [ 42%]
tests/test_eval_adapters.py::test_mgsm_telugu_rejected PASSED            [ 47%]
tests/test_eval_adapters.py::test_msvamp_rejects_unknown_language PASSED [ 52%]
tests/test_eval_adapters.py::test_msvamp_accepts_each_listed_language PASSED [ 57%]
tests/test_eval_adapters.py::test_build_prompt_default_system PASSED     [ 61%]
tests/test_eval_adapters.py::test_build_prompt_with_tokenizer_chat_template PASSED [ 66%]
tests/test_eval_adapters.py::test_normalize_number_string_strip_commas_and_dollar PASSED [ 71%]
tests/test_eval_adapters.py::test_numeric_match_float_equivalence PASSED [ 76%]
tests/test_eval_adapters.py::test_majority_vote PASSED                   [ 80%]
tests/test_eval_adapters.py::test_normalize_fasttext_label PASSED        [ 85%]
tests/test_eval_adapters.py::test_parse_filename_with_lang PASSED        [ 90%]
tests/test_eval_adapters.py::test_parse_filename_without_lang PASSED     [ 95%]
tests/test_eval_adapters.py::test_process_files_with_mock_fasttext PASSED [100%]

============================== 21 passed in 0.13s ==============================
```

Full suite (eval-related, ignoring rewards/decontaminate which other agents own): **37/37 passed**.

---

## 7. Risks / open issues

1. **vLLM tokenizer ≠ HuggingFace tokenizer (CLAUDE.md pitfall #7).** I load the HF tokenizer separately via `AutoTokenizer.from_pretrained(checkpoint)` to apply the chat template *before* feeding to vLLM. vLLM internally re-tokenizes with its fast tokenizer; whitespace/special-token edge cases can desynchronize. Mitigation: in `runner.py` I pass already-templated *strings*, not pre-tokenized ids, so vLLM's tokenizer is only used for input ids — same path as production training. Still, before the W2 reproduction gate, run a tokenization-parity check (compare HF `tokenizer(prompt).input_ids` vs `vllm_engine.tokenizer.encode(prompt)`) on a 10-sample fixture and fail loudly on mismatch.

2. **Math-Verify edge cases (CLAUDE.md pitfall #4).** Sympy ≥1.13 breaks Math-Verify; pyproject pins `sympy<1.13`. My `math_verify_match` lazy-imports and falls back to `numeric_match` on any exception, so missing math-verify does not crash MATH-500 eval — but pass@1 will be biased low for symbolic answers (e.g., `\frac{1}{2}`). Action: confirm math-verify installs on RunPod A100 image during W1.

3. **MGSM-Pro fallback (CLAUDE.md pitfall #2).** Only 9 langs of mgsm-pro exist on Hub (per phase0_01); my pipeline currently supports MGSM (10 langs) + MSVAMP (10 langs). Adding mgsm-pro is a config-only change (`benchmarks.mgsm_pro: ...`) plus a thin adapter; deferred until McGill releases the 47-lang version or paper deadline forces the hand-translated VI MATH-500 backup.

4. **AIME field-caps regression risk (pitfall #18 in phase 0 summary).** Tests `test_aime_uses_capitalized_fields` + `test_aime_lowercase_fields_would_raise` lock the contract. Any future code that lowercases AIME field names will fail in CI.

5. **MSVAMP config name uppercasing.** Mathoctopus' HF release inconsistently uses `bn` vs `BN` config names across snapshots. I try lowercase first then uppercase in a try/except. If both fail, an exception bubbles up — preferable to silent empty result. Verify which case is current at W1.

6. **`maj@8` with vLLM `n=8` vs loop:** I implemented as a Python loop over K=8 rebuilt SamplingParams (with seed = base+k). vLLM supports `SamplingParams(n=8)` natively, which is more efficient. Reason for loop: per-sample reproducible seeds and easier integration with mock LLM in tests. Optimization deferred until profiling shows AIME eval >5 min (30 problems × 8 = 240 generations is small).

7. **`fasttext` import vs `fasttext-langdetect`.** `pyproject.toml` pins `fasttext>=0.9.2`. On macOS, `fasttext` build sometimes fails (needs `pybind11`). Lang-consistency CLI handles ImportError with a clear message but cannot run without it. Option: the lighter `fasttext-wheel` or `lid.176.ftz` (smaller). Decide before W7 paper deadline.

8. **`responses` array disk usage (~5GB across full sweep, CLAUDE.md pitfall #8):** every eval JSON keeps full text. Acceptable per CLAUDE.md; mitigated by keeping JSON in `results/eval/` outside git.

---

## 8. Module API summary (for downstream `aggregate.py`)

Output JSON conforms exactly to CLAUDE.md schema. File naming:

- `{run_id}_{benchmark}.json` for gsm8k, math500, aime2024.
- `{run_id}_{benchmark}_{lang}.json` for mgsm, msvamp.

`run_id` is auto-derived from checkpoint path: `results/grpo/qwen15b_en_42/checkpoint-500` → `qwen15b_en_42`. Override via `--run_id` flag.

CLI:

```bash
python src/eval/runner.py \
    --checkpoint results/grpo/qwen15b_en_42/checkpoint-500 \
    --benchmarks gsm8k math500 aime2024 mgsm msvamp \
    --output_dir results/eval/

# Post-hoc lang consistency
python src/eval/lang_consistency.py \
    --input "results/eval/*_mgsm_*.json" "results/eval/*_msvamp_*.json" \
    --output results/lang_consist.csv \
    --fasttext_model data/raw/lid.176.bin
```

Both honor configs/eval.yaml via `extends: base.yaml`.
