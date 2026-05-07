# Phase 6 — Polish Additions (2026-05-06)

**Status:** ✅ 4 follow-up tasks complete; **108 tests pass / 4 skip / 0 fail**.

---

## Mục đích

Sau Phase 5 (build complete), session tiếp tục thêm các polish items quan trọng phát hiện
qua review:

1. **AMC23 eval adapter** — gap critical: AMC23 là gating benchmark cho W2 reproduction
   nhưng chưa có trong eval pipeline.
2. **Pre-flight check script** — verify env trước khi pay $$ cho cloud GPU.
3. **Paper main.tex content scaffolding** — TBD placeholders → actual section
   structure with boxed claim, related-work differentiation, method equations,
   results paragraphs.
4. **GitHub Actions CI** — auto pytest on push/PR (no GPU subset).

---

## Tasks completed

### Task 8 — AMC23 eval adapter (CRITICAL fix)

**Gap discovered:** `scripts/reproduce_open_rs.sh` script luôn evaluates `math500 aime2024`
nhưng **không có AMC23** — đây chính là gating benchmark target 80% ±3pp.

**Research (huggingface-skills agent):**
- HF dataset chính xác: `knoveleng/AMC-23` (with hyphen).
- Open-RS evaluate.py reference đúng repo này (verified via WebFetch).
- 40 problems, single `train` split, AoPS 2023 AMC 12A+12B.
- Schema: `{id, problem, question, answer, url}`.
- License unspecified — paper citation phải note "AoPS wiki, redistributed".

**Files modified/created:**
- `src/eval/amc23.py` (NEW, 130 lines) — evaluate() với pass@1 + maj@4. Pattern matches
  aime.py: injectable `dataset` param cho test, `_amc_match()` numeric→math-verify
  fallback, `_maybe_with_seed()` cho rollout determinism.
- `configs/eval.yaml` — thêm `amc23` benchmark entry.
- `src/eval/runner.py` — register `amc23` trong `_BENCHMARK_MODULES` + dispatch path.
- `scripts/reproduce_open_rs.sh` — eval `amc23 math500 aime2024` thay vì `math500 aime2024`.
- `tests/test_eval_amc23.py` (NEW) — 7 tests: schema compliance, partial correct,
  n_samples limit, metadata.hf_dataset, maj@4 rotation logic, runner dispatch,
  empty dataset.

**Tests:** 7/7 passed.

---

### Task 9 — Pre-flight check script

**Mục đích:** Run trước `bash scripts/reproduce_open_rs.sh` hoặc `train_grpo.sh` để
catch missing deps / GPU / disk / fastText / HF / wandb auth — tránh job crash sau
20 phút training.

**File:** `scripts/preflight.sh` (NEW, executable).

**Checks:**
- Python version (warn nếu khác 3.11).
- Disk free ≥ 30GB (fail), ≥ 60GB (pass), 30-60 (warn) — cross-platform via Python `shutil.disk_usage`.
- RAM ≥ 32GB (warn nếu thiếu).
- GPU via `nvidia-smi` — VRAM ≥ 70GB (pass), 40-70 (warn LoRA fallback), <40 (fail).
- Python deps: 15 packages với version pin verify (torch, transformers, trl, vllm,
  accelerate, peft, datasets, sympy, math-verify, fasttext, datasketch, xxhash,
  wandb, pandas, matplotlib).
- `data/raw/lid.176.bin` ≥ 100MB.
- HF Hub reachability via `HfApi.list_repo_files`.
- `HF_TOKEN` set (cần cho Llama-3.2 gated).
- Wandb authenticated.

**Output:** colored pass/warn/fail per check, final verdict + exit code (0 OK, 1 fail).

**Smoke test (Mac CPU):** correctly fails với 5 missing hard deps (transformers, trl, vllm,
accelerate, peft) — exit 1 như expected.

---

### Task 10 — Paper main.tex content scaffolding

**Trước:** TBD placeholders ở mọi section — khi viết paper W7 phải refresh from
zero context.

**Sau:** Concrete structure với:
- **Abstract** template với 4-5 sentence pattern + headline @ sentence 3 placeholder.
- **Intro paragraph 1** — concise problem statement với 3 cited refs.
- **Boxed main claim** (`\boxedclaim{...}` macro mới) — quantitative pattern với TBD% values.
- **Intro paragraph 2** — motivation cho sub-3B + 6 langs (4 scripts × 3 resource tiers).
- **Intro paragraph 3** — explicit differentiation từ Open-RS / M-Thinker / Cross-lingual Collapse.
- **Contributions** numbered list (3 items) — paper convention.
- **Related Work** 2 paragraphs: GRPO+small models, cross-lingual reasoning. Each với
  explicit "differs in (i)/(ii)/(iii)" wrt closest concurrent paper.
- **Method §3** với:
  - 3.1 Training pipeline diagram (text)
  - 3.2 Reward functions với LaTeX equations cho R1-R5 (đặc biệt R5 với 3-case piecewise)
  - 3.3 Three conditions với `\condA{}`, `\condB{}`, `\condC{}` macros
  - 3.4 Models + seeds
- **Experiments §4**:
  - 4.1 Decontamination spec (8-gram MinHashLSH threshold 0.5)
  - 4.2 Eval setup (vLLM 0.7.2, greedy + maj@N)
  - 4.3 Headline benchmarks list
  - 4.4 Open-RS reproduction paragraph với target band [77, 83]
  - Table inputs (3 tables)
  - Figure inputs (3 figures với captions placeholder)
  - 4 result subsections: transfer / EN sanity / lang dynamics / Pareto
- **Discussion** với 3 bullet questions (training language matter? R5 mechanism?
  MT pipeline implications?)
- **Limitations** 5 specific items (scale, langs, MT quality, single SFT recipe,
  single seed default) — anti-boilerplate per CLAUDE.md.
- **Conclusion** placeholder với 1-paragraph spec.

**Macros added:** `\condA`, `\condB`, `\condC` (typeset conditions consistently),
`\boxedclaim{}` (highlights main claim with tcolorbox).

**File:** `paper/main.tex` (rewritten, 200+ lines vs ~80 trước).

---

### Task 11 — GitHub Actions CI

**File:** `.github/workflows/test.yml` (NEW).

**Jobs:**
1. **pytest** trên Python 3.11 Ubuntu, install lightweight deps only (pyyaml, datasets,
   datasketch, math-verify, sympy, pandas, matplotlib + pytest). KHÔNG cài
   torch/transformers/trl/vllm (quá nặng cho free CI minutes).
2. Test subset: 16 file whitelist, skip những test cần torch/transformers/trl
   (rewards correctness OK vì math-verify đủ; trainer integration tests OK vì lazy
   import; chỉ trainer smoke tests skip).
3. **lint** job với ruff, `continue-on-error: true` (warn only, không block PR).

**Coverage:** placeholder `pytest --cov=src --cov-report=xml` cho codecov upload
(disabled until codecov token added, comment đã ghi).

---

## Tổng kết Phase 6

| Metric | Trước | Sau |
|---|---|---|
| Tests | 105 (101 pass, 4 skip, 0 fail) | **112 (108 pass, 4 skip, 0 fail)** |
| Files trong xling-grpo-sub3b/ | 159 | **165+** |
| Eval benchmarks | 5 | **6 (added AMC23)** |
| CI workflow | none | **GitHub Actions configured** |
| Pre-flight check | none | **scripts/preflight.sh ready** |
| Paper structure | TBD placeholders | **section-by-section scaffolding** |

---

## Files modified/created (phase 6)

```
xling-grpo-sub3b/
├── src/eval/
│   ├── amc23.py                     [NEW] AMC23 adapter
│   └── runner.py                    [MODIFIED] register amc23 dispatch
├── configs/
│   └── eval.yaml                    [MODIFIED] amc23 benchmark entry
├── scripts/
│   ├── preflight.sh                 [NEW] env verification
│   └── reproduce_open_rs.sh         [MODIFIED] eval amc23 trong gating
├── tests/
│   └── test_eval_amc23.py           [NEW] 7 tests cho AMC23 adapter
├── paper/
│   └── main.tex                     [REWRITTEN] section scaffolding
└── .github/workflows/
    └── test.yml                     [NEW] CI workflow
```

---

## Còn lại sau Phase 6

Các tasks còn lại đều là **execution** (W2-8 timeline), KHÔNG phải build:
1. W2: download datasets thật, decontaminate, reproduce Open-RS RS2
2. W3: train first 3 cells (qwen15b × 3 conditions)
3. W3 sanity gate
4. W4-5: train remaining 6 cells
5. W6: eval all on 6 benchmarks (gsm8k + math500 + aime2024 + **amc23** + mgsm + msvamp)
6. W7: aggregate → master.csv → tables/figures → paper draft fill TBD
7. W8: polish + arXiv submit

Pipeline build complete. Repo sẵn sàng để execute.
