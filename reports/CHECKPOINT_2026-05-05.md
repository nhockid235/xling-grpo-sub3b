# Checkpoint — Resume Guide (2026-05-05)

**Mục đích:** Nếu usage limits hit 90%, đây là tài liệu để resume từ đúng vị trí khi limits reset.

---

## Tóm tắt 30 giây

Project `xling-grpo-sub3b` đang ở **giữa Phase 3 (Trainer integration)**. Đã hoàn thành 5/7 tasks. Còn lại Task 6 (Glue scripts + paper LaTeX) và Task 7 (Validation pytest sweep). Skeleton + rewards + data + eval + trainers core code đã viết xong.

---

## Trạng thái 7 tasks

| # | Task | Status | Output |
|---|------|--------|--------|
| 1 | Save Phase 0 research reports | ✅ DONE | 5 files trong `reports/phase0_*.md` |
| 2 | Update CLAUDE.md với critical fixes | ✅ DONE | 5 edits applied; TRL pin `>=0.15.0,<0.16.0` |
| 3 | Phase 1 — Bootstrap skeleton | ✅ DONE | 65 files trong `xling-grpo-sub3b/` |
| 4 | Phase 2 — Implementation parallel agents | ✅ DONE | 4 agents → 4 reports `phase2_0[1-4]_*.md`; 46+ tests pass |
| 5 | Phase 3 — Trainer integration | ✅ DONE | `sft.py`, `grpo.py`, `dataset_utils.py`, `checkpoint_utils.py` + 13 tests pass |
| 6 | Phase 4 — Glue scripts + paper LaTeX | ⏳ PENDING | Need: aggregate.py impl, plot_curves.py impl, make_tables.py impl, sanity_check.py impl |
| 7 | Phase 5 — Validation pytest sweep | ⏳ PENDING | Launch 1 agent: full pytest, fix trivial failures |

---

## Files tạo/sửa trong session 2026-05-05

### Reports (lưu vĩnh viễn — đừng xoá)

```
reports/
├── README.md                                 (37 dòng — index)
├── CHECKPOINT_2026-05-05.md                  (← file này)
├── phase0_00_summary_critical_fixes.md       (195 dòng — 4 critical, 5 medium, 7 confirmed)
├── phase0_01_datasets_verification.md        (110 dòng — Agent 1)
├── phase0_02_trl_api_research.md             (196 dòng — Agent 2)
├── phase0_03_open_rs_reproduction.md         (180 dòng — Agent 3)
├── phase2_01_rewards.md                      (~1100 từ — Agent #3)
├── phase2_02_data.md                         (~750 từ — Agent #4)
├── phase2_03_eval.md                         (~1100 từ — Agent #5)
└── phase2_04_trainer_plan.md                 (Plan agent)
```

### CLAUDE.md (project root)

5 edits applied (xem `phase0_00_summary` Edit 1-5) + 1 follow-up edit cho TRL 0.15.0+ pin sau khi Plan agent verify `reward_weights` field availability.

### xling-grpo-sub3b/ (~70 files)

```
xling-grpo-sub3b/
├── pyproject.toml         ✅ deps pinned (trl>=0.15.0, datasketch, fasttext-wheel)
├── README.md, LICENSE, .gitignore
├── configs/               ✅ 9 YAML files (base, 3 sft_*, 3 grpo_*, eval, accelerate, reproduce_open_rs)
├── data/                  ✅ 5 scripts implemented (Phase 2 agent)
├── src/
│   ├── rewards/           ✅ 5 rewards implemented (Phase 2 agent) + registry
│   ├── trainers/          ✅ sft.py, grpo.py, dataset_utils.py, checkpoint_utils.py (Phase 3 main)
│   ├── eval/              ✅ runner + 5 benchmarks + lang_consistency (Phase 2 agent)
│   ├── analysis/          ⏳ aggregate.py, plot_curves.py, bootstrap.py — STUBS (Phase 4 chưa làm)
│   └── utils/             ✅ seed.py, io.py, parsing.py
├── scripts/               ⚠️ shell launchers OK, sanity_check.py + make_tables.py vẫn stub
├── tests/                 ✅ 13 test files; agent reports 46+ pass; main agent verified 29 pass
└── paper/
    ├── main.tex           ✅ ACL 2024 skeleton với TBD sections
    ├── refs.bib           ✅ 16 refs
    ├── appendix.tex       ✅ reproducibility appendix với hyperparams + datasets
    ├── figures/.gitkeep
    └── tables/            ⏳ table[1-3].tex chưa generate (cần Phase 4 + master.csv)
```

---

## Critical findings cần ghi nhớ (đừng forget)

1. **TRL pin = `>=0.15.0,<0.16.0`** (NOT 0.13.0, NOT 0.14.0). `reward_weights` field chỉ có từ 0.15.0.
2. **Open-RS base = `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`** (NOT Qwen2.5-1.5B-Instruct) cho W2 gating.
3. **Open-RS RS2 hyperparams** (target AMC23 80% ±3pp ckpt-50): G=6, max_completion=3584, T=0.7, reward `accuracy+format` weights `[1.0, 1.0]`.
4. **MGSM-Pro 47-langs KHÔNG TỒN TẠI** — backup = MGSM 10 + MSVAMP 10 + 100 hand-translated VI MATH-500.
5. **MGSM 10 langs guaranteed:** `en, es, fr, de, ru, zh, ja, th, sw, bn` (Telugu mention nhưng tag không có).
6. **AIME-2024 fields CAPITALIZED:** `Problem`, `Solution`, `Answer`, `ID`.
7. **GSM8K canonical:** `openai/gsm8k` config `main`.
8. **Reward signature:** `def f(prompts, completions, **kwargs) -> list[float]` — return scalar = silent break.
9. **Dataset GRPO column = `"prompt"`** (NOT `"text"`/`"question"`).
10. **`5CD-AI/Vietnamese-MetaMathQA` license unset** — blocker arXiv reproducibility.

---

## Tasks còn lại

### Task #6 — Phase 4 (Glue scripts + paper LaTeX)

**Files cần implement** (hiện đang `NotImplementedError`):

1. `src/analysis/aggregate.py`:
   - Walk `results/eval/*.json`, parse mỗi file ra row.
   - Schema CSV: `run_id, model, condition, seed, stage, step, benchmark, language, pass_at_1, maj_at_8, lang_consistency, avg_tokens, n_samples`.
   - Parse run_id từ filename hoặc JSON metadata.
   - Output `results/master.csv`.

2. `src/analysis/bootstrap.py`:
   - `bootstrap_ci(values, n_bootstrap=1000, confidence=0.95, seed=42) -> (mean, lower, upper)`.
   - Pure numpy, percentile method.

3. `src/analysis/plot_curves.py`:
   - 3 figures (Fig 1: per-lang transfer gap; Fig 2: lang-consistency curve; Fig 3: Pareto frontier tokens vs MGSM-vi).
   - matplotlib, ACL serif font, `pdf.fonttype=42`, 300 DPI, palette `tab10`.

4. `scripts/make_tables.py`:
   - Generate `paper/tables/table[1,2,3].tex` từ master.csv.
   - Booktabs, no vertical rules, bold best per row, bootstrap CI in cells.

5. `scripts/sanity_check.py`:
   - W3 mid-run gate: load 3 first-cell evals, check EN-VI gap >= 1pp signal.
   - Output `results/sanity_w3.json`.

**Estimate:** 1 main agent session, ~30K tokens (analysis code không phức tạp).

### Task #7 — Phase 5 (Validation)

**Mục tiêu:** Đảm bảo `pytest tests/` chạy thành công cho FULL skeleton trên Mac (no GPU required).

**Steps:**
1. Launch 1 general-purpose agent với prompt:
   - `cd xling-grpo-sub3b && pip install -e ".[dev]"` để có pytest.
   - `pytest tests/ -v` toàn bộ.
   - Fix trivial errors (missing deps, import order, edge cases).
   - Report số pass/skip/fail.
2. Save report tại `reports/phase5_validation.md`.

**Estimate:** ~10K tokens.

---

## Quick resume commands

```bash
cd /Users/vudang/PythonLab/Papper

# Sanity check current state
ls reports/ xling-grpo-sub3b/
cat reports/CHECKPOINT_2026-05-05.md   # ← bạn đang ở đây

# Re-run tests đã pass (verify state intact)
cd xling-grpo-sub3b
/usr/local/bin/python -m pytest tests/test_io.py tests/test_seed.py \
    tests/test_eval_parsing.py tests/test_dataset_utils.py \
    tests/test_grpo_reward_path.py -v
# Expected: 29 passed

# Continue Task 6 — implement analysis tools
# (xem section "Task #6" ở trên)
```

---

## Memory items quan trọng

(Đã lưu trong session — sẽ không mất khi reset)

User profile:
- ML researcher, đang viết arXiv preprint về cross-lingual GRPO
- Compute budget $180, 8 weeks, target arXiv cs.CL
- Native Vietnamese — comments inline = VI, docstrings = EN

Workflow preferences:
- Lưu reports đầy đủ (không ngắn lại) — paper upgrade cần
- Sequential task execution (auto mode active)
- Verify mọi finding trước khi assert

---

## Ai đang chờ?

Khi resume:
1. Đọc file này (đã có context đầy đủ).
2. Tiếp tục với **Task #6 — Phase 4 (Glue scripts + paper LaTeX)**.
3. Sau đó **Task #7 — Phase 5 (Validation)** để đảm bảo skeleton runnable.
4. Tổng kết toàn bộ project trong final report.
