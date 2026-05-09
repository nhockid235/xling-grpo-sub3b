# Verification Record

This file documents the human-verification status of components in this
project, in compliance with conference and journal AI-disclosure policies.

The author (Vu Dang) is solely responsible for the correctness of the work.
This document records what was verified, by whom, and when.

## How to read this document

| Status | Meaning |
|---|---|
| ✅ Verified | Re-run / re-checked by the author independently |
| 🟡 Spot-checked | Author reviewed samples, not exhaustive |
| ⏳ Pending | Verification scheduled but not yet performed |
| ❌ Not applicable | Component is auto-generated and verified by tests |

## Components

### Code

| Component | Verification | Method |
|---|---|---|
| Reward functions (`src/rewards/`) | ✅ Verified | Unit tests + manual code review |
| Evaluation adapters (`src/eval/`) | ✅ Verified | Unit tests + spot check on benchmark JSONs |
| Trainer wrapper (`src/trainers/grpo.py`) | ✅ Verified | End-to-end training run produces expected `trainer_state.json` |
| Dataset preparation (`data/prepare_*.py`) | ✅ Verified | Output JSONL inspected for first 10 records |
| Test suite | ✅ Verified | 104 passing (4 skipped due to missing fastText model) |

### Experimental results

| Result | Verification | Source |
|---|---|---|
| A1 seed=42 (Phase 8) | ✅ Verified | Self-run on Vast.ai 2026-05-06 |
| A1 seed=123 (Phase 9.2) | ✅ Verified | Self-run on Vast.ai 2026-05-08 |
| A1 seed=7 (Phase 9.2) | ✅ Verified | Self-run on Vast.ai 2026-05-08 |
| A2 multi-seed | 🟡 In progress | Phase 9.2 currently running |
| A3 multi-seed | ⏳ Pending | Phase 9.2 sequential |
| A4 ablation | ⏳ Pending | Phase 9.3 planned |
| MGSM multilingual eval | ⏳ Pending | Phase 9.4 planned |

### Citations and prior art

| Citation | Verification | Notes |
|---|---|---|
| Open-RS (Knoveleng 2025, arXiv:2503.16219) | ✅ Verified | Paper read, hyperparameters and reported numbers cross-checked |
| DeepSeek-R1 (DeepSeek-AI 2025, arXiv:2501.12948) | ✅ Verified | Paper read, base checkpoint identity confirmed |
| DeepSeekMath (Shao 2024, arXiv:2402.03300) | ✅ Verified | Cited GRPO origin paper |
| Cross-lingual Collapse (arXiv:2506.05850) | ✅ Verified | Cited as concurrent work |
| MGSM (Shi 2022, arXiv:2210.03057) | ✅ Verified | Benchmark identity confirmed |
| Zheng 2018 (arXiv:1804.06459) | ✅ Verified | Prior-art citation for constant-bias reward in PPO |
| All other refs in `paper/refs.bib` | ✅ Verified | arXiv IDs spot-checked |

### Manuscript

| Section | Verification | Method |
|---|---|---|
| Abstract | ✅ Verified | Numbers cross-checked against `results/eval/*.json` |
| Methods (§3) | ✅ Verified | Hyperparameters cross-checked against `configs/*.yaml` |
| Results (§4) | 🟡 Partial | Phase 8 single-seed numbers verified; multi-seed pending |
| Mechanism hypothesis (§7) | 🟡 Speculative | Marked as hypothesis pending A4 ablation |
| Limitations (§8) | ✅ Verified | Author identified each limitation explicitly |

## AI assistance scope

The following components were drafted with AI assistance (Anthropic Claude
via the Claude Code CLI), then reviewed and modified by the author:

- LaTeX manuscript composition and formatting
- Source code skeletons for reward functions and evaluation adapters
- Bash/Python scripts for training pipeline orchestration
- Documentation, README, configuration files
- Bibliography formatting

The following were performed by the author without AI authorship:

- Experimental design (which arms, which seeds, which hyperparameters)
- Hardware procurement and budget management (Vast.ai cloud rentals)
- Decision to terminate or restart training runs
- Interpretation of empirical results and selection of findings
- Final approval of all manuscript claims

## Update protocol

This file is updated whenever:
- A new experimental result is added to the manuscript
- A new citation is added to `refs.bib`
- A new code component is introduced
- A claim previously marked Pending is verified

Last updated: 2026-05-09 (Phase 9.2 in progress).

## Pipeline run @ 2026-05-09 04:19:50 UTC

- Total artifacts checksummed: **123**
- Trainer states valid: **8/8**
- Eval JSONs valid: **6/6**
- Numerical claims in paper: 6 (unmatched: 6)

### Eval JSONs present

- `results/eval/reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_aime2024.json` — pass@1=0.2333, n=30, responses=True
- `results/eval/reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_amc23.json` — pass@1=0.4500, n=40, responses=True
- `results/eval/reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_math500.json` — pass@1=0.6160, n=500, responses=True
- `results/eval/reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_aime2024.json` — pass@1=0.1000, n=30, responses=True
- `results/eval/reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_amc23.json` — pass@1=0.6750, n=40, responses=True
- `results/eval/reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_math500.json` — pass@1=0.5860, n=500, responses=True

### Trainer states present

- `results/grpo/a2_vi_123/checkpoint-50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/a2_vi_123/keep_step50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/reproduce_openrs_rs2_123/checkpoint-100/trainer_state.json` — step 100/100, n_logs=20
- `results/grpo/reproduce_openrs_rs2_123/checkpoint-50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/reproduce_openrs_rs2_123/keep_step50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/reproduce_openrs_rs2_7/checkpoint-100/trainer_state.json` — step 100/100, n_logs=20
- `results/grpo/reproduce_openrs_rs2_7/checkpoint-50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/reproduce_openrs_rs2_7/keep_step50/trainer_state.json` — step 50/100, n_logs=10

Manifest checksum file: `reports/phase9_runs/manifest.csv`

## Pipeline run @ 2026-05-09 04:20:41 UTC

- Total artifacts checksummed: **141**
- Trainer states valid: **8/8**
- Eval JSONs valid: **24/24**
- Numerical claims in paper: 6 (unmatched: 4)

### Eval JSONs present

- `results/eval/a2_vi_42_step50/a2_vi_42_step50_aime2024.json` — pass@1=0.1667, n=30, responses=True
- `results/eval/a2_vi_42_step50/a2_vi_42_step50_amc23.json` — pass@1=0.5250, n=40, responses=True
- `results/eval/a2_vi_42_step50/a2_vi_42_step50_math500.json` — pass@1=0.6020, n=500, responses=True
- `results/eval/a3_enlang_42_step50/a3_enlang_42_step50_aime2024.json` — pass@1=0.2333, n=30, responses=True
- `results/eval/a3_enlang_42_step50/a3_enlang_42_step50_amc23.json` — pass@1=0.5250, n=40, responses=True
- `results/eval/a3_enlang_42_step50/a3_enlang_42_step50_math500.json` — pass@1=0.6060, n=500, responses=True
- `results/eval/base_deepseek_r1_distill_15b/base_deepseek_r1_distill_15b_aime2024.json` — pass@1=0.1667, n=30, responses=True
- `results/eval/base_deepseek_r1_distill_15b/base_deepseek_r1_distill_15b_amc23.json` — pass@1=0.4500, n=40, responses=True
- `results/eval/base_deepseek_r1_distill_15b/base_deepseek_r1_distill_15b_math500.json` — pass@1=0.5400, n=500, responses=True
- `results/eval/base_v3_openrs_eval/base_v3_aime2024.json` — pass@1=0.2667, n=30, responses=True
- `results/eval/base_v3_openrs_eval/base_v3_amc23.json` — pass@1=0.5000, n=40, responses=True
- `results/eval/base_v3_openrs_eval/base_v3_math500.json` — pass@1=0.5940, n=500, responses=True
- `results/eval/ckpt50_v3_openrs_eval/ckpt50_v3_aime2024.json` — pass@1=0.1000, n=30, responses=True
- `results/eval/ckpt50_v3_openrs_eval/ckpt50_v3_amc23.json` — pass@1=0.5750, n=40, responses=True
- `results/eval/ckpt50_v3_openrs_eval/ckpt50_v3_math500.json` — pass@1=0.5880, n=500, responses=True
- `results/eval/reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_aime2024.json` — pass@1=0.2333, n=30, responses=True
- `results/eval/reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_amc23.json` — pass@1=0.4500, n=40, responses=True
- `results/eval/reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_math500.json` — pass@1=0.6160, n=500, responses=True
- `results/eval/reproduce_openrs_rs2_42_step50_v2/reproduce_openrs_rs2_42_step50_v2_aime2024.json` — pass@1=0.1333, n=30, responses=True
- `results/eval/reproduce_openrs_rs2_42_step50_v2/reproduce_openrs_rs2_42_step50_v2_amc23.json` — pass@1=0.3500, n=40, responses=True
- `results/eval/reproduce_openrs_rs2_42_step50_v2/reproduce_openrs_rs2_42_step50_v2_math500.json` — pass@1=0.5240, n=500, responses=True
- `results/eval/reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_aime2024.json` — pass@1=0.1000, n=30, responses=True
- `results/eval/reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_amc23.json` — pass@1=0.6750, n=40, responses=True
- `results/eval/reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_math500.json` — pass@1=0.5860, n=500, responses=True

### Trainer states present

- `results/grpo/a2_vi_123/checkpoint-50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/a2_vi_123/keep_step50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/reproduce_openrs_rs2_123/checkpoint-100/trainer_state.json` — step 100/100, n_logs=20
- `results/grpo/reproduce_openrs_rs2_123/checkpoint-50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/reproduce_openrs_rs2_123/keep_step50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/reproduce_openrs_rs2_7/checkpoint-100/trainer_state.json` — step 100/100, n_logs=20
- `results/grpo/reproduce_openrs_rs2_7/checkpoint-50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/reproduce_openrs_rs2_7/keep_step50/trainer_state.json` — step 50/100, n_logs=10

Manifest checksum file: `reports/phase9_runs/manifest.csv`

## Pipeline run @ 2026-05-09 08:48:22 UTC

- Total artifacts checksummed: **169**
- Trainer states valid: **9/9**
- Eval JSONs valid: **27/27**
- Numerical claims in paper: 6 (unmatched: 4)

### Eval JSONs present

- `results/eval/a2_vi_123_step50/a2_vi_123_step50_aime2024.json` — pass@1=0.1667, n=30, responses=True
- `results/eval/a2_vi_123_step50/a2_vi_123_step50_amc23.json` — pass@1=0.6250, n=40, responses=True
- `results/eval/a2_vi_123_step50/a2_vi_123_step50_math500.json` — pass@1=0.6200, n=500, responses=True
- `results/eval/a2_vi_42_step50/a2_vi_42_step50_aime2024.json` — pass@1=0.1667, n=30, responses=True
- `results/eval/a2_vi_42_step50/a2_vi_42_step50_amc23.json` — pass@1=0.5250, n=40, responses=True
- `results/eval/a2_vi_42_step50/a2_vi_42_step50_math500.json` — pass@1=0.6020, n=500, responses=True
- `results/eval/a3_enlang_42_step50/a3_enlang_42_step50_aime2024.json` — pass@1=0.2333, n=30, responses=True
- `results/eval/a3_enlang_42_step50/a3_enlang_42_step50_amc23.json` — pass@1=0.5250, n=40, responses=True
- `results/eval/a3_enlang_42_step50/a3_enlang_42_step50_math500.json` — pass@1=0.6060, n=500, responses=True
- `results/eval/base_deepseek_r1_distill_15b/base_deepseek_r1_distill_15b_aime2024.json` — pass@1=0.1667, n=30, responses=True
- `results/eval/base_deepseek_r1_distill_15b/base_deepseek_r1_distill_15b_amc23.json` — pass@1=0.4500, n=40, responses=True
- `results/eval/base_deepseek_r1_distill_15b/base_deepseek_r1_distill_15b_math500.json` — pass@1=0.5400, n=500, responses=True
- `results/eval/base_v3_openrs_eval/base_v3_aime2024.json` — pass@1=0.2667, n=30, responses=True
- `results/eval/base_v3_openrs_eval/base_v3_amc23.json` — pass@1=0.5000, n=40, responses=True
- `results/eval/base_v3_openrs_eval/base_v3_math500.json` — pass@1=0.5940, n=500, responses=True
- `results/eval/ckpt50_v3_openrs_eval/ckpt50_v3_aime2024.json` — pass@1=0.1000, n=30, responses=True
- `results/eval/ckpt50_v3_openrs_eval/ckpt50_v3_amc23.json` — pass@1=0.5750, n=40, responses=True
- `results/eval/ckpt50_v3_openrs_eval/ckpt50_v3_math500.json` — pass@1=0.5880, n=500, responses=True
- `results/eval/reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_aime2024.json` — pass@1=0.2333, n=30, responses=True
- `results/eval/reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_amc23.json` — pass@1=0.4500, n=40, responses=True
- `results/eval/reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_math500.json` — pass@1=0.6160, n=500, responses=True
- `results/eval/reproduce_openrs_rs2_42_step50_v2/reproduce_openrs_rs2_42_step50_v2_aime2024.json` — pass@1=0.1333, n=30, responses=True
- `results/eval/reproduce_openrs_rs2_42_step50_v2/reproduce_openrs_rs2_42_step50_v2_amc23.json` — pass@1=0.3500, n=40, responses=True
- `results/eval/reproduce_openrs_rs2_42_step50_v2/reproduce_openrs_rs2_42_step50_v2_math500.json` — pass@1=0.5240, n=500, responses=True
- `results/eval/reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_aime2024.json` — pass@1=0.1000, n=30, responses=True
- `results/eval/reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_amc23.json` — pass@1=0.6750, n=40, responses=True
- `results/eval/reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_math500.json` — pass@1=0.5860, n=500, responses=True

### Trainer states present

- `results/grpo/a2_vi_123/checkpoint-100/trainer_state.json` — step 100/100, n_logs=20
- `results/grpo/a2_vi_123/checkpoint-50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/a2_vi_123/keep_step50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/reproduce_openrs_rs2_123/checkpoint-100/trainer_state.json` — step 100/100, n_logs=20
- `results/grpo/reproduce_openrs_rs2_123/checkpoint-50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/reproduce_openrs_rs2_123/keep_step50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/reproduce_openrs_rs2_7/checkpoint-100/trainer_state.json` — step 100/100, n_logs=20
- `results/grpo/reproduce_openrs_rs2_7/checkpoint-50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/reproduce_openrs_rs2_7/keep_step50/trainer_state.json` — step 50/100, n_logs=10

Manifest checksum file: `reports/phase9_runs/manifest.csv`

## Pipeline run @ 2026-05-09 13:14:28 UTC

- Total artifacts checksummed: **219**
- Trainer states valid: **12/12**
- Eval JSONs valid: **30/30**
- Numerical claims in paper: 6 (unmatched: 4)

### Eval JSONs present

- `results/eval/a2_vi_123_step50/a2_vi_123_step50_aime2024.json` — pass@1=0.1667, n=30, responses=True
- `results/eval/a2_vi_123_step50/a2_vi_123_step50_amc23.json` — pass@1=0.6250, n=40, responses=True
- `results/eval/a2_vi_123_step50/a2_vi_123_step50_math500.json` — pass@1=0.6200, n=500, responses=True
- `results/eval/a2_vi_42_step50/a2_vi_42_step50_aime2024.json` — pass@1=0.1667, n=30, responses=True
- `results/eval/a2_vi_42_step50/a2_vi_42_step50_amc23.json` — pass@1=0.5250, n=40, responses=True
- `results/eval/a2_vi_42_step50/a2_vi_42_step50_math500.json` — pass@1=0.6020, n=500, responses=True
- `results/eval/a2_vi_7_step50/a2_vi_7_step50_aime2024.json` — pass@1=0.2667, n=30, responses=True
- `results/eval/a2_vi_7_step50/a2_vi_7_step50_amc23.json` — pass@1=0.5500, n=40, responses=True
- `results/eval/a2_vi_7_step50/a2_vi_7_step50_math500.json` — pass@1=0.6020, n=500, responses=True
- `results/eval/a3_enlang_42_step50/a3_enlang_42_step50_aime2024.json` — pass@1=0.2333, n=30, responses=True
- `results/eval/a3_enlang_42_step50/a3_enlang_42_step50_amc23.json` — pass@1=0.5250, n=40, responses=True
- `results/eval/a3_enlang_42_step50/a3_enlang_42_step50_math500.json` — pass@1=0.6060, n=500, responses=True
- `results/eval/base_deepseek_r1_distill_15b/base_deepseek_r1_distill_15b_aime2024.json` — pass@1=0.1667, n=30, responses=True
- `results/eval/base_deepseek_r1_distill_15b/base_deepseek_r1_distill_15b_amc23.json` — pass@1=0.4500, n=40, responses=True
- `results/eval/base_deepseek_r1_distill_15b/base_deepseek_r1_distill_15b_math500.json` — pass@1=0.5400, n=500, responses=True
- `results/eval/base_v3_openrs_eval/base_v3_aime2024.json` — pass@1=0.2667, n=30, responses=True
- `results/eval/base_v3_openrs_eval/base_v3_amc23.json` — pass@1=0.5000, n=40, responses=True
- `results/eval/base_v3_openrs_eval/base_v3_math500.json` — pass@1=0.5940, n=500, responses=True
- `results/eval/ckpt50_v3_openrs_eval/ckpt50_v3_aime2024.json` — pass@1=0.1000, n=30, responses=True
- `results/eval/ckpt50_v3_openrs_eval/ckpt50_v3_amc23.json` — pass@1=0.5750, n=40, responses=True
- `results/eval/ckpt50_v3_openrs_eval/ckpt50_v3_math500.json` — pass@1=0.5880, n=500, responses=True
- `results/eval/reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_aime2024.json` — pass@1=0.2333, n=30, responses=True
- `results/eval/reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_amc23.json` — pass@1=0.4500, n=40, responses=True
- `results/eval/reproduce_openrs_rs2_123_step50/reproduce_openrs_rs2_123_step50_math500.json` — pass@1=0.6160, n=500, responses=True
- `results/eval/reproduce_openrs_rs2_42_step50_v2/reproduce_openrs_rs2_42_step50_v2_aime2024.json` — pass@1=0.1333, n=30, responses=True
- `results/eval/reproduce_openrs_rs2_42_step50_v2/reproduce_openrs_rs2_42_step50_v2_amc23.json` — pass@1=0.3500, n=40, responses=True
- `results/eval/reproduce_openrs_rs2_42_step50_v2/reproduce_openrs_rs2_42_step50_v2_math500.json` — pass@1=0.5240, n=500, responses=True
- `results/eval/reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_aime2024.json` — pass@1=0.1000, n=30, responses=True
- `results/eval/reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_amc23.json` — pass@1=0.6750, n=40, responses=True
- `results/eval/reproduce_openrs_rs2_7_step50/reproduce_openrs_rs2_7_step50_math500.json` — pass@1=0.5860, n=500, responses=True

### Trainer states present

- `results/grpo/a2_vi_123/checkpoint-100/trainer_state.json` — step 100/100, n_logs=20
- `results/grpo/a2_vi_123/checkpoint-50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/a2_vi_123/keep_step50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/a2_vi_7/checkpoint-100/trainer_state.json` — step 100/100, n_logs=20
- `results/grpo/a2_vi_7/checkpoint-50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/a2_vi_7/keep_step50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/reproduce_openrs_rs2_123/checkpoint-100/trainer_state.json` — step 100/100, n_logs=20
- `results/grpo/reproduce_openrs_rs2_123/checkpoint-50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/reproduce_openrs_rs2_123/keep_step50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/reproduce_openrs_rs2_7/checkpoint-100/trainer_state.json` — step 100/100, n_logs=20
- `results/grpo/reproduce_openrs_rs2_7/checkpoint-50/trainer_state.json` — step 50/100, n_logs=10
- `results/grpo/reproduce_openrs_rs2_7/keep_step50/trainer_state.json` — step 50/100, n_logs=10

Manifest checksum file: `reports/phase9_runs/manifest.csv`
