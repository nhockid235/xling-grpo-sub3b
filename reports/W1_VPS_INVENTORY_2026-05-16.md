# W1 VPS → Local Inventory (2026-05-16, before VPS destruction)

This file is the final manifest of artifacts pulled back from the
A100 VPS at `root@202.214.223.66:22408` (vast.ai) after the W1 paper-revision
work-stream completed.

## Local repo state

| Path | What | Size |
|---|---|---|
| `paper/main.tex` | Updated manuscript (A4 multi-seed + bootstrap CI) | 26 KB |
| `paper/ieee/main.tex` | IEEE Access version, same content | 36 KB |
| `paper/tables/table_main_v2.tex` | Regenerated 3-seed × 4 arms table | 1.0 KB |
| `paper/tables/table_delta_v2.tex` | Regenerated Δ vs base table | 0.7 KB |
| `paper/rebuttal_prep.md` | Point-by-point response (Issues #1–6 filled) | 16 KB |
| `paper/ieee/rebuttal_letter_template.md` | Cover letter draft for IEEE Access | 5 KB |
| `paper/figures/*.pdf` | 9 figures from prior version (need regen with W1.7 numbers; see TODO) | — |
| `VERIFICATION.md` | Component verification record (A4 multi-seed marked verified) | 4 KB |
| `CHANGELOG.md` | W1 fixes logged under `[Unreleased]` | 3 KB |
| `reports/W1_2026-05-15_VPS_EXECUTION.md` | Full execution log (env setup, bugs found, fixes, results) | 5 KB |
| `reports/W1_VPS_INVENTORY_2026-05-16.md` | This file | — |
| `reports/vps_setup/` | install/fix scripts + install.log (env reproducibility) | 100 KB |
| `scripts/w17_reeval_amc23_majfix.sh` | Re-eval all LoRAs with post-fix maj@4 | — |
| `scripts/w18_eval_openrs.sh` | Eval Open-RS2 public ckpt + base distill | — |
| `scripts/w19_train_a4_multiseed.sh` | Train A4 seeds 123 + 7 | — |
| `scripts/w110_bootstrap_ci.py` | Bootstrap 95% CI computation | — |
| `scripts/w20_update_paper_tables.py` | Regen tables from manifest | — |
| `scripts/w_master_chain.sh` | Master orchestration (W1.7 → W1.8 → W1.9) | — |

## Data artifacts on local

### Eval JSONs: 137 files (matches VPS) — `results/eval/`

| Bucket | Dirs |
|---|---|
| W1.7 post-fix AMC23 (new) | `w17_base_distill_v2/`, `w17_reproduce_openrs_rs2_{42,123,7}_v2/`, `w17_a2_vi_{42,123,7}_v2/`, `w17_a3_enlang_{42,123,7}_v2/` |
| W1.8 Open-RS2 + base diagnostics (new) | `openrs2_public/`, `base_distill/` |
| W1.9 A4 multi-seed step-50 (new) | `a4_const_bias_{123,7}_step50/` |
| W1.9 chain log (new) | `w17_chain.log`, `w17_*.log`, `w18_run.log`, `w18_*.log` |
| Pre-W1 (paper v2 published) | All other dirs (a2_vi_*, a3_enlang_*, etc.) — unchanged |

### LoRA adapters: 34 `adapter_model.safetensors` (~17 MB each) — `results/grpo/`

| Arm | Seed 42 | Seed 123 | Seed 7 | Notes |
|---|---|---|---|---|
| A1 (`reproduce_openrs_rs2`) | ✓ | ✓ | ✓ | All from pre-W1; W1.7 re-eval used these |
| A2 (`a2_vi`) | ✓ | ✓ | ✓ | All from pre-W1 |
| A3 (`a3_enlang`) | ✓ | ✓ | ✓ | All from pre-W1 |
| A4 (`a4_const_bias`) | ⚠️ EMPTY DIR | ✓ NEW (W1.9) | ✓ NEW (W1.9) | seed=42 LoRA was lost between training and now; eval JSON preserved |

### Training metadata (new from W1.9)

- `results/grpo/a4_const_bias_123/wandb/` — 2.4 MB offline wandb run (training curves)
- `results/grpo/a4_const_bias_7/wandb/` — 2.4 MB offline wandb run
- `results/grpo/w19_chain.log` — full chain log
- `results/grpo/w19_*.log` — chain restart logs

## What was NOT downloaded (intentionally)

- `/workspace/.hf_home/` — HF cache (3.4 GB models). Public, available on HF Hub.
- `**/merged/` directories — base + LoRA merged for vLLM serving (3.4 GB each × 10 = ~34 GB). Regenerable from base + adapter.
- `**/optimizer.pt`, `**/scheduler.pt`, `**/rng_state.pth`, `**/training_args.bin` — large training state files for resume. Not needed for paper.
- `~/.cache/pip` — pip wheel cache (~1.5 MB).
- `~/.env_papper` — contains HF token; never to be downloaded to repo.

## Verification commands

```bash
# Eval JSON count (137 expected)
find /Users/vudang/PythonLab/Papper/xling-grpo-sub3b/results/eval -name "*.json" | wc -l

# LoRA adapter count (34 expected, including legacy copies)
find /Users/vudang/PythonLab/Papper/xling-grpo-sub3b/results/grpo -name "adapter_model.safetensors" | wc -l

# Table regeneration smoke test
cd /Users/vudang/PythonLab/Papper/xling-grpo-sub3b && python3 scripts/w20_update_paper_tables.py --print-only | grep -E "^\[|n_seeds_found"

# Pytest sanity
python3 -m pytest tests/test_eval_parsing.py tests/test_rewards_correctness.py tests/test_make_tables.py tests/test_io.py -q
```

## Safe to destroy VPS

All paper-critical artifacts pulled to local. VPS can be destroyed via vast.ai
web UI or `vast destroy <instance_id>`. Total VPS rental: ~17 hours @ $1.5-2/hr =
~$25-34.

## Recommended next steps (paper revision)

1. **Recompile PDFs**: `cd paper && tectonic main.tex`; `cd paper/ieee && bash build.sh`.
2. **Visually verify** Table 1 + Table 2 in both PDFs show the new numbers (post-fix maj@4, A4 3-seed).
3. **Regenerate figures** with new AMC23 numbers — `paper/figures/make_figures_v2.py` reads from `results/eval/` (currently references old dirs; needs minor update to point to `w17_*` for AMC23 + `a4_const_bias_{123,7}_step50` for A4).
4. (Optional) Re-archive on Zenodo as new version of DOI 10.5281/zenodo.20061328.
5. **Wait for reviewer feedback** → submit revision with rebuttal_prep.md content + updated PDF.
