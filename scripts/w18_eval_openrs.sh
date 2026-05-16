#!/bin/bash
# W1.8 — Eval Open-RS2 public checkpoint on AMC23
# Mục đích: đóng Issue #2 (eval-gap −12.9pp). Diagnose: bug pipeline hay LoRA constraint?
# Output:
#   results/eval/openrs2_public_amc23/ → AMC23 pass@1 + maj@4
#   results/eval/base_distill_amc23/    → re-eval base với fixed code
set -euo pipefail

source ~/.env_papper 2>/dev/null || true
source /venv/main/bin/activate
cd /workspace/xling-grpo-sub3b

mkdir -p results/eval/openrs2_public results/eval/base_distill

LOG=results/eval/w18_$(date -u +%Y%m%dT%H%M%SZ).log
echo "[W1.8] start $(date -u +%FT%TZ)" | tee -a "$LOG"

# 1) Eval Open-RS2 public checkpoint
echo "[W1.8] Eval knoveleng/Open-RS2 on AMC23..." | tee -a "$LOG"
python -u src/eval/runner.py \
    --checkpoint "knoveleng/Open-RS2" \
    --benchmarks amc23 \
    --config configs/eval.yaml \
    --output_dir results/eval/openrs2_public/ \
    --run_id openrs2_public_step50 \
    --seed 42 2>&1 | tee -a "$LOG"

# 2) Eval base DeepSeek-R1-Distill-Qwen-1.5B on AMC23 (re-baseline với fixed code)
echo "[W1.8] Eval base deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B on AMC23..." | tee -a "$LOG"
python -u src/eval/runner.py \
    --checkpoint "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --benchmarks amc23 \
    --config configs/eval.yaml \
    --output_dir results/eval/base_distill/ \
    --run_id base_distill_v2 \
    --seed 42 2>&1 | tee -a "$LOG"

echo "[W1.8] done $(date -u +%FT%TZ)" | tee -a "$LOG"

# Quick summary
python - <<'PY' | tee -a "$LOG"
import json, glob
for p in sorted(glob.glob("results/eval/openrs2_public/*.json") + glob.glob("results/eval/base_distill/*.json")):
    d = json.load(open(p))
    print(f"  {p}")
    print(f"    pass@1 = {d.get('pass_at_1'):.4f}  ({int(d.get('pass_at_1', 0)*40)}/40)")
    print(f"    maj@4  = {d.get('maj_at_4'):.4f}  ({int(d.get('maj_at_4', 0)*40)}/40)")
    print(f"    n_samples = {d.get('n_samples')}, n_responses = {len(d.get('responses', []))}")
PY
