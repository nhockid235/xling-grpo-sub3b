#!/bin/bash
# W1.9 — Train A4 (constant-bias ablation) seeds 123 + 7.
# Mục đích: đóng Issue #4 (A4 single-seed limitation).
# Sequential: seed 123 first, then seed 7. Each ~3-4h on A100 80GB.
# Train script tự động eval AMC23 + MATH-500 + AIME-2024 sau khi train xong.
set -euo pipefail

source ~/.env_papper 2>/dev/null || true
source /venv/main/bin/activate
cd /workspace/xling-grpo-sub3b

LOG=results/grpo/w19_$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p results/grpo
echo "[W1.9] start $(date -u +%FT%TZ)" | tee -a "$LOG"

for SEED in 123 7; do
  echo "[W1.9] ============ A4 seed=$SEED ============" | tee -a "$LOG"
  bash scripts/train_a2_a3.sh a4 --seed "$SEED" 2>&1 | tee -a "$LOG"
done

echo "[W1.9] done $(date -u +%FT%TZ)" | tee -a "$LOG"
