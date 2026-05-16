#!/bin/bash
# Master chain: W1.8 (eval Open-RS2 + base) → W1.7 (re-eval all 11 ckpts AMC23) → W1.9 (train A4 multi-seed).
# Run inside tmux 'jobs' session to survive disconnections.
set -euo pipefail

source ~/.env_papper 2>/dev/null || true
source /venv/main/bin/activate
cd /workspace/xling-grpo-sub3b

START=$(date -u +%s)
LOG=results/eval/master_chain_$(date -u +%Y%m%dT%H%M%SZ).log

echo "[MASTER] start $(date -u +%FT%TZ)" | tee -a "$LOG"

echo "[MASTER] === W1.8 (~20 min) ===" | tee -a "$LOG"
bash scripts/w18_eval_openrs.sh 2>&1 | tee -a "$LOG" || { echo "W1.8 FAILED" | tee -a "$LOG"; exit 1; }

echo "[MASTER] === W1.7 (~1.5h) ===" | tee -a "$LOG"
bash scripts/w17_reeval_amc23_majfix.sh 2>&1 | tee -a "$LOG" || { echo "W1.7 FAILED" | tee -a "$LOG"; exit 1; }

echo "[MASTER] === W1.9 (~8h) ===" | tee -a "$LOG"
bash scripts/w19_train_a4_multiseed.sh 2>&1 | tee -a "$LOG" || { echo "W1.9 FAILED" | tee -a "$LOG"; exit 1; }

END=$(date -u +%s)
ELAPSED=$(( (END - START) / 60 ))
echo "[MASTER] done $(date -u +%FT%TZ)  total ${ELAPSED} min" | tee -a "$LOG"
