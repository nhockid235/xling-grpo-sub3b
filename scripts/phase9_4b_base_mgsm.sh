#!/bin/bash
# Phase 9.4b — Base MGSM eval
# Eval untrained DeepSeek-R1-Distill-Qwen-1.5B on MGSM 10 langs.
# Comparison anchor for A1/A2/A3/A4 Δ computation.
#
# Total: ~30 min, ~$1

set -uo pipefail
cd /workspace/xling-grpo-sub3b
LOGS=/workspace/phase9_logs

START=$(date +%s)
echo "[$(date)] === Base MGSM eval ==="

OUT="results/eval/base_distill15b_mgsm/"
LOG="$LOGS/base_distill15b_mgsm.log"

{
    date
    echo "Base model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    python3 src/eval/runner.py \
        --checkpoint deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --benchmarks mgsm \
        --config configs/eval.yaml \
        --output_dir "$OUT" \
        --run_id "base_distill15b_mgsm"
    EXIT=$?
    date
    echo "Exit: $EXIT"
} 2>&1 | tee "$LOG"

TOTAL=$(($(date +%s) - START))
H=$((TOTAL / 3600)); M=$(((TOTAL % 3600) / 60))
echo "[$(date)] Base MGSM done: wall=${H}h${M}m"
