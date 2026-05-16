#!/bin/bash
# Phase 9.2 recovery sequence after Cell 1/2 eval crash + Cell 3/4 data missing.
#
# State at recovery:
#   - reproduce_openrs_rs2_123/checkpoint-50 saved (eval failed → re-eval here)
#   - reproduce_openrs_rs2_7/checkpoint-50 saved   (eval failed → re-eval here)
#   - a2_vi_*  not started (data was missing, now present)
#   - a3_enlang_* not started
#   - a4_const_bias_42 not started
#
# Recovery plan: re-eval the two A1 ckpts, then run remaining 5 cells with the
# already-fixed scripts (merge_lora_if_needed handled in train_a2_a3.sh + runner.py).
#
# Per-cell stdout streams to its own tee log file under /workspace/phase9_logs/
# (no more tail -100 buffering blackout).
#
# Usage on Vast.ai (in tmux):
#   tmux new-session -d -s phase9 \
#     "WANDB_DISABLED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 bash scripts/phase9_recovery.sh"

set -uo pipefail

cd /workspace/xling-grpo-sub3b
LOGS=/workspace/phase9_logs
mkdir -p "$LOGS"

START_TIME=$(date +%s)
echo "[$(date)] === Phase 9.2 recovery start ==="
echo "[$(date)] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Stage 1: re-eval cells 1-2 (A1 multi-seed) — ckpt-50 already saved
declare -a EVAL_CELLS=(
    "reproduce_openrs_rs2_123:results/grpo/reproduce_openrs_rs2_123/checkpoint-50"
    "reproduce_openrs_rs2_7:results/grpo/reproduce_openrs_rs2_7/checkpoint-50"
)

for spec in "${EVAL_CELLS[@]}"; do
    RUN_ID="${spec%:*}"
    CKPT="${spec#*:}"
    LOG="$LOGS/${RUN_ID}_eval.log"

    if [[ ! -d "$CKPT" ]]; then
        echo "[$(date)] SKIP $RUN_ID: $CKPT does not exist"
        continue
    fi

    echo "[$(date)] === Re-eval $RUN_ID @ ckpt-50 → $LOG ==="
    {
        date
        echo "Re-eval $RUN_ID ckpt-50 (after merge_lora_if_needed fix)"
        python3 src/eval/runner.py \
            --checkpoint "$CKPT" \
            --benchmarks amc23 math500 aime2024 \
            --config configs/eval.yaml \
            --output_dir "results/eval/${RUN_ID}_step50/" \
            --run_id "${RUN_ID}_step50"
        date
    } 2>&1 | tee "$LOG"

    if [[ $? -eq 0 ]]; then
        echo "[$(date)] ✓ $RUN_ID eval PASS"
    else
        echo "[$(date)] ✗ $RUN_ID eval FAIL"
    fi
done

# Stage 2: training cells 3-7 (A2 ×2 seeds, A3 ×2 seeds, A4 ×1 seed)
declare -a TRAIN_CELLS=(
    "a2:123"
    "a2:7"
    "a3:123"
    "a3:7"
    "a4:42"
)

PASS_COUNT=0
FAIL_CELLS=""

for spec in "${TRAIN_CELLS[@]}"; do
    CELL="${spec%:*}"
    SEED="${spec#*:}"
    RUN_ID="${CELL}_$([ "$CELL" = "a2" ] && echo "vi" || ([ "$CELL" = "a3" ] && echo "enlang" || echo "const_bias"))_${SEED}"
    LOG="$LOGS/${RUN_ID}_train.log"
    CELL_START=$(date +%s)

    echo
    echo "[$(date)] === Cell $CELL seed=$SEED → $LOG ==="
    bash scripts/train_a2_a3.sh "$CELL" --seed "$SEED" 2>&1 | tee "$LOG"
    EXIT=${PIPESTATUS[0]}

    CELL_END=$(date +%s)
    ELAPSED=$((CELL_END - CELL_START))
    H=$((ELAPSED / 3600))
    M=$(((ELAPSED % 3600) / 60))

    if [[ $EXIT -eq 0 ]]; then
        PASS_COUNT=$((PASS_COUNT + 1))
        echo "[$(date)] ✓ $CELL seed=$SEED PASS in ${H}h${M}m"
    else
        FAIL_CELLS="$FAIL_CELLS $CELL:$SEED"
        echo "[$(date)] ✗ $CELL seed=$SEED FAIL after ${H}h${M}m — continuing"
    fi

    nvidia-smi --query-gpu=memory.used --format=csv,noheader
    df -h / | tail -1
done

TOTAL=$(($(date +%s) - START_TIME))
H=$((TOTAL / 3600))
M=$(((TOTAL % 3600) / 60))

echo
echo "[$(date)] === Phase 9.2 recovery summary ==="
echo "  Train cells pass: $PASS_COUNT / 5"
echo "  Train cells fail: ${FAIL_CELLS:-none}"
echo "  Per-cell logs:    $LOGS/"
echo "  Total wallclock:  ${H}h${M}m"
