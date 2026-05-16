#!/bin/bash
# Phase 9 multi-seed + ablation training sequence (1× A100 80GB).
# 7 cells run sequentially. Total ~25h.
#
# Usage on Vast.ai (in tmux):
#   tmux new-session -d -s phase9 \
#     "bash scripts/phase9_run_sequence.sh 2>&1 | tee /workspace/phase9.log"
#
# Order strategy:
#   1-2: A1 multi-seed (validate seed determinism on known-good config)
#   3-4: A2 multi-seed
#   5-6: A3 multi-seed
#   7:   A4 ablation (new bias_const reward — last so other cells safe if A4 fails)
#
# Each cell auto-evals on AMC23 + MATH-500 + AIME-2024 after training.

set -uo pipefail

cd /workspace/xling-grpo-sub3b

START_TIME=$(date +%s)
echo "[$(date)] === Phase 9 run sequence start ==="
echo "[$(date)] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Cell schedule
declare -a CELLS=(
    "reproduce_open_rs:123"   # A1 seed 123
    "reproduce_open_rs:7"     # A1 seed 7
    "a2:123"                  # A2 seed 123
    "a2:7"                    # A2 seed 7
    "a3:123"                  # A3 seed 123
    "a3:7"                    # A3 seed 7
    "a4:42"                   # A4 ablation (new config)
)

PASS_COUNT=0
FAIL_CELLS=""

for spec in "${CELLS[@]}"; do
    CELL="${spec%:*}"
    SEED="${spec#*:}"
    CELL_START=$(date +%s)

    echo
    echo "[$(date)] ╔══════════════════════════════════════════════════════════"
    echo "[$(date)] ║ Cell $((PASS_COUNT + 1))/7: $CELL seed=$SEED"
    echo "[$(date)] ╚══════════════════════════════════════════════════════════"

    case "$CELL" in
        reproduce_open_rs)
            bash scripts/reproduce_open_rs.sh --seed "$SEED" 2>&1 | tail -100
            EXIT=${PIPESTATUS[0]}
            ;;
        a2|a3|a4)
            bash scripts/train_a2_a3.sh "$CELL" --seed "$SEED" 2>&1 | tail -100
            EXIT=${PIPESTATUS[0]}
            ;;
        *)
            echo "Unknown cell: $CELL"
            EXIT=1
            ;;
    esac

    CELL_END=$(date +%s)
    ELAPSED=$((CELL_END - CELL_START))
    H=$((ELAPSED / 3600))
    M=$(((ELAPSED % 3600) / 60))

    if [[ $EXIT -eq 0 ]]; then
        PASS_COUNT=$((PASS_COUNT + 1))
        echo "[$(date)] ✓ Cell $CELL seed=$SEED PASS in ${H}h${M}m"
    else
        FAIL_CELLS="$FAIL_CELLS $CELL:$SEED"
        echo "[$(date)] ✗ Cell $CELL seed=$SEED FAIL after ${H}h${M}m — continuing"
    fi

    # Cleanup after each cell to free disk for next
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
    df -h / | tail -1
done

TOTAL=$(($(date +%s) - START_TIME))
H=$((TOTAL / 3600))
M=$(((TOTAL % 3600) / 60))

echo
echo "[$(date)] ╔══════════════════════════════════════════════════════════"
echo "[$(date)] ║ Phase 9 sequence summary"
echo "[$(date)] ║   Pass: $PASS_COUNT / 7"
echo "[$(date)] ║   Fail: ${FAIL_CELLS:-none}"
echo "[$(date)] ║   Wall: ${H}h${M}m"
echo "[$(date)] ╚══════════════════════════════════════════════════════════"
echo "PHASE9_EXIT=$([[ $PASS_COUNT -eq 7 ]] && echo 0 || echo 1)"
