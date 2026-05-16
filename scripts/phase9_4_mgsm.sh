#!/bin/bash
# Phase 9.4 — MGSM multilingual eval
# Eval 7 cells × 6 headline languages = 42 JSONs.
#
# Cells available on this Vast.ai instance:
#   A1 ×2 (seed=123, 7), A2 ×2, A3 ×2, A4 ×1
#   Phase 8 seed=42 cells were on the previous (destroyed) instance.
#
# Each cell auto-merges LoRA + base via runner.py, evals 6 langs (en, fr, de,
# zh, th, sw — 4 scripts), then we delete merged/ to free disk.
#
# Total: ~7 × 30 min = 3.5h, ~$5.5

set -uo pipefail
cd /workspace/xling-grpo-sub3b
LOGS=/workspace/phase9_logs
mkdir -p "$LOGS"

START=$(date +%s)
echo "[$(date)] === Phase 9.4 MGSM multilingual eval ==="
echo "[$(date)] Disk: $(df -h /workspace | tail -1 | awk '{print $4}') free"

declare -a CELLS=(
    "reproduce_openrs_rs2_123:results/grpo/reproduce_openrs_rs2_123/checkpoint-50"
    "reproduce_openrs_rs2_7:results/grpo/reproduce_openrs_rs2_7/checkpoint-50"
    "a2_vi_123:results/grpo/a2_vi_123/checkpoint-50"
    "a2_vi_7:results/grpo/a2_vi_7/checkpoint-50"
    "a3_enlang_123:results/grpo/a3_enlang_123/checkpoint-50"
    "a3_enlang_7:results/grpo/a3_enlang_7/checkpoint-50"
    "a4_const_bias_42:results/grpo/a4_const_bias_42/checkpoint-50"
)

PASS=0
FAIL=""

for spec in "${CELLS[@]}"; do
    CELL="${spec%:*}"
    CKPT="${spec#*:}"
    LOG="$LOGS/${CELL}_mgsm.log"
    OUT="results/eval/${CELL}_mgsm/"

    if [[ ! -d "$CKPT" ]]; then
        echo "[$(date)] SKIP $CELL — checkpoint missing: $CKPT"
        continue
    fi

    echo
    echo "[$(date)] === Eval $CELL → $OUT ==="
    {
        date
        echo "Cell: $CELL — MGSM 6 langs"
        python3 src/eval/runner.py \
            --checkpoint "$CKPT" \
            --benchmarks mgsm \
            --config configs/eval.yaml \
            --output_dir "$OUT" \
            --run_id "${CELL}_mgsm"
        EXIT=$?
        date
        echo "Exit: $EXIT"
    } 2>&1 | tee "$LOG"
    EXIT=${PIPESTATUS[0]}

    if [[ -d "$CKPT/merged" ]]; then
        rm -rf "$CKPT/merged" && echo "[$(date)] Freed merged/ for $CELL"
    fi

    if [[ $EXIT -eq 0 ]]; then
        PASS=$((PASS + 1))
        echo "[$(date)] ✓ $CELL MGSM PASS"
    else
        FAIL="$FAIL $CELL"
        echo "[$(date)] ✗ $CELL MGSM FAIL"
    fi
    df -h /workspace | tail -1
done

TOTAL=$(($(date +%s) - START))
H=$((TOTAL / 3600)); M=$(((TOTAL % 3600) / 60))
echo
echo "[$(date)] Phase 9.4 done: pass=$PASS/${#CELLS[@]} fail='${FAIL:-none}' wall=${H}h${M}m"
