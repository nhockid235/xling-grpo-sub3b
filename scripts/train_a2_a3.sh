#!/bin/bash
# Usage: bash scripts/train_a2_a3.sh {a2|a3}
# Example: bash scripts/train_a2_a3.sh a3

set -euo pipefail

CELL="${1:?cell required: a2 or a3}"
case "${CELL}" in
    a2) CONFIG="configs/grpo_a2_vi.yaml"; RUN_ID="a2_vi_42" ;;
    a3) CONFIG="configs/grpo_a3_enlang.yaml"; RUN_ID="a3_enlang_42" ;;
    *) echo "ERROR: cell must be 'a2' or 'a3'"; exit 1 ;;
esac

OUTPUT="results/grpo/${RUN_ID}"

echo "[${CELL}] training ${RUN_ID}"
echo "[${CELL}] config = ${CONFIG}"
echo "[${CELL}] output = ${OUTPUT}"

mkdir -p "${OUTPUT}"

accelerate launch \
    --config_file configs/accelerate_single_a100.yaml \
    src/trainers/grpo.py \
        --config "${CONFIG}" \
        --model deepseek_r1_distill_15b \
        --seed 42 \
        --output_dir "${OUTPUT}" \
        --wandb_run_name "${RUN_ID}"

# Eval ckpt-50 (gating step) on AMC23 + MATH-500 + AIME-2024
CKPT="${OUTPUT}/checkpoint-50"
if [[ -d "${CKPT}" ]]; then
    echo "[${CELL}] eval ckpt-50..."
    python3 -c "
from src.trainers.checkpoint_utils import merge_lora_if_needed
merged = merge_lora_if_needed('${CKPT}')
print('Merged at:', merged)
"
    python3 src/eval/runner.py \
        --checkpoint "${CKPT}/merged" \
        --benchmarks amc23 math500 aime2024 \
        --config configs/eval.yaml \
        --output_dir "results/eval/${RUN_ID}_step50/" \
        --run_id "${RUN_ID}_step50"
fi

echo "[${CELL}] DONE"
