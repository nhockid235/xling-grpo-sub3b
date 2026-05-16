#!/bin/bash
# Train A2 (VI), A3 (enlang), or A4 (mechanism ablation) cell.
#
# Usage:
#   bash scripts/train_a2_a3.sh {a2|a3|a4}                  # default seed=42
#   bash scripts/train_a2_a3.sh a3 --seed 123                # multi-seed
#   bash scripts/train_a2_a3.sh a2 --seed 7

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 {a2|a3|a4} [--seed N]" >&2
    exit 1
fi

CELL="$1"
shift

# Parse --seed arg (default 42)
SEED=42
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        --seed=*) SEED="${1#*=}"; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

case "${CELL}" in
    a2) CONFIG="configs/grpo_a2_vi.yaml";       RUN_ID="a2_vi_${SEED}" ;;
    a3) CONFIG="configs/grpo_a3_enlang.yaml";   RUN_ID="a3_enlang_${SEED}" ;;
    a4) CONFIG="configs/grpo_a4_const_bias.yaml"; RUN_ID="a4_const_bias_${SEED}" ;;
    *) echo "ERROR: cell must be 'a2', 'a3', or 'a4'" >&2; exit 1 ;;
esac

OUTPUT="results/grpo/${RUN_ID}"

echo "[${CELL}] training ${RUN_ID}"
echo "[${CELL}] config = ${CONFIG}"
echo "[${CELL}] seed = ${SEED}"
echo "[${CELL}] output = ${OUTPUT}"

mkdir -p "${OUTPUT}"

accelerate launch \
    --config_file configs/accelerate_single_a100.yaml \
    src/trainers/grpo.py \
        --config "${CONFIG}" \
        --model deepseek_r1_distill_15b \
        --seed "${SEED}" \
        --output_dir "${OUTPUT}" \
        --wandb_run_name "${RUN_ID}"

# Eval ckpt-50 (gating step) on AMC23 + MATH-500 + AIME-2024
CKPT="${OUTPUT}/checkpoint-50"
if [[ -d "${CKPT}" ]]; then
    echo "[${CELL}] merge LoRA..."
    python3 -c "
from src.trainers.checkpoint_utils import merge_lora_if_needed
merged = merge_lora_if_needed('${CKPT}')
print('Merged at:', merged)
"
    echo "[${CELL}] eval ckpt-50..."
    python3 src/eval/runner.py \
        --checkpoint "${CKPT}/merged" \
        --benchmarks amc23 math500 aime2024 \
        --config configs/eval.yaml \
        --output_dir "results/eval/${RUN_ID}_step50/" \
        --run_id "${RUN_ID}_step50"
fi

echo "[${CELL}] DONE (seed=${SEED})"
