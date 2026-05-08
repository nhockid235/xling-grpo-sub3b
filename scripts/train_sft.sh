#!/bin/bash
# Usage: bash scripts/train_sft.sh <model_short> <condition> <seed>
# Example: bash scripts/train_sft.sh qwen15b en 42

set -euo pipefail

MODEL="${1:?model_short required: qwen15b | qwen3b | llama3b}"
CONDITION="${2:?condition required: en | vi | enlang}"
SEED="${3:-42}"

RUN_ID="${MODEL}_${CONDITION}_${SEED}"
CONFIG="configs/sft_${MODEL}.yaml"
OUTPUT="results/sft/${RUN_ID}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config not found: ${CONFIG}" >&2
    exit 1
fi

echo "[train_sft] run_id=${RUN_ID}"
echo "[train_sft] config=${CONFIG}"
echo "[train_sft] output=${OUTPUT}"

mkdir -p "${OUTPUT}"

accelerate launch \
    --config_file configs/accelerate_single_a100.yaml \
    src/trainers/sft.py \
        --config "${CONFIG}" \
        --condition "${CONDITION}" \
        --seed "${SEED}" \
        --output_dir "${OUTPUT}" \
        --wandb_run_name "${RUN_ID}_sft"
