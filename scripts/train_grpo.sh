#!/bin/bash
# Usage: bash scripts/train_grpo.sh <model_short> <condition> <seed>
# Example: bash scripts/train_grpo.sh qwen15b en 42

set -euo pipefail

MODEL="${1:?model_short required: qwen15b | qwen3b | llama3b}"
CONDITION="${2:?condition required: en | vi | enlang}"
SEED="${3:-42}"

RUN_ID="${MODEL}_${CONDITION}_${SEED}"
GRPO_CONFIG="configs/grpo_${CONDITION}.yaml"
SFT_CKPT="results/sft/${RUN_ID}/checkpoint-final"
OUTPUT="results/grpo/${RUN_ID}"

if [[ ! -f "${GRPO_CONFIG}" ]]; then
    echo "ERROR: config not found: ${GRPO_CONFIG}" >&2
    exit 1
fi

if [[ ! -d "${SFT_CKPT}" ]]; then
    echo "WARNING: SFT checkpoint not found: ${SFT_CKPT}" >&2
    echo "  Trying latest checkpoint in results/sft/${RUN_ID}/" >&2
    SFT_CKPT=$(ls -td "results/sft/${RUN_ID}"/checkpoint-* 2>/dev/null | head -1 || true)
    if [[ -z "${SFT_CKPT}" ]]; then
        echo "ERROR: no SFT checkpoint found. Run train_sft.sh first." >&2
        exit 1
    fi
fi

echo "[train_grpo] run_id=${RUN_ID}"
echo "[train_grpo] grpo_config=${GRPO_CONFIG}"
echo "[train_grpo] sft_ckpt=${SFT_CKPT}"
echo "[train_grpo] output=${OUTPUT}"

mkdir -p "${OUTPUT}"

accelerate launch \
    --config_file configs/accelerate_single_a100.yaml \
    src/trainers/grpo.py \
        --config "${GRPO_CONFIG}" \
        --model "${MODEL}" \
        --seed "${SEED}" \
        --sft_checkpoint "${SFT_CKPT}" \
        --output_dir "${OUTPUT}" \
        --wandb_run_name "${RUN_ID}_grpo"
