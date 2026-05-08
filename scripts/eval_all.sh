#!/bin/bash
# Usage: bash scripts/eval_all.sh <checkpoint_dir>
# Example: bash scripts/eval_all.sh results/grpo/qwen15b_en_42/checkpoint-500

set -euo pipefail

CKPT="${1:?checkpoint dir required}"

if [[ ! -d "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
fi

# Run vLLM-based eval
python src/eval/runner.py \
    --checkpoint "${CKPT}" \
    --benchmarks gsm8k math500 aime2024 mgsm msvamp \
    --config configs/eval.yaml \
    --output_dir results/eval/

echo "[eval_all] done. Update master.csv:"
echo "  python src/analysis/aggregate.py --output results/master.csv"
