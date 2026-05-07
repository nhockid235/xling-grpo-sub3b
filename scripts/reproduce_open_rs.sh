#!/bin/bash
# W2 GATING — reproduce Open-RS RS2 baseline.
# Target: AMC23 pass@1 ∈ [77, 83] (80.0 ± 3pp) sau 50 GRPO steps.
# Source: reports/phase0_03_open_rs_reproduction.md
#
# Cost: ~$10-12 trên 1× A100 80GB (RunPod $1.19/hr).
#
# Usage: bash scripts/reproduce_open_rs.sh

set -euo pipefail

RUN_ID="reproduce_openrs_rs2_42"
CONFIG="configs/reproduce_open_rs.yaml"
OUTPUT="results/grpo/${RUN_ID}"

echo "[gating] reproducing Open-RS RS2 (W2 gating decision point)"
echo "[gating] base = deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
echo "[gating] reward = accuracy + format, weights [1.0, 1.0]"
echo "[gating] target = AMC23 pass@1 ∈ [77, 83] @ ckpt-50"
echo "[gating] output = ${OUTPUT}"

mkdir -p "${OUTPUT}"

# Train
accelerate launch \
    --config_file configs/accelerate_single_a100.yaml \
    src/trainers/grpo.py \
        --config "${CONFIG}" \
        --model deepseek_r1_distill_15b \
        --seed 42 \
        --output_dir "${OUTPUT}" \
        --wandb_run_name "${RUN_ID}"

# Eval ckpt-50 (target match) và ckpt-100 (sanity)
for STEP in 50 100; do
    CKPT="${OUTPUT}/checkpoint-${STEP}"
    if [[ -d "${CKPT}" ]]; then
        echo "[gating] eval ckpt-${STEP} on AMC23 + MATH-500 + AIME-2024..."
        python src/eval/runner.py \
            --checkpoint "${CKPT}" \
            --benchmarks amc23 math500 aime2024 \
            --config configs/eval.yaml \
            --output_dir "results/eval/${RUN_ID}_step${STEP}/" \
            --run_id "${RUN_ID}_step${STEP}"
    fi
done

echo
echo "[gating] DONE. Inspect AMC23 pass@1 in:"
echo "  results/eval/${RUN_ID}_step50/${RUN_ID}_step50_amc23.json"
echo
echo "[gating] Decision tree (AMC23 pass@1 @ step 50):"
echo "  - ∈ [77, 83]: ✅ PASS — continue Idea 1 (full 9-cell sweep)"
echo "  - ∈ [70, 77): ⚠️  PARTIAL — check ckpt-100, debug tokenization/reward signal"
echo "  - < 70:       ❌ FAIL — pivot to Idea 4 (LIMO-style SFT)"
