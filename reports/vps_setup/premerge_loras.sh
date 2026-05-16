#!/bin/bash
set -euo pipefail
source ~/.env_papper 2>/dev/null || true
source /venv/main/bin/activate
cd /workspace/xling-grpo-sub3b

CKPTS=(
  "results/grpo/reproduce_openrs_rs2_42/keep_step50"
  "results/grpo/reproduce_openrs_rs2_123/keep_step50"
  "results/grpo/reproduce_openrs_rs2_7/keep_step50"
  "results/grpo/a2_vi_42/keep_step50"
  "results/grpo/a2_vi_123/keep_step50"
  "results/grpo/a2_vi_7/keep_step50"
  "results/grpo/a3_enlang_42/keep_step50"
  "results/grpo/a3_enlang_123/keep_step50"
  "results/grpo/a3_enlang_7/keep_step50"
  "results/grpo/a4_const_bias_42/keep_step50"
)

merge_one() {
    local CKPT="$1"
    if [ -f "${CKPT}/merged/config.json" ]; then
        echo "[SKIP] ${CKPT}/merged already exists"
        return 0
    fi
    echo "[MERGE] ${CKPT}  start $(date -u +%FT%TZ)"
    python -c "
from src.trainers.checkpoint_utils import merge_lora_if_needed
m = merge_lora_if_needed('$CKPT')
print(f'  merged -> {m}')
" 2>&1
    echo "[MERGE] ${CKPT}  done $(date -u +%FT%TZ)"
}

export -f merge_one

printf '%s\n' "${CKPTS[@]}" | xargs -P 2 -I {} bash -c 'merge_one "$@"' _ {}

echo "[DONE] all 10 LoRAs pre-merged"
ls -la results/grpo/*/keep_step50/merged/config.json 2>/dev/null | wc -l
