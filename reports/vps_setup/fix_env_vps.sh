#!/bin/bash
# Post-install fix: pin transformers, peft to Phase 7 exact + download fasttext + verify.
set -euo pipefail
source ~/.env_papper 2>/dev/null || true
source /venv/main/bin/activate
cd /workspace/xling-grpo-sub3b

echo "=== Pin transformers==4.57.6 (Phase 7 exact) ==="
pip install --no-cache-dir transformers==4.57.6

echo "=== Re-verify imports ==="
python <<'PY'
import torch, vllm, transformers, trl, peft, accelerate, datasets, math_verify, sympy, fasttext, numpy
print(f"py        {__import__('sys').version_info[:3]}")
print(f"torch     {torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else None}")
print(f"vllm      {vllm.__version__}")
print(f"transformers  {transformers.__version__}")
print(f"trl       {trl.__version__}")
print(f"peft      {peft.__version__}")
print(f"accelerate    {accelerate.__version__}")
print(f"datasets  {datasets.__version__}")
print(f"sympy     {sympy.__version__}")
print(f"math_verify   {math_verify.__version__}")
print(f"numpy     {numpy.__version__}")
import flash_attn
print(f"flash_attn   {flash_attn.__version__}")
# GRPOTrainer import check (trl+transformers compatibility)
from trl import GRPOTrainer, GRPOConfig
print("trl GRPOTrainer + GRPOConfig import OK")
# Math-verify smoke
from math_verify import parse, verify
print(f"math_verify test: 1/2 == 0.5 → {verify(parse('0.5'), parse('1/2'))}")
print("=== ALL IMPORTS OK ===")
PY

echo "=== Download fasttext lid.176.bin ==="
mkdir -p data/raw
if [[ ! -f data/raw/lid.176.bin ]]; then
    wget -q --show-progress -O data/raw/lid.176.bin \
        https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
fi
ls -la data/raw/lid.176.bin

echo "=== POST-INSTALL FIX COMPLETE ==="
