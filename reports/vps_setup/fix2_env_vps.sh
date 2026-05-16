#!/bin/bash
# Fix v2: skip math_verify.__version__ (not available), do imports + GRPOTrainer + fasttext.
set -euo pipefail
source ~/.env_papper 2>/dev/null || true
source /venv/main/bin/activate
cd /workspace/xling-grpo-sub3b

echo "=== Verify imports (fixed) ==="
python <<'PY'
import sys
import torch, vllm, transformers, trl, peft, accelerate, datasets, math_verify, sympy, fasttext, numpy, flash_attn
print(f"py        {sys.version_info[:3]}")
print(f"torch     {torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else None}")
print(f"vllm      {vllm.__version__}")
print(f"transformers  {transformers.__version__}")
print(f"trl       {trl.__version__}")
print(f"peft      {peft.__version__}")
print(f"accelerate    {accelerate.__version__}")
print(f"datasets  {datasets.__version__}")
print(f"sympy     {sympy.__version__}")
print(f"math_verify   module={math_verify.__file__}")
print(f"numpy     {numpy.__version__}")
print(f"flash_attn   {flash_attn.__version__}")
# GRPOTrainer + transformers compat
from trl import GRPOTrainer, GRPOConfig
print("trl GRPOTrainer + GRPOConfig import OK")
# Math-verify smoke
from math_verify import parse, verify
test = verify(parse("0.5"), parse("1/2"))
print(f"math_verify test: 1/2 == 0.5 → {test}")
assert test is True, "math_verify smoke test failed"
print("=== ALL IMPORTS OK ===")
PY

echo "=== Download fasttext lid.176.bin ==="
mkdir -p data/raw
if [[ ! -f data/raw/lid.176.bin ]]; then
    wget -q --show-progress -O data/raw/lid.176.bin \
        https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
fi
ls -la data/raw/lid.176.bin

echo "=== pytest sanity ==="
pytest tests/ -q --tb=no -x 2>&1 | tail -10

echo "=== FIX2 COMPLETE ==="
