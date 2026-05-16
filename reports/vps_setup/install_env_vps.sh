#!/bin/bash
# Install env matching Phase 7 paper pin on VPS A100
# Target: torch 2.5.1+cu124, vllm 0.7.2, transformers 4.57.6, trl 0.15.2, flash-attn 2.7.2.post1
set -euo pipefail

source ~/.env_papper 2>/dev/null || true
source /venv/main/bin/activate

cd /workspace/xling-grpo-sub3b

echo "=== STEP 1: Python version check ==="
python --version
test "$(python -c 'import sys; print(sys.version_info[:2])')" = "(3, 12)" || { echo "FAIL: expected Python 3.12"; exit 1; }

echo "=== STEP 2: Pre-clean conflicting packages ==="
pip uninstall -y torch torchvision torchaudio torchcodec torchdata torchtext sympy numpy 2>/dev/null || true

echo "=== STEP 3: Install torch 2.5.1 + cu124 ==="
pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

echo "=== STEP 4: Install vllm 0.7.2 (will pull transformers 4.57.6 + sympy 1.13.1) ==="
pip install --no-cache-dir vllm==0.7.2

echo "=== STEP 5: Install project + dev + analysis extras ==="
pip install --no-cache-dir -e ".[dev,analysis]"

echo "=== STEP 6: Install flash-attn 2.7.2.post1 (uses prebuilt wheel) ==="
pip install --no-cache-dir flash-attn==2.7.2.post1 --no-build-isolation || {
    echo "WARN: flash-attn install failed, retrying without --no-build-isolation"
    pip install --no-cache-dir flash-attn==2.7.2.post1
}

echo "=== STEP 7: Force numpy < 2.0 (vllm/torch may pull newer) ==="
pip install --no-cache-dir "numpy<2.0" --force-reinstall --no-deps

echo "=== STEP 8: Verify imports + versions ==="
python <<'PY'
import torch, vllm, transformers, trl, peft, accelerate, datasets, math_verify, sympy, fasttext, numpy
assert torch.__version__.startswith("2.5.1"), f"torch={torch.__version__}"
assert vllm.__version__ == "0.7.2", f"vllm={vllm.__version__}"
assert transformers.__version__ == "4.57.6", f"transformers={transformers.__version__}"
assert trl.__version__ == "0.15.2", f"trl={trl.__version__}"
assert sympy.__version__ == "1.13.1", f"sympy={sympy.__version__}"
assert math_verify.__version__ == "0.9.0", f"math_verify={math_verify.__version__}"
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
print("=== IMPORTS OK — match Phase 7 ===")
try:
    import flash_attn
    print(f"flash_attn   {flash_attn.__version__}")
except ImportError as e:
    print(f"flash_attn   NOT INSTALLED: {e}")
# Math-verify smoke test
from math_verify import parse, verify
half = parse("1/2"); zero5 = parse("0.5")
print(f"math_verify test: 1/2 == 0.5 → {verify(zero5, half)}")
PY

echo "=== STEP 9: Download fasttext lid.176.bin ==="
mkdir -p data/raw
if [[ ! -f data/raw/lid.176.bin ]]; then
    wget -q --show-progress -O data/raw/lid.176.bin \
        https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
fi
ls -la data/raw/lid.176.bin
echo "=== ENV INSTALL COMPLETE ==="
