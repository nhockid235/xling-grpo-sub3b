#!/bin/bash
# Pre-flight check trước khi launch training trên RunPod.
# Verify env hoàn chỉnh để tránh lãng phí $$ vào job sẽ crash.
#
# Usage: bash scripts/preflight.sh
# Exit code 0 = pass; 1 = fail

set -uo pipefail

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

_pass() { echo "  ✅ $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
_fail() { echo "  ❌ $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
_warn() { echo "  ⚠️  $1"; WARN_COUNT=$((WARN_COUNT + 1)); }
_section() { echo; echo "=== $1 ==="; }

# --- System ---
_section "System"

PY_VERSION=$(python --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" == "3" && "$PY_MINOR" == "11" ]]; then
    _pass "Python $PY_VERSION"
elif [[ "$PY_MAJOR" == "3" ]]; then
    _warn "Python $PY_VERSION (CLAUDE.md pin 3.11; có thể work nhưng untested)"
else
    _fail "Python $PY_VERSION (cần >=3.11)"
fi

# Disk space — cross-platform via Python (df flag khác BSD/GNU)
DISK_FREE_GB=$(python -c "import shutil; print(int(shutil.disk_usage('.').free / 1024**3))" 2>/dev/null || echo "0")
if [[ "${DISK_FREE_GB:-0}" -ge 60 ]]; then
    _pass "Disk free: ${DISK_FREE_GB}GB"
elif [[ "${DISK_FREE_GB:-0}" -ge 30 ]]; then
    _warn "Disk free: ${DISK_FREE_GB}GB (recommend 60GB+; có thể OOM disk khi save ckpts)"
else
    _fail "Disk free: ${DISK_FREE_GB}GB (cần tối thiểu 30GB)"
fi

# RAM
if command -v free >/dev/null 2>&1; then
    RAM_GB=$(free -g | awk '/^Mem:/ {print $2}')
    if [[ "$RAM_GB" -ge 32 ]]; then
        _pass "RAM: ${RAM_GB}GB"
    else
        _warn "RAM: ${RAM_GB}GB (recommend 32GB+ cho dataset preprocessing)"
    fi
fi

# --- GPU ---
_section "GPU"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [[ -n "$GPU_INFO" ]]; then
        GPU_NAME=$(echo "$GPU_INFO" | cut -d, -f1 | xargs)
        GPU_MEM_MB=$(echo "$GPU_INFO" | cut -d, -f2 | xargs)
        GPU_MEM_GB=$((GPU_MEM_MB / 1024))
        if [[ "$GPU_MEM_GB" -ge 70 ]]; then
            _pass "GPU: $GPU_NAME (${GPU_MEM_GB}GB) — đủ cho 1.5B/3B GRPO full-param"
        elif [[ "$GPU_MEM_GB" -ge 40 ]]; then
            _warn "GPU: $GPU_NAME (${GPU_MEM_GB}GB) — OK cho 1.5B; 3B cần LoRA fallback"
        else
            _fail "GPU: $GPU_NAME (${GPU_MEM_GB}GB) — không đủ; cần ≥40GB"
        fi
    else
        _fail "nvidia-smi available but không detect GPU"
    fi
else
    _warn "nvidia-smi không có (CPU-only env? OK cho unit tests, FAIL cho training)"
fi

# --- Python dependencies ---
_section "Python dependencies"

python <<'PYEOF' || true
import sys
import importlib.metadata as md

CHECKS = [
    ("torch", "2.4.1", True),
    ("transformers", "4.46.3", True),
    ("trl", ">=0.15.0,<0.16.0", True),
    ("vllm", "0.7.2", True),
    ("accelerate", "1.2.1", True),
    ("peft", "0.14.0", True),
    ("datasets", ">=2.21.0", True),
    ("sympy", "<1.13", True),
    ("math_verify", ">=0.5.0", False),
    ("fasttext", None, False),
    ("datasketch", ">=1.6.0", False),
    ("xxhash", ">=3.0", False),
    ("wandb", None, False),
    ("pandas", None, False),
    ("matplotlib", None, False),
]

passed, failed, warned = 0, 0, 0

for pkg, expected, hard in CHECKS:
    try:
        actual = md.version(pkg.replace("_", "-"))
    except md.PackageNotFoundError:
        try:
            actual = md.version(pkg)
        except md.PackageNotFoundError:
            actual = None

    if actual is None:
        msg = f"  ❌ {pkg} NOT INSTALLED" if hard else f"  ⚠️  {pkg} not installed (optional)"
        print(msg)
        if hard:
            failed += 1
        else:
            warned += 1
        continue

    if expected and actual != expected.lstrip(">=<"):
        # Loose check — just print actual
        print(f"  ⚠️  {pkg} {actual} (expected {expected})")
        warned += 1
    else:
        print(f"  ✅ {pkg} {actual}")
        passed += 1

print(f"\n  [deps] {passed} ok, {warned} warn, {failed} fail", file=sys.stderr)
sys.exit(0 if failed == 0 else 1)
PYEOF
PY_DEPS_RC=$?
if [[ $PY_DEPS_RC -eq 0 ]]; then
    PASS_COUNT=$((PASS_COUNT + 1))
else
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# --- Data files ---
_section "Data assets"

FT_PATH="data/raw/lid.176.bin"
if [[ -f "$FT_PATH" ]]; then
    FT_SIZE=$(stat -f%z "$FT_PATH" 2>/dev/null || stat -c%s "$FT_PATH" 2>/dev/null)
    FT_MB=$((FT_SIZE / 1024 / 1024))
    if [[ "$FT_MB" -ge 100 ]]; then
        _pass "fastText langID model: $FT_PATH (${FT_MB}MB)"
    else
        _warn "fastText file present but size ${FT_MB}MB (expected ~131MB) — corrupt?"
    fi
else
    _warn "fastText langID missing. Cond C (enlang) sẽ skip R5 reward."
    echo "       Download: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P data/raw/"
fi

# --- HuggingFace access ---
_section "HuggingFace access"

if python -c "from huggingface_hub import HfApi; HfApi().list_repo_files('Qwen/Qwen2.5-1.5B-Instruct', timeout=10)" 2>/dev/null; then
    _pass "HF Hub reachable + can list Qwen2.5-1.5B-Instruct files"
else
    _warn "HF Hub không reachable hoặc rate-limited. Verify HF_TOKEN nếu private."
fi

# Check HF_TOKEN cho gated models (Llama)
if [[ -n "${HF_TOKEN:-}" ]] || [[ -f "$HOME/.cache/huggingface/token" ]]; then
    _pass "HF_TOKEN present (cần cho meta-llama/Llama-3.2-3B-Instruct)"
else
    _warn "HF_TOKEN không set. Nếu plan train Llama-3.2 → cần huggingface-cli login"
fi

# --- Wandb ---
_section "Wandb"

if [[ -n "${WANDB_API_KEY:-}" ]] || wandb login --verify >/dev/null 2>&1; then
    _pass "Wandb authenticated"
else
    _warn "Wandb chưa login. Run: wandb login"
fi

# --- Final verdict ---
_section "Verdict"

echo "  Pass: $PASS_COUNT, Warn: $WARN_COUNT, Fail: $FAIL_COUNT"
echo

if [[ $FAIL_COUNT -gt 0 ]]; then
    echo "❌ PRE-FLIGHT FAILED — fix errors trước khi launch training."
    exit 1
elif [[ $WARN_COUNT -gt 5 ]]; then
    echo "⚠️  PRE-FLIGHT PASSED WITH MANY WARNINGS — review trước khi launch."
    exit 0
else
    echo "✅ PRE-FLIGHT PASSED — sẵn sàng launch training."
    exit 0
fi
