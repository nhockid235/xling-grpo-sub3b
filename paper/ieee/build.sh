#!/bin/bash
# Build script for IEEE Access submission PDF.
# Requires: tectonic (https://tectonic-typesetting.github.io/)
#   brew install tectonic   # macOS
#   apt install tectonic    # Linux

set -e
cd "$(dirname "$0")"

if ! command -v tectonic >/dev/null 2>&1; then
    echo "ERROR: tectonic not found. Install via: brew install tectonic"
    exit 1
fi

echo "Building IEEE PDF..."
tectonic -X compile main.tex --keep-intermediates 2>&1 | tail -5

if [[ -f main.pdf ]]; then
    SIZE=$(ls -l main.pdf | awk '{print $5}')
    echo
    echo "Built main.pdf ($((SIZE/1024)) KB)"
    if command -v /Users/vudang/miniconda3/bin/python3 >/dev/null 2>&1; then
        /Users/vudang/miniconda3/bin/python3 -c "
from pypdf import PdfReader
r = PdfReader('main.pdf')
print(f'Pages: {len(r.pages)}')
" 2>/dev/null || true
    fi
else
    echo "ERROR: main.pdf not generated"
    exit 1
fi
