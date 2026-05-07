#!/bin/bash
# Build arXiv submission bundle.
# Usage: bash build_arxiv_bundle.sh
# Output: arxiv_submission/ folder + arxiv_submission.tar.gz

set -euo pipefail

cd "$(dirname "$0")"

echo "[arxiv] === Building arXiv submission bundle ==="

# 0. Clean intermediates
echo "[arxiv] Cleanup intermediate files..."
rm -f *.aux *.log *.out *.toc *.bbl *.blg main.pdf
rm -rf arxiv_submission arxiv_submission.tar.gz

# 1. Verify required files exist
REQ=(main.tex appendix.tex refs.bib acl.sty acl_natbib.bst)
for f in "${REQ[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "[arxiv] ERROR: required file missing: $f" >&2
        exit 1
    fi
done

# 2. Sanity check critical fixes
echo "[arxiv] Checking for unresolved <TBD> markers..."
if grep -q "<TBD>" main.tex appendix.tex; then
    echo "[arxiv] WARNING: <TBD> still present in:"
    grep -l "<TBD>" main.tex appendix.tex
    echo "[arxiv] Replace với GitHub URL trước khi submit:"
    echo "  sed -i '' 's|<TBD>|<your-username>|g' main.tex appendix.tex"
    echo
fi

if grep -q '\\cite{}' main.tex; then
    echo "[arxiv] ERROR: empty \\cite{} found in main.tex"
    grep -n '\\cite{}' main.tex
    exit 1
fi

# 3. Build bundle dir
echo "[arxiv] Creating bundle..."
mkdir -p arxiv_submission/figures arxiv_submission/tables

cp main.tex appendix.tex refs.bib acl.sty acl_natbib.bst arxiv_submission/

# Copy only used tables (not the empty stubs from earlier)
cp tables/table_results.tex tables/table_delta_vs_base.tex arxiv_submission/tables/

# Copy figures (PDFs only, not .py source)
cp figures/fig1_arm_comparison.pdf figures/fig2_training_curves.pdf figures/fig3_effect_sizes.pdf arxiv_submission/figures/

# 4. Verify standalone build
echo "[arxiv] Verifying standalone compile..."
cd arxiv_submission
tectonic main.tex 2>&1 | tail -5
if [[ ! -f main.pdf ]]; then
    echo "[arxiv] ERROR: main.pdf was not created" >&2
    exit 1
fi
PAGES=$(mdls -name kMDItemNumberOfPages main.pdf 2>/dev/null | grep -oE '[0-9]+' | head -1)
SIZE=$(stat -f%z main.pdf 2>/dev/null || stat -c%s main.pdf)
echo "[arxiv] PDF: ${PAGES:-?} pages, $((SIZE/1024)) KB"
cd ..

# 5. Tar bundle (without intermediate compile artifacts)
echo "[arxiv] Tarballing..."
cd arxiv_submission
rm -f *.aux *.log *.out *.toc *.bbl *.blg main.pdf
cd ..
tar czf arxiv_submission.tar.gz arxiv_submission/
TARSIZE=$(stat -f%z arxiv_submission.tar.gz 2>/dev/null || stat -c%s arxiv_submission.tar.gz)
echo "[arxiv] Bundle ready: arxiv_submission.tar.gz ($((TARSIZE/1024)) KB)"

echo
echo "[arxiv] === DONE ==="
echo "Next steps:"
echo "  1. Open https://arxiv.org/submit (after endorsement)"
echo "  2. Upload: $(pwd)/arxiv_submission.tar.gz"
echo "  3. Primary category: cs.CL ; Cross-list: cs.LG"
echo "  4. License: CC BY 4.0"
echo "  5. Comments: '7 pages, 3 figures, 2 tables. Code: <github>. Single seed.'"
