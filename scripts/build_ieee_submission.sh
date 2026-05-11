#!/bin/bash
# Build IEEE Access submission bundle.
# Output: paper/ieee/submission_bundle.zip — ready to upload as "Main Manuscript"

set -euo pipefail
cd "$(dirname "$0")/.."

BUNDLE_DIR=paper/ieee/_bundle_tmp
ZIP_OUT=paper/ieee/submission_bundle.zip

rm -rf "$BUNDLE_DIR" "$ZIP_OUT"
mkdir -p "$BUNDLE_DIR"

echo "Copying LaTeX source + assets..."
cp paper/ieee/main.tex      "$BUNDLE_DIR/"
cp paper/ieee/refs.bib      "$BUNDLE_DIR/"
cp paper/ieee/main.pdf      "$BUNDLE_DIR/"

mkdir -p "$BUNDLE_DIR/tables"
cp paper/ieee/tables/table_main_v2.tex   "$BUNDLE_DIR/tables/"
cp paper/ieee/tables/table_delta_v2.tex  "$BUNDLE_DIR/tables/"
cp paper/ieee/tables/table_mgsm.tex      "$BUNDLE_DIR/tables/"

mkdir -p "$BUNDLE_DIR/figures"
cp paper/ieee/figures/fig1_arm_means_with_ci.pdf         "$BUNDLE_DIR/figures/"
cp paper/ieee/figures/fig4_effect_vs_base.pdf            "$BUNDLE_DIR/figures/"
cp paper/ieee/figures/fig5_training_curves_multiseed.pdf "$BUNDLE_DIR/figures/"

echo "Creating zip..."
( cd "$BUNDLE_DIR" && zip -r "../../../$ZIP_OUT" . > /dev/null )

rm -rf "$BUNDLE_DIR"

SIZE=$(ls -l "$ZIP_OUT" | awk '{print $5}')
echo
echo "Bundle ready: $ZIP_OUT ($((SIZE/1024)) KB)"
echo
echo "Contents:"
unzip -l "$ZIP_OUT"
