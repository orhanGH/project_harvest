#!/bin/bash
#SBATCH --job-name=e05_visual_all
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=intelsr_devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/e05_visual_all_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/e05_visual_all_%j.err

set -euo pipefail

WORK=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work
PYTHON="$HOME/.conda/envs/harvest-paddleocr/bin/python"

RESULTS="$WORK/paddle_ocr/results/mixed_layout_test_74p/e02_e03_e05_comparison"

OUT="$RESULTS/E05_diagnostics/visual_review_all74"

if [ -e "$OUT" ]; then
    echo "Output already exists: $OUT"
    exit 1
fi

mkdir -p "$OUT"

PAGES=$(seq -s, 1 74)

cd "${HARVEST_REPO:-$HOME/project_harvest/HARVEST}"

"$PYTHON" visualisation/layout_analysis/visualize_e05_changes.py \
    --pages "$PAGES" \
    --out "$OUT"

cp visualisation/layout_analysis/visualize_e05_changes.py "$OUT/"

echo "=== Verify output ==="

N=$(find "$OUT" -name '*_GT_E02_E05.jpg' | wc -l)
M=$(find "$OUT" -name '*_NEW_ONLY.jpg' | wc -l)

echo "Comparison images: $N"
echo "New-only images: $M"

test "$N" -eq 74
test "$M" -eq 74

echo "E05 all-page visualization completed successfully."
