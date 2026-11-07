#!/bin/bash

#SBATCH --job-name=mixed_layout050
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/mixed_layout_test/mixed_050_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/mixed_layout_test/mixed_050_%j.err

set -euo pipefail

WORK=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work
REPO="${HARVEST_REPO:-$HOME/project_harvest/HARVEST}"

PYTHON="$HOME/.conda/envs/harvest-paddleocr/bin/python"

PDF="$WORK/input/mixed_layout_test/mixed_layout_test_59p.pdf"
OUTPUT="$WORK/paddle_ocr/runs/mixed_layout_test/baseline_050"

export PYTHONNOUSERSITE=1
export PADDLE_PDX_CACHE_HOME="$WORK/paddle_ocr/cache/paddlex"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

cd "$REPO"

echo "=== MIXED LAYOUT BASELINE ==="
echo "PDF: $PDF"
echo "Output: $OUTPUT"
echo "Model: PP-DocLayout_plus-L"
echo "DPI: 180"
echo "Layout threshold: 0.50"
echo "Table threshold: 0.50"
echo

"$PYTHON" -u paddleocr/layout_analysis/detection/detect_tables_pdf.py \
    --pdf "$PDF" \
    --output-dir "$OUTPUT" \
    --model PP-DocLayout_plus-L \
    --device gpu:0 \
    --dpi 180 \
    --layout-threshold 0.50 \
    --table-score-threshold 0.50

echo
echo "=== COMPLETE ==="
