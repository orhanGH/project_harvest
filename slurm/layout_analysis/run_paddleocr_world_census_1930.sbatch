#!/bin/bash

#SBATCH --job-name=wca1930_layout
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-4%5

#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/world_census_1930/layout_%A_%a.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/world_census_1930/layout_%A_%a.err

set -euo pipefail

WORK=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work
REPO="${HARVEST_REPO:-$HOME/project_harvest/HARVEST}"

INPUT="$WORK/input/World_Census_of_Agriculture"
OUTPUT_ROOT="$WORK/paddle_ocr/runs/world_census_1930"

PYTHON="$HOME/.conda/envs/harvest-paddleocr/bin/python"

export PYTHONNOUSERSITE=1
export PADDLE_PDX_CACHE_HOME="$WORK/paddle_ocr/cache/paddlex"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1

PDFS=(
    "1930 VOLUME I.pdf"
    "1930 VOLUME II.pdf"
    "1930 VOLUME III.pdf"
    "1930 VOLUME IV.pdf"
    "1930 VOLUME V.pdf"
)

NAMES=(
    "volume_I"
    "volume_II"
    "volume_III"
    "volume_IV"
    "volume_V"
)

PDF="$INPUT/${PDFS[$SLURM_ARRAY_TASK_ID]}"
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
OUTPUT="$OUTPUT_ROOT/$NAME"

echo "=========================================="
echo "World Census of Agriculture 1930"
echo "Task:             $SLURM_ARRAY_TASK_ID"
echo "PDF:              $PDF"
echo "Output:           $OUTPUT"
echo "Model:            PP-DocLayout_plus-L"
echo "DPI:              180"
echo "Layout threshold: 0.50"
echo "Table threshold:  0.50"
echo "=========================================="

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

cd "$REPO"

"$PYTHON" -u paddleocr/layout_analysis/detection/detect_tables_pdf.py \
    --pdf "$PDF" \
    --output-dir "$OUTPUT" \
    --model PP-DocLayout_plus-L \
    --device gpu:0 \
    --dpi 180 \
    --layout-threshold 0.50 \
    --table-score-threshold 0.50

echo
echo "=== COMPLETE: $NAME ==="
