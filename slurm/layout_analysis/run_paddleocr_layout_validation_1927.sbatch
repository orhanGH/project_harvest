#!/bin/bash

#SBATCH --job-name=layout_val27
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/layout_validation_1927_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/layout_validation_1927_%j.err

set -euo pipefail

WORK=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work

PDF="$WORK/input/International_Statistical_Yearbook/1927/0000765724_D0001.pdf"

OUTPUT="$WORK/paddle_ocr/runs/layout_validation/1927_threshold020"

PYTHON="$HOME/.conda/envs/harvest-paddleocr/bin/python"

REPO="${HARVEST_REPO:-$HOME/project_harvest/HARVEST}"

export PYTHONNOUSERSITE=1
export PADDLE_PDX_CACHE_HOME="$WORK/paddle_ocr/cache/paddlex"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

cd "$REPO"

"$PYTHON" -u paddleocr/layout_analysis/detection/detect_tables_pdf.py \
    --pdf "$PDF" \
    --output-dir "$OUTPUT" \
    --pages "28,29,48,49,116,134,149,151,160,165" \
    --model PP-DocLayout_plus-L \
    --device gpu:0 \
    --dpi 180 \
    --layout-threshold 0.20 \
    --table-score-threshold 0.20
