#!/bin/bash
#SBATCH --job-name=isy26_more_tables
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/isy26_more_tables_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/isy26_more_tables_%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1
WORK=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work
export PADDLE_PDX_CACHE_HOME="$WORK/paddle_ocr/cache/paddlex"
REPO="${HARVEST_REPO:-$HOME/project_harvest/HARVEST}"
PYTHON="$HOME/.conda/envs/harvest-paddleocr/bin/python"
BASE="$WORK/paddle_ocr/runs/pipeline_dev/isy_1926_first50"
CROPS="$BASE/more_agricultural_tables/crops"
OCR="$BASE/more_agricultural_tables/ocr"
PARSE="$BASE/more_agricultural_tables/parsed"
cd "$REPO"
echo "MARVIN ISY agricultural table OCR + parser batch"
echo "Host: $(hostname) | Job: $SLURM_JOB_ID | Account: $SLURM_JOB_ACCOUNT"
echo "Crops: $CROPS"
echo "OCR output: $OCR"
echo "Parser output: $PARSE"
"$PYTHON" paddleocr/ocr/recognition/recognize_table_crops.py \
  --crops-root "$CROPS" \
  --output-dir "$OCR" \
  --device gpu:0
"$PYTHON" paddleocr/extraction/batch_parse_isy_tables.py \
  --ocr-root "$OCR" \
  --output-dir "$PARSE" \
  --repo "$REPO" \
  --python "$PYTHON" \
  --target-period 1925
