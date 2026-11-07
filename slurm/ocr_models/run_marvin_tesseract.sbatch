#!/bin/bash
#SBATCH --job-name=harvest_tess
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-"${HARVEST_REPO:-$HOME/project_harvest/HARVEST}"}
WORKSPACE=${HARVEST_WORKSPACE:-/lustre/scratch/data/s6oraydi_hpc-project_harvest}
SOURCE_PDF=${HARVEST_SOURCE_PDF:-"$WORKSPACE/s6oraydi_work/old/data/0000765723_D0001.pdf"}
FINAL_ROOT=${HARVEST_FINAL_ROOT:-"$WORKSPACE/s6oraydi_work/results/harvest_1926/tesseract"}
CONFIG=${1:-"$REPO_DIR/configs/harvest_1926.yaml"}
JOB_SCRATCH=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/harvest_1926_${SLURM_JOB_ID:-manual}
FINAL_DIR="$FINAL_ROOT/${SLURM_JOB_ID:-manual}"

mkdir -p "$REPO_DIR/runs/slurm" "$JOB_SCRATCH" "$FINAL_DIR"
test -f "$SOURCE_PDF"
cp "$SOURCE_PDF" "$JOB_SCRATCH/0000765723_D0001.pdf"

module purge
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate harvest-tesseract

export HARVEST_INPUT_PDF="$JOB_SCRATCH/0000765723_D0001.pdf"
export HARVEST_RUN_DIR="$JOB_SCRATCH/run"
export TESSDATA_PREFIX=${HARVEST_TESSDATA_DIR:-"$HOME/.local/share/harvest-tessdata"}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

test -s "$TESSDATA_PREFIX/eng.traineddata"
test -s "$TESSDATA_PREFIX/fra.traineddata"

cd "$REPO_DIR"
harvest-ocr --config "$CONFIG" run --model tesseract

cp -a "$HARVEST_RUN_DIR/exports" "$FINAL_DIR/"
cp -a "$HARVEST_RUN_DIR/validated" "$FINAL_DIR/"
cp -a "$HARVEST_RUN_DIR/parsed" "$FINAL_DIR/"
tar -czf "$FINAL_DIR/layout_diagnostics.tar.gz" -C "$HARVEST_RUN_DIR/layout" geometry.json debug
python -m pytest -q tests > "$FINAL_DIR/test_report.txt"
sha256sum "$SOURCE_PDF" > "$FINAL_DIR/source_pdf.sha256"

echo "Final results: $FINAL_DIR"
