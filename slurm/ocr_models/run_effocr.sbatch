#!/bin/bash
#SBATCH --job-name=harvest_effocr
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=intelsr_devel
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail

REGIONS=${1:?Usage: sbatch run_effocr.sbatch REGIONS_MANIFEST [OUTPUT_DIR]}
OUTPUT_DIR=${2:-runs/effocr}
WORKSPACE=${WORKSPACE:-/lustre/scratch/data/s6oraydi_hpc-project_harvest}

mkdir -p runs/slurm "$OUTPUT_DIR" "$WORKSPACE/models/effocr"

module purge
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate harvest-effocr
export PYTHONNOUSERSITE=1

python ocr_models/effocr/run_effocr_ocr.py \
  --regions-manifest "$REGIONS" \
  --output-dir "$OUTPUT_DIR" \
  --model-dir "$WORKSPACE/models/effocr" \
  --device cpu \
  --region-type cell
