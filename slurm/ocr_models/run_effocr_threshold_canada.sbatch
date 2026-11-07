#!/bin/bash
#SBATCH --job-name=effocr_canada_threshold
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=intelsr_devel
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=runs/slurm/effocr_canada_threshold_%j.out
#SBATCH --error=runs/slurm/effocr_canada_threshold_%j.err

set -euo pipefail

REPO="$HOME/project_harvest/HARVEST_effocr"
WORKSPACE="/lustre/scratch/data/s6oraydi_hpc-project_harvest"
EXPERIMENT="$REPO/runs/effocr_threshold_grid_canada"
MANIFEST="$REPO/runs/effocr_preprocessing_ab_40/manifests_canada/original.jsonl"

cd "$REPO"

module purge
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate harvest-effocr

export PYTHONNOUSERSITE=1
export HF_HOME="$WORKSPACE/models/huggingface"

mkdir -p "$EXPERIMENT"

for CONF in 0.25 0.40 0.55
do
    for IOU in 0.05 0.10
    do
        CONF_TAG="${CONF/./p}"
        IOU_TAG="${IOU/./p}"
        NAME="conf${CONF_TAG}_iou${IOU_TAG}"

        echo "===== $NAME ====="

        export EFFOCR_LOCALIZER_CONF="$CONF"
        export EFFOCR_LOCALIZER_IOU="$IOU"

        python \
          ocr_models/effocr/run_effocr_threshold_override.py \
          --regions-manifest "$MANIFEST" \
          --output-dir "$EXPERIMENT/$NAME" \
          --model-dir "$WORKSPACE/models/effocr" \
          --device cpu \
          --region-type cell \
          --limit 4
    done
done
