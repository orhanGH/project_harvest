#!/bin/bash
#SBATCH --job-name=effocr_canada_ab
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=intelsr_devel
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=runs/slurm/effocr_canada_ab_%j.out
#SBATCH --error=runs/slurm/effocr_canada_ab_%j.err

set -euo pipefail

REPO="$HOME/project_harvest/HARVEST_effocr"
WORKSPACE="/lustre/scratch/data/s6oraydi_hpc-project_harvest"
EXPERIMENT="$REPO/runs/effocr_preprocessing_ab_40"

cd "$REPO"

module purge
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate harvest-effocr

export PYTHONNOUSERSITE=1
export HF_HOME="$WORKSPACE/models/huggingface"

mkdir -p "$EXPERIMENT/outputs" "$HF_HOME"

for VARIANT in \
    original \
    pad2x \
    auto_pad2x \
    trim_auto_pad2x
do
    echo "===== VARIANT: $VARIANT ====="

    python ocr_models/effocr/run_effocr_ocr.py \
      --regions-manifest \
        "$EXPERIMENT/manifests_canada/${VARIANT}.jsonl" \
      --output-dir \
        "$EXPERIMENT/outputs_canada/${VARIANT}" \
      --model-dir \
        "$WORKSPACE/models/effocr" \
      --device cpu \
      --region-type cell \
      --limit 4 \
      --log-every 10
done

echo "All preprocessing variants completed."
