#!/bin/bash
#SBATCH --job-name=harvest_gpu
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail

MODEL=${1:?Usage: sbatch run_gpu_ocr.sbatch paddle|chandra|surya [config]}
CONFIG=${2:-configs/harvest_1926.yaml}
mkdir -p runs/slurm

case "$MODEL" in
  paddle) ENVIRONMENT=harvest-paddle ;;
  chandra) ENVIRONMENT=harvest-chandra ;;
  surya) ENVIRONMENT=harvest-surya ;;
  *) echo "Unsupported model: $MODEL" >&2; exit 2 ;;
esac

module purge
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate "$ENVIRONMENT"

harvest-ocr --config "$CONFIG" ocr --model "$MODEL"
harvest-ocr --config "$CONFIG" parse --model "$MODEL"
harvest-ocr --config "$CONFIG" validate --model "$MODEL"
