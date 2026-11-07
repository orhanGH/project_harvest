#!/bin/bash
#SBATCH --job-name=harvest_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail

CONFIG=${1:-configs/harvest_1926.yaml}
mkdir -p runs/slurm

module purge
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate harvest-tesseract

harvest-ocr --config "$CONFIG" extract
harvest-ocr --config "$CONFIG" preprocess
harvest-ocr --config "$CONFIG" layout
harvest-ocr --config "$CONFIG" ocr --model tesseract
harvest-ocr --config "$CONFIG" parse --model tesseract
harvest-ocr --config "$CONFIG" validate --model tesseract
harvest-ocr --config "$CONFIG" export --model tesseract
