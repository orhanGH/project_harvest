#!/bin/bash
#SBATCH --job-name=harvest_e04
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/e04_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/e04_%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1
WORK=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work
PADDLE_WORK="$WORK/paddle_ocr"
export PADDLE_PDX_CACHE_HOME="$PADDLE_WORK/cache/paddlex"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1
PYTHON="$HOME/.conda/envs/harvest-paddleocr/bin/python"
REPO="${HARVEST_REPO:-$HOME/project_harvest/HARVEST}"
INPUT_DIR="$PADDLE_WORK/runs/mixed_layout_test_74p/baseline_050/rendered"
GT_DIR="$WORK/input/mixed_layout_test_74p/labels"
E02_RAW="$PADDLE_WORK/runs/mixed_layout_test_74p/9labels_050/raw"
E03_RAW="$PADDLE_WORK/runs/mixed_layout_test_74p/9labels_diagnostic025/raw"
PAGES="${PAGES:-2,6}"
OUTPUT_DIR="${OUTPUT_DIR:-$PADDLE_WORK/runs/mixed_layout_test_74p/e04_lowconf_test}"
mkdir -p "$PADDLE_WORK/logs"
cd "$REPO"
"$PYTHON" --version
"$PYTHON" - <<'PY'
import paddle, paddleocr
print('Paddle:', paddle.__version__)
print('PaddleOCR:', getattr(paddleocr, '__version__', 'unknown'))
print('CUDA:', paddle.device.is_compiled_with_cuda())
print('GPU count:', paddle.device.cuda.device_count())
PY
nvidia-smi
"$PYTHON" -u paddleocr/layout_analysis/diagnostics/e04_low_confidence.py \
  --images "$INPUT_DIR" \
  --labels "$GT_DIR" \
  --e02-raw "$E02_RAW" \
  --e03-raw "$E03_RAW" \
  --out "$OUTPUT_DIR" \
  --pages "$PAGES" \
  --threshold 0.01 \
  --device gpu:0
