#!/bin/bash
#SBATCH --job-name=isy26_ocr5
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/isy1926_region_ocr_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/isy1926_region_ocr_%j.err

set -euo pipefail

export PYTHONNOUSERSITE=1

WORK=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work

PADDLE_WORK="$WORK/paddle_ocr"

export PADDLE_PDX_CACHE_HOME="$PADDLE_WORK/cache/paddlex"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1

REPO="${HARVEST_REPO:-$HOME/project_harvest/HARVEST}"

PYTHON="$HOME/.conda/envs/harvest-paddleocr/bin/python"

PIPE="$PADDLE_WORK/runs/pipeline_dev/isy_1926_first50"

RENDERED="$PIPE/layout_9labels_050/rendered"

ORIGINAL="$PIPE/layout_9labels_050/raw"

MERGED="$PIPE/layout_5class_merged/raw"

OUTPUT="$PIPE/ocr_smoke_5pages"

rm -rf "$OUTPUT"

mkdir -p "$OUTPUT"

cd "$REPO"

echo "=============================================="
echo " HARVEST — ISY 1926 REGION OCR SMOKE"
echo "=============================================="
echo
echo "Host: $(hostname)"
echo "Job:  $SLURM_JOB_ID"
echo
echo "Pages:"
echo "20,40,41,46,50"
echo

"$PYTHON" - <<'PY'
import paddle
import paddleocr

print(
    "Paddle:",
    paddle.__version__,
)

print(
    "PaddleOCR:",
    getattr(
        paddleocr,
        "__version__",
        "unknown",
    ),
)

print(
    "CUDA:",
    paddle.device.is_compiled_with_cuda(),
)

print(
    "GPU count:",
    paddle.device.cuda.device_count(),
)
PY

echo

nvidia-smi \
  --query-gpu=name,memory.total,memory.used \
  --format=csv,noheader

echo
echo "=== Starting region OCR ==="
echo

"$PYTHON" -u \
  paddleocr/ocr/recognition/ocr_task_aligned_regions.py \
  --rendered-dir "$RENDERED" \
  --merged-dir "$MERGED" \
  --original-dir "$ORIGINAL" \
  --output-dir "$OUTPUT" \
  --pages "20,40,41,46,50" \
  --device gpu:0

echo
echo "=== Structured outputs ==="
echo

find "$OUTPUT/structured" \
  -maxdepth 1 \
  -name 'page_*.json' \
  -print

echo
echo "=============================================="
echo " OCR SMOKE COMPLETED"
echo "=============================================="
