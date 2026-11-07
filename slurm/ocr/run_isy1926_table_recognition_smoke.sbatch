#!/bin/bash
#SBATCH --job-name=isy26_table
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/isy1926_table_rec_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/isy1926_table_rec_%j.err

set -euo pipefail

export PYTHONNOUSERSITE=1

WORK=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work

PADDLE_WORK="$WORK/paddle_ocr"

export PADDLE_PDX_CACHE_HOME="$PADDLE_WORK/cache/paddlex"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1

REPO="${HARVEST_REPO:-$HOME/project_harvest/HARVEST}"

PYTHON="$HOME/.conda/envs/harvest-paddleocr/bin/python"

PIPE="$PADDLE_WORK/runs/pipeline_dev/isy_1926_first50"

CROPS="$PIPE/ocr_smoke_5pages/crops"

OUTPUT="$PIPE/table_recognition_smoke"

rm -rf "$OUTPUT"

mkdir -p "$OUTPUT"

cd "$REPO"

echo "============================================================"
echo " HARVEST — ISY 1926 TABLE RECOGNITION SMOKE"
echo "============================================================"
echo
echo "Host:    $(hostname)"
echo "Job:     $SLURM_JOB_ID"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo
echo "Crops:"
echo "$CROPS"
echo
echo "Output:"
echo "$OUTPUT"
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
echo "=== TABLE CROPS ==="
echo

find "$CROPS" \
  -type f \
  -name 'region_*_table.png' \
  | sort

echo
echo "=== START TABLE RECOGNITION ==="
echo

"$PYTHON" -u \
  paddleocr/ocr/recognition/recognize_table_crops.py \
  --crops-root "$CROPS" \
  --output-dir "$OUTPUT" \
  --device gpu:0

echo
echo "=== OUTPUT FILES ==="
echo

find "$OUTPUT" \
  -type f \
  | sort

echo
echo "============================================================"
echo " TABLE RECOGNITION SMOKE FINISHED"
echo "============================================================"
