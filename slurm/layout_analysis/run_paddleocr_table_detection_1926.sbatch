#!/bin/bash
#SBATCH --job-name=paddle_tables_1926
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/paddle_tables_1926_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/paddle_tables_1926_%j.err

set -euo pipefail

export PYTHONNOUSERSITE=1

PADDLE_WORK=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr
export PADDLE_PDX_CACHE_HOME="$PADDLE_WORK/cache/paddlex"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1

REPO="${HARVEST_REPO:-$HOME/project_harvest/HARVEST}"

PYTHON="$HOME/.conda/envs/harvest-paddleocr/bin/python"

PDF=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/input/statistical-yearbook-1926/0000765723_D0001.pdf

OUTPUT="$PADDLE_WORK/runs/table_detection/pages_047_192"

mkdir -p "$PADDLE_WORK/logs"

# Start this run cleanly.
rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

cd "$REPO"

echo "=== HARVEST PaddleOCR table detection ==="
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "PDF: $PDF"
echo "Output: $OUTPUT"
echo "Python: $PYTHON"
echo "Paddle cache: $PADDLE_PDX_CACHE_HOME"
echo

"$PYTHON" --version

"$PYTHON" - <<'PY'
import paddle
import paddleocr

print("Paddle:", paddle.__version__)
print("PaddleOCR:", getattr(paddleocr, "__version__", "unknown"))
print("CUDA compiled:", paddle.device.is_compiled_with_cuda())
print("GPU count:", paddle.device.cuda.device_count())
print("Device:", paddle.device.get_device())
PY

echo
nvidia-smi

echo
echo "=== Running pages 47-192 ==="

"$PYTHON" paddleocr/layout_analysis/detection/detect_tables_pdf.py \
  --pdf "$PDF" \
  --output-dir "$OUTPUT" \
  --pages 47-192 \
  --model PP-DocLayout_plus-L \
  --device gpu:0 \
  --dpi 180 \
  --table-score-threshold 0.50

echo
echo "=== Completed successfully ==="