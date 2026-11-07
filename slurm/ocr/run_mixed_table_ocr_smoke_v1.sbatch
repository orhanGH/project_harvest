#!/bin/bash
#SBATCH --job-name=harvest_mixed_ocr_v1
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/mixed_table_ocr_v1_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/mixed_table_ocr_v1_%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1
WORK=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work
PADDLE_WORK="$WORK/paddle_ocr"
export PADDLE_PDX_CACHE_HOME="$PADDLE_WORK/cache/paddlex"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1
REPO="${HARVEST_REPO:-$HOME/project_harvest/HARVEST}"
PYTHON="$HOME/.conda/envs/harvest-paddleocr/bin/python"
OUTPUT="$PADDLE_WORK/runs/mixed_layout_test_74p/table_ocr_smoke_v1"

if [[ -e "$OUTPUT" ]]; then
    echo "Refusing to overwrite existing OCR experiment: $OUTPUT" >&2
    exit 1
fi
mkdir -p "$OUTPUT/reproducibility" "$PADDLE_WORK/logs"
cd "$REPO"
cp paddleocr/ocr/preprocessing/prepare_mixed_table_crops.py "$OUTPUT/reproducibility/"
cp slurm/ocr/run_mixed_table_ocr_smoke_v1.sbatch "$OUTPUT/reproducibility/"
cp paddleocr/ocr/recognition/recognize_table_crops.py "$OUTPUT/reproducibility/"

echo "=== HARVEST mixed-dataset table OCR smoke v1 ==="
echo "Host: $(hostname); Job: ${SLURM_JOB_ID:-unknown}"
"$PYTHON" - <<'PY'
import paddle, paddleocr
print('Paddle:', paddle.__version__, 'PaddleOCR:', getattr(paddleocr, '__version__', 'unknown'))
print('CUDA:', paddle.device.is_compiled_with_cuda(), 'GPU count:', paddle.device.cuda.device_count())
PY
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

echo "=== Cropping five selected table regions ==="
"$PYTHON" -u paddleocr/ocr/preprocessing/prepare_mixed_table_crops.py --work "$WORK" --out "$OUTPUT"
echo "=== Running existing TableRecognitionPipelineV2 on five crops ==="
"$PYTHON" -u paddleocr/ocr/recognition/recognize_table_crops.py \
  --crops-root "$OUTPUT/crops" \
  --output-dir "$OUTPUT/recognition" \
  --device gpu:0

echo "=== Output summary ==="
test -s "$OUTPUT/crop_manifest.csv"
test -s "$OUTPUT/recognition/table_recognition_summary.json"
find "$OUTPUT/recognition" -type f \( -name '*.json' -o -name '*.html' -o -name '*.xlsx' \) | sort | head -35
echo "HARVEST MIXED TABLE OCR SMOKE COMPLETED"
