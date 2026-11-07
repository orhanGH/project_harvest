#!/bin/bash
#SBATCH --job-name=isy26_layout50
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/isy1926_first50_layout_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/isy1926_first50_layout_%j.err

set -euo pipefail

export PYTHONNOUSERSITE=1

WORK=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work
PADDLE_WORK="$WORK/paddle_ocr"

export PADDLE_PDX_CACHE_HOME="$PADDLE_WORK/cache/paddlex"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1

REPO="${HARVEST_REPO:-$HOME/project_harvest/HARVEST}"

PYTHON="$HOME/.conda/envs/harvest-paddleocr/bin/python"

PDF="$WORK/input/pipeline_dev/isy_1926_first50/ISY_1926_pages_001_050.pdf"

OUTPUT="$PADDLE_WORK/runs/pipeline_dev/isy_1926_first50/layout_9labels_050"

mkdir -p "$PADDLE_WORK/logs"

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

cd "$REPO"

echo "=============================================="
echo "HARVEST — ISY 1926 pages 1-50"
echo "=============================================="
echo
echo "Host:    $(hostname)"
echo "Job:     $SLURM_JOB_ID"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "PDF:     $PDF"
echo "Output:  $OUTPUT"
echo

"$PYTHON" --version

echo
echo "=== Paddle environment ==="

"$PYTHON" - <<'PY'
import paddle
import paddleocr

print("Paddle:", paddle.__version__)
print(
    "PaddleOCR:",
    getattr(paddleocr, "__version__", "unknown"),
)
print(
    "CUDA compiled:",
    paddle.device.is_compiled_with_cuda(),
)
print(
    "GPU count:",
    paddle.device.cuda.device_count(),
)
print(
    "Device:",
    paddle.device.get_device(),
)
PY

echo

nvidia-smi \
  --query-gpu=name,memory.total,memory.used \
  --format=csv,noheader

echo
echo "=== Running 9-label layout detector ==="
echo

"$PYTHON" -u \
  paddleocr/layout_analysis/detection/detect_layout_9labels_pdf.py \
  --pdf "$PDF" \
  --output-dir "$OUTPUT" \
  --model PP-DocLayout_plus-L \
  --device gpu:0 \
  --dpi 180

echo
echo "=== Checking result ==="
echo

RAW="$OUTPUT/raw"

COUNT=$(
    find "$RAW" \
        -maxdepth 1 \
        -type f \
        -name 'page_*.json' \
        | wc -l
)

echo "Page JSON files: $COUNT"

if [ "$COUNT" -ne 50 ]; then
    echo "ERROR: expected 50 page JSON files."
    exit 1
fi

echo
echo "=== Detected label counts ==="
echo

"$PYTHON" - "$RAW" <<'PY'
import json
import sys

from collections import Counter
from pathlib import Path


raw_dir = Path(sys.argv[1])

counts = Counter()


for path in sorted(
    raw_dir.glob("page_*.json")
):
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    payload = data.get(
        "res",
        data,
    )

    for box in payload.get(
        "boxes",
        [],
    ):
        label = str(
            box.get(
                "label",
                "UNKNOWN",
            )
        )

        counts[label] += 1


print(
    f"{'Label':25s} Count"
)

print("-" * 34)

for label in sorted(counts):
    print(
        f"{label:25s} "
        f"{counts[label]:5d}"
    )

print()
print(
    "TOTAL:",
    sum(counts.values()),
)
PY

echo
echo "=============================================="
echo "LAYOUT RUN FINISHED SUCCESSFULLY"
echo "=============================================="
