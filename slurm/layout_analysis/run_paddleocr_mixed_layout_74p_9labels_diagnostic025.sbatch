#!/bin/bash
#SBATCH --job-name=mixed9_diag25
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/mixed_layout_test_74p_9labels_diag025_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/mixed_layout_test_74p_9labels_diag025_%j.err

set -euo pipefail

export PYTHONNOUSERSITE=1

WORK=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work
PADDLE_WORK="$WORK/paddle_ocr"

export PADDLE_PDX_CACHE_HOME="$PADDLE_WORK/cache/paddlex"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1

REPO="${HARVEST_REPO:-$HOME/project_harvest/HARVEST}"

PYTHON="$HOME/.conda/envs/harvest-paddleocr/bin/python"

PDF="$WORK/input/mixed_layout_test_74p/mixed_layout_test_74p.pdf"

OUTPUT="$PADDLE_WORK/runs/mixed_layout_test_74p/9labels_diagnostic025"

mkdir -p "$PADDLE_WORK/logs"

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

cd "$REPO"

echo "============================================================"
echo "HARVEST 74-page mixed layout diagnostic"
echo "PP-DocLayout_plus-L"
echo "============================================================"
echo
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo
echo "Thresholds:"
echo "  paragraph_title = 0.50"
echo "  text            = 0.50"
echo "  number          = 0.50"
echo "  content         = 0.25"
echo "  figure_title    = 0.25"
echo "  table           = 0.50"
echo "  doc_title       = 0.50"
echo "  footnote        = 0.25"
echo "  header          = 0.25"
echo
echo "All other pretrained classes = 1.00"
echo
echo "PDF: $PDF"
echo "Output: $OUTPUT"
echo

"$PYTHON" --version

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
PY

echo
nvidia-smi
echo

"$PYTHON" -u \
  paddleocr/layout_analysis/detection/detect_layout_9labels_diagnostic025_pdf.py \
  --pdf "$PDF" \
  --output-dir "$OUTPUT" \
  --pages 1-74 \
  --model PP-DocLayout_plus-L \
  --device gpu:0 \
  --dpi 180 \
  --table-score-threshold 0.50

echo
echo "=== CHECKING OUTPUT ==="

RAW_COUNT=$(
    find "$OUTPUT/raw" \
      -maxdepth 1 \
      -name 'page_*.json' \
      | wc -l
)

echo "Raw pages: $RAW_COUNT"

if [ "$RAW_COUNT" -ne 74 ]; then
    echo "ERROR: expected 74 pages"
    exit 1
fi

"$PYTHON" - "$OUTPUT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1]) / "raw"

counts = Counter()
scores = {}

for path in sorted(
    root.glob("page_*.json")
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
        label = box["label"]
        score = float(
            box["score"]
        )

        counts[label] += 1

        scores.setdefault(
            label,
            [],
        ).append(score)

print()
print("=== DETECTION COUNTS ===")
print()

for label in sorted(counts):
    values = scores[label]

    print(
        f"{label:25s} "
        f"{counts[label]:4d} "
        f"mean={sum(values)/len(values):.4f} "
        f"min={min(values):.4f}"
    )

print()
print(
    "Total:",
    sum(counts.values()),
)
PY

echo
echo "============================================================"
echo "Diagnostic run finished."
echo "$OUTPUT"
echo "============================================================"
