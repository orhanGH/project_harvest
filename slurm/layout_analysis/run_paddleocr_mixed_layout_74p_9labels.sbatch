#!/bin/bash
#SBATCH --job-name=mixed9_layout
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/mixed_layout_test_74p_9labels_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/mixed_layout_test_74p_9labels_%j.err

set -euo pipefail

# ============================================================
# HARVEST - Mixed 74-page layout detection
# PP-DocLayout_plus-L
#
# Selected classes kept at threshold 0.50:
#
#   paragraph_title
#   text
#   number
#   content
#   figure_title
#   table
#   doc_title
#   footnote
#   header
#
# All remaining pretrained classes use threshold 1.00.
# ============================================================


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

export PYTHONNOUSERSITE=1

WORK=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work
PADDLE_WORK="$WORK/paddle_ocr"

export PADDLE_PDX_CACHE_HOME="$PADDLE_WORK/cache/paddlex"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1

REPO="${HARVEST_REPO:-$HOME/project_harvest/HARVEST}"

PYTHON="$HOME/.conda/envs/harvest-paddleocr/bin/python"

PDF="$WORK/input/mixed_layout_test_74p/mixed_layout_test_74p.pdf"

OUTPUT="$PADDLE_WORK/runs/mixed_layout_test_74p/9labels_050"

mkdir -p "$PADDLE_WORK/logs"


# ------------------------------------------------------------
# Start clean
# ------------------------------------------------------------

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

cd "$REPO"


# ------------------------------------------------------------
# Run information
# ------------------------------------------------------------

echo "============================================================"
echo "HARVEST - PP-DocLayout_plus-L"
echo "Mixed 74-page benchmark"
echo "Selected 9-label experiment"
echo "============================================================"
echo

echo "Host:"
hostname

echo
echo "Job ID:"
echo "${SLURM_JOB_ID}"

echo
echo "Account:"
echo "${SLURM_JOB_ACCOUNT:-unknown}"

echo
echo "PDF:"
echo "$PDF"

echo
echo "Output:"
echo "$OUTPUT"

echo
echo "Detector script:"
echo "paddleocr/layout_analysis/detection/detect_layout_9labels_pdf.py"

echo
echo "Model:"
echo "PP-DocLayout_plus-L"

echo
echo "DPI:"
echo "180"

echo
echo "Selected classes @ threshold 0.50:"
echo "  paragraph_title"
echo "  text"
echo "  number"
echo "  content"
echo "  figure_title"
echo "  table"
echo "  doc_title"
echo "  footnote"
echo "  header"

echo
echo "Remaining pretrained classes:"
echo "  threshold = 1.00"

echo


# ------------------------------------------------------------
# Python / Paddle check
# ------------------------------------------------------------

"$PYTHON" --version

echo

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
    "Current Paddle device:",
    paddle.device.get_device(),
)
PY

echo

nvidia-smi

echo


# ------------------------------------------------------------
# Run layout detection
# ------------------------------------------------------------

echo "=== Starting layout detection ==="

"$PYTHON" -u \
  paddleocr/layout_analysis/detection/detect_layout_9labels_pdf.py \
  --pdf "$PDF" \
  --output-dir "$OUTPUT" \
  --pages 1-74 \
  --model PP-DocLayout_plus-L \
  --device gpu:0 \
  --dpi 180 \
  --table-score-threshold 0.50

echo
echo "=== Layout detection finished ==="
echo


# ------------------------------------------------------------
# Verify output
# ------------------------------------------------------------

RAW_COUNT=$(
    find "$OUTPUT/raw" \
      -maxdepth 1 \
      -type f \
      -name 'page_*.json' \
      | wc -l
)

RENDERED_COUNT=$(
    find "$OUTPUT/rendered" \
      -maxdepth 1 \
      -type f \
      -name 'page_*.png' \
      | wc -l
)

VIS_COUNT=$(
    find "$OUTPUT/visualizations" \
      -maxdepth 1 \
      -type f \
      -name '*.png' \
      | wc -l
)

echo "Raw JSON files:"
echo "$RAW_COUNT"

echo
echo "Rendered pages:"
echo "$RENDERED_COUNT"

echo
echo "Visualization files:"
echo "$VIS_COUNT"

if [ "$RAW_COUNT" -ne 74 ]; then
    echo
    echo "ERROR: expected 74 raw JSON files."
    exit 1
fi


# ------------------------------------------------------------
# Print detected label counts
# ------------------------------------------------------------

"$PYTHON" - "$OUTPUT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

output = Path(sys.argv[1])
raw_dir = output / "raw"

counts = Counter()

scores = {}

for path in sorted(raw_dir.glob("page_*.json")):

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    payload = data.get("res", data)

    for box in payload.get("boxes", []):

        label = str(
            box.get("label", "UNKNOWN")
        )

        score = float(
            box.get("score", 0.0)
        )

        counts[label] += 1

        scores.setdefault(
            label,
            [],
        ).append(score)


print()
print("=== DETECTED LABEL COUNTS ===")
print()

for label in sorted(counts):

    values = scores[label]

    mean_score = (
        sum(values) / len(values)
        if values
        else 0.0
    )

    print(
        f"{label:25s} "
        f"{counts[label]:4d} "
        f"mean_score={mean_score:.4f}"
    )

print()
print(
    "Total retained detections:",
    sum(counts.values()),
)

print()
print("Expected selected classes:")

expected = [
    "paragraph_title",
    "text",
    "number",
    "content",
    "figure_title",
    "table",
    "doc_title",
    "footnote",
    "header",
]

for label in expected:
    print(
        f"  {label:25s} "
        f"{counts.get(label, 0):4d}"
    )

unexpected = sorted(
    set(counts) - set(expected)
)

print()

if unexpected:
    print(
        "WARNING: unexpected classes were still returned:"
    )

    for label in unexpected:
        print(
            f"  {label:25s} "
            f"{counts[label]:4d}"
        )
else:
    print(
        "OK: only the selected 9 classes were retained."
    )
PY


echo
echo "============================================================"
echo "Experiment completed successfully."
echo
echo "Output:"
echo "$OUTPUT"
echo "============================================================"
