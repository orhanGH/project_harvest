#!/bin/bash

#SBATCH --job-name=yearbook_layout
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

#SBATCH --array=0-16%4

#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/yearbook_layout_analysis/layout_%A_%a.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/yearbook_layout_analysis/layout_%A_%a.err


set -euo pipefail


YEAR=$((1926 + SLURM_ARRAY_TASK_ID))

WORK=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work

INPUT_ROOT="$WORK/input/International_Statistical_Yearbook"
OUTPUT_ROOT="$WORK/paddle_ocr/runs/yearbook_layout_analysis"

YEAR_INPUT="$INPUT_ROOT/$YEAR"
OUTPUT="$OUTPUT_ROOT/$YEAR"


PYTHON="$HOME/.conda/envs/harvest-paddleocr/bin/python"

export PYTHONNOUSERSITE=1

export PADDLE_PDX_CACHE_HOME="$WORK/paddle_ocr/cache/paddlex"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1


REPO="${HARVEST_REPO:-$HOME/project_harvest/HARVEST}"


echo "=============================================="
echo " HARVEST — International Statistical Yearbook"
echo "=============================================="
echo "Year:       $YEAR"
echo "Job:        $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Host:       $(hostname)"
echo "Account:    ${SLURM_JOB_ACCOUNT:-unknown}"
echo


# ------------------------------------------------------------
# Locate exactly one PDF for this year
# ------------------------------------------------------------

mapfile -t PDFS < <(
    find "$YEAR_INPUT" \
        -maxdepth 1 \
        -type f \
        -iname '*.pdf' \
        | sort
)

if [ "${#PDFS[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one PDF in:"
    echo "  $YEAR_INPUT"
    echo "Found: ${#PDFS[@]}"

    printf '%s\n' "${PDFS[@]:-}"

    exit 1
fi


PDF="${PDFS[0]}"


echo "PDF:"
echo "  $PDF"

echo
echo "Output:"
echo "  $OUTPUT"

echo


# ------------------------------------------------------------
# Clean this year's output only
# ------------------------------------------------------------

rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"


cd "$REPO"


echo "Python:"
"$PYTHON" --version

echo

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


# ------------------------------------------------------------
# Layout detection
# ------------------------------------------------------------

echo "=== Starting layout detection for $YEAR ==="

"$PYTHON" -u paddleocr/layout_analysis/detection/detect_tables_pdf.py \
    --pdf "$PDF" \
    --output-dir "$OUTPUT" \
    --model PP-DocLayout_plus-L \
    --device gpu:0 \
    --dpi 180 \
    --table-score-threshold 0.50


# ------------------------------------------------------------
# Generate compact per-year summary
# ------------------------------------------------------------

"$PYTHON" - "$OUTPUT" "$YEAR" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
year = int(sys.argv[2])

tables_path = output / "tables.jsonl"

rows = [
    json.loads(line)
    for line in tables_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

with_tables = []
without_tables = []
scores = []
total_tables = 0

for row in rows:
    tables = row["tables"]
    n = len(tables)

    total_tables += n

    if n:
        with_tables.append(row["page_number"])
    else:
        without_tables.append(row["page_number"])

    scores.extend(
        float(table["confidence"])
        for table in tables
    )


summary = {
    "year": year,
    "pages_processed": len(rows),
    "pages_with_detected_tables": len(with_tables),
    "pages_without_detected_tables": len(without_tables),
    "detected_table_regions": total_tables,
    "table_confidence_min": min(scores) if scores else None,
    "table_confidence_max": max(scores) if scores else None,
    "table_confidence_mean": (
        sum(scores) / len(scores)
        if scores
        else None
    ),
    "pages_with_detected_tables_list": with_tables,
    "pages_without_detected_tables_list": without_tables,
}

summary_path = output / "summary.json"

summary_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print()
print("=== YEAR SUMMARY ===")
print("Year:", year)
print("Pages processed:", len(rows))
print("Pages with detected tables:", len(with_tables))
print("Pages without detected tables:", len(without_tables))
print("Detected table regions:", total_tables)

if scores:
    print("Mean table confidence:", round(sum(scores) / len(scores), 4))
    print("Minimum table confidence:", round(min(scores), 4))
    print("Maximum table confidence:", round(max(scores), 4))

print("Summary:", summary_path)
PY


echo
echo "=== $YEAR completed successfully ==="
