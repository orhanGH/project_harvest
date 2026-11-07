#!/bin/bash
#SBATCH --job-name=e05_table038
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/e05_table038_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/paddle_ocr/logs/e05_table038_%j.err

set -euo pipefail

export PYTHONNOUSERSITE=1

WORK=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work
REPO="${HARVEST_REPO:-$HOME/project_harvest/HARVEST}"
PYTHON="$HOME/.conda/envs/harvest-paddleocr/bin/python"

export PADDLE_PDX_CACHE_HOME="$WORK/paddle_ocr/cache/paddlex"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1

PDF="$WORK/input/mixed_layout_test_74p/mixed_layout_test_74p.pdf"

case "${RUN_MODE:-smoke}" in
    smoke)
        PAGES="2,6"
        RUN_NAME="e05_table038_smoke"
        EXPECTED=2
        ;;
    full)
        PAGES="1-74"
        RUN_NAME="e05_table038_full"
        EXPECTED=74
        ;;
    *)
        echo "Invalid RUN_MODE" >&2
        exit 1
        ;;
esac

OUTPUT="$WORK/paddle_ocr/runs/mixed_layout_test_74p/$RUN_NAME"

if [ -e "$OUTPUT" ]; then
    echo "Output already exists: $OUTPUT" >&2
    exit 1
fi

mkdir -p "$OUTPUT/reproducibility"

cp "$REPO/paddleocr/layout_analysis/detection/detect_layout_9labels_table038_pdf.py" \
   "$OUTPUT/reproducibility/"

cp "$REPO/slurm/layout_analysis/run_e05_table038.sbatch" \
   "$OUTPUT/reproducibility/"

cd "$REPO"

echo "=== HARVEST E05 ==="
echo "Pages: $PAGES"
echo "Table threshold: 0.38"
echo "Other eight selected classes: 0.50"
echo "Other pretrained classes: 1.00"
echo "Output: $OUTPUT"

"$PYTHON" -u \
    paddleocr/layout_analysis/detection/detect_layout_9labels_table038_pdf.py \
    --pdf "$PDF" \
    --output-dir "$OUTPUT" \
    --pages "$PAGES" \
    --model PP-DocLayout_plus-L \
    --device gpu:0 \
    --dpi 180 \
    --table-score-threshold 0.38

COUNT=$(find "$OUTPUT/raw" -maxdepth 1 \
    -name 'page_*.json' | wc -l)

echo "Processed pages: $COUNT"

if [ "$COUNT" -ne "$EXPECTED" ]; then
    echo "Incomplete E05 run" >&2
    exit 1
fi

echo "E05 completed successfully."
