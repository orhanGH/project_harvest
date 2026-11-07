#!/bin/bash
#SBATCH --job-name=harvest_effocr_rye
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=intelsr_devel
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail

REPO="$HOME/project_harvest/HARVEST_effocr"
CONFIG="$REPO/configs/harvest_1926_effocr_test.yaml"
RUN_DIR="$REPO/runs/effocr_test_rye_page46_v2"
REGIONS="$RUN_DIR/layout/regions.jsonl"
OUTPUT_DIR="$RUN_DIR/ocr/effocr"
WORKSPACE="/lustre/scratch/data/s6oraydi_hpc-project_harvest"

cd "$REPO"

module purge
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate harvest-effocr

export PYTHONNOUSERSITE=1
export HF_HOME="$WORKSPACE/models/huggingface"
export CONFIG RUN_DIR OUTPUT_DIR

mkdir -p \
  "$OUTPUT_DIR" \
  "$WORKSPACE/models/effocr" \
  "$HF_HOME"

test -s "$REGIONS"

echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Regions: $REGIONS"
echo "Number of cells: $(wc -l < "$REGIONS")"

python ocr_models/effocr/run_effocr_ocr.py \
  --regions-manifest "$REGIONS" \
  --output-dir "$OUTPUT_DIR" \
  --model-dir "$WORKSPACE/models/effocr" \
  --device cpu \
  --region-type cell \
  --limit "${EFFOCR_LIMIT:-0}" \
  --log-every 10

python - <<'PYRUN'
import json
import os
from pathlib import Path

from harvest_ocr.pipeline import HarvestPipeline
from harvest_ocr.utils import write_jsonl

config = Path(os.environ["CONFIG"])
run_dir = Path(os.environ["RUN_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])

raw_path = output_dir / "effocr_raw.jsonl"
standard_path = output_dir / "ocr_results.jsonl"

records = []
with raw_path.open(encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        records.append(
            {
                "region_id": row["region_id"],
                "source_model": "effocr",
                "text": row.get("text", ""),
                "confidence": None,
                "items": [],
                "raw_output_path": str(raw_path),
                "error": row.get("error"),
                "metadata": {"lines": row.get("lines", [])},
            }
        )

write_jsonl(standard_path, records)
print(f"Converted {len(records)} OCR records: {standard_path}")

pipeline = HarvestPipeline(config)
parsed = pipeline.parse("effocr")
validated, summary = pipeline.validate("effocr")
exports = pipeline.export("effocr")

print(f"Parsed: {parsed}")
print(f"Validated: {validated}")
print(f"Validation summary: {summary}")
print(f"Exports: {exports}")
PYRUN

echo "===== GENERATED CSV FILES ====="
find "$RUN_DIR/exports" -type f -name '*.csv' -print

echo "===== CSV PREVIEW ====="
find "$RUN_DIR/exports" -type f -name '*.csv' \
  -exec sed -n '1,25p' {} \;
