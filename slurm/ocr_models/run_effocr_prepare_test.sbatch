#!/bin/bash
#SBATCH --job-name=harvest_effocr_prep
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=intelsr_devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail

REPO="$HOME/project_harvest/HARVEST_effocr"
CONFIG="$REPO/configs/harvest_1926_effocr_test.yaml"
RUN_DIR="$REPO/runs/effocr_test_rye_page46_v2"
WORKSPACE="/lustre/scratch/data/s6oraydi_hpc-project_harvest"

cd "$REPO"

module purge
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate harvest-effocr
export PYTHONNOUSERSITE=1

echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Configuration: $CONFIG"

python -m harvest_ocr.cli --config "$CONFIG" extract
python -m harvest_ocr.cli --config "$CONFIG" preprocess
python -m harvest_ocr.cli --config "$CONFIG" layout

REGIONS="$RUN_DIR/layout/regions.jsonl"

python ocr_models/effocr/run_effocr_ocr.py   --regions-manifest "$REGIONS"   --output-dir "$RUN_DIR/ocr/effocr"   --model-dir "$WORKSPACE/models/effocr"   --region-type cell   --validate-only

python - <<'CHECK'
import json
from collections import Counter
from pathlib import Path

manifest = Path("runs/effocr_test_rye_page46_v2/layout/regions.jsonl")
records = [
    json.loads(line)
    for line in manifest.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

cells = [row for row in records if row["region_type"] == "cell"]
fields = Counter(row["column_id"] for row in cells)
rows = {row["row_index"] for row in cells}

print("Total regions:", len(records))
print("Detected rows:", len(rows))
print("Cell crops:", len(cells))
print("Cells by field:", dict(fields))
print("Debug overlay: runs/effocr_test_rye_page46_v2/layout/debug/page-0001.png")

expected = {"area_1925", "production_1925", "yield_1925", "country_en"}
if set(fields) != expected:
    raise SystemExit(f"Unexpected fields: {sorted(fields)}")
if len(cells) != 4 * len(rows):
    raise SystemExit("The layout did not create exactly four cells per row")
CHECK
