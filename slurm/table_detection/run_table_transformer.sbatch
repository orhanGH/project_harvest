#!/bin/bash
#SBATCH --job-name=tatr_detect
#SBATCH --account=ag_ilr_wuepper
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/gap_visio/table_transformer_test/tatr_%j.out
#SBATCH --error=/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work/gap_visio/table_transformer_test/tatr_%j.err

set -euo pipefail

BASE=/lustre/scratch/data/s6oraydi_hpc-project_harvest
WORK="$BASE/s6oraydi_work/gap_visio/table_transformer_test"
PDF="$BASE/s6oraydi_work/input/statistical-yearbook-1926/0000765723_D0001.pdf"

export HF_HOME="$BASE/models/huggingface"
export HF_HUB_OFFLINE=1

source /opt/software/easybuild-INTEL/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate pbpb_pp_mamba

echo "===== HARVEST TABLE TRANSFORMER ====="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "PDF: $PDF"

echo
nvidia-smi --query-gpu=name,memory.total --format=csv

echo
python - <<PY
import pymupdf
doc = pymupdf.open("$PDF")
print("PDF pages:", len(doc))
assert len(doc) > 0
PY

rm -rf "$WORK/output"
mkdir -p "$WORK/output"

echo
echo "Starting detection..."

python -u "$WORK/detect_tables_pdf.py" \
    --pdf "$PDF" \
    --output-dir "$WORK/output" \
    --start-page 20 \
    --end-page 45 \
    --dpi 180 \
    --threshold 0.50

echo
echo "Finished: $(date)"
