# project_harvest

Research scripts and Python package for **HARVEST**, a pipeline for detecting and reading statistical tables in historical agricultural publications. This repository consolidates scripts developed locally and on the University of Bonn Marvin HPC system (snapshot: 7 November 2026).

> **Status:** Archived experimental scripts plus an evolving OCR package. The source and job scripts were reorganized and checked for Python/Bash syntax; the reorganized HPC workflows have **not** been re-run on Marvin. The original experiments and output datasets remain in the original Marvin workspace. This repository does **not** ship source PDFs, manual annotations, model weights, or the original `configs/` files.

## What we did

- **Layout analysis:** PP-DocLayout_plus-L detection on a mixed 74-page dataset; baseline and nine-class threshold experiments (E01–E03).
- **Low-confidence diagnostics (E04):** inspected table and other region predictions surviving a score cutoff of 0.01 to explain missed annotations.
- **Table threshold test (E05):** compared table score 0.38 with baseline score 0.50; table recall increased, while table F1 decreased due to extra false positives.
- **Table OCR smoke test:** cropped five selected regions from E02/E05 predictions and ran PaddleOCR `TableRecognitionPipelineV2` with `PP-OCRv6_medium_rec`, producing image, JSON, HTML and XLSX outputs. These outputs have not yet been independently assessed for extraction accuracy.
- **Related workflows:** Tesseract, EffOCR, Surya, Chandra adapters; deterministic extraction/validation; Table Transformer experimentation.

## Repository layout

| Directory | Contents |
|---|---|
| `paddleocr/layout_analysis/detection/` | Layout inference scripts |
| `paddleocr/layout_analysis/evaluation/` | IoU and detection diagnostic evaluation |
| `paddleocr/layout_analysis/diagnostics/` | Low-confidence and layout postprocessing experiments |
| `paddleocr/datasets/` | Mixed test-set construction scripts |
| `paddleocr/ocr/preprocessing/` | Region crop preparation |
| `paddleocr/ocr/recognition/` | Table and text recognition |
| `paddleocr/extraction/` | Table parsing and evaluation |
| `visualisation/` | Rendered comparisons and OCR visualization code (not outputs) |
| `slurm/` | Marvin CPU/GPU job scripts, organized by task |
| `ocr_models/` | Other OCR model runners |
| `table_detection/` | Table Transformer detection script |
| `src/harvest_ocr/` | Reusable Python package and CLI |
| `tests/` | Unit tests from the original HARVEST checkout |
| `pipeline/`, `postprocess/`, `marvin/` | Pipeline wrapper, validation and HPC setup helpers |
| `docs/` | Original README snapshots and migration notes |

There is intentionally **no top-level `scripts/` directory**. Entrypoints were moved directly under their functional categories.

## Requirements and running

The common package metadata is in `pyproject.toml` and requires Python >= 3.10. Optional OCR engines need separate installations and, for PaddleOCR GPU, a compatible PaddlePaddle CUDA environment. The historical OCR experiments were run using existing environments on Marvin; installing `pip install -e '.[base,dev]'` is only the common baseline and does **not** establish GPU compatibility.

To inspect the existing pipeline's CLI after supplying its missing configuration and data files:

```bash
python -m pip install -e '.[base,dev]'
python -m pytest
```

**Known limitation:** `tests/test_target_config.py` requires `configs/harvest_1926.yaml`, which was not provided in the Marvin snapshots. Other workflows can require files in `configs/`, `docs/` and raw historical scans. The two original config-specific tests are skipped until the missing configuration is supplied; this is not evidence that the complete pipeline is reproducible.

### Marvin Slurm scripts

The Slurm job templates retain experiment-specific paths (such as output directories, account/partition names and conda environments). They are **historical reproducibility records, not portable defaults**. Most jobs accept an optional `HARVEST_REPO` environment override for the reorganized copy; jobs originally run from `HARVEST_effocr` may still expect that separate checkout. Check paths and data dependencies before submitting.

```bash
export HARVEST_REPO="$HOME/project_harvest/project_harvest"
sbatch slurm/ocr/run_mixed_table_ocr_smoke_v1.sbatch
```

Run inference only in the appropriate Slurm compute allocation, not on Marvin's login node. The example above assumes you first cloned this repository at the path shown and that the existing Lustre inputs/model cache are available. For a safe first attempt, use an unused output directory or review the job's overwrite guard.

## Source and data provenance

- Code: 93 files from `HARVEST_marvin_code_20261107.tar.gz` plus two unique local visualization scripts from `scripts.zip`.
- Config and original explanation: `HARVEST_project_metadata_20261107.tar.gz`.
- Large experiment outputs, images and restricted historical source documents are **not included**.
- Research results and recovered OCR text need to be checked against the original scanned tables before being treated as correct.

See [`docs/MIGRATION_NOTES.md`](docs/MIGRATION_NOTES.md) for source-to-destination changes and caveats. The earlier full HARVEST CLI instructions are preserved in [`docs/ORIGINAL_HARVEST_README.md`](docs/ORIGINAL_HARVEST_README.md) but use the *original* repository layout.
