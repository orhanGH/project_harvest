# Migration notes

This is a **copy** of the supplied code: nothing on Marvin or in the local original folder was modified.

Script code references of the form `scripts/<path>` were updated where the corresponding source file was moved.
Slurm jobs may still contain historical Marvin absolute paths, output paths and models. Their run-time behavior has **not been validated after relocation**.
`HARVEST_REPO` is supported by most jobs that originally had a fixed `REPO="$HOME/project_harvest/HARVEST"` assignment. Other jobs may rely on `HARVEST_effocr`.
`configs/`, full `docs/` and `environments/` are **not present** in the supplied Marvin archives, so the common HARVEST CLI and some tests are not fully reproducible from this source snapshot.

## Original-to-new paths

| Original relative path | New path |
|---|---|
| `scripts/marvin/setup_tesseract.sh` | `marvin/setup_tesseract.sh` |
| `scripts/ocr_models/chandra/run_chandra_ocr.py` | `ocr_models/chandra/run_chandra_ocr.py` |
| `scripts/ocr_models/effocr/run_effocr_ocr.py` | `ocr_models/effocr/run_effocr_ocr.py` |
| `scripts/ocr_models/effocr/run_effocr_threshold_override.py` | `ocr_models/effocr/run_effocr_threshold_override.py` |
| `scripts/ocr_models/paddle/run_paddle_ocr.py` | `ocr_models/paddle/run_paddle_ocr.py` |
| `scripts/ocr_models/surya/run_surya_ocr.py` | `ocr_models/surya/run_surya_ocr.py` |
| `scripts/ocr_models/tesseract/ocr_crop.py` | `ocr_models/tesseract/ocr_crop.py` |
| `scripts/ocr_models/tesseract/tesseract_ocr.py` | `ocr_models/tesseract/tesseract_ocr.py` |
| `scripts/paddleocr/analyze_layout_confusion.py` | `paddleocr/layout_analysis/evaluation/analyze_layout_confusion.py` |
| `scripts/paddleocr/analyze_layout_geometry.py` | `paddleocr/layout_analysis/evaluation/analyze_layout_geometry.py` |
| `scripts/paddleocr/batch_parse_isy_tables.py` | `paddleocr/extraction/batch_parse_isy_tables.py` |
| `scripts/paddleocr/build_mixed_layout_test.py` | `paddleocr/datasets/build_mixed_layout_test.py` |
| `scripts/paddleocr/debug_isy_symbols.py` | `paddleocr/extraction/debug_isy_symbols.py` |
| `scripts/paddleocr/detect_layout_9labels_diagnostic025_pdf.py` | `paddleocr/layout_analysis/detection/detect_layout_9labels_diagnostic025_pdf.py` |
| `scripts/paddleocr/detect_layout_9labels_pdf.py` | `paddleocr/layout_analysis/detection/detect_layout_9labels_pdf.py` |
| `scripts/paddleocr/detect_layout_9labels_table038_pdf.py` | `paddleocr/layout_analysis/detection/detect_layout_9labels_table038_pdf.py` |
| `scripts/paddleocr/detect_tables_pdf.py` | `paddleocr/layout_analysis/detection/detect_tables_pdf.py` |
| `scripts/paddleocr/e04_low_confidence.py` | `paddleocr/layout_analysis/diagnostics/e04_low_confidence.py` |
| `scripts/paddleocr/evaluate_isy_dev_extraction.py` | `paddleocr/extraction/evaluate_isy_dev_extraction.py` |
| `scripts/paddleocr/evaluate_isy_strict.py` | `paddleocr/extraction/evaluate_isy_strict.py` |
| `scripts/paddleocr/evaluate_layout_detection.py` | `paddleocr/layout_analysis/evaluation/evaluate_layout_detection.py` |
| `scripts/paddleocr/extract_1925_geometry.py` | `paddleocr/extraction/extract_1925_geometry.py` |
| `scripts/paddleocr/inspect_table_cells.py` | `visualisation/ocr/inspect_table_cells.py` |
| `scripts/paddleocr/merge_layout_predictions.py` | `paddleocr/layout_analysis/diagnostics/merge_layout_predictions.py` |
| `scripts/paddleocr/ocr_task_aligned_regions.py` | `paddleocr/ocr/recognition/ocr_task_aligned_regions.py` |
| `scripts/paddleocr/parse_isy_table.py` | `paddleocr/extraction/parse_isy_table.py` |
| `scripts/paddleocr/parse_isy_table_v2.py` | `paddleocr/extraction/parse_isy_table_v2.py` |
| `scripts/paddleocr/prepare_isy_crops.py` | `paddleocr/ocr/preprocessing/prepare_isy_crops.py` |
| `scripts/paddleocr/prepare_mixed_table_crops.py` | `paddleocr/ocr/preprocessing/prepare_mixed_table_crops.py` |
| `scripts/paddleocr/recognize_table_crops.py` | `paddleocr/ocr/recognition/recognize_table_crops.py` |
| `scripts/paddleocr/remap_layout_labels.py` | `paddleocr/layout_analysis/diagnostics/remap_layout_labels.py` |
| `scripts/paddleocr/visualize_e05_changes.py` | `visualisation/layout_analysis/visualize_e05_changes.py` |
| `scripts/paddleocr/visualize_layout_diagnostics.py` | `visualisation/layout_analysis/visualize_layout_diagnostics.py` |
| `scripts/paddleocr/visualize_layout_evaluation.py` | `visualisation/layout_analysis/visualize_layout_evaluation.py` |
| `scripts/paddleocr/visualize_task_aligned_layout.py` | `visualisation/layout_analysis/visualize_task_aligned_layout.py` |
| `scripts/pipeline/run_pipeline.py` | `pipeline/run_pipeline.py` |
| `scripts/postprocess/validate_results.py` | `postprocess/validate_results.py` |
| `scripts/slurm/run_cpu_pipeline.sbatch` | `slurm/ocr_models/run_cpu_pipeline.sbatch` |
| `scripts/slurm/run_e04.sbatch` | `slurm/layout_analysis/run_e04.sbatch` |
| `scripts/slurm/run_e05_table038.sbatch` | `slurm/layout_analysis/run_e05_table038.sbatch` |
| `scripts/slurm/run_e05_visual_all74.sbatch` | `slurm/layout_analysis/run_e05_visual_all74.sbatch` |
| `scripts/slurm/run_effocr.sbatch` | `slurm/ocr_models/run_effocr.sbatch` |
| `scripts/slurm/run_effocr_prepare_test.sbatch` | `slurm/ocr_models/run_effocr_prepare_test.sbatch` |
| `scripts/slurm/run_effocr_preprocessing_ab.sbatch` | `slurm/ocr_models/run_effocr_preprocessing_ab.sbatch` |
| `scripts/slurm/run_effocr_preprocessing_canada.sbatch` | `slurm/ocr_models/run_effocr_preprocessing_canada.sbatch` |
| `scripts/slurm/run_effocr_rye_test.sbatch` | `slurm/ocr_models/run_effocr_rye_test.sbatch` |
| `scripts/slurm/run_effocr_threshold_canada.sbatch` | `slurm/ocr_models/run_effocr_threshold_canada.sbatch` |
| `scripts/slurm/run_effocr_threshold_grid.sbatch` | `slurm/ocr_models/run_effocr_threshold_grid.sbatch` |
| `scripts/slurm/run_gpu_ocr.sbatch` | `slurm/ocr_models/run_gpu_ocr.sbatch` |
| `scripts/slurm/run_isy1926_first50_layout.sbatch` | `slurm/layout_analysis/run_isy1926_first50_layout.sbatch` |
| `scripts/slurm/run_isy1926_more_tables.sbatch` | `slurm/ocr/run_isy1926_more_tables.sbatch` |
| `scripts/slurm/run_isy1926_region_ocr_smoke.sbatch` | `slurm/ocr/run_isy1926_region_ocr_smoke.sbatch` |
| `scripts/slurm/run_isy1926_table_recognition_smoke.sbatch` | `slurm/ocr/run_isy1926_table_recognition_smoke.sbatch` |
| `scripts/slurm/run_marvin_tesseract.sbatch` | `slurm/ocr_models/run_marvin_tesseract.sbatch` |
| `scripts/slurm/run_mixed_table_ocr_smoke_v1.sbatch` | `slurm/ocr/run_mixed_table_ocr_smoke_v1.sbatch` |
| `scripts/slurm/run_paddle_on_test_data.slurm` | `slurm/other/run_paddle_on_test_data.slurm` |
| `scripts/slurm/run_paddleocr_layout_validation_1927.sbatch` | `slurm/layout_analysis/run_paddleocr_layout_validation_1927.sbatch` |
| `scripts/slurm/run_paddleocr_mixed_layout_74p_9labels.sbatch` | `slurm/layout_analysis/run_paddleocr_mixed_layout_74p_9labels.sbatch` |
| `scripts/slurm/run_paddleocr_mixed_layout_74p_9labels_diagnostic025.sbatch` | `slurm/layout_analysis/run_paddleocr_mixed_layout_74p_9labels_diagnostic025.sbatch` |
| `scripts/slurm/run_paddleocr_mixed_layout_74p_baseline.sbatch` | `slurm/layout_analysis/run_paddleocr_mixed_layout_74p_baseline.sbatch` |
| `scripts/slurm/run_paddleocr_mixed_layout_baseline.sbatch` | `slurm/layout_analysis/run_paddleocr_mixed_layout_baseline.sbatch` |
| `scripts/slurm/run_paddleocr_table_detection_1926.sbatch` | `slurm/layout_analysis/run_paddleocr_table_detection_1926.sbatch` |
| `scripts/slurm/run_paddleocr_world_census_1930.sbatch` | `slurm/layout_analysis/run_paddleocr_world_census_1930.sbatch` |
| `scripts/slurm/run_paddleocr_yearbook_layout_analysis.sbatch` | `slurm/layout_analysis/run_paddleocr_yearbook_layout_analysis.sbatch` |
| `scripts/slurm/table_detection/run_table_transformer.sbatch` | `slurm/table_detection/run_table_transformer.sbatch` |
| `scripts/table_detection/detect_tables_pdf.py` | `table_detection/detect_tables_pdf.py` |
| `scripts/visualisation/make_e04_winner_comparisons.py` | `visualisation/layout_analysis/make_e04_winner_comparisons.py` |
| `scripts/visualisation/make_layout_comparisons.py` | `visualisation/layout_analysis/make_layout_comparisons.py` |

The EffOCR adapter unit test also now locates the runner at `ocr_models/effocr/run_effocr_ocr.py`.

Test adjustments: the EffOCR adapter test now checks the established fields while allowing additional character provenance fields. `test_target_config.py` explicitly skips its two cases until the original `configs/harvest_1926.yaml` is added. No production inference algorithm or output data was changed for these test updates.
