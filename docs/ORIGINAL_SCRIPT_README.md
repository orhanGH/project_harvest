# HARVEST OCR Benchmark Structure

## ocr_models/

Model-specific wrappers.

Each OCR wrapper should take PNG images and produce model-native raw outputs:

- Markdown: `.md`
- JSON: `.json`
- plain OCR text: `.txt`
- word boxes/cell boxes: `.csv` or `.json`

The OCR model should not decide the final HARVEST values.

The EfficientOCR branch follows the same rule: it recognizes pre-cropped cells
and preserves their geometry metadata, while deterministic HARVEST code assigns
cells to fields and creates the final rows.

## parsers/

Dataset-specific deterministic parsers.

These convert OCR outputs into the HARVEST schema:

- product_id
- table_number
- row_type: country / continent_total / world_total
- french_name
- english_name
- area_1925
- production_1925
- yield_1925
- status
- raw_source

Core rule:

Area 1925 = 3rd value in the Area block.
Production 1925 = 3rd value in the Production block.
Yield 1925 = 3rd value in the Yield block.

Equivalently, if a row has 9 numeric values:

numbers[2], numbers[5], numbers[8]

## pipeline/

Shell scripts that run OCR model wrappers and parsers in the right order.

## common/

Shared helper utilities.
