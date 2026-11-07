from harvest_ocr.parsers.agricultural import AgriculturalTableParser, parse_numeric_cell
from harvest_ocr.utils import read_jsonl, write_jsonl


def test_thousands_decimal_and_marker():
    assert parse_numeric_cell("* 10,193").value == 10193
    assert parse_numeric_cell("12.5").value == 12.5
    assert parse_numeric_cell("b) 125").markers == ["b)"]


def test_missing_values_are_distinct():
    assert parse_numeric_cell("—").missing_code == "dash"
    assert parse_numeric_cell("...").missing_code == "ellipsis"
    assert parse_numeric_cell("").missing_code == "blank"


def test_raw_silk_tonnes_are_scaled_to_quintals(tmp_path):
    regions = tmp_path / "regions.jsonl"
    ocr = tmp_path / "ocr.jsonl"
    output = tmp_path / "records.jsonl"
    write_jsonl(
        regions,
        [
            {
                "region_id": "country-fr",
                "document_id": "book",
                "pdf_page": 83,
                "region_type": "cell",
                "row_index": 0,
                "column_id": "country_fr",
                "bbox": [0, 0, 1, 1],
            },
            {
                "region_id": "production",
                "document_id": "book",
                "pdf_page": 83,
                "region_type": "cell",
                "row_index": 0,
                "column_id": "production_1925",
                "bbox": [1, 0, 2, 1],
            },
            {
                "region_id": "country-en",
                "document_id": "book",
                "pdf_page": 83,
                "region_type": "cell",
                "row_index": 0,
                "column_id": "country_en",
                "bbox": [2, 0, 3, 1],
            },
        ],
    )
    write_jsonl(
        ocr,
        [
            {"region_id": "country-fr", "source_model": "test", "text": "Japon", "confidence": 1},
            {"region_id": "production", "source_model": "test", "text": "1,234", "confidence": 1},
            {"region_id": "country-en", "source_model": "test", "text": "Japan", "confidence": 1},
        ],
    )
    parser = AgriculturalTableParser(
        columns=["country_fr", "production_1925", "country_en"],
        table_catalog=[
            {
                "pages": "83",
                "table_number": "40",
                "product_id": "raw_silk",
                "production_scale": 10,
            }
        ],
    )
    parser.parse(regions, ocr, output)
    record = next(read_jsonl(output))
    assert record["production_1925"] == 12340
