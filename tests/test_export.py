import csv

from harvest_ocr.export import OUTPUT_COLUMNS, export_product_csvs
from harvest_ocr.utils import write_jsonl


def test_product_export_has_exact_schema_and_blank_missing_fields(tmp_path):
    records = tmp_path / "records.jsonl"
    write_jsonl(
        records,
        [
            {
                "product_id": "raw_silk",
                "row_type": "country",
                "country_en": "Japan",
                "country_fr": "Japon",
                "area_1925": None,
                "production_1925": 12340,
                "yield_1925": None,
            },
            {
                "product_id": "raw_silk",
                "row_type": "continent_total",
                "country_en": "Asia",
                "country_fr": "Asie",
                "production_1925": 99999,
            },
        ],
    )
    output = export_product_csvs(
        records,
        tmp_path / "exports",
        [{"table_number": "40", "product_id": "raw_silk"}],
    )
    csv_path = output / "table_40_raw_silk_1925.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == OUTPUT_COLUMNS
    assert rows == [
        {
            "country_name_english": "Japan",
            "area_hectares_1925": "",
            "production_quintals_1925": "12340",
            "yield_per_hectare_1925": "",
        }
    ]
