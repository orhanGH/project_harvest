from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from .utils import read_jsonl, safe_stem


OUTPUT_COLUMNS = [
    "country_name_english",
    "area_hectares_1925",
    "production_quintals_1925",
    "yield_per_hectare_1925",
]


def _display_number(value: Any) -> str:
    if value is None:
        return ""
    number = float(value)
    return str(int(number)) if number.is_integer() else format(number, ".12g")


def export_product_csvs(
    records_path: str | Path,
    output_dir: str | Path,
    table_catalog: list[dict[str, Any]],
) -> Path:
    """Write one stable four-column CSV per product, including empty products."""
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(records_path):
        if record.get("row_type") != "country":
            continue
        product_id = record.get("product_id")
        if product_id:
            grouped[str(product_id)].append(record)

    for product in table_catalog:
        product_id = str(product["product_id"])
        table_number = int(product["table_number"])
        output = target / f"table_{table_number:02d}_{safe_stem(product_id)}_1925.csv"
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for record in grouped.get(product_id, []):
                writer.writerow(
                    {
                        "country_name_english": record.get("country_en", ""),
                        "area_hectares_1925": _display_number(record.get("area_1925")),
                        "production_quintals_1925": _display_number(record.get("production_1925")),
                        "yield_per_hectare_1925": _display_number(record.get("yield_1925")),
                    }
                )
    return target
