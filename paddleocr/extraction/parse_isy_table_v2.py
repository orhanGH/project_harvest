#!/usr/bin/env python3
"""ISY v2 semantic row reconstruction from a frozen v1 JSON output.

This phase splits leading region headers accidentally merged with their first
country, carries forward the current region, and preserves v1 numeric fields.
No OCR is rerun; no absent source symbols are fabricated.
"""
import argparse
import csv
import json
import re
from copy import deepcopy
from pathlib import Path

REGIONS = (
    ("AFRICA", "AFRIQUE", "AFRICA"),
    ("NORTH AMERICA", "AMÉRIQUE DU NORD", "NORTH AMERICA"),
    ("SOUTH AMERICA", "AMÉRIQUE DU SUD", "SOUTH AMERICA"),
    ("ASIA", "ASIE", "ASIA"),
    ("EUROPE", "EUROPE", "EUROPE"),
    ("OCEANIA", "OCÉANIE", "OCEANIA"),
)

CSV_FIELDS = (
    "row_type", "country_fr_raw", "country_en_raw", "parent_region",
    "entity_qualifier", "area_1925_raw", "production_1925_raw",
    "yield_1925_raw", "area_1925_qualifier",
    "production_1925_qualifier", "yield_1925_qualifier",
)


def take_leading_region(text, region_label):
    """Return (remaining_country, region_marker) or None."""
    pattern = (
        r"^\s*" + re.escape(region_label) +
        r"\b\s*[.:]?\s*(?P<marker>[*†‡]?)\s+"
        r"(?P<country>.+?)\s*$"
    )
    match = re.match(pattern, text or "", flags=re.IGNORECASE)
    return ((match.group("country"), match.group("marker") or None)
            if match else None)


def peel_entity_marker(text):
    """Remove a final printed superscript from a country name."""
    match = re.search(r"([⁰¹²³⁴⁵⁶⁷⁸⁹]+)([.]?)\s*$", text)
    if not match:
        return text, None
    marker = match.group(1).translate(str.maketrans(
        "⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789"))
    cleaned = (text[:match.start()] + match.group(2)).strip()
    return cleaned, marker


def is_region_total(row, fr, en):
    if row.get("row_type") != "region_total":
        return False
    names = {str(row.get("country_fr_raw") or "").strip(" .").casefold(),
             str(row.get("country_en_raw") or "").strip(" .").casefold()}
    return fr.casefold() in names or en.casefold() in names


def reconstruct(rows):
    result = []
    current_region = None
    split_count = 0
    for source_index, source_row in enumerate(rows):
        row = deepcopy(source_row)
        if row.get("row_type") == "world_total":
            current_region = None
            result.append(row)
            continue
        if row.get("row_type") == "region_total":
            for canonical, fr, en in REGIONS:
                if is_region_total(row, fr, en):
                    current_region = canonical
                    break
            result.append(row)
            continue

        found = None
        if row.get("row_type") == "country":
            for canonical, fr, en in REGIONS:
                fr_part = take_leading_region(row.get("country_fr_raw"), fr)
                en_part = take_leading_region(row.get("country_en_raw"), en)
                # Require independent evidence on both sides of the table.
                if fr_part and en_part:
                    found = canonical, fr, en, fr_part, en_part
                    break

        if found:
            canonical, fr, en, fr_part, en_part = found
            fr_country, fr_header_marker = fr_part
            en_country, en_header_marker = en_part
            header = {
                "row_type": "region_header",
                "country_fr_raw": fr,
                "country_en_raw": en,
                "parent_region": None,
                "entity_qualifier": fr_header_marker or en_header_marker,
                "row_y": row.get("row_y"),
                "derived_from_v1_row_index": source_index,
                "reconstruction_rule": "split_leading_bilingual_region_header",
            }
            for metric in ("area", "production", "yield"):
                for suffix in ("raw", "value", "qualifier", "ocr_confidence"):
                    header[f"{metric}_1925_{suffix}"] = None
                header[f"{metric}_1925_source_tokens"] = []
            result.append(header)
            current_region = canonical
            fr_country, fr_marker = peel_entity_marker(fr_country)
            en_country, en_marker = peel_entity_marker(en_country)
            row["country_fr_raw"] = fr_country
            row["country_en_raw"] = en_country
            row["entity_qualifier"] = fr_marker or en_marker or row.get("entity_qualifier")
            row["derived_from_v1_row_index"] = source_index
            row["reconstruction_rule"] = "remove_leading_region_header"
            split_count += 1

        if row.get("row_type") == "country" and current_region:
            row["parent_region"] = current_region
        result.append(row)
    return result, split_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input_json.read_text(encoding="utf-8"))
    rows, splits = reconstruct(source["rows"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = dict(source)
    result["rows"] = rows
    result["semantic_parser_version"] = "isy-v2-region-headers"
    result["source_v1_json"] = str(args.input_json)
    json_path = args.output_dir / "isy_target_period.json"
    csv_path = args.output_dir / "isy_target_period.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_FIELDS})
    print("Inserted region headers:", splits)
    print("Rows:", len(source["rows"]), "→", len(rows))
    for row in rows[:15]:
        print(f"{row['row_type']:13} {str(row.get('country_en_raw') or ''):27} "
              f"parent={str(row.get('parent_region') or ''):13} "
              f"marker={row.get('entity_qualifier') or ''}")
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
