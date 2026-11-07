from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils import parse_page_range, read_jsonl, write_jsonl


NUMERIC_COLUMNS = [
    "area_1925",
    "production_1925",
    "yield_1925",
]

CONTINENT_TERMS = {
    "afrique", "africa", "amerique du nord", "north america", "amerique du sud",
    "south america", "asie", "asia", "europe", "oceanie", "oceania",
}
WORLD_TERMS = {"monde entier", "world", "world total"}


@dataclass(slots=True)
class ParsedNumber:
    value: float | None
    raw: str
    missing_code: str | None
    markers: list[str]


def _plain(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return " ".join("".join(ch for ch in normalized if not unicodedata.combining(ch)).lower().split())


def parse_numeric_cell(raw: str | None) -> ParsedNumber:
    text = " ".join((raw or "").replace("\n", " ").split())
    if not text:
        return ParsedNumber(None, text, "blank", [])
    plain = text.strip()
    if re.fullmatch(r"(?:[—–-]|\.{2,}|…)+", plain):
        code = "dash" if any(ch in plain for ch in "—–-") else "ellipsis"
        return ParsedNumber(None, text, code, [])

    markers: list[str] = []
    markers.extend(re.findall(r"\*", plain))
    markers.extend(re.findall(r"\b[a-zA-Z]\)", plain))
    markers.extend(re.findall(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]", plain))
    if plain.startswith("(") and plain.endswith(")"):
        markers.append("parenthesized")

    candidate = plain.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"}))
    match = re.search(r"(?<![A-Za-z])\d[\d.,]*", candidate)
    if not match:
        return ParsedNumber(None, text, "unparsed", sorted(set(markers)))
    number = match.group(0).rstrip(".,")
    # The yearbook uses commas as thousands separators and periods for decimals.
    number = number.replace(",", "")
    if number.count(".") > 1:
        parts = number.split(".")
        number = "".join(parts[:-1]) + "." + parts[-1]
    try:
        value = float(number)
    except ValueError:
        return ParsedNumber(None, text, "unparsed", sorted(set(markers)))
    return ParsedNumber(value, text, None, sorted(set(markers)))


def classify_row(country_fr: str, country_en: str) -> str:
    labels = {
        " ".join(re.sub(r"[^a-z ]", " ", _plain(value)).split())
        for value in (country_fr, country_en)
    }
    if any(label in WORLD_TERMS or label.startswith("monde entier") for label in labels):
        return "world_total"
    simplified = {
        re.sub(r"\s*\((?:suite|continued?)\)\s*$", "", label).strip(" .*0123456789")
        for label in labels
    }
    if simplified & CONTINENT_TERMS:
        return "continent_total"
    return "country"


def _has_numeric(record: dict[str, Any]) -> bool:
    return any(record.get(column) is not None for column in NUMERIC_COLUMNS)


def _looks_like_continuation(record: dict[str, Any]) -> bool:
    if record.get("row_type") != "country" or _has_numeric(record):
        return False
    french = (record.get("country_fr") or "").strip()
    english = (record.get("country_en") or "").strip()
    combined = f"{french} {english}".lower()
    return (
        french.endswith((",", "-"))
        or english.endswith((",", "-"))
        or "anglo-" in combined
        or "union of" in combined
    )


def _merge_continuation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rows)
    merged: list[dict[str, Any]] = []
    index = 0
    while index < len(rows):
        current = rows[index]
        if (
            _looks_like_continuation(current)
            and index + 1 < len(rows)
            and rows[index + 1]["pdf_page"] == current["pdf_page"]
        ):
            following = rows[index + 1]
            following["country_fr"] = " ".join(
                part for part in (current.get("country_fr", ""), following.get("country_fr", "")) if part
            )
            following["country_en"] = " ".join(
                part for part in (current.get("country_en", ""), following.get("country_en", "")) if part
            )
            following["row_type"] = classify_row(following["country_fr"], following["country_en"])
            following.setdefault("merged_from", []).append(current["record_id"])
            index += 1
            continue
        else:
            labels = {_plain(current.get("country_fr", "")), _plain(current.get("country_en", ""))}
            if not (labels & {"pays", "country"}):
                merged.append(current)
        index += 1
    return merged


def _catalog_by_page(catalog: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for entry in catalog:
        pages = parse_page_range(str(entry["pages"]))
        for index, page in enumerate(pages):
            item = dict(entry)
            item["page_role"] = "first" if index == 0 else "continued"
            result[page] = item
    return result


class AgriculturalTableParser:
    def __init__(
        self,
        columns: list[str],
        table_catalog: list[dict[str, Any]] | None = None,
        keep_empty_rows: bool = False,
    ) -> None:
        self.columns = columns
        self.catalog = _catalog_by_page(table_catalog or [])
        self.keep_empty_rows = keep_empty_rows

    def _cell_rows(
        self,
        regions: dict[str, dict[str, Any]],
        ocr_results: list[dict[str, Any]],
    ) -> dict[tuple[str, int, int], dict[str, dict[str, Any]]]:
        rows: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
        for result in ocr_results:
            region = regions.get(result["region_id"])
            if not region or region.get("region_type") != "cell":
                continue
            key = (region["document_id"], int(region["pdf_page"]), int(region["row_index"]))
            rows[key][region["column_id"]] = {
                "text": result.get("text", ""),
                "confidence": result.get("confidence"),
                "bbox": region.get("bbox"),
                "region_id": region["region_id"],
            }
        return rows

    def _structured_rows(
        self,
        regions: dict[str, dict[str, Any]],
        ocr_results: list[dict[str, Any]],
    ) -> dict[tuple[str, int, int], dict[str, dict[str, Any]]]:
        rows: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
        for result in ocr_results:
            region = regions.get(result["region_id"])
            if not region or region.get("region_type") != "table":
                continue
            document_id = region["document_id"]
            pdf_page = int(region["pdf_page"])
            items = result.get("items", [])
            for item in items:
                row_index = item.get("row_index")
                column_index = item.get("column_index")
                if row_index is None or column_index is None:
                    continue
                if not (0 <= int(column_index) < len(self.columns)):
                    continue
                column_id = self.columns[int(column_index)]
                key = (document_id, pdf_page, int(row_index))
                rows[key][column_id] = {
                    "text": item.get("text", ""),
                    "confidence": item.get("confidence"),
                    "bbox": item.get("bbox"),
                    "region_id": region["region_id"],
                }
        return rows

    def parse(
        self,
        regions_manifest: str | Path,
        ocr_results_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        regions = {row["region_id"]: row for row in read_jsonl(regions_manifest)}
        ocr_results = list(read_jsonl(ocr_results_path))
        model_name = ocr_results[0].get("source_model", "unknown") if ocr_results else "unknown"
        rows = self._cell_rows(regions, ocr_results)
        if not rows:
            rows = self._structured_rows(regions, ocr_results)

        output_rows: list[dict[str, Any]] = []
        for (document_id, pdf_page, row_index), cells in sorted(rows.items()):
            country_fr = cells.get("country_fr", {}).get("text", "").strip(" .")
            country_en = cells.get("country_en", {}).get("text", "").strip(" .")
            numeric = {column: parse_numeric_cell(cells.get(column, {}).get("text")) for column in NUMERIC_COLUMNS}
            if not self.keep_empty_rows and not country_fr and not country_en and not any(
                item.value is not None for item in numeric.values()
            ):
                continue

            catalog = self.catalog.get(pdf_page, {})
            confidences = [
                cell["confidence"]
                for cell in cells.values()
                if cell.get("confidence") is not None
            ]
            record: dict[str, Any] = {
                "record_id": f"{document_id}:p{pdf_page:04d}:r{row_index:03d}",
                "document_id": document_id,
                "pdf_page": pdf_page,
                "row_index": row_index,
                "table_number": catalog.get("table_number"),
                "product_id": catalog.get("product_id"),
                "product_fr": catalog.get("product_fr"),
                "product_en": catalog.get("product_en"),
                "area_scale": float(catalog.get("area_scale", 1)),
                "production_scale": float(catalog.get("production_scale", 1)),
                "page_role": catalog.get("page_role"),
                "row_type": classify_row(country_fr, country_en),
                "country_fr": country_fr,
                "country_en": country_en,
                "source_model": model_name,
                "confidence": sum(confidences) / len(confidences) if confidences else None,
                "status": "parsed",
                "validation_flags": [],
                "raw_cells": {column: cells.get(column, {}) for column in self.columns},
                "markers": {column: number.markers for column, number in numeric.items() if number.markers},
                "missing_codes": {
                    column: number.missing_code
                    for column, number in numeric.items()
                    if number.missing_code
                },
            }
            for column, number in numeric.items():
                scale = 1.0
                if column == "area_1925":
                    scale = record["area_scale"]
                elif column == "production_1925":
                    scale = record["production_scale"]
                record[column] = number.value * scale if number.value is not None else None
                record[f"{column}_raw"] = number.raw
            output_rows.append(record)

        write_jsonl(output_path, _merge_continuation_rows(output_rows))
        return Path(output_path)
