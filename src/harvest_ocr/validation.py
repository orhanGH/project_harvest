from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .utils import read_jsonl, write_json, write_jsonl


def validate_record(record: dict[str, Any], settings: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if not record.get("country_fr") and not record.get("country_en"):
        flags.append("missing_country_label")

    numeric_fields = [
        "area_1925", "production_1925", "yield_1925",
    ]
    if not any(record.get(field) is not None for field in numeric_fields):
        flags.append("no_numeric_values")
    if any(record.get(field) is not None and record[field] < 0 for field in numeric_fields):
        flags.append("negative_value")

    confidence = record.get("confidence")
    threshold = float(settings.get("review_confidence_threshold", 0.75))
    if confidence is not None and confidence < threshold:
        flags.append("low_ocr_confidence")

    relative_tolerance = float(settings.get("yield_relative_tolerance", 0.18))
    absolute_tolerance = float(settings.get("yield_absolute_tolerance", 0.8))
    periods = ("1925",)
    for period in periods:
        area = record.get(f"area_{period}")
        production = record.get(f"production_{period}")
        printed_yield = record.get(f"yield_{period}")
        if area not in (None, 0) and production is not None and printed_yield is not None:
            expected = production / area
            difference = abs(expected - printed_yield)
            allowed = max(absolute_tolerance, abs(printed_yield) * relative_tolerance)
            if difference > allowed:
                flags.append(f"yield_mismatch_{period}")
    return flags


def validate_dataset(
    parsed_path: str | Path,
    validated_path: str | Path,
    summary_path: str | Path,
    settings: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    settings = settings or {}
    validated = []
    counts: Counter[str] = Counter()
    for record in read_jsonl(parsed_path):
        flags = validate_record(record, settings)
        record["validation_flags"] = flags
        record["status"] = "review" if flags else "validated"
        counts.update(flags)
        validated.append(record)
    write_jsonl(validated_path, validated)
    summary = {
        "records": len(validated),
        "validated": sum(1 for row in validated if row["status"] == "validated"),
        "review": sum(1 for row in validated if row["status"] == "review"),
        "flag_counts": dict(counts.most_common()),
    }
    write_json(summary_path, summary)
    return Path(validated_path), Path(summary_path)
