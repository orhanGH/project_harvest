from __future__ import annotations

import math
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .utils import read_jsonl, write_json


FIELDS = [
    "area_1925", "production_1925", "yield_1925",
]


def _normalize_label(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    return " ".join("".join(ch for ch in text if not unicodedata.combining(ch)).lower().split())


def _numeric_equal(left, right, tolerance: float) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def evaluate(
    gold_path: str | Path,
    prediction_path: str | Path,
    output_path: str | Path,
    numeric_tolerance: float = 1e-6,
) -> Path:
    gold = list(read_jsonl(gold_path))
    predictions = list(read_jsonl(prediction_path))
    by_id = {row["record_id"]: row for row in predictions}
    by_page = {}
    for row in predictions:
        by_page.setdefault(int(row["pdf_page"]), []).append(row)

    matches: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for gold_row in gold:
        prediction = by_id.get(gold_row.get("record_id"))
        if prediction is None:
            label = _normalize_label(gold_row.get("country_en") or gold_row.get("country_fr"))
            candidates = by_page.get(int(gold_row["pdf_page"]), [])
            scored = [
                (
                    SequenceMatcher(
                        None,
                        label,
                        _normalize_label(row.get("country_en") or row.get("country_fr")),
                    ).ratio(),
                    row,
                )
                for row in candidates
            ]
            if scored and max(scored, key=lambda item: item[0])[0] >= 0.72:
                prediction = max(scored, key=lambda item: item[0])[1]
        matches.append((gold_row, prediction))

    total_cells = correct_cells = 0
    field_metrics: dict[str, dict[str, int]] = {field: {"correct": 0, "total": 0} for field in FIELDS}
    country_scores = []
    for gold_row, prediction in matches:
        if prediction is None:
            for field in FIELDS:
                field_metrics[field]["total"] += 1
                total_cells += 1
            country_scores.append(0.0)
            continue
        gold_label = _normalize_label(gold_row.get("country_en") or gold_row.get("country_fr"))
        predicted_label = _normalize_label(prediction.get("country_en") or prediction.get("country_fr"))
        country_scores.append(SequenceMatcher(None, gold_label, predicted_label).ratio())
        for field in FIELDS:
            field_metrics[field]["total"] += 1
            total_cells += 1
            if _numeric_equal(gold_row.get(field), prediction.get(field), numeric_tolerance):
                field_metrics[field]["correct"] += 1
                correct_cells += 1

    report = {
        "gold_rows": len(gold),
        "matched_rows": sum(1 for _, prediction in matches if prediction is not None),
        "row_coverage": sum(1 for _, prediction in matches if prediction is not None) / len(gold) if gold else 0,
        "numeric_cell_accuracy": correct_cells / total_cells if total_cells else 0,
        "mean_country_similarity": sum(country_scores) / len(country_scores) if country_scores else 0,
        "fields": {
            field: {
                **counts,
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0,
            }
            for field, counts in field_metrics.items()
        },
    }
    write_json(output_path, report)
    return Path(output_path)
