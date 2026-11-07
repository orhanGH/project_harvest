#!/usr/bin/env python3
"""Strict ISY annotated-table evaluator: unmatched rows count as failures.

Use full-table ground truth for precision, and partial ground truth for recall/
annotated-field accuracy only. Raw formatting is scored separately from values.
"""
import argparse
import csv
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path

VALUE_FIELDS = ("area_1925_raw", "production_1925_raw", "yield_1925_raw")
QUALIFIER_FIELDS = {
    "area_qualifier": "area_1925_qualifier",
    "production_qualifier": "production_1925_qualifier",
    "yield_qualifier": "yield_1925_qualifier",
}


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = len(reader.fieldnames or [])
        rows = list(reader)
    if any(None in row or len(row) != expected for row in rows):
        raise ValueError(f"Malformed CSV: {path}")
    return rows


def clean(value):
    return str(value or "").strip()


def norm_name(value):
    value = unicodedata.normalize("NFKD", clean(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def as_decimal(value):
    text = clean(value).replace(",", "").replace(" ", "")
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return text


def f_ratio(ok, total):
    return f"{ok}/{total} ({ok / total * 100:.1f}%)" if total else "0/0 (not scored)"


def match_rows(gt_rows, pred_rows):
    result = {}
    used = set()
    for gi, gt in enumerate(gt_rows):
        gt_names = {norm_name(gt.get(field)) for field in ("entity_fr", "entity_en")}
        gt_names.discard("")
        for pi, pred in enumerate(pred_rows):
            if pi in used:
                continue
            names = {norm_name(pred.get(field)) for field in ("country_fr_raw", "country_en_raw")}
            names.discard("")
            if names & gt_names:
                result[gi] = pi
                used.add(pi)
                break
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--partial-ground-truth", action="store_true",
                        help="Suppress row precision/F1 when GT is only a selected subset.")
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()
    gt_rows = read_rows(args.ground_truth)
    pred_rows = read_rows(args.prediction)
    matches = match_rows(gt_rows, pred_rows)
    scores = {
        "entity_type": [0, 0], "parent_region": [0, 0],
        "entity_qualifier": [0, 0], "target_values_numeric": [0, 0],
        "target_values_raw": [0, 0], "cell_qualifiers": [0, 0],
    }

    print("\nISY STRICT EVALUATION")
    print("GT rows:", len(gt_rows), "| prediction rows:", len(pred_rows),
          "| matched:", len(matches))
    print()
    for gi, gt in enumerate(gt_rows):
        pred = pred_rows[matches[gi]] if gi in matches else {}
        print(f"{gt.get('entity_en') or gt.get('entity_fr')}: "
              f"{'MATCH' if pred else 'UNMATCHED'}")
        pairs = [("entity_type", "entity_type", "row_type"),
                 ("parent_region", "parent_region", "parent_region"),
                 ("entity_qualifier", "entity_qualifier", "entity_qualifier")]
        for category, gt_key, pred_key in pairs:
            gt_value = clean(gt.get(gt_key))
            if category != "entity_type" and not gt_value:
                continue
            pred_value = clean(pred.get(pred_key))
            correct = (norm_name(gt_value) == norm_name(pred_value)
                       if category == "parent_region" else gt_value == pred_value)
            scores[category][1] += 1
            scores[category][0] += int(correct)
            if not correct:
                print(f"  {category}: GT={gt_value!r}, PRED={pred_value!r}")

        for field in VALUE_FIELDS:
            gt_value = clean(gt.get(field))
            if not gt_value:
                continue
            pred_value = clean(pred.get(field))
            numeric_ok = bool(pred) and as_decimal(gt_value) == as_decimal(pred_value)
            raw_ok = bool(pred) and gt_value == pred_value
            scores["target_values_numeric"][1] += 1
            scores["target_values_raw"][1] += 1
            scores["target_values_numeric"][0] += int(numeric_ok)
            scores["target_values_raw"][0] += int(raw_ok)
            if not numeric_ok or not raw_ok:
                print(f"  {field}: GT={gt_value!r}, PRED={pred_value!r} "
                      f"(numeric={'OK' if numeric_ok else 'WRONG'}, "
                      f"raw={'OK' if raw_ok else 'WRONG'})")

        for gt_field, pred_field in QUALIFIER_FIELDS.items():
            gt_value = clean(gt.get(gt_field))
            if not gt_value:
                continue
            pred_value = clean(pred.get(pred_field))
            ok = bool(pred) and gt_value == pred_value
            scores["cell_qualifiers"][1] += 1
            scores["cell_qualifiers"][0] += int(ok)
            if not ok:
                print(f"  {gt_field}: GT={gt_value!r}, PRED={pred_value!r}")

    tp = len(matches)
    recall = tp / len(gt_rows) if gt_rows else 0
    report = {
        "ground_truth_rows": len(gt_rows),
        "predicted_rows": len(pred_rows),
        "matched_rows": tp,
        "row_recall": recall,
        "metrics": {key: {"correct": value[0], "total": value[1],
                          "accuracy": value[0] / value[1] if value[1] else None}
                    for key, value in scores.items()},
    }
    print("\n--- SUMMARY ---")
    print("Row recall:", f_ratio(tp, len(gt_rows)))
    if not args.partial_ground_truth:
        precision = tp / len(pred_rows) if pred_rows else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        print("Row precision:", f_ratio(tp, len(pred_rows)))
        print(f"Row F1: {100 * f1:.1f}%")
        report.update(row_precision=precision, row_f1=f1)
    else:
        print("Row precision / F1: not scored (partial GT)")
    for key, (correct, total) in scores.items():
        print(f"{key}: {f_ratio(correct, total)}")
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Report:", args.report_json)


if __name__ == "__main__":
    main()
