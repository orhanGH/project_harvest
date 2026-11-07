#!/usr/bin/env python3

import argparse
import csv
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


VALUE_FIELDS = [
    "area_1925_raw",
    "production_1925_raw",
    "yield_1925_raw",
]

QUALIFIER_FIELDS = [
    "area_qualifier",
    "production_qualifier",
    "yield_qualifier",
]


def clean(value):
    if value is None:
        return ""

    return str(value).strip()


def normalize_name(value):
    value = clean(value)

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        ch
        for ch in value
        if not unicodedata.combining(ch)
    )

    value = value.casefold()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return " ".join(
        value.split()
    )


def normalize_number(value):
    value = clean(value)

    if not value:
        return ""

    value = value.replace(
        ",",
        "",
    )

    value = value.replace(
        " ",
        "",
    )

    if not re.fullmatch(
        r"[-+]?\d+(?:\.\d+)?",
        value,
    ):
        return value

    try:
        number = float(value)
    except ValueError:
        return value

    if number.is_integer():
        return str(int(number))

    return (
        f"{number:.12f}"
        .rstrip("0")
        .rstrip(".")
    )


def name_score(
    gt_fr,
    gt_en,
    pred_fr,
    pred_en,
):
    gt_names = [
        normalize_name(gt_fr),
        normalize_name(gt_en),
    ]

    pred_names = [
        normalize_name(pred_fr),
        normalize_name(pred_en),
    ]

    scores = []

    for gt in gt_names:
        if not gt:
            continue

        for pred in pred_names:
            if not pred:
                continue

            if gt == pred:
                scores.append(1.0)
                continue

            if (
                gt in pred
                or pred in gt
            ):
                shorter = min(
                    len(gt),
                    len(pred),
                )

                longer = max(
                    len(gt),
                    len(pred),
                )

                containment_score = (
                    shorter / longer
                )

                scores.append(
                    max(
                        0.80,
                        containment_score,
                    )
                )

            scores.append(
                SequenceMatcher(
                    None,
                    gt,
                    pred,
                ).ratio()
            )

    return max(
        scores,
        default=0.0,
    )


def read_csv(path):
    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def prediction_names(row):
    return (
        row.get(
            "country_fr_raw",
            "",
        ),
        row.get(
            "country_en_raw",
            "",
        ),
    )


def match_rows(
    gt_rows,
    pred_rows,
    threshold,
):
    matches = []

    next_pred_index = 0

    for gt_index, gt in enumerate(
        gt_rows
    ):
        best = None

        for pred_index in range(
            next_pred_index,
            len(pred_rows),
        ):
            pred = pred_rows[
                pred_index
            ]

            pred_fr, pred_en = (
                prediction_names(pred)
            )

            score = name_score(
                gt.get(
                    "entity_fr",
                    "",
                ),
                gt.get(
                    "entity_en",
                    "",
                ),
                pred_fr,
                pred_en,
            )

            if (
                best is None
                or score > best[
                    "score"
                ]
            ):
                best = {
                    "gt_index":
                        gt_index,
                    "pred_index":
                        pred_index,
                    "score":
                        score,
                    "gt":
                        gt,
                    "pred":
                        pred,
                }

            if score >= 0.999:
                break

        if (
            best is not None
            and best["score"]
            >= threshold
        ):
            matches.append(
                best
            )

            next_pred_index = (
                best[
                    "pred_index"
                ]
                + 1
            )

        else:
            matches.append(
                {
                    "gt_index":
                        gt_index,
                    "pred_index":
                        None,
                    "score":
                        (
                            best["score"]
                            if best
                            else 0.0
                        ),
                    "gt":
                        gt,
                    "pred":
                        None,
                }
            )

    return matches


def pred_field_name(
    gt_field,
):
    mapping = {
        "area_qualifier":
            "area_1925_qualifier",

        "production_qualifier":
            "production_1925_qualifier",

        "yield_qualifier":
            "yield_1925_qualifier",
    }

    return mapping.get(
        gt_field,
        gt_field,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ground-truth",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--prediction",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.55,
    )

    args = parser.parse_args()

    gt_rows = read_csv(
        args.ground_truth
    )

    pred_rows = read_csv(
        args.prediction
    )

    matches = match_rows(
        gt_rows,
        pred_rows,
        args.match_threshold,
    )

    print()
    print(
        "========================================"
    )
    print(
        "ISY DEVELOPMENT EXTRACTION EVALUATION"
    )
    print(
        "========================================"
    )
    print()

    print(
        "Ground-truth rows:",
        len(gt_rows),
    )

    print(
        "Prediction rows:",
        len(pred_rows),
    )

    matched_count = sum(
        match["pred"] is not None
        for match in matches
    )

    print(
        "Matched GT rows:",
        matched_count,
        "/",
        len(gt_rows),
    )

    print()

    value_total = 0
    value_correct = 0

    qualifier_total = 0
    qualifier_correct = 0

    type_total = 0
    type_correct = 0

    print(
        "--- ROW-BY-ROW ---"
    )
    print()

    for match in matches:
        gt = match["gt"]
        pred = match["pred"]

        entity = (
            gt.get(
                "entity_en",
                "",
            )
            or gt.get(
                "entity_fr",
                "",
            )
        )

        print(
            f"{entity}"
        )

        if pred is None:
            print(
                "  ROW: UNMATCHED"
            )
            print(
                f"  best name score: "
                f"{match['score']:.3f}"
            )
            print()
            continue

        pred_fr, pred_en = (
            prediction_names(pred)
        )

        print(
            f"  row match score: "
            f"{match['score']:.3f}"
        )

        print(
            "  predicted label:",
            pred_en or pred_fr,
        )

        gt_type = clean(
            gt.get(
                "entity_type"
            )
        )

        pred_type = clean(
            pred.get(
                "row_type"
            )
        )

        if pred_type == "aggregate":
            pred_type = (
                "region_total"
            )

        type_total += 1

        type_ok = (
            gt_type == pred_type
        )

        type_correct += int(
            type_ok
        )

        print(
            "  entity_type:",
            "OK"
            if type_ok
            else (
                f"WRONG "
                f"(GT={gt_type}, "
                f"PRED={pred_type})"
            ),
        )

        for field in VALUE_FIELDS:
            gt_value = clean(
                gt.get(field)
            )

            if not gt_value:
                continue

            pred_field = (
                pred_field_name(
                    field
                )
            )

            pred_value = clean(
                pred.get(
                    pred_field
                )
            )

            value_total += 1

            ok = (
                normalize_number(
                    gt_value
                )
                ==
                normalize_number(
                    pred_value
                )
            )

            value_correct += int(
                ok
            )

            print(
                f"  {field}:",
                "OK"
                if ok
                else (
                    f"WRONG "
                    f"(GT={gt_value!r}, "
                    f"PRED={pred_value!r})"
                ),
            )

        for field in (
            QUALIFIER_FIELDS
        ):
            gt_value = clean(
                gt.get(field)
            )

            if not gt_value:
                continue

            pred_field = (
                pred_field_name(
                    field
                )
            )

            pred_value = clean(
                pred.get(
                    pred_field
                )
            )

            qualifier_total += 1

            ok = (
                gt_value
                == pred_value
            )

            qualifier_correct += int(
                ok
            )

            print(
                f"  {field}:",
                "OK"
                if ok
                else (
                    f"WRONG "
                    f"(GT={gt_value!r}, "
                    f"PRED={pred_value!r})"
                ),
            )

        print()

    print(
        "========================================"
    )
    print(
        "SUMMARY"
    )
    print(
        "========================================"
    )
    print()

    row_rate = (
        matched_count
        / len(gt_rows)
        if gt_rows
        else 0.0
    )

    print(
        f"GT row match rate: "
        f"{matched_count}/{len(gt_rows)} "
        f"({100 * row_rate:.1f}%)"
    )

    if type_total:
        print(
            f"Entity-type accuracy: "
            f"{type_correct}/{type_total} "
            f"({100 * type_correct / type_total:.1f}%)"
        )

    if value_total:
        print(
            f"Target-value accuracy: "
            f"{value_correct}/{value_total} "
            f"({100 * value_correct / value_total:.1f}%)"
        )

    if qualifier_total:
        print(
            f"Qualifier accuracy: "
            f"{qualifier_correct}/{qualifier_total} "
            f"({100 * qualifier_correct / qualifier_total:.1f}%)"
        )

    print()
    print(
        "NOTE:"
    )
    print(
        "This is a development-subset baseline."
    )
    print(
        "parent_region and entity_qualifier "
        "are not scored yet because the current "
        "geometry parser does not predict them."
    )


if __name__ == "__main__":
    main()
