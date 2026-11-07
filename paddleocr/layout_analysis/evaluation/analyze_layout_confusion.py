#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = [
    "table",
    "header",
    "figure_title",
    "paragraph_title",
    "footnote",
    "number",
    "text",
    "content",
    "doc_title",
]


def norm(label):
    return (
        str(label)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def gt_bbox(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]

    return [
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    ]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_gt(path):
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    result = []

    for shape in data.get("shapes", []):
        label = norm(
            shape.get("label", "")
        )

        if label not in CLASSES:
            continue

        points = shape.get(
            "points",
            [],
        )

        if not points:
            continue

        result.append({
            "label": label,
            "bbox": gt_bbox(points),
        })

    return result


def load_pred(path):
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    payload = data.get(
        "res",
        data,
    )

    result = []

    for box in payload.get(
        "boxes",
        [],
    ):
        coordinate = box.get(
            "coordinate"
        )

        if (
            coordinate is None
            or len(coordinate) != 4
        ):
            continue

        result.append({
            "label": norm(
                box.get("label", "")
            ),
            "score": float(
                box.get("score", 0.0)
            ),
            "bbox": [
                float(v)
                for v in coordinate
            ],
        })

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.30,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix = defaultdict(Counter)

    details = []

    gt_files = sorted(
        args.labels_dir.glob(
            "page_*.json"
        )
    )

    for gt_path in gt_files:
        stem = gt_path.stem

        pred_path = (
            args.pred_dir
            / f"{stem}.json"
        )

        if not pred_path.exists():
            raise RuntimeError(
                f"Missing {pred_path}"
            )

        gt_objects = load_gt(
            gt_path
        )

        predictions = load_pred(
            pred_path
        )

        for gt_index, gt in enumerate(
            gt_objects
        ):
            best = None
            best_overlap = 0.0

            for pred_index, pred in enumerate(
                predictions
            ):
                overlap = iou(
                    gt["bbox"],
                    pred["bbox"],
                )

                if overlap > best_overlap:
                    best_overlap = overlap
                    best = (
                        pred_index,
                        pred,
                    )

            if (
                best is None
                or best_overlap
                < args.min_overlap
            ):
                predicted_label = "NO_MATCH"
                score = None
            else:
                _, pred = best

                predicted_label = (
                    pred["label"]
                )

                score = pred["score"]

            matrix[
                gt["label"]
            ][predicted_label] += 1

            details.append({
                "page": stem,
                "gt_index": gt_index,
                "gt_label": gt["label"],
                "pred_label": (
                    predicted_label
                ),
                "iou": best_overlap,
                "pred_score": score,
            })

    all_pred_labels = sorted(
        {
            pred
            for counts in matrix.values()
            for pred in counts
            if pred != "NO_MATCH"
        }
    )

    columns = (
        all_pred_labels
        + ["NO_MATCH"]
    )

    print()
    print(
        "=== GT -> BEST OVERLAPPING "
        "PADDLE CLASS ==="
    )
    print(
        f"Minimum IoU = "
        f"{args.min_overlap:.2f}"
    )
    print()

    for gt_label in CLASSES:
        counts = matrix[
            gt_label
        ]

        total = sum(
            counts.values()
        )

        if total == 0:
            continue

        print(
            f"{gt_label} "
            f"(GT={total})"
        )

        for predicted, count in (
            counts.most_common()
        ):
            percentage = (
                100.0
                * count
                / total
            )

            print(
                f"  -> "
                f"{predicted:25s} "
                f"{count:4d} "
                f"({percentage:5.1f}%)"
            )

        print()

    matrix_path = (
        args.output_dir
        / "confusion_matrix.csv"
    )

    with matrix_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            ["gt_label"]
            + columns
        )

        for gt_label in CLASSES:
            writer.writerow(
                [gt_label]
                + [
                    matrix[
                        gt_label
                    ][pred]
                    for pred in columns
                ]
            )

    detail_path = (
        args.output_dir
        / "confusion_details.csv"
    )

    with detail_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "page",
                "gt_index",
                "gt_label",
                "pred_label",
                "iou",
                "pred_score",
            ],
        )

        writer.writeheader()
        writer.writerows(
            details
        )

    print("Saved:")
    print(matrix_path)
    print(detail_path)


if __name__ == "__main__":
    main()
