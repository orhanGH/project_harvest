#!/usr/bin/env python3

import argparse
import csv
import json
import statistics
from collections import defaultdict
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


def normalize_label(label):
    return (
        str(label)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def bbox_from_points(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]

    return [
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    ]


def area(box):
    x1, y1, x2, y2 = box

    return (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )


def intersection_box(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def intersection_area(a, b):
    inter = intersection_box(a, b)

    if inter is None:
        return 0.0

    return area(inter)


def iou(a, b):
    inter = intersection_area(a, b)

    if inter <= 0:
        return 0.0

    union = (
        area(a)
        + area(b)
        - inter
    )

    if union <= 0:
        return 0.0

    return inter / union


def rectangle_union_area(rectangles):
    """
    Exact union area for axis-aligned rectangles.

    This avoids double-counting when several Paddle
    predictions overlap each other inside one GT box.
    """

    if not rectangles:
        return 0.0

    xs = sorted(
        {
            coordinate
            for rect in rectangles
            for coordinate in (
                rect[0],
                rect[2],
            )
        }
    )

    total = 0.0

    for left, right in zip(
        xs[:-1],
        xs[1:],
    ):
        width = right - left

        if width <= 0:
            continue

        intervals = []

        for x1, y1, x2, y2 in rectangles:
            if x1 < right and x2 > left:
                intervals.append(
                    (y1, y2)
                )

        if not intervals:
            continue

        intervals.sort()

        merged_length = 0.0

        current_start = intervals[0][0]
        current_end = intervals[0][1]

        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(
                    current_end,
                    end,
                )
            else:
                merged_length += (
                    current_end
                    - current_start
                )

                current_start = start
                current_end = end

        merged_length += (
            current_end
            - current_start
        )

        total += (
            width
            * merged_length
        )

    return total


def load_ground_truth(path):
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    objects = []

    for shape in data.get(
        "shapes",
        [],
    ):
        label = normalize_label(
            shape.get(
                "label",
                "",
            )
        )

        if label not in CLASSES:
            continue

        points = shape.get(
            "points",
            [],
        )

        if not points:
            continue

        objects.append(
            {
                "label": label,
                "bbox": bbox_from_points(
                    points
                ),
            }
        )

    return objects


def load_predictions(path):
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    payload = data.get(
        "res",
        data,
    )

    objects = []

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

        objects.append(
            {
                "label": normalize_label(
                    box.get(
                        "label",
                        "",
                    )
                ),
                "score": float(
                    box.get(
                        "score",
                        0.0,
                    )
                ),
                "bbox": [
                    float(v)
                    for v in coordinate
                ],
            }
        )

    return objects


def analyze_gt_object(
    gt,
    predictions,
):
    gt_box = gt["bbox"]
    gt_area = area(gt_box)

    same_class = [
        pred
        for pred in predictions
        if pred["label"] == gt["label"]
    ]

    # ------------------------------------------
    # Traditional best same-class IoU
    # ------------------------------------------

    best_iou = 0.0

    for pred in same_class:
        best_iou = max(
            best_iou,
            iou(
                gt_box,
                pred["bbox"],
            ),
        )

    # ------------------------------------------
    # Same-class geometry
    # ------------------------------------------

    same_intersections = []

    same_overlap_predictions = []

    best_pred_purity = 0.0

    for pred in same_class:
        inter = intersection_box(
            gt_box,
            pred["bbox"],
        )

        if inter is None:
            continue

        inter_area = area(inter)

        if inter_area <= 0:
            continue

        same_intersections.append(
            inter
        )

        same_overlap_predictions.append(
            pred
        )

        pred_area = area(
            pred["bbox"]
        )

        if pred_area > 0:
            purity = (
                inter_area
                / pred_area
            )

            best_pred_purity = max(
                best_pred_purity,
                purity,
            )

    same_union = rectangle_union_area(
        same_intersections
    )

    same_class_coverage = (
        same_union / gt_area
        if gt_area > 0
        else 0.0
    )

    # ------------------------------------------
    # Any-class geometry
    # ------------------------------------------

    any_intersections = []

    overlapping_labels = defaultdict(
        int
    )

    for pred in predictions:
        inter = intersection_box(
            gt_box,
            pred["bbox"],
        )

        if inter is None:
            continue

        if area(inter) <= 0:
            continue

        any_intersections.append(
            inter
        )

        overlapping_labels[
            pred["label"]
        ] += 1

    any_union = rectangle_union_area(
        any_intersections
    )

    any_class_coverage = (
        any_union / gt_area
        if gt_area > 0
        else 0.0
    )

    return {
        "best_iou": best_iou,
        "same_class_coverage":
            same_class_coverage,
        "any_class_coverage":
            any_class_coverage,
        "best_pred_purity":
            best_pred_purity,
        "same_class_fragments":
            len(
                same_overlap_predictions
            ),
        "all_overlap_fragments":
            sum(
                overlapping_labels.values()
            ),
        "overlap_labels":
            dict(
                sorted(
                    overlapping_labels.items()
                )
            ),
    }


def mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def median(values):
    if not values:
        return 0.0

    return statistics.median(
        values
    )


def percentage_at_least(
    values,
    threshold,
):
    if not values:
        return 0.0

    count = sum(
        value >= threshold
        for value in values
    )

    return (
        count / len(values)
    )


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
        "--run-name",
        default="run",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt_files = sorted(
        args.labels_dir.glob(
            "page_*.json"
        )
    )

    if not gt_files:
        raise RuntimeError(
            "No ground-truth JSON files found."
        )

    detail_rows = []

    per_class = defaultdict(list)

    for gt_path in gt_files:
        stem = gt_path.stem

        pred_path = (
            args.pred_dir
            / f"{stem}.json"
        )

        if not pred_path.exists():
            raise RuntimeError(
                f"Missing prediction: "
                f"{pred_path}"
            )

        ground_truth = (
            load_ground_truth(
                gt_path
            )
        )

        predictions = (
            load_predictions(
                pred_path
            )
        )

        for gt_index, gt in enumerate(
            ground_truth
        ):
            metrics = (
                analyze_gt_object(
                    gt,
                    predictions,
                )
            )

            row = {
                "run":
                    args.run_name,
                "page":
                    stem,
                "gt_index":
                    gt_index,
                "label":
                    gt["label"],
                "best_iou":
                    metrics[
                        "best_iou"
                    ],
                "same_class_coverage":
                    metrics[
                        "same_class_coverage"
                    ],
                "any_class_coverage":
                    metrics[
                        "any_class_coverage"
                    ],
                "best_pred_purity":
                    metrics[
                        "best_pred_purity"
                    ],
                "same_class_fragments":
                    metrics[
                        "same_class_fragments"
                    ],
                "all_overlap_fragments":
                    metrics[
                        "all_overlap_fragments"
                    ],
                "overlap_labels":
                    json.dumps(
                        metrics[
                            "overlap_labels"
                        ],
                        sort_keys=True,
                    ),
            }

            detail_rows.append(
                row
            )

            per_class[
                gt["label"]
            ].append(row)

    summary_rows = []

    print()
    print(
        "===================================================="
    )
    print(
        f"GEOMETRY ANALYSIS: {args.run_name}"
    )
    print(
        "===================================================="
    )
    print()

    print(
        f"{'Class':20s}"
        f"{'GT':>5s}"
        f"{'IoU50':>8s}"
        f"{'Cov50':>8s}"
        f"{'Cov75':>8s}"
        f"{'MeanCov':>10s}"
        f"{'Any75':>8s}"
        f"{'MedFrag':>9s}"
    )

    print("-" * 76)

    for label in CLASSES:
        rows = per_class.get(
            label,
            [],
        )

        if not rows:
            continue

        best_ious = [
            row["best_iou"]
            for row in rows
        ]

        same_coverages = [
            row[
                "same_class_coverage"
            ]
            for row in rows
        ]

        any_coverages = [
            row[
                "any_class_coverage"
            ]
            for row in rows
        ]

        fragment_counts = [
            row[
                "same_class_fragments"
            ]
            for row in rows
        ]

        iou50 = percentage_at_least(
            best_ious,
            0.50,
        )

        cov50 = percentage_at_least(
            same_coverages,
            0.50,
        )

        cov75 = percentage_at_least(
            same_coverages,
            0.75,
        )

        any75 = percentage_at_least(
            any_coverages,
            0.75,
        )

        mean_cov = mean(
            same_coverages
        )

        median_fragments = median(
            fragment_counts
        )

        summary_rows.append(
            {
                "run":
                    args.run_name,
                "label":
                    label,
                "gt_count":
                    len(rows),
                "iou_ge_050":
                    iou50,
                "same_coverage_ge_050":
                    cov50,
                "same_coverage_ge_075":
                    cov75,
                "mean_same_class_coverage":
                    mean_cov,
                "median_same_class_coverage":
                    median(
                        same_coverages
                    ),
                "any_coverage_ge_075":
                    any75,
                "mean_any_class_coverage":
                    mean(
                        any_coverages
                    ),
                "median_fragments":
                    median_fragments,
                "mean_fragments":
                    mean(
                        fragment_counts
                    ),
            }
        )

        print(
            f"{label:20s}"
            f"{len(rows):5d}"
            f"{100*iou50:7.1f}%"
            f"{100*cov50:7.1f}%"
            f"{100*cov75:7.1f}%"
            f"{100*mean_cov:9.1f}%"
            f"{100*any75:7.1f}%"
            f"{median_fragments:9.1f}"
        )

    detail_path = (
        args.output_dir
        / f"{args.run_name}_geometry_details.csv"
    )

    with detail_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "page",
                "gt_index",
                "label",
                "best_iou",
                "same_class_coverage",
                "any_class_coverage",
                "best_pred_purity",
                "same_class_fragments",
                "all_overlap_fragments",
                "overlap_labels",
            ],
        )

        writer.writeheader()
        writer.writerows(
            detail_rows
        )

    summary_path = (
        args.output_dir
        / f"{args.run_name}_geometry_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "label",
                "gt_count",
                "iou_ge_050",
                "same_coverage_ge_050",
                "same_coverage_ge_075",
                "mean_same_class_coverage",
                "median_same_class_coverage",
                "any_coverage_ge_075",
                "mean_any_class_coverage",
                "median_fragments",
                "mean_fragments",
            ],
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print()
    print("Saved:")
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
