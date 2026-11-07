#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


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


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_ground_truth(path, allowed_classes):
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    objects = []
    ignored = Counter()

    for shape in data.get("shapes", []):
        label = normalize_label(
            shape.get("label", "")
        )

        if label not in allowed_classes:
            ignored[label] += 1
            continue

        points = shape.get("points", [])

        if not points:
            continue

        objects.append(
            {
                "label": label,
                "bbox": bbox_from_points(points),
            }
        )

    return objects, ignored


def load_predictions(
    path,
    score_threshold,
    allowed_classes,
):
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    payload = data.get("res", data)

    objects = []
    ignored = Counter()

    for box in payload.get("boxes", []):
        score = float(
            box.get("score", 0.0)
        )

        if score < score_threshold:
            continue

        label = normalize_label(
            box.get("label", "")
        )

        if label not in allowed_classes:
            ignored[label] += 1
            continue

        coordinate = box.get("coordinate")

        if (
            coordinate is None
            or len(coordinate) != 4
        ):
            continue

        objects.append(
            {
                "label": label,
                "score": score,
                "bbox": [
                    float(x)
                    for x in coordinate
                ],
            }
        )

    return objects, ignored


def match_class(
    ground_truth,
    predictions,
    iou_threshold,
):
    candidates = []

    for gt_index, gt in enumerate(
        ground_truth
    ):
        for pred_index, pred in enumerate(
            predictions
        ):
            overlap = iou(
                gt["bbox"],
                pred["bbox"],
            )

            if overlap >= iou_threshold:
                candidates.append(
                    (
                        overlap,
                        gt_index,
                        pred_index,
                    )
                )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    matched_gt = set()
    matched_pred = set()
    matches = []

    for overlap, gt_index, pred_index in candidates:
        if gt_index in matched_gt:
            continue

        if pred_index in matched_pred:
            continue

        matched_gt.add(gt_index)
        matched_pred.add(pred_index)

        matches.append(
            (
                gt_index,
                pred_index,
                overlap,
            )
        )

    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp

    return tp, fp, fn, matches


def safe_div(numerator, denominator):
    if denominator == 0:
        return None

    return numerator / denominator


def f1_score(precision, recall):
    if precision is None or recall is None:
        return None

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


def format_metric(value):
    if value is None:
        return "-"

    return f"{value:.4f}"


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
        "--classes",
        required=True,
        help=(
            "Comma-separated class list, e.g. "
            "table,text,footnote"
        ),
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
    )

    args = parser.parse_args()

    selected_classes = [
        normalize_label(label)
        for label in args.classes.split(",")
        if label.strip()
    ]

    allowed_classes = set(
        selected_classes
    )

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
            f"No ground-truth JSON files found: "
            f"{args.labels_dir}"
        )

    stats = defaultdict(
        lambda: {
            "gt": 0,
            "pred": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matched_ious": [],
        }
    )

    gt_label_counts = Counter()
    pred_label_counts = Counter()

    ignored_gt_counts = Counter()
    ignored_pred_counts = Counter()

    page_results = []

    for gt_path in gt_files:
        page_name = gt_path.stem

        pred_path = (
            args.pred_dir
            / f"{page_name}.json"
        )

        if not pred_path.exists():
            raise RuntimeError(
                f"Missing prediction JSON: "
                f"{pred_path}"
            )

        gt_objects, ignored_gt = (
            load_ground_truth(
                gt_path,
                allowed_classes,
            )
        )

        pred_objects, ignored_pred = (
            load_predictions(
                pred_path,
                args.score_threshold,
                allowed_classes,
            )
        )

        ignored_gt_counts.update(
            ignored_gt
        )

        ignored_pred_counts.update(
            ignored_pred
        )

        gt_by_class = defaultdict(list)
        pred_by_class = defaultdict(list)

        for obj in gt_objects:
            gt_by_class[
                obj["label"]
            ].append(obj)

            gt_label_counts[
                obj["label"]
            ] += 1

        for obj in pred_objects:
            pred_by_class[
                obj["label"]
            ].append(obj)

            pred_label_counts[
                obj["label"]
            ] += 1

        page_tp = 0
        page_fp = 0
        page_fn = 0

        for label in selected_classes:
            gt_class = gt_by_class[label]
            pred_class = pred_by_class[label]

            tp, fp, fn, matches = (
                match_class(
                    gt_class,
                    pred_class,
                    args.iou_threshold,
                )
            )

            stats[label]["gt"] += len(
                gt_class
            )

            stats[label]["pred"] += len(
                pred_class
            )

            stats[label]["tp"] += tp
            stats[label]["fp"] += fp
            stats[label]["fn"] += fn

            stats[label][
                "matched_ious"
            ].extend(
                overlap
                for _, _, overlap
                in matches
            )

            page_tp += tp
            page_fp += fp
            page_fn += fn

        page_results.append(
            {
                "page": page_name,
                "gt_objects": len(
                    gt_objects
                ),
                "pred_objects": len(
                    pred_objects
                ),
                "tp": page_tp,
                "fp": page_fp,
                "fn": page_fn,
            }
        )

    print()
    print(
        "=== SELECTED CLASSES ==="
    )

    for label in selected_classes:
        print(f"  {label}")

    print()
    print(
        "=== GROUND TRUTH COUNTS ==="
    )

    for label in selected_classes:
        print(
            f"  {label:25s} "
            f"{gt_label_counts[label]:4d}"
        )

    print()
    print(
        "=== FILTERED PREDICTION COUNTS ==="
    )

    for label in selected_classes:
        print(
            f"  {label:25s} "
            f"{pred_label_counts[label]:4d}"
        )

    print()
    print(
        "=== IGNORED PADDLE CLASSES ==="
    )

    if ignored_pred_counts:
        for label, count in sorted(
            ignored_pred_counts.items()
        ):
            print(
                f"  {label:25s} "
                f"{count:4d}"
            )
    else:
        print("  none")

    if ignored_gt_counts:
        print()
        print(
            "=== IGNORED GROUND-TRUTH "
            "CLASSES ==="
        )

        for label, count in sorted(
            ignored_gt_counts.items()
        ):
            print(
                f"  {label:25s} "
                f"{count:4d}"
            )

    results = []

    for label in selected_classes:
        class_stats = stats[label]

        precision = safe_div(
            class_stats["tp"],
            (
                class_stats["tp"]
                + class_stats["fp"]
            ),
        )

        recall = safe_div(
            class_stats["tp"],
            (
                class_stats["tp"]
                + class_stats["fn"]
            ),
        )

        f1 = f1_score(
            precision,
            recall,
        )

        if class_stats["matched_ious"]:
            mean_iou = (
                sum(
                    class_stats[
                        "matched_ious"
                    ]
                )
                / len(
                    class_stats[
                        "matched_ious"
                    ]
                )
            )
        else:
            mean_iou = None

        results.append(
            {
                "label": label,
                "gt": class_stats["gt"],
                "pred": class_stats["pred"],
                "tp": class_stats["tp"],
                "fp": class_stats["fp"],
                "fn": class_stats["fn"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_matched_iou": mean_iou,
            }
        )

    print()
    print(
        "=== PER-CLASS RESULTS ==="
    )
    print()

    print(
        f"{'Label':25s}"
        f"{'GT':>6s}"
        f"{'Pred':>7s}"
        f"{'TP':>6s}"
        f"{'FP':>6s}"
        f"{'FN':>6s}"
        f"{'Precision':>11s}"
        f"{'Recall':>10s}"
        f"{'F1':>10s}"
        f"{'Mean IoU':>11s}"
    )

    print("-" * 98)

    for result in results:
        print(
            f"{result['label']:25s}"
            f"{result['gt']:6d}"
            f"{result['pred']:7d}"
            f"{result['tp']:6d}"
            f"{result['fp']:6d}"
            f"{result['fn']:6d}"
            f"{format_metric(result['precision']):>11s}"
            f"{format_metric(result['recall']):>10s}"
            f"{format_metric(result['f1']):>10s}"
            f"{format_metric(result['mean_matched_iou']):>11s}"
        )

    total_tp = sum(
        result["tp"]
        for result in results
    )

    total_fp = sum(
        result["fp"]
        for result in results
    )

    total_fn = sum(
        result["fn"]
        for result in results
    )

    micro_precision = safe_div(
        total_tp,
        total_tp + total_fp,
    )

    micro_recall = safe_div(
        total_tp,
        total_tp + total_fn,
    )

    micro_f1 = f1_score(
        micro_precision,
        micro_recall,
    )

    valid_f1_values = [
        result["f1"]
        for result in results
        if result["f1"] is not None
    ]

    macro_f1 = (
        sum(valid_f1_values)
        / len(valid_f1_values)
        if valid_f1_values
        else None
    )

    all_matched_ious = []

    for label in selected_classes:
        all_matched_ious.extend(
            stats[label][
                "matched_ious"
            ]
        )

    overall_mean_iou = (
        sum(all_matched_ious)
        / len(all_matched_ious)
        if all_matched_ious
        else None
    )

    print()
    print(
        "=== OVERALL / SELECTED CLASSES ==="
    )

    print(
        f"Pages:             "
        f"{len(gt_files)}"
    )

    print(
        f"Selected classes:  "
        f"{len(selected_classes)}"
    )

    print(
        f"GT objects:        "
        f"{sum(r['gt'] for r in results)}"
    )

    print(
        f"Pred objects:      "
        f"{sum(r['pred'] for r in results)}"
    )

    print(f"TP:                {total_tp}")
    print(f"FP:                {total_fp}")
    print(f"FN:                {total_fn}")

    print(
        f"Micro precision:   "
        f"{format_metric(micro_precision)}"
    )

    print(
        f"Micro recall:      "
        f"{format_metric(micro_recall)}"
    )

    print(
        f"Micro F1:          "
        f"{format_metric(micro_f1)}"
    )

    print(
        f"Macro F1:          "
        f"{format_metric(macro_f1)}"
    )

    print(
        f"Mean matched IoU:  "
        f"{format_metric(overall_mean_iou)}"
    )

    per_class_csv = (
        args.output_dir
        / "per_class_metrics.csv"
    )

    with per_class_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                "label",
                "gt",
                "pred",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "mean_matched_iou",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    per_page_csv = (
        args.output_dir
        / "per_page_metrics.csv"
    )

    with per_page_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                "page",
                "gt_objects",
                "pred_objects",
                "tp",
                "fp",
                "fn",
            ],
        )

        writer.writeheader()
        writer.writerows(
            page_results
        )

    summary = {
        "pages": len(gt_files),
        "selected_classes": selected_classes,
        "score_threshold": (
            args.score_threshold
        ),
        "iou_threshold": (
            args.iou_threshold
        ),
        "ground_truth_counts": {
            label: gt_label_counts[label]
            for label in selected_classes
        },
        "prediction_counts": {
            label: pred_label_counts[label]
            for label in selected_classes
        },
        "ignored_prediction_classes": dict(
            sorted(
                ignored_pred_counts.items()
            )
        ),
        "micro": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": (
                micro_precision
            ),
            "recall": (
                micro_recall
            ),
            "f1": micro_f1,
        },
        "macro_f1": macro_f1,
        "mean_matched_iou": (
            overall_mean_iou
        ),
        "per_class": results,
    }

    summary_path = (
        args.output_dir
        / "evaluation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=== SAVED ===")
    print(per_class_csv)
    print(per_page_csv)
    print(summary_path)


if __name__ == "__main__":
    main()
