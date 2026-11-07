#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SELECTED_CLASSES = [
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


# ------------------------------------------------------------
# Colors
# ------------------------------------------------------------

GREEN = "#198754"
YELLOW = "#d39e00"
BLUE = "#0d6efd"
RED = "#dc3545"

TEXT_COLOR = "white"


# ------------------------------------------------------------
# Label / box helpers
# ------------------------------------------------------------

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


def box_area(box):
    x1, y1, x2, y2 = box

    return (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )


def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return width * height


def iou(a, b):
    inter = intersection_area(a, b)

    if inter <= 0:
        return 0.0

    union = (
        box_area(a)
        + box_area(b)
        - inter
    )

    if union <= 0:
        return 0.0

    return inter / union


def overlap_coefficient(a, b):
    """
    Intersection / smaller-box area.

    This is useful when one box is largely contained
    inside the other but IoU is low because their sizes
    differ strongly.
    """

    inter = intersection_area(a, b)

    if inter <= 0:
        return 0.0

    smaller = min(
        box_area(a),
        box_area(b),
    )

    if smaller <= 0:
        return 0.0

    return inter / smaller


def gt_coverage(gt_box, pred_box):
    inter = intersection_area(
        gt_box,
        pred_box,
    )

    area = box_area(
        gt_box
    )

    if area <= 0:
        return 0.0

    return inter / area


def pred_purity(gt_box, pred_box):
    inter = intersection_area(
        gt_box,
        pred_box,
    )

    area = box_area(
        pred_box
    )

    if area <= 0:
        return 0.0

    return inter / area


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def load_gt(path):
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

        if label not in SELECTED_CLASSES:
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


def load_predictions(
    path,
    score_threshold,
):
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
        label = normalize_label(
            box.get(
                "label",
                "",
            )
        )

        if label not in SELECTED_CLASSES:
            continue

        score = float(
            box.get(
                "score",
                0.0,
            )
        )

        if score < score_threshold:
            continue

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
                "label": label,
                "score": score,
                "bbox": [
                    float(value)
                    for value in coordinate
                ],
                "merged_from_count":
                    int(
                        box.get(
                            "merged_from_count",
                            1,
                        )
                    ),
            }
        )

    return objects


# ------------------------------------------------------------
# Strict IoU matching
# ------------------------------------------------------------

def strict_match(
    gt,
    pred,
    iou_threshold,
):
    """
    One-to-one same-class greedy IoU matching.
    """

    candidates = []

    for gt_index, gt_obj in enumerate(gt):
        for pred_index, pred_obj in enumerate(
            pred
        ):
            if (
                gt_obj["label"]
                != pred_obj["label"]
            ):
                continue

            overlap = iou(
                gt_obj["bbox"],
                pred_obj["bbox"],
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
        reverse=True
    )

    matched_gt = set()
    matched_pred = set()

    matches = []

    for (
        overlap,
        gt_index,
        pred_index,
    ) in candidates:

        if gt_index in matched_gt:
            continue

        if pred_index in matched_pred:
            continue

        matched_gt.add(
            gt_index
        )

        matched_pred.add(
            pred_index
        )

        matches.append(
            {
                "gt_index":
                    gt_index,
                "pred_index":
                    pred_index,
                "iou":
                    overlap,
            }
        )

    return (
        matched_gt,
        matched_pred,
        matches,
    )


# ------------------------------------------------------------
# Diagnostic relationship
# ------------------------------------------------------------

def best_relation(
    source_obj,
    candidates,
    same_class_only=False,
):
    """
    Find the candidate with the strongest containment-style
    spatial overlap.

    Primary sort:
        overlap coefficient

    Tie-breakers:
        IoU
        GT coverage
        prediction purity
    """

    best = None

    for index, candidate in enumerate(
        candidates
    ):
        if (
            same_class_only
            and candidate["label"]
            != source_obj["label"]
        ):
            continue

        overlap = overlap_coefficient(
            source_obj["bbox"],
            candidate["bbox"],
        )

        pair_iou = iou(
            source_obj["bbox"],
            candidate["bbox"],
        )

        coverage = gt_coverage(
            source_obj["bbox"],
            candidate["bbox"],
        )

        purity = pred_purity(
            source_obj["bbox"],
            candidate["bbox"],
        )

        key = (
            overlap,
            pair_iou,
            coverage,
            purity,
        )

        if (
            best is None
            or key > best["key"]
        ):
            best = {
                "index": index,
                "object": candidate,
                "overlap":
                    overlap,
                "iou":
                    pair_iou,
                "coverage":
                    coverage,
                "purity":
                    purity,
                "key":
                    key,
            }

    return best


def classify_gt(
    gt_index,
    gt,
    pred,
    matched_gt,
    strict_by_gt,
    overlap_threshold,
):
    if gt_index in matched_gt:
        match = strict_by_gt[
            gt_index
        ]

        return {
            "status": "STRICT",
            "color": GREEN,
            "related_label":
                gt[gt_index]["label"],
            "iou":
                match["iou"],
            "overlap":
                1.0,
        }

    obj = gt[
        gt_index
    ]

    same = best_relation(
        obj,
        pred,
        same_class_only=True,
    )

    if (
        same is not None
        and same["overlap"]
        >= overlap_threshold
    ):
        return {
            "status": "GEOMETRY",
            "color": YELLOW,
            "related_label":
                same["object"]["label"],
            "iou":
                same["iou"],
            "overlap":
                same["overlap"],
        }

    any_relation = best_relation(
        obj,
        pred,
        same_class_only=False,
    )

    if (
        any_relation is not None
        and any_relation["overlap"]
        >= overlap_threshold
    ):
        return {
            "status": "WRONG_CLASS",
            "color": BLUE,
            "related_label":
                any_relation[
                    "object"
                ]["label"],
            "iou":
                any_relation["iou"],
            "overlap":
                any_relation["overlap"],
        }

    return {
        "status": "MISS",
        "color": RED,
        "related_label": None,
        "iou": 0.0,
        "overlap": 0.0,
    }


def classify_prediction(
    pred_index,
    gt,
    pred,
    matched_pred,
    strict_by_pred,
    overlap_threshold,
):
    if pred_index in matched_pred:
        match = strict_by_pred[
            pred_index
        ]

        return {
            "status": "STRICT",
            "color": GREEN,
            "related_label":
                pred[pred_index]["label"],
            "iou":
                match["iou"],
            "overlap":
                1.0,
        }

    obj = pred[
        pred_index
    ]

    same = best_relation(
        obj,
        gt,
        same_class_only=True,
    )

    if (
        same is not None
        and same["overlap"]
        >= overlap_threshold
    ):
        return {
            "status": "GEOMETRY",
            "color": YELLOW,
            "related_label":
                same["object"]["label"],
            "iou":
                same["iou"],
            "overlap":
                same["overlap"],
        }

    any_relation = best_relation(
        obj,
        gt,
        same_class_only=False,
    )

    if (
        any_relation is not None
        and any_relation["overlap"]
        >= overlap_threshold
    ):
        return {
            "status": "WRONG_CLASS",
            "color": BLUE,
            "related_label":
                any_relation[
                    "object"
                ]["label"],
            "iou":
                any_relation["iou"],
            "overlap":
                any_relation["overlap"],
        }

    return {
        "status": "FP",
        "color": RED,
        "related_label": None,
        "iou": 0.0,
        "overlap": 0.0,
    }


# ------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------

def load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/"
        "dejavu/DejaVuSans-Bold.ttf",

        "/usr/share/fonts/truetype/"
        "dejavu/DejaVuSans.ttf",
    ]

    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(
                candidate,
                size=size,
            )

    return ImageFont.load_default()


def prepare_panel(
    source,
    max_height,
):
    image = source.convert(
        "RGB"
    )

    if image.height <= max_height:
        return (
            image.copy(),
            1.0,
        )

    scale = (
        max_height
        / image.height
    )

    new_width = int(
        round(
            image.width
            * scale
        )
    )

    new_height = int(
        round(
            image.height
            * scale
        )
    )

    resized = image.resize(
        (
            new_width,
            new_height,
        ),
        Image.Resampling.LANCZOS,
    )

    return (
        resized,
        scale,
    )


def scale_box(
    box,
    scale,
):
    return [
        int(
            round(
                value
                * scale
            )
        )
        for value in box
    ]


def draw_tag(
    draw,
    x,
    y,
    text,
    font,
    color,
):
    text_bbox = draw.textbbox(
        (x, y),
        text,
        font=font,
    )

    padding = 3

    background = [
        text_bbox[0] - padding,
        text_bbox[1] - padding,
        text_bbox[2] + padding,
        text_bbox[3] + padding,
    ]

    draw.rectangle(
        background,
        fill=color,
    )

    draw.text(
        (x, y),
        text,
        font=font,
        fill=TEXT_COLOR,
    )


def draw_gt_panel(
    source,
    gt,
    gt_statuses,
    max_height,
):
    image, scale = prepare_panel(
        source,
        max_height,
    )

    draw = ImageDraw.Draw(
        image
    )

    font = load_font(
        max(
            15,
            int(
                22
                * scale
            ),
        )
    )

    line_width = max(
        2,
        int(
            5
            * scale
        ),
    )

    for index, obj in enumerate(
        gt
    ):
        status = gt_statuses[
            index
        ]

        color = status[
            "color"
        ]

        box = scale_box(
            obj["bbox"],
            scale,
        )

        draw.rectangle(
            box,
            outline=color,
            width=line_width,
        )

        if (
            status["status"]
            == "STRICT"
        ):
            label = (
                f"{obj['label']} | MATCH"
            )

        elif (
            status["status"]
            == "GEOMETRY"
        ):
            label = (
                f"{obj['label']} | GEOM "
                f"ov={status['overlap']:.2f}"
            )

        elif (
            status["status"]
            == "WRONG_CLASS"
        ):
            label = (
                f"{obj['label']} <- "
                f"{status['related_label']} "
                f"| CLASS"
            )

        else:
            label = (
                f"{obj['label']} | MISS"
            )

        draw_tag(
            draw,
            box[0] + 2,
            max(
                2,
                box[1] + 2,
            ),
            label,
            font,
            color,
        )

    return image


def draw_prediction_panel(
    source,
    pred,
    pred_statuses,
    max_height,
):
    image, scale = prepare_panel(
        source,
        max_height,
    )

    draw = ImageDraw.Draw(
        image
    )

    font = load_font(
        max(
            15,
            int(
                22
                * scale
            ),
        )
    )

    line_width = max(
        2,
        int(
            5
            * scale
        ),
    )

    for index, obj in enumerate(
        pred
    ):
        status = pred_statuses[
            index
        ]

        color = status[
            "color"
        ]

        box = scale_box(
            obj["bbox"],
            scale,
        )

        draw.rectangle(
            box,
            outline=color,
            width=line_width,
        )

        merge_info = ""

        if (
            obj[
                "merged_from_count"
            ]
            > 1
        ):
            merge_info = (
                f" m{obj['merged_from_count']}"
            )

        if (
            status["status"]
            == "STRICT"
        ):
            label = (
                f"{obj['label']} "
                f"{obj['score']:.2f}"
                f"{merge_info} | MATCH"
            )

        elif (
            status["status"]
            == "GEOMETRY"
        ):
            label = (
                f"{obj['label']} "
                f"{obj['score']:.2f}"
                f"{merge_info} | GEOM "
                f"ov={status['overlap']:.2f}"
            )

        elif (
            status["status"]
            == "WRONG_CLASS"
        ):
            label = (
                f"{obj['label']} "
                f"{obj['score']:.2f}"
                f"{merge_info} -> "
                f"{status['related_label']} "
                f"| CLASS"
            )

        else:
            label = (
                f"{obj['label']} "
                f"{obj['score']:.2f}"
                f"{merge_info} | FP"
            )

        draw_tag(
            draw,
            box[0] + 2,
            max(
                2,
                box[1] + 2,
            ),
            label,
            font,
            color,
        )

    return image


def add_header(
    image,
    title,
    subtitle,
):
    height = 88

    result = Image.new(
        "RGB",
        (
            image.width,
            image.height
            + height,
        ),
        "white",
    )

    result.paste(
        image,
        (
            0,
            height,
        ),
    )

    draw = ImageDraw.Draw(
        result
    )

    title_font = load_font(
        24
    )

    subtitle_font = load_font(
        16
    )

    draw.text(
        (
            14,
            8,
        ),
        title,
        fill="black",
        font=title_font,
    )

    draw.text(
        (
            14,
            42,
        ),
        subtitle,
        fill="#333333",
        font=subtitle_font,
    )

    return result


def combine_panels(
    left,
    right,
):
    gap = 20

    height = max(
        left.height,
        right.height,
    )

    result = Image.new(
        "RGB",
        (
            left.width
            + gap
            + right.width,
            height,
        ),
        "#d0d0d0",
    )

    result.paste(
        left,
        (
            0,
            0,
        ),
    )

    result.paste(
        right,
        (
            left.width
            + gap,
            0,
        ),
    )

    return result


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
    )

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
        "--score-threshold",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--max-height",
        type=int,
        default=2200,
    )

    args = parser.parse_args()

    pages_dir = (
        args.output_dir
        / "pages"
    )

    pages_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    page_rows = []

    global_gt_counts = Counter()
    global_pred_counts = Counter()

    gt_files = sorted(
        args.labels_dir.glob(
            "page_*.json"
        )
    )

    if not gt_files:
        raise RuntimeError(
            "No Labelme JSON files found."
        )

    for gt_path in gt_files:
        stem = gt_path.stem

        image_path = (
            args.images_dir
            / f"{stem}.png"
        )

        pred_path = (
            args.pred_dir
            / f"{stem}.json"
        )

        if not image_path.exists():
            raise RuntimeError(
                f"Missing image: "
                f"{image_path}"
            )

        if not pred_path.exists():
            raise RuntimeError(
                f"Missing prediction: "
                f"{pred_path}"
            )

        gt = load_gt(
            gt_path
        )

        pred = load_predictions(
            pred_path,
            args.score_threshold,
        )

        (
            matched_gt,
            matched_pred,
            strict_matches,
        ) = strict_match(
            gt,
            pred,
            args.iou_threshold,
        )

        strict_by_gt = {
            match["gt_index"]:
                match
            for match in strict_matches
        }

        strict_by_pred = {
            match["pred_index"]:
                match
            for match in strict_matches
        }

        gt_statuses = []

        for gt_index in range(
            len(gt)
        ):
            status = classify_gt(
                gt_index,
                gt,
                pred,
                matched_gt,
                strict_by_gt,
                args.overlap_threshold,
            )

            gt_statuses.append(
                status
            )

            global_gt_counts[
                status["status"]
            ] += 1

        pred_statuses = []

        for pred_index in range(
            len(pred)
        ):
            status = classify_prediction(
                pred_index,
                gt,
                pred,
                matched_pred,
                strict_by_pred,
                args.overlap_threshold,
            )

            pred_statuses.append(
                status
            )

            global_pred_counts[
                status["status"]
            ] += 1

        gt_page_counts = Counter(
            status["status"]
            for status in gt_statuses
        )

        pred_page_counts = Counter(
            status["status"]
            for status in pred_statuses
        )

        with Image.open(
            image_path
        ) as source:

            left = draw_gt_panel(
                source,
                gt,
                gt_statuses,
                args.max_height,
            )

            right = draw_prediction_panel(
                source,
                pred,
                pred_statuses,
                args.max_height,
            )

        left = add_header(
            left,
            f"{stem} - HUMAN GROUND TRUTH",
            (
                f"green=strict  "
                f"yellow=geometry  "
                f"blue=wrong class  "
                f"red=miss"
            ),
        )

        right = add_header(
            right,
            f"{stem} - PADDLE DIAGNOSTIC",
            (
                f"green=strict  "
                f"yellow=geometry  "
                f"blue=wrong class  "
                f"red=false positive"
            ),
        )

        comparison = combine_panels(
            left,
            right,
        )

        output_path = (
            pages_dir
            / f"{stem}_diagnostic.jpg"
        )

        comparison.save(
            output_path,
            format="JPEG",
            quality=90,
            optimize=True,
        )

        row = {
            "page":
                stem,

            "gt_total":
                len(gt),

            "pred_total":
                len(pred),

            "gt_strict":
                gt_page_counts[
                    "STRICT"
                ],

            "gt_geometry":
                gt_page_counts[
                    "GEOMETRY"
                ],

            "gt_wrong_class":
                gt_page_counts[
                    "WRONG_CLASS"
                ],

            "gt_miss":
                gt_page_counts[
                    "MISS"
                ],

            "pred_strict":
                pred_page_counts[
                    "STRICT"
                ],

            "pred_geometry":
                pred_page_counts[
                    "GEOMETRY"
                ],

            "pred_wrong_class":
                pred_page_counts[
                    "WRONG_CLASS"
                ],

            "pred_fp":
                pred_page_counts[
                    "FP"
                ],
        }

        page_rows.append(
            row
        )

        print(
            f"{stem}: "
            f"GT "
            f"strict={row['gt_strict']} "
            f"geom={row['gt_geometry']} "
            f"class={row['gt_wrong_class']} "
            f"miss={row['gt_miss']} | "
            f"PRED "
            f"strict={row['pred_strict']} "
            f"geom={row['pred_geometry']} "
            f"class={row['pred_wrong_class']} "
            f"fp={row['pred_fp']}"
        )

    summary_path = (
        args.output_dir
        / "diagnostic_page_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "page",
                "gt_total",
                "pred_total",
                "gt_strict",
                "gt_geometry",
                "gt_wrong_class",
                "gt_miss",
                "pred_strict",
                "pred_geometry",
                "pred_wrong_class",
                "pred_fp",
            ],
        )

        writer.writeheader()
        writer.writerows(
            page_rows
        )

    totals_path = (
        args.output_dir
        / "diagnostic_totals.txt"
    )

    with totals_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "HARVEST layout diagnostic visualization\n"
        )

        handle.write(
            f"IoU strict threshold: "
            f"{args.iou_threshold:.2f}\n"
        )

        handle.write(
            f"Overlap diagnostic threshold: "
            f"{args.overlap_threshold:.2f}\n\n"
        )

        handle.write(
            "GROUND TRUTH STATUS\n"
        )

        for name in [
            "STRICT",
            "GEOMETRY",
            "WRONG_CLASS",
            "MISS",
        ]:
            handle.write(
                f"{name:15s} "
                f"{global_gt_counts[name]}\n"
            )

        handle.write(
            "\nPREDICTION STATUS\n"
        )

        for name in [
            "STRICT",
            "GEOMETRY",
            "WRONG_CLASS",
            "FP",
        ]:
            handle.write(
                f"{name:15s} "
                f"{global_pred_counts[name]}\n"
            )

    print()
    print(
        "========================================"
    )
    print(
        "DIAGNOSTIC TOTALS"
    )
    print(
        "========================================"
    )

    print()
    print(
        "Ground truth:"
    )

    for name in [
        "STRICT",
        "GEOMETRY",
        "WRONG_CLASS",
        "MISS",
    ]:
        print(
            f"  {name:15s} "
            f"{global_gt_counts[name]}"
        )

    print()
    print(
        "Predictions:"
    )

    for name in [
        "STRICT",
        "GEOMETRY",
        "WRONG_CLASS",
        "FP",
    ]:
        print(
            f"  {name:15s} "
            f"{global_pred_counts[name]}"
        )

    print()
    print(
        "Images:",
        pages_dir,
    )

    print(
        "Page summary:",
        summary_path,
    )

    print(
        "Totals:",
        totals_path,
    )


if __name__ == "__main__":
    main()
