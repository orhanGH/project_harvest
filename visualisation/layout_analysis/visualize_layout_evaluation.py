#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
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

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def load_gt(path):
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    result = []

    for shape in data.get("shapes", []):
        label = normalize_label(
            shape.get("label", "")
        )

        if label not in SELECTED_CLASSES:
            continue

        points = shape.get("points", [])

        if not points:
            continue

        result.append(
            {
                "label": label,
                "bbox": bbox_from_points(points),
            }
        )

    return result


def load_predictions(path, score_threshold):
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    payload = data.get("res", data)

    result = []

    for box in payload.get("boxes", []):
        label = normalize_label(
            box.get("label", "")
        )

        score = float(
            box.get("score", 0.0)
        )

        if label not in SELECTED_CLASSES:
            continue

        if score < score_threshold:
            continue

        coordinate = box.get("coordinate")

        if (
            coordinate is None
            or len(coordinate) != 4
        ):
            continue

        result.append(
            {
                "label": label,
                "score": score,
                "bbox": [
                    float(x)
                    for x in coordinate
                ],
            }
        )

    return result


def match_objects(gt, pred, iou_threshold):
    gt_by_class = defaultdict(list)
    pred_by_class = defaultdict(list)

    for index, obj in enumerate(gt):
        gt_by_class[obj["label"]].append(
            (index, obj)
        )

    for index, obj in enumerate(pred):
        pred_by_class[obj["label"]].append(
            (index, obj)
        )

    matched_gt = set()
    matched_pred = set()

    matches = []

    for label in SELECTED_CLASSES:
        candidates = []

        for gt_index, gt_obj in gt_by_class[label]:
            for pred_index, pred_obj in pred_by_class[label]:
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
            key=lambda x: x[0],
            reverse=True,
        )

        for overlap, gt_index, pred_index in candidates:
            if gt_index in matched_gt:
                continue

            if pred_index in matched_pred:
                continue

            matched_gt.add(gt_index)
            matched_pred.add(pred_index)

            matches.append(
                {
                    "label": label,
                    "gt_index": gt_index,
                    "pred_index": pred_index,
                    "iou": overlap,
                }
            )

    return matched_gt, matched_pred, matches


def load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]

    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(
                candidate,
                size=size,
            )

    return ImageFont.load_default()


def scaled_bbox(bbox, scale):
    return [
        int(round(value * scale))
        for value in bbox
    ]


def draw_label(
    draw,
    xy,
    text,
    font,
    fill,
):
    x, y = xy

    bbox = draw.textbbox(
        (x, y),
        text,
        font=font,
    )

    padding = 4

    background = [
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding,
    ]

    draw.rectangle(
        background,
        fill=fill,
    )

    draw.text(
        (x, y),
        text,
        fill="white",
        font=font,
    )


def prepare_panel(
    source_image,
    max_height,
):
    image = source_image.convert("RGB")

    if image.height <= max_height:
        return image.copy(), 1.0

    scale = max_height / image.height

    width = int(
        round(image.width * scale)
    )

    height = int(
        round(image.height * scale)
    )

    image = image.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    return image, scale


def draw_gt_panel(
    source_image,
    gt,
    matched_gt,
    max_height,
):
    image, scale = prepare_panel(
        source_image,
        max_height,
    )

    draw = ImageDraw.Draw(image)

    font_size = max(
        16,
        int(24 * scale),
    )

    font = load_font(font_size)

    line_width = max(
        2,
        int(5 * scale),
    )

    for index, obj in enumerate(gt):
        box = scaled_bbox(
            obj["bbox"],
            scale,
        )

        matched = index in matched_gt

        color = (
            "#198754"
            if matched
            else "#dc3545"
        )

        status = (
            "MATCH"
            if matched
            else "MISS"
        )

        draw.rectangle(
            box,
            outline=color,
            width=line_width,
        )

        label = (
            f"{obj['label']} | {status}"
        )

        draw_label(
            draw,
            (
                box[0] + 2,
                max(2, box[1] + 2),
            ),
            label,
            font,
            color,
        )

    return image


def draw_prediction_panel(
    source_image,
    pred,
    matched_pred,
    max_height,
):
    image, scale = prepare_panel(
        source_image,
        max_height,
    )

    draw = ImageDraw.Draw(image)

    font_size = max(
        16,
        int(24 * scale),
    )

    font = load_font(font_size)

    line_width = max(
        2,
        int(5 * scale),
    )

    for index, obj in enumerate(pred):
        box = scaled_bbox(
            obj["bbox"],
            scale,
        )

        matched = index in matched_pred

        color = (
            "#198754"
            if matched
            else "#fd7e14"
        )

        status = (
            "MATCH"
            if matched
            else "FP"
        )

        draw.rectangle(
            box,
            outline=color,
            width=line_width,
        )

        label = (
            f"{obj['label']} "
            f"{obj['score']:.2f} | "
            f"{status}"
        )

        draw_label(
            draw,
            (
                box[0] + 2,
                max(2, box[1] + 2),
            ),
            label,
            font,
            color,
        )

    return image


def add_title(image, title, subtitle):
    bar_height = 72

    result = Image.new(
        "RGB",
        (
            image.width,
            image.height + bar_height,
        ),
        "white",
    )

    result.paste(
        image,
        (0, bar_height),
    )

    draw = ImageDraw.Draw(result)

    title_font = load_font(25)
    subtitle_font = load_font(17)

    draw.text(
        (16, 8),
        title,
        fill="black",
        font=title_font,
    )

    draw.text(
        (16, 40),
        subtitle,
        fill="#444444",
        font=subtitle_font,
    )

    return result


def combine_panels(left, right):
    gap = 20

    height = max(
        left.height,
        right.height,
    )

    result = Image.new(
        "RGB",
        (
            left.width
            + right.width
            + gap,
            height,
        ),
        "#dddddd",
    )

    result.paste(
        left,
        (0, 0),
    )

    result.paste(
        right,
        (
            left.width + gap,
            0,
        ),
    )

    return result


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
        default=0.50,
    )

    parser.add_argument(
        "--iou-threshold",
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
        args.output_dir / "pages"
    )

    pages_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    label_jsons = sorted(
        args.labels_dir.glob(
            "page_*.json"
        )
    )

    if not label_jsons:
        raise RuntimeError(
            "No Labelme annotations found."
        )

    for gt_path in label_jsons:
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
                f"Missing image: {image_path}"
            )

        if not pred_path.exists():
            raise RuntimeError(
                f"Missing prediction: {pred_path}"
            )

        gt = load_gt(gt_path)

        pred = load_predictions(
            pred_path,
            args.score_threshold,
        )

        (
            matched_gt,
            matched_pred,
            matches,
        ) = match_objects(
            gt,
            pred,
            args.iou_threshold,
        )

        tp = len(matches)
        fn = len(gt) - tp
        fp = len(pred) - tp

        with Image.open(image_path) as source:
            left = draw_gt_panel(
                source,
                gt,
                matched_gt,
                args.max_height,
            )

            right = draw_prediction_panel(
                source,
                pred,
                matched_pred,
                args.max_height,
            )

        left = add_title(
            left,
            f"{stem} - HUMAN GROUND TRUTH",
            (
                f"GT={len(gt)} | "
                f"matched={tp} | "
                f"missed={fn}"
            ),
        )

        right = add_title(
            right,
            f"{stem} - PADDLE 9 LABELS",
            (
                f"pred={len(pred)} | "
                f"matched={tp} | "
                f"false positives={fp}"
            ),
        )

        comparison = combine_panels(
            left,
            right,
        )

        output_path = (
            pages_dir
            / f"{stem}_comparison.jpg"
        )

        comparison.save(
            output_path,
            format="JPEG",
            quality=88,
            optimize=True,
        )

        mean_iou = (
            sum(
                match["iou"]
                for match in matches
            )
            / len(matches)
            if matches
            else 0.0
        )

        summary_rows.append(
            {
                "page": stem,
                "gt": len(gt),
                "pred": len(pred),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "errors": fp + fn,
                "mean_matched_iou": mean_iou,
            }
        )

        print(
            f"{stem}: "
            f"GT={len(gt)} "
            f"Pred={len(pred)} "
            f"TP={tp} "
            f"FP={fp} "
            f"FN={fn}"
        )

    summary_rows.sort(
        key=lambda row: (
            row["errors"],
            row["fp"],
            row["fn"],
        ),
        reverse=True,
    )

    csv_path = (
        args.output_dir
        / "page_error_summary.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "page",
                "gt",
                "pred",
                "tp",
                "fp",
                "fn",
                "errors",
                "mean_matched_iou",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    worst_path = (
        args.output_dir
        / "worst_pages.txt"
    )

    with worst_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "HARVEST 9-label visual error analysis\n"
        )
        handle.write(
            "confidence >= 0.50, IoU >= 0.50\n\n"
        )

        for rank, row in enumerate(
            summary_rows,
            start=1,
        ):
            handle.write(
                f"{rank:02d}. "
                f"{row['page']}  "
                f"GT={row['gt']}  "
                f"Pred={row['pred']}  "
                f"TP={row['tp']}  "
                f"FP={row['fp']}  "
                f"FN={row['fn']}  "
                f"errors={row['errors']}\n"
            )

    print()
    print("=== FINISHED ===")
    print(
        "Comparison pages:",
        len(summary_rows),
    )
    print("Images:", pages_dir)
    print("CSV:", csv_path)
    print("Worst pages:", worst_path)


if __name__ == "__main__":
    main()
