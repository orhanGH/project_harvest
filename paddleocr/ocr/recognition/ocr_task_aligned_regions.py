#!/usr/bin/env python3

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


TASK_MAPPING = {
    "table": "table",
    "text": "text",
    "footnote": "text",
    "content": "content",
    "figure_title": "title",
    "paragraph_title": "title",
    "doc_title": "title",
    "header": "page_metadata",
    "number": "page_metadata",
}


def parse_pages(value):
    pages = []

    for part in value.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            start, end = part.split(
                "-",
                1,
            )

            pages.extend(
                range(
                    int(start),
                    int(end) + 1,
                )
            )

        else:
            pages.append(
                int(part)
            )

    return sorted(set(pages))


def load_payload(path):
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return data.get(
        "res",
        data,
    )


def area(box):
    x1, y1, x2, y2 = box

    return max(
        0.0,
        x2 - x1,
    ) * max(
        0.0,
        y2 - y1,
    )


def intersection(a, b):
    x1 = max(
        a[0],
        b[0],
    )

    y1 = max(
        a[1],
        b[1],
    )

    x2 = min(
        a[2],
        b[2],
    )

    y2 = min(
        a[3],
        b[3],
    )

    return max(
        0.0,
        x2 - x1,
    ) * max(
        0.0,
        y2 - y1,
    )


def overlap_coefficient(a, b):
    denom = min(
        area(a),
        area(b),
    )

    if denom <= 0:
        return 0.0

    return (
        intersection(a, b)
        / denom
    )


def to_builtin(value):
    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        dict,
    ):
        return {
            key: to_builtin(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            to_builtin(item)
            for item in value
        ]

    return value


def recover_sources(
    merged_box,
    original_boxes,
):
    target_box = [
        float(value)
        for value
        in merged_box["coordinate"]
    ]

    task_class = str(
        merged_box["label"]
    )

    sources = []

    for source in original_boxes:
        original_label = str(
            source.get(
                "label",
                "UNKNOWN",
            )
        )

        mapped_label = TASK_MAPPING.get(
            original_label
        )

        if mapped_label != task_class:
            continue

        source_box = source.get(
            "coordinate"
        )

        if not source_box:
            continue

        source_box = [
            float(value)
            for value
            in source_box
        ]

        overlap = overlap_coefficient(
            target_box,
            source_box,
        )

        if overlap < 0.20:
            continue

        sources.append(
            {
                "original_label":
                    original_label,

                "original_cls_id":
                    source.get(
                        "cls_id"
                    ),

                "score":
                    float(
                        source.get(
                            "score",
                            0.0,
                        )
                    ),

                "bbox":
                    source_box,

                "overlap_with_region":
                    overlap,
            }
        )

    sources.sort(
        key=lambda item:
            item[
                "overlap_with_region"
            ],
        reverse=True,
    )

    return sources


def extract_ocr_result(
    result_objects,
):
    lines = []

    for result in result_objects:
        data = to_builtin(
            result.json
        )

        payload = data.get(
            "res",
            data,
        )

        texts = payload.get(
            "rec_texts",
            [],
        )

        scores = payload.get(
            "rec_scores",
            [],
        )

        boxes = payload.get(
            "rec_boxes",
            [],
        )

        for index, text in enumerate(
            texts
        ):
            score = (
                float(scores[index])
                if index < len(scores)
                else None
            )

            bbox = (
                boxes[index]
                if index < len(boxes)
                else None
            )

            lines.append(
                {
                    "text": str(text),
                    "confidence": score,
                    "bbox_in_crop": bbox,
                }
            )

    confidence_values = [
        line["confidence"]
        for line in lines
        if line["confidence"]
        is not None
    ]

    mean_confidence = (
        statistics.mean(
            confidence_values
        )
        if confidence_values
        else None
    )

    text = "\n".join(
        line["text"]
        for line in lines
        if line["text"].strip()
    )

    return {
        "text": text,
        "mean_confidence":
            mean_confidence,
        "lines": lines,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rendered-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--merged-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--original-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--pages",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="gpu:0",
    )

    args = parser.parse_args()

    structured_dir = (
        args.output_dir
        / "structured"
    )

    crops_root = (
        args.output_dir
        / "crops"
    )

    structured_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    crops_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    pages = parse_pages(
        args.pages
    )

    print(
        "Initializing PP-OCRv5..."
    )

    ocr = PaddleOCR(
        text_detection_model_name=
            "PP-OCRv5_server_det",

        text_recognition_model_name=
            "PP-OCRv5_server_rec",

        use_doc_orientation_classify=
            False,

        use_doc_unwarping=
            False,

        use_textline_orientation=
            False,

        device=args.device,
    )

    print(
        "OCR initialized."
    )

    for page in pages:
        print()
        print(
            "=" * 60
        )
        print(
            f"PAGE {page}"
        )
        print(
            "=" * 60
        )

        image_path = (
            args.rendered_dir
            / f"page_{page:04d}.png"
        )

        merged_path = (
            args.merged_dir
            / f"page_{page:04d}.json"
        )

        original_path = (
            args.original_dir
            / f"page_{page:04d}.json"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                image_path
            )

        if not merged_path.exists():
            raise FileNotFoundError(
                merged_path
            )

        if not original_path.exists():
            raise FileNotFoundError(
                original_path
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        merged_payload = load_payload(
            merged_path
        )

        original_payload = load_payload(
            original_path
        )

        merged_boxes = merged_payload.get(
            "boxes",
            [],
        )

        original_boxes = (
            original_payload.get(
                "boxes",
                [],
            )
        )

        page_crop_dir = (
            crops_root
            / f"page_{page:04d}"
        )

        page_crop_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        page_record = {
            "page": page,
            "image":
                str(image_path),
            "regions": [],
        }

        for index, box in enumerate(
            merged_boxes,
            start=1,
        ):
            label = str(
                box.get(
                    "label",
                    "UNKNOWN",
                )
            )

            score = float(
                box.get(
                    "score",
                    0.0,
                )
            )

            coordinate = [
                float(value)
                for value
                in box["coordinate"]
            ]

            x1, y1, x2, y2 = coordinate

            margin = 4

            left = max(
                0,
                int(x1) - margin,
            )

            top = max(
                0,
                int(y1) - margin,
            )

            right = min(
                image.width,
                int(x2) + margin,
            )

            bottom = min(
                image.height,
                int(y2) + margin,
            )

            crop = image.crop(
                (
                    left,
                    top,
                    right,
                    bottom,
                )
            )

            crop_path = (
                page_crop_dir
                / (
                    f"region_{index:03d}_"
                    f"{label}.png"
                )
            )

            crop.save(
                crop_path
            )

            sources = recover_sources(
                box,
                original_boxes,
            )

            source_labels = sorted(
                {
                    item[
                        "original_label"
                    ]
                    for item
                    in sources
                }
            )

            region = {
                "region_id":
                    (
                        f"page{page:04d}_"
                        f"region{index:03d}"
                    ),

                "layout_class":
                    label,

                "bbox":
                    coordinate,

                "layout_confidence":
                    score,

                "merged_from_count":
                    int(
                        box.get(
                            "merged_from_count",
                            1,
                        )
                    ),

                "source_original_labels":
                    source_labels,

                "source_regions":
                    sources,

                "crop_path":
                    str(crop_path),
            }

            if label == "table":
                region["ocr"] = None

                region["next_stage"] = (
                    "table_recognition"
                )

                print(
                    f"{index:03d} "
                    f"{label:14s} "
                    "-> deferred to "
                    "table recognition"
                )

            else:
                crop_array = np.array(
                    crop
                )

                results = list(
                    ocr.predict(
                        crop_array
                    )
                )

                ocr_result = (
                    extract_ocr_result(
                        results
                    )
                )

                region["ocr"] = (
                    ocr_result
                )

                region["next_stage"] = (
                    "structured_page"
                )

                preview = (
                    ocr_result["text"]
                    .replace(
                        "\n",
                        " | ",
                    )
                )

                if len(preview) > 100:
                    preview = (
                        preview[:97]
                        + "..."
                    )

                print(
                    f"{index:03d} "
                    f"{label:14s} "
                    f"lines="
                    f"{len(ocr_result['lines']):3d} "
                    f"text={preview}"
                )

            page_record[
                "regions"
            ].append(
                region
            )

        output_path = (
            structured_dir
            / f"page_{page:04d}.json"
        )

        output_path.write_text(
            json.dumps(
                page_record,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print(
            "Saved:",
            output_path,
        )


if __name__ == "__main__":
    main()
