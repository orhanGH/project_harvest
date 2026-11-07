#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def flatten(value):
    if isinstance(value, (list, tuple)):
        result = []

        for item in value:
            result.extend(
                flatten(item)
            )

        return result

    return [value]


def to_bbox(value):
    values = [
        float(item)
        for item in flatten(value)
    ]

    if len(values) == 4:
        x1, y1, x2, y2 = values

        return [
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        ]

    if (
        len(values) >= 8
        and len(values) % 2 == 0
    ):
        xs = values[0::2]
        ys = values[1::2]

        return [
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        ]

    raise ValueError(
        f"Unsupported box format: {value}"
    )


def center(box):
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def contains(cell, point):
    x1, y1, x2, y2 = cell
    x, y = point

    return (
        x1 <= x <= x2
        and y1 <= y <= y2
    )


def intersection_area(a, b):
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

    width = max(
        0.0,
        x2 - x1,
    )

    height = max(
        0.0,
        y2 - y1,
    )

    return width * height


def area(box):
    return max(
        0.0,
        box[2] - box[0],
    ) * max(
        0.0,
        box[3] - box[1],
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-json",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    data = json.loads(
        args.input_json.read_text(
            encoding="utf-8"
        )
    )

    payload = data.get(
        "res",
        data,
    )

    ocr = payload.get(
        "overall_ocr_res",
        {},
    )

    texts = ocr.get(
        "rec_texts",
        [],
    )

    token_boxes = ocr.get(
        "rec_boxes",
        [],
    )

    scores = ocr.get(
        "rec_scores",
        [],
    )

    tables = payload.get(
        "table_res_list",
        [],
    )

    if not tables:
        raise RuntimeError(
            "No table_res_list found."
        )

    raw_cells = tables[0].get(
        "cell_box_list",
        [],
    )

    cells = []

    for index, raw_box in enumerate(
        raw_cells,
        start=1,
    ):
        bbox = to_bbox(
            raw_box
        )

        cx, cy = center(
            bbox
        )

        cells.append(
            {
                "index": index,
                "bbox": bbox,
                "cx": cx,
                "cy": cy,
                "tokens": [],
            }
        )

    tokens = []

    for index, (
        text,
        raw_box,
    ) in enumerate(
        zip(
            texts,
            token_boxes,
        ),
        start=1,
    ):
        bbox = to_bbox(
            raw_box
        )

        cx, cy = center(
            bbox
        )

        score = (
            float(scores[index - 1])
            if index - 1 < len(scores)
            else None
        )

        tokens.append(
            {
                "index": index,
                "text": str(text),
                "bbox": bbox,
                "cx": cx,
                "cy": cy,
                "score": score,
            }
        )

    print()
    print("=== TABLE CELL DIAGNOSTIC ===")
    print()

    print(
        "OCR tokens:",
        len(tokens),
    )

    print(
        "Detected cells:",
        len(cells),
    )

    print()

    unassigned = []

    for token in tokens:
        token_point = (
            token["cx"],
            token["cy"],
        )

        candidates = []

        for cell in cells:
            inside = contains(
                cell["bbox"],
                token_point,
            )

            overlap = intersection_area(
                token["bbox"],
                cell["bbox"],
            )

            token_area = area(
                token["bbox"]
            )

            overlap_ratio = (
                overlap / token_area
                if token_area > 0
                else 0.0
            )

            if (
                inside
                or overlap_ratio >= 0.50
            ):
                candidates.append(
                    (
                        cell,
                        overlap_ratio,
                    )
                )

        if not candidates:
            unassigned.append(
                token
            )
            continue

        candidates.sort(
            key=lambda item: (
                not contains(
                    item[0]["bbox"],
                    token_point,
                ),
                -item[1],
                area(item[0]["bbox"]),
            )
        )

        chosen = candidates[0][0]

        chosen["tokens"].append(
            token
        )

    cells.sort(
        key=lambda item: (
            item["bbox"][1],
            item["bbox"][0],
        )
    )

    print(
        f"{'Cell':>5s} "
        f"{'x1':>6s} "
        f"{'y1':>6s} "
        f"{'x2':>6s} "
        f"{'y2':>6s} "
        f"Text"
    )

    print("-" * 110)

    nonempty = 0

    for cell in cells:
        if not cell["tokens"]:
            continue

        nonempty += 1

        cell["tokens"].sort(
            key=lambda item: (
                item["cy"],
                item["cx"],
            )
        )

        text = " | ".join(
            token["text"]
            for token
            in cell["tokens"]
        )

        x1, y1, x2, y2 = (
            cell["bbox"]
        )

        print(
            f"{cell['index']:5d} "
            f"{x1:6.0f} "
            f"{y1:6.0f} "
            f"{x2:6.0f} "
            f"{y2:6.0f} "
            f"{text}"
        )

    print()
    print(
        "Non-empty cells:",
        nonempty,
    )

    print(
        "Unassigned OCR tokens:",
        len(unassigned),
    )

    print()
    print("=== IMPORTANT TOKENS ===")
    print()

    keywords = {
        "PAYS",
        "COUNTRY",
        "1925",
        "Canada",
        "Etats-Unis..",
        "United States.",
        "France..",
        "France.",
        "Allemagne",
        "Germany.",
    }

    for cell in cells:
        matching = [
            token
            for token
            in cell["tokens"]
            if token["text"]
            in keywords
        ]

        if not matching:
            continue

        cell_text = " | ".join(
            token["text"]
            for token
            in cell["tokens"]
        )

        print(
            f"cell={cell['index']:03d} "
            f"bbox={cell['bbox']} "
            f"text={cell_text}"
        )

    print()
    print("=== UNASSIGNED IMPORTANT TOKENS ===")
    print()

    for token in unassigned:
        if token["text"] in keywords:
            print(
                token["index"],
                token["text"],
                token["bbox"],
            )


if __name__ == "__main__":
    main()
