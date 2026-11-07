#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "table": (0, 150, 136),
    "text": (30, 136, 229),
    "content": (67, 160, 71),
    "title": (123, 31, 162),
    "page_metadata": (245, 124, 0),
}


def parse_pages(value):
    pages = []

    for part in value.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            start, end = part.split("-", 1)

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


def load_boxes(path):
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    payload = data.get(
        "res",
        data,
    )

    return payload.get(
        "boxes",
        [],
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rendered-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--pred-dir",
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

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pages = parse_pages(
        args.pages
    )

    font = ImageFont.load_default()

    for page in pages:
        image_path = (
            args.rendered_dir
            / f"page_{page:04d}.png"
        )

        json_path = (
            args.pred_dir
            / f"page_{page:04d}.json"
        )

        if not image_path.exists():
            print(
                "Missing image:",
                image_path,
            )
            continue

        if not json_path.exists():
            print(
                "Missing JSON:",
                json_path,
            )
            continue

        image = Image.open(
            image_path
        ).convert("RGB")

        draw = ImageDraw.Draw(
            image
        )

        boxes = load_boxes(
            json_path
        )

        for index, box in enumerate(
            boxes,
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

            coordinate = box.get(
                "coordinate"
            )

            if not coordinate:
                continue

            x1, y1, x2, y2 = [
                float(value)
                for value in coordinate
            ]

            color = COLORS.get(
                label,
                (255, 0, 0),
            )

            draw.rectangle(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                outline=color,
                width=4,
            )

            merged_count = int(
                box.get(
                    "merged_from_count",
                    1,
                )
            )

            text = (
                f"{index} "
                f"{label} "
                f"{score:.2f}"
            )

            if merged_count > 1:
                text += (
                    f" [{merged_count}]"
                )

            text_bbox = draw.textbbox(
                (x1, y1),
                text,
                font=font,
            )

            draw.rectangle(
                text_bbox,
                fill=(255, 255, 255),
            )

            draw.text(
                (x1, y1),
                text,
                fill=color,
                font=font,
            )

        output_path = (
            args.output_dir
            / f"page_{page:04d}_merged.png"
        )

        image.save(
            output_path
        )

        print(
            f"page {page:04d}: "
            f"{len(boxes)} regions "
            f"-> {output_path}"
        )


if __name__ == "__main__":
    main()
