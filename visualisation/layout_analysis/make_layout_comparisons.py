#!/usr/bin/env python3
"""HARVEST: create four visual comparisons per page; no prediction JSON required."""
import argparse
import csv
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path.home() / "Desktop" / "Arbeit" / "HARVEST"

COLORS = {
    "table": "#d32323",
    "footnote": "#128a36",
    "number": "#858100",
    "header": "#2255d5",
    "figure_title": "#a000b0",
    "reference": "#06949c",
    "paragraph_title": "#777777",
    "text": "#c95c00",
    "content": "#008b95",
    "doc_title": "#984319",
    "seal": "#333333",
}

# Every comparison contains ground truth.
# Each panel has a maximum of three columns.

COMPARISONS = {
    "E01_GT_vs_baseline": [
        ("Ground truth", "gt"),
        ("E01 | All labels, 0.50", "e01"),
    ],
    "E02_GT_vs_9labels": [
        ("Ground truth", "gt"),
        ("E02 | Nine labels, 0.50", "e02"),
    ],
    "E01_vs_E02_label_filtering": [
        ("Ground truth", "gt"),
        ("E01 | All labels, 0.50", "e01"),
        ("E02 | Nine labels, 0.50", "e02"),
    ],
    "E02_vs_E03_threshold_change": [
        ("Ground truth", "gt"),
        ("E02 | Nine labels, 0.50", "e02"),
        ("E03 | Class-specific thresholds", "e03"),
    ],
}


def font(size):
    for name in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)

    return ImageFont.load_default()


def read_image(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as image:
        return image.convert("RGB")


def ground_truth(image, label_path):
    data = json.loads(label_path.read_text(encoding="utf-8"))

    expected = (
        data.get("imageWidth"),
        data.get("imageHeight"),
    )

    if expected != image.size:
        raise ValueError(
            f"GT/PNG size mismatch in {label_path.name}: "
            f"{expected} vs {image.size}"
        )

    result = image.copy()
    draw = ImageDraw.Draw(result)

    width, height = image.size

    f = font(max(18, min(36, int(width * 0.013))))
    stroke = max(3, int(min(width, height) / 450))

    captions = []

    for shape in data.get("shapes", []):
        name = (
            str(shape.get("label", "unknown"))
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        points = shape.get("points") or []

        if len(points) < 2:
            continue

        color = COLORS.get(name, "#444444")
        coords = [(float(x), float(y)) for x, y in points]

        if shape.get("shape_type", "rectangle") == "rectangle":
            x1, x2 = sorted((coords[0][0], coords[1][0]))
            y1, y2 = sorted((coords[0][1], coords[1][1]))

            draw.rectangle(
                (x1, y1, x2, y2),
                outline=color,
                width=stroke,
            )
        else:
            x1 = min(x for x, y in coords)
            y1 = min(y for x, y in coords)

            draw.line(
                coords + [coords[0]],
                fill=color,
                width=stroke,
            )

        captions.append((name, color, x1, y1))

    for name, color, x, y in captions:
        bounds = draw.textbbox((0, 0), name, font=f)

        tw = bounds[2] - bounds[0]
        th = bounds[3] - bounds[1]

        tx = max(0, min(int(x), width - tw - 10))
        ty = max(0, min(int(y) - th - 8, height - th - 6))

        draw.rectangle(
            (tx, ty, tx + tw + 10, ty + th + 6),
            fill=color,
        )

        draw.text(
            (tx + 5 - bounds[0], ty + 3 - bounds[1]),
            name,
            fill="white",
            font=f,
        )

    return result


def make_panel(columns, title):
    w, h = columns[0][1].size

    if any(img.size != (w, h) for _, img in columns):
        raise ValueError(
            "Input PNG and model visualization sizes differ; "
            "refusing to misalign boxes"
        )

    pad = 28
    gap = 25

    text_size = max(24, min(48, int(w * 0.023)))
    header = 130

    output_width = (
        len(columns) * w
        + (len(columns) - 1) * gap
        + 2 * pad
    )

    output_height = h + header + pad

    out = Image.new(
        "RGB",
        (output_width, output_height),
        "#f3f5f7",
    )

    draw = ImageDraw.Draw(out)

    draw.text(
        (pad, 12),
        title,
        font=font(text_size),
        fill="#172338",
    )

    for i, (caption, img) in enumerate(columns):
        x = pad + i * (w + gap)

        draw.text(
            (x, 67),
            caption,
            font=font(max(18, text_size - 5)),
            fill="#172338",
        )

        out.paste(img, (x, header))

        draw.rectangle(
            (x, header, x + w - 1, header + h - 1),
            outline="#708090",
            width=2,
        )

    return out


def pages_to_process(spec):
    if spec.lower() == "all":
        return range(1, 75)

    pages = set()

    for part in spec.split(","):
        if "-" in part:
            first, last = map(int, part.split("-", 1))
            pages.update(range(first, last + 1))
        else:
            pages.add(int(part))

    if not pages or min(pages) < 1 or max(pages) > 74:
        raise ValueError("Pages must be between 1 and 74")

    return sorted(pages)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pages",
        default="all",
        help="6, 1-10, 1,3,6 or all",
    )

    parser.add_argument(
        "--format",
        choices=("jpg", "png"),
        default="jpg",
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
    )

    args = parser.parse_args()

    repo = args.root / "paddle_harvest"
    vis = args.root / "paddle_harvest_visuals"

    dataset = repo / "dataset"

    manifest_path = dataset / "mixed_layout_test_74p_manifest.csv"

    with manifest_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as fh:
        manifest = {
            int(r["mixed_page"]): r
            for r in csv.DictReader(fh)
        }

    folders = {
        "e01": vis / "E01_baseline_050",
        "e02": vis / "E02_9labels_050",
        "e03": vis / "E03_diagnostic025",
    }

    for page in pages_to_process(args.pages):
        stem = f"page_{page:04d}"

        original = read_image(
            vis / "original_pages" / f"{stem}.png"
        )

        gt = ground_truth(
            original,
            dataset / "labels" / f"{stem}.json",
        )

        images = {
            key: read_image(folder / f"{stem}_res.png")
            for key, folder in folders.items()
        }

        images["gt"] = gt

        source = manifest[page]

        title = (
            f"HARVEST | Page {page:04d} | "
            f"{source['source_id']} | "
            f"Source page {source['source_page']}"
        )

        for output_folder, columns in COMPARISONS.items():
            selected = [
                (caption, images[key])
                for caption, key in columns
            ]

            panel = make_panel(selected, title)

            dest = (
                vis
                / "comparisons"
                / output_folder
                / f"{stem}.{args.format}"
            )

            dest.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if args.format == "jpg":
                panel.save(
                    dest,
                    quality=95,
                    subsampling=0,
                )
            else:
                panel.save(dest)

        print(f"Finished {stem} (4 comparisons)")

    print(
        "Done. Originals and LabelMe JSONs were not modified."
    )


if __name__ == "__main__":
    main()