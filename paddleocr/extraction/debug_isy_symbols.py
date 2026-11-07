#!/usr/bin/env python3
"""Create visual diagnostics for ISY symbols missing from OCR.

Reads already-saved PaddleOCR table JSON plus the corresponding table-crop
image. Does not launch PaddleOCR or modify any extraction output.
"""

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw

CASES = {
    "0046": [
        ("africa_area_1925_star", 420, 180, 140, 46),
        ("africa_production_1925_star", 695, 180, 140, 46),
        ("south_africa_area_1925_missing", 420, 222.5, 140, 45),
        ("south_africa_production_1925_missing", 695, 222.5, 140, 45),
        ("south_africa_yield_1925_missing", 948, 222.5, 100, 45),
        ("south_america_production_star", 695, 367, 140, 44),
        ("brazil_production_1925_missing", 695, 410, 140, 44),
        ("brazil_yield_1925_missing", 948, 410, 100, 44),
    ],
    "0050": [
        ("asia_entity_star", 123, "ASIE", 190, 55),
        ("europe_entity_star", 123, "EUROPE", 190, 55),
        ("austria_area_1925_dash", 423, 330.5, 130, 36),
        ("czechoslovakia_area_1925_dash", 423, 442, 130, 36),
    ],
}


def payload_from_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("res", data)


def get_image_path(payload, pipe, page):
    candidates = []
    for key in ("input_path", "image_path", "img_path"):
        raw = payload.get(key)
        if isinstance(raw, str):
            candidate = Path(raw).expanduser()
            if candidate.is_file():
                candidates.append(candidate)

    if not candidates:
        crop_dir = pipe / "ocr_smoke_5pages" / "crops" / f"page_{page}"
        if crop_dir.is_dir():
            for extension in ("*.png", "*.jpg", "*.jpeg"):
                for candidate in crop_dir.rglob(extension):
                    if "region_003" in candidate.name.lower() and "table" in candidate.name.lower():
                        candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            f"Couldn't find the table crop for page {page}. "
            f"Inspect JSON input_path and {pipe / 'ocr_smoke_5pages' / 'crops' / f'page_{page}'}"
        )
    return candidates[0]


def label_y(payload, label):
    ocr = payload.get("overall_ocr_res", {})
    texts = ocr.get("rec_texts", [])
    boxes = ocr.get("rec_boxes", [])
    matches = []
    for text, box in zip(texts, boxes):
        cleaned = re.sub(r"[^a-z]+", "", str(text).casefold())
        if cleaned.startswith(label.casefold()):
            x1, y1, x2, y2 = [float(v) for v in box]
            if x1 < 210 and y1 > 150:
                matches.append((y1 + y2) / 2)
    if not matches:
        raise ValueError(f"Can't find header label {label!r} in raw OCR boxes")
    return min(matches)


def crop_about(img, cx, cy, width, height):
    left = max(0, round(cx - width / 2))
    top = max(0, round(cy - height / 2))
    right = min(img.width, round(cx + width / 2))
    bottom = min(img.height, round(cy + height / 2))
    if right <= left or bottom <= top:
        raise ValueError(f"Empty crop centered at {cx}, {cy}; image={img.size}")
    return img.crop((left, top, right, bottom))


def contact_sheet(patches, output):
    panel_w = 600
    panel_h = 216
    sheet = Image.new("RGB", (panel_w, panel_h * len(patches)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, patch) in enumerate(patches):
        y0 = index * panel_h
        draw.text((14, y0 + 8), name, fill="black")
        zoom = 3
        bigger = patch.resize(
            (patch.width * zoom, patch.height * zoom),
            Image.Resampling.NEAREST,
        )
        if bigger.width > panel_w - 30 or bigger.height > panel_h - 40:
            bigger.thumbnail((panel_w - 30, panel_h - 40))
        sheet.paste(bigger, (14, y0 + 30))
        draw.line((0, y0 + panel_h - 1, panel_w, y0 + panel_h - 1), fill="gray")
    sheet.save(output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipe", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for page, cases in CASES.items():
        source_json = (
            args.pipe / "table_recognition_smoke" / f"page_{page}"
            / "region_003_table" / "result_01" / "region_003_table_res.json"
        )
        payload = payload_from_json(source_json)
        source_image = get_image_path(payload, args.pipe, page)
        im = Image.open(source_image).convert("RGB")
        page_output = args.output_dir / f"page_{page}"
        page_output.mkdir(parents=True, exist_ok=True)
        print(f"PAGE {page}: image={source_image}, size={im.size}")

        patches = []
        for name, cx, cy, width, height in cases:
            if isinstance(cy, str):
                cy = label_y(payload, cy)
            patch = crop_about(im, cx, cy, width, height)
            patch.save(page_output / f"{name}.png")
            patches.append((name, patch))
            print(f"  {name}: x={cx}, y={cy:.1f}")
        contact_sheet(patches, page_output / "CONTACT_SHEET.png")

    print(f"Saved symbol inspection crops under: {args.output_dir}")


if __name__ == "__main__":
    main()
