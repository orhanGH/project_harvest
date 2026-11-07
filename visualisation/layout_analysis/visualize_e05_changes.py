#!/usr/bin/env python3
"""HARVEST E05 visual review: GT tables | E02 tables | E05 tables.

Runs on Marvin using saved JSONs/images: no PaddleOCR inference required.
Only TABLE regions are shown, as E05 modified only the table threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_WORK = Path(
    '/lustre/scratch/data/s6oraydi_hpc-project_harvest/s6oraydi_work'
)
BLUE = (36, 97, 240)
GREEN = (12, 147, 89)
ORANGE = (240, 125, 20)
DARK = (29, 39, 52)
WHITE = (255, 255, 255)
LIGHT = (242, 245, 249)


def font(size: int) -> ImageFont.ImageFont:
    for filename in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                     '/usr/share/fonts/dejavu/DejaVuSans.ttf',
                     'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(filename, size)
        except OSError:
            continue
    return ImageFont.load_default()


def boxed_label(draw: ImageDraw.ImageDraw, position, text, color, body_font, image_size):
    x, y = position
    left, top, right, bottom = draw.textbbox((0, 0), text, font=body_font)
    tw, th = right - left, bottom - top
    x = min(max(0, int(x)), max(0, image_size[0] - tw - 12))
    y = int(y)
    if y - th - 12 >= 0:
        y = y - th - 12
    else:
        y = min(image_size[1] - th - 10, y + 3)
    draw.rectangle((x, y, x + tw + 10, y + th + 9), fill=WHITE, outline=color, width=2)
    draw.text((x + 5, y + 3), text, font=body_font, fill=color)


def draw_gt_tables(base: Image.Image, shapes: list, line_width: int, body_font):
    result = base.copy()
    draw = ImageDraw.Draw(result)
    count = 0
    for shape in shapes:
        if str(shape.get('label', '')).lower() != 'table':
            continue
        points = [tuple(map(float, pt)) for pt in shape.get('points', [])]
        if len(points) < 2:
            continue
        x1 = min(p[0] for p in points)
        y1 = min(p[1] for p in points)
        x2 = max(p[0] for p in points)
        y2 = max(p[1] for p in points)
        if shape.get('shape_type') == 'rectangle' or len(points) == 2:
            draw.rectangle((x1, y1, x2, y2), outline=BLUE, width=line_width)
        else:
            draw.line(points + [points[0]], fill=BLUE, width=line_width, joint='curve')
        boxed_label(draw, (x1, y1), 'GT table', BLUE, body_font, result.size)
        count += 1
    return result, count


def prediction_boxes(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    prediction = payload.get('res', payload)
    if not isinstance(prediction, dict) or not isinstance(prediction.get('boxes'), list):
        raise ValueError(f'Unexpected prediction JSON structure: {path}')
    return [item for item in prediction['boxes']
            if str(item.get('label', '')).lower() == 'table']


def draw_tables(base: Image.Image, boxes: list[dict], threshold: float,
                line_width: int, body_font):
    result = base.copy()
    draw = ImageDraw.Draw(result)
    high, low = 0, 0
    # Draw high-confidence boxes first, then low-confidence boxes on top.
    for item in sorted(boxes, key=lambda b: float(b['score']), reverse=True):
        score = float(item['score'])
        if score < threshold - 1e-6:
            raise ValueError(f'Table score {score:.6f} is below expected threshold {threshold}')
        xyxy = [float(z) for z in item['coordinate']]
        if len(xyxy) != 4:
            raise ValueError(f'Invalid table box: {xyxy}')
        is_new = score < 0.5
        color = ORANGE if is_new else GREEN
        draw.rectangle(xyxy, outline=color, width=line_width)
        boxed_label(draw, (xyxy[0], xyxy[1]), f'table {score:.3f}', color,
                    body_font, result.size)
        low += int(is_new)
        high += int(not is_new)
    return result, high, low


def save_triptych(gt, e02, e05, output, page, body_font, heading_font):
    w, h = gt.size
    pad, gap, bar = 28, 18, 105
    dest = Image.new('RGB', (3 * w + 2 * gap + 2 * pad, h + bar + 2 * pad), LIGHT)
    x_positions = [pad, pad + w + gap, pad + 2 * (w + gap)]
    draw = ImageDraw.Draw(dest)
    for x, title, img in zip(x_positions,
        ('GROUND TRUTH  |  tables only',
         'E02  |  table threshold 0.50',
         'E05  |  table threshold 0.38'), (gt, e02, e05)):
        dest.paste(img, (x, bar + pad))
        draw.text((x + 4, 24), title, fill=DARK, font=heading_font)
    draw.text((pad + 3, bar - 38),
              f'{page}: blue=GT; green=score >= 0.50; orange=NEW 0.38-0.50',
              fill=DARK, font=body_font)
    dest.save(output, quality=92, subsampling=0, optimize=True)


def save_delta(base, gt_shapes, e05_boxes, output, body_font, heading_font, line_width):
    panel, _ = draw_gt_tables(base, gt_shapes, line_width, body_font)
    canvas_draw = ImageDraw.Draw(panel)
    count = 0
    for item in sorted(e05_boxes, key=lambda b: float(b['score']), reverse=True):
        score = float(item['score'])
        if not 0.38 - 1e-6 <= score < 0.5:
            continue
        box = [float(v) for v in item['coordinate']]
        canvas_draw.rectangle(box, outline=ORANGE, width=line_width)
        boxed_label(canvas_draw, (box[0], box[1]), f'NEW {score:.3f}',
                    ORANGE, body_font, panel.size)
        count += 1
    w, h = panel.size
    header = 90
    dest = Image.new('RGB', (w, h + header), LIGHT)
    dest.paste(panel, (0, header))
    d = ImageDraw.Draw(dest)
    d.text((20, 15), f'GT tables (blue) + NEW E05 candidates (orange): {count}',
           font=heading_font, fill=DARK)
    dest.save(output, quality=93, subsampling=0, optimize=True)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=DEFAULT_WORK)
    parser.add_argument('--pages', default='2,6,27,35,73',
                        help='Comma-separated PDF page numbers (default: five changed pages)')
    parser.add_argument('--out', type=Path, help='Override default visual_review output directory')
    args = parser.parse_args()

    work = args.work.expanduser()
    runs = work / 'paddle_ocr/runs/mixed_layout_test_74p'
    gt_dir = work / 'input/mixed_layout_test_74p/labels'
    output = args.out or (
        work / 'paddle_ocr/results/mixed_layout_test_74p/'
        'e02_e03_e05_comparison/E05_diagnostics/visual_review'
    )
    pages = sorted(set(int(x.strip()) for x in args.pages.split(',') if x.strip()))
    if not pages or any(p < 1 or p > 74 for p in pages):
        parser.error('Expected PDF page numbers between 1 and 74')
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for number in pages:
        stem = f'page_{number:04d}'
        image_file = runs / 'baseline_050/rendered' / f'{stem}.png'
        gt_file = gt_dir / f'{stem}.json'
        e02_file = runs / '9labels_050/raw' / f'{stem}.json'
        e05_file = runs / 'e05_table038_full/raw' / f'{stem}.json'
        for file in (image_file, gt_file, e02_file, e05_file):
            if not file.is_file():
                raise FileNotFoundError(f'Missing required file for {stem}: {file}')
        with Image.open(image_file) as raw_img:
            base = raw_img.convert('RGB')
        data = json.loads(gt_file.read_text(encoding='utf-8'))
        if ('imageWidth' in data and 'imageHeight' in data and
            (int(data['imageWidth']), int(data['imageHeight'])) != base.size):
            raise ValueError(f'{stem}: LabelMe dimensions mismatch page PNG')
        fsize = max(16, int(base.width / 115))
        body_font = font(fsize)
        heading_font = font(int(fsize * 1.3))
        line_width = max(4, round(base.width / 450))
        gt, n_gt = draw_gt_tables(base, data.get('shapes', []), line_width, body_font)
        e02_boxes = prediction_boxes(e02_file)
        e05_boxes = prediction_boxes(e05_file)
        old, _, _ = draw_tables(base, e02_boxes, 0.5, line_width, body_font)
        new, _, _ = draw_tables(base, e05_boxes, 0.38, line_width, body_font)
        triple = output / f'{stem}_GT_E02_E05.jpg'
        new_only = output / f'{stem}_NEW_ONLY.jpg'
        save_triptych(gt, old, new, triple, stem, body_font, heading_font)
        n_low = save_delta(base, data.get('shapes', []), e05_boxes, new_only,
                           body_font, heading_font, line_width)
        summary.append(dict(page=stem, gt_tables=n_gt,
                            e02_tables=len(e02_boxes), e05_tables=len(e05_boxes),
                            new_low_confidence_boxes=n_low,
                            comparison_image=triple.name, new_only_image=new_only.name))
        print(f'{stem}: GT={n_gt}, E02={len(e02_boxes)}, '
              f'E05={len(e05_boxes)}, NEW={n_low}', flush=True)
    with (output / 'visual_review_manifest.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f'DONE: {len(pages)} pages, {2 * len(pages)} images, '
          f'{sum(row["new_low_confidence_boxes"] for row in summary)} new boxes')
    print('Output:', output)


if __name__ == '__main__':
    main()
