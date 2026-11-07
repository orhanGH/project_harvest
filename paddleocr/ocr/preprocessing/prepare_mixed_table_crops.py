#!/usr/bin/env python3
"""Create five hand-picked table crops from existing HARVEST E02/E05 predictions.

This is a qualitative OCR smoke test, NOT an unbiased evaluation.  No ground-truth
information is read by this script or used during OCR inference.
"""
import argparse
import csv
import json
from math import ceil, floor
from pathlib import Path

from PIL import Image


def read_boxes(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    payload = data.get('res', data)
    boxes = payload.get('boxes', [])
    return [b for b in boxes if str(b.get('label', '')).lower() == 'table']


def nearest(boxes, score):
    candidate = min(boxes, key=lambda b: abs(float(b['score']) - score))
    actual = float(candidate['score'])
    if abs(actual - score) > 0.002:
        raise ValueError(f'No table near score {score}; nearest={actual:.4f}')
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    runs = args.work / 'paddle_ocr/runs/mixed_layout_test_74p'
    source_images = runs / 'baseline_050/rendered'

    selections = [
        ('page_0006', '9labels_050', 'top2', None),
        ('page_0002', 'e05_table038_full', 'score', 0.4752),
        ('page_0002', 'e05_table038_full', 'score', 0.3812),
        ('page_0027', 'e05_table038_full', 'score', 0.4112),
    ]

    chosen = []
    for page, run, method, target in selections:
        path = runs / run / 'raw' / (page + '.json')
        boxes = read_boxes(path)
        if not boxes:
            raise ValueError(f'No table detections in {path}')
        if method == 'top2':
            selected = sorted(boxes, key=lambda b: float(b['score']), reverse=True)[:2]
            if len(selected) != 2:
                raise ValueError(f'Expected 2 tables on {page}, found {len(selected)}')
        else:
            selected = [nearest(boxes, target)]
        chosen.extend((page, run, b) for b in selected)

    if len(chosen) != 5:
        raise RuntimeError(f'Expected 5 crops, selected {len(chosen)}')

    crops_dir = args.out / 'crops'
    if crops_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing crops: {crops_dir}')
    crops_dir.mkdir(parents=True)

    counters = {}
    rows = []
    for page, run, box in chosen:
        image_path = source_images / (page + '.png')
        with Image.open(image_path) as img:
            image = img.convert('RGB')
        W, H = image.size
        x1, y1, x2, y2 = map(float, box['coordinate'])
        # Small 8-pixel margin retains nearby characters at crop boundaries.
        rect = [
            max(0, floor(min(x1, x2)) - 8),
            max(0, floor(min(y1, y2)) - 8),
            min(W, ceil(max(x1, x2)) + 8),
            min(H, ceil(max(y1, y2)) + 8),
        ]
        if rect[2] - rect[0] < 16 or rect[3] - rect[1] < 16:
            raise ValueError(f'Invalid crop rectangle for {page}: {rect}')
        counters[page] = counters.get(page, 0) + 1
        crop_name = f'region_{counters[page]:03d}_table.png'
        destination = crops_dir / page / crop_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.crop(tuple(rect)).save(destination)
        rows.append({
            'page': page,
            'detection_run': run,
            'crop': str(destination),
            'score': f"{float(box['score']):.6f}",
            'predicted_bbox': json.dumps(box['coordinate']),
            'crop_bbox_with_padding': json.dumps(rect),
            'page_image': str(image_path),
        })
        print(f"{page}: {run}, score={float(box['score']):.4f}, crop={crop_name}, size={rect[2]-rect[0]}x{rect[3]-rect[1]}")

    manifest = args.out / 'crop_manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'CREATED {len(rows)} crops; manifest: {manifest}')


if __name__ == '__main__':
    main()
