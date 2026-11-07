#!/usr/bin/env python3
"""Create focused HARVEST E04 comparisons using one *winner* per missed GT region.

Outputs (three panels each):
    E02_vs_E04_lowconf_winner: GT | E02 | strongest suppressed E04 candidate
    E03_vs_E04_lowconf_winner: GT | E03 | strongest suppressed E04 candidate

A winner is the highest-confidence E04 candidate that:
  * belongs to the same nine selected layout classes as E02/E03,
  * overlaps a GT region not matched by the baseline (IoU >= --min-iou),
  * has confidence BELOW the threshold of its predicted class in that baseline.

Among good matches (IoU >= 0.50), choose the highest score. If there are no
such matches, optionally display the HIGHEST-IOU partial candidate as a
fallback, labelled 'partial' in the CSV. This prevents a barely overlapping
high-score region from hiding a much better lower-scoring detection.

This is a qualitative visualization; it is NOT a new independent detection result
or a raw per-proposal class-probability ranking. E04 candidates were already NMS-
filtered. GT polygons use their bounding rectangles for IoU, but the GT overlay
itself preserves the polygon shape.

Input layout (default --root is ~/Desktop/Arbeit/HARVEST):
  paddle_harvest/dataset/labels/page_XXXX.json
  paddle_harvest/dataset/mixed_layout_test_74p_manifest.csv
  No dependency on make_layout_comparisons.py (standalone script).
  paddle_harvest_visuals/original_pages/page_XXXX.png
  paddle_harvest_visuals/E02_9labels_050/page_XXXX_res.png
  paddle_harvest_visuals/E03_diagnostic025/page_XXXX_res.png
  paddle_harvest_visuals/E04_lowconf_all/{gt_candidate_summary.csv,all_candidates.csv}

Usage:
  python visualisation/layout_analysis/make_e04_winner_comparisons.py --pages 2,6
  python visualisation/layout_analysis/make_e04_winner_comparisons.py --pages all
  python visualisation/layout_analysis/make_e04_winner_comparisons.py --pages all --min-iou 0.50
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# This script is intentionally SELF-CONTAINED: it does not import the older
# make_layout_comparisons.py, whose function names vary between revisions.
LABEL_COLORS = {
    "table": "#D32323", "footnote": "#128A36", "number": "#858100",
    "header": "#2255D5", "figure_title": "#A000B0", "reference": "#06949C",
    "paragraph_title": "#777777", "text": "#C95C00", "content": "#008B95",
    "doc_title": "#984319", "seal": "#313131", "aside_text": "#7A479D",
    "footer": "#476E91",
}

def font_at(size):
    paths = [
        Path('C:/Windows/Fonts/arial.ttf'),
        Path('C:/Windows/Fonts/segoeui.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    ]
    for path in paths:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()

def normalize_label(raw):
    return str(raw).strip().lower().replace(' ', '_').replace('-', '_')

def parse_pages(raw):
    if raw.lower() == 'all':
        return list(range(1, 75))
    pages = set()
    for group in raw.split(','):
        part = group.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            start, end = int(a), int(b)
            if start > end:
                raise ValueError(f'Invalid page range: {part}')
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    if not pages or any(not 1 <= p <= 74 for p in pages):
        raise ValueError('Pages must be between 1 and 74')
    return sorted(pages)

def read_manifest(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing dataset manifest: {path}')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        result[int(row['mixed_page'])] = row
    return result

def load_image(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing image: {path}')
    with Image.open(path) as im:
        return im.convert('RGB')

def gt_overlay(original, labels_path):
    if not labels_path.is_file():
        raise FileNotFoundError(f'Missing ground-truth annotation: {labels_path}')
    with labels_path.open(encoding='utf-8') as handle:
        data = json.load(handle)

    expected = (data.get('imageWidth'), data.get('imageHeight'))
    if expected != original.size:
        raise ValueError(
            f'GT image size mismatch: {labels_path.name} says {expected}, '
            f'but original PNG is {original.size}. '
            'Check that PNG and JSON belong to the same rendering; no automatic scaling performed.'
        )

    img = original.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    border_px = max(3, round(min(width, height) / 450))
    font = font_at(max(18, min(36, round(width * 0.013))))
    label_positions = []

    for shape in data.get('shapes', []):
        points = shape.get('points') or []
        if len(points) < 2:
            continue
        try:
            pts = [(float(x), float(y)) for x, y in points]
        except (TypeError, ValueError):
            continue
        name = normalize_label(shape.get('label', 'unknown'))
        color = LABEL_COLORS.get(name, '#444444')
        shape_type = shape.get('shape_type', 'rectangle')
        if shape_type == 'rectangle':
            # LabelMe rectangles use two opposite corners; this also handles
            # reversed corner order correctly.
            x1, x2 = sorted((pts[0][0], pts[1][0]))
            y1, y2 = sorted((pts[0][1], pts[1][1]))
            draw.rectangle((x1, y1, x2, y2), outline=color, width=border_px)
            label_positions.append((x1, y1, name, color))
        else:
            # Retain non-rectangle shapes rather than silently converting to a
            # bounding rectangle (LabelMe supports polygons).
            draw.line(pts + [pts[0]], fill=color, width=border_px, joint='curve')
            x1 = min(x for x, _ in pts)
            y1 = min(y for _, y in pts)
            label_positions.append((x1, y1, name, color))

    for x, y, name, color in label_positions:
        box = draw.textbbox((0, 0), name, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        pad_x, pad_y = 5, 3
        tx = max(0, min(int(x), width - tw - 2 * pad_x))
        # Prefer outside/above box; use inside only when near the page top.
        ty = int(y) - th - 2 * pad_y - 2
        if ty < 0:
            ty = max(0, int(y) + 3)
        ty = max(0, min(ty, height - th - 2 * pad_y))
        draw.rectangle((tx, ty, tx + tw + 2 * pad_x, ty + th + 2 * pad_y), fill=color)
        draw.text((tx + pad_x - box[0], ty + pad_y - box[1]), name, fill='white', font=font)
    return img

def build_panel(columns, main_title):
    """Build a high-resolution, two- or three-column comparison.

    Page pixel dimensions are preserved exactly; no resizing or cropping.
    The images must share a coordinate system to be compared meaningfully.
    """
    if len(columns) not in (2, 3):
        raise ValueError(f'Expected two or three columns, got {len(columns)}')

    page_size = columns[0][1].size
    for caption, img in columns[1:]:
        if img.size != page_size:
            raise ValueError(
                f'Visualization size {img.size} ({caption}) differs from page size {page_size}. '
                'This script intentionally does not resize/misalign images. '
                'Check model rendering DPI and source image DPI.'
            )

    w, h = page_size
    n = len(columns)
    gap = max(18, round(w * 0.018))
    pad = max(20, round(w * 0.018))
    font_size = max(22, min(48, round(w * 0.024)))
    title_font = font_at(font_size)
    subtitle_font = font_at(max(18, round(font_size * 0.83)))
    title_h = font_size * 2 + 42
    header_h = font_size * 2 + 35
    width = w * n + gap * (n - 1) + 2 * pad
    panel = Image.new('RGB', (width, h + title_h + header_h + pad), '#F3F5F7')
    draw = ImageDraw.Draw(panel)
    draw.text((pad, 10), main_title, font=title_font, fill='#152238')
    top = title_h + header_h

    for idx, (caption, img) in enumerate(columns):
        x = pad + idx * (w + gap)
        draw.text((x, title_h), caption, font=subtitle_font, fill='#152238')
        panel.paste(img, (x, top))
        draw.rectangle((x - 1, top - 1, x + w, top + h), outline='#708090', width=2)

    return panel

def save_panel(im, out, fmt, quality):
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == 'png':
        im.save(out, format='PNG', compress_level=6)
    else:
        # 4:4:4 chroma, high quality to keep thin annotation edges readable.
        im.save(out, format='JPEG', quality=quality, subsampling=0, optimize=True)

ROOT = Path.home() / 'Desktop' / 'Arbeit' / 'HARVEST'
CLASSES = {
    'paragraph_title', 'text', 'number', 'content', 'figure_title',
    'table', 'doc_title', 'footnote', 'header',
}
LOW_E03 = {'content', 'figure_title', 'footnote', 'header'}
TEAL = '#008B83'       # E04 winner label agrees with GT
ORANGE = '#E16A00'     # E04 winner label disagrees with GT
MISS = '#9B9DA1'       # reference outline of a GT region missed by baseline


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing E04 file: {path}')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def parse_box(s):
    box = [float(x) for x in json.loads(s)]
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f'Invalid bounding box: {s}')
    return box


def box_iou(a, b):
    x = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = x * y
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.


def threshold(label: str, baseline: str):
    if label not in CLASSES:
        return 1.00  # E02/E03 intentionally exclude other pretrained classes.
    return 0.25 if baseline == 'e03' and label in LOW_E03 else 0.50


def find_e04(base: Path) -> Path:
    options = (base / 'E04_lowconf_all', base / 'e04_lowconf_all',
               base.parent / 'paddle_harvest' / 'e04_lowconf_all')
    for p in options:
        if (p / 'all_candidates.csv').is_file() and (p / 'gt_candidate_summary.csv').is_file():
            return p
    raise FileNotFoundError(
        'Cannot find E04 CSVs. Tried:\n' + '\n'.join(str(x) for x in options) +
        '\nUse --e04 "C:/.../E04_lowconf_all" if your folder is elsewhere.'
    )


def missing_outline(draw, box, stroke=2, dash=22, gap=12):
    """Light dashed outline, only for GT objects the baseline missed."""
    x1, y1, x2, y2 = map(int, box)
    for x in range(x1, x2, dash + gap):
        draw.line((x, y1, min(x + dash, x2), y1), fill=MISS, width=stroke)
        draw.line((x, y2, min(x + dash, x2), y2), fill=MISS, width=stroke)
    for y in range(y1, y2, dash + gap):
        draw.line((x1, y, x1, min(y + dash, y2)), fill=MISS, width=stroke)
        draw.line((x2, y, x2, min(y + dash, y2)), fill=MISS, width=stroke)


def draw_winners(original, missed, winners):
    """Show original image and just the winning candidate boxes, once each."""
    out = original.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    stroke = max(3, round(min(w, h) / 530))
    font = font_at(max(16, min(30, round(w * 0.012))))
    for gt in missed:
        missing_outline(d, gt['box'], stroke=2)
    unique = {}
    for item in winners:
        cand = item['candidate']
        # One candidate may overlap two GT shapes; display its box only once.
        key = cand['candidate_id']
        if key not in unique or item['iou'] > unique[key]['iou']:
            unique[key] = item
    labels = []
    for item in unique.values():
        cand = item['candidate']
        color = TEAL if cand['class'] == item['gt_class'] else ORANGE
        box = cand['bbox']
        d.rectangle(box, outline=color, width=stroke + 1)
        labels.append((box, cand, color, item['iou']))
    for box, cand, color, iou in labels:
        text = f"{cand['class']} {cand['score']:.3f} | IoU {iou:.2f}"
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        px, py = 5, 3
        x = min(max(0, round(box[0])), w - tw - 2 * px)
        y = max(0, round(box[1]) - th - 2 * py - 3)
        d.rectangle((x, y, x + tw + 2*px, y + th + 2*py), fill=color)
        d.text((x + px - bbox[0], y + py - bbox[1]), text, font=font, fill='white')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--e04', type=Path, default=None, help='Override E04_lowconf_all folder')
    ap.add_argument('--pages', default='2,6', help='e.g. 2,6; 1-10; all')
    ap.add_argument('--include-excluded', action='store_true',
                    help='Also consider excluded pretrained classes (threshold 1.00 in E02/E03)')
    ap.add_argument('--min-iou', type=float, default=0.10,
                    help='Minimum IoU for partial fallback candidates (default 0.10; at >=0.50 choose highest score)')
    ap.add_argument('--format', choices=('jpg','png'), default='jpg')
    ap.add_argument('--quality', type=int, default=95)
    args = ap.parse_args()
    if not 0.0 < args.min_iou <= 1.0:
        ap.error('--min-iou must be in (0,1]')
    if not 1 <= args.quality <= 100:
        ap.error('--quality must be in 1..100')
    pages = parse_pages(args.pages)
    root = args.root.expanduser()
    repo = root / 'paddle_harvest'
    vis = root / 'paddle_harvest_visuals'
    e04 = args.e04.expanduser() if args.e04 else find_e04(vis)
    summaries = read_rows(e04 / 'gt_candidate_summary.csv')
    candidates = read_rows(e04 / 'all_candidates.csv')
    manifest = read_manifest(repo / 'dataset' / 'mixed_layout_test_74p_manifest.csv')
    summaries_by_page = defaultdict(list)
    for r in summaries:
        summaries_by_page[r['page']].append({
            'gt_id': int(r['gt_id']), 'class': r['gt_class'],
            'box': parse_box(r['gt_bbox']), 'e02': r['e02_matched_iou050'],
            'e03': r['e03_matched_iou050'],
        })
    candidates_by_page = defaultdict(list)
    for r in candidates:
        candidates_by_page[r['page']].append({
            'candidate_id': int(r['candidate_id']), 'class': r['class'],
            'score': float(r['score']), 'bbox': parse_box(r['bbox']),
        })
    output_root = vis / 'comparisons'
    result_rows = []
    results = (
        ('e02', 'E02_vs_E04_lowconf_winner',
         'E02 | Nine labels, threshold 0.50',
         'E04 | Strongest below-E02 candidate per missed GT'),
        ('e03', 'E03_vs_E04_lowconf_winner',
         'E03 | Class-adjusted thresholds',
         'E04 | Strongest below-E03 candidate per missed GT'),
    )
    for p in pages:
        stem = f'page_{p:04d}'
        original = load_image(vis / 'original_pages' / f'{stem}.png')
        labels_path = repo / 'dataset' / 'labels' / f'{stem}.json'
        # gt_overlay validates exact shape-to-PNG pixel dimensions and retains polygons.
        gt_img = gt_overlay(original, labels_path)
        gt_data = json.loads(labels_path.read_text(encoding='utf-8'))
        gts = summaries_by_page.get(stem)
        if gts is None:
            raise ValueError(f'No E04 GT rows for {stem}')
        for g in gts:
            shape = gt_data['shapes'][g['gt_id']]
            label = str(shape['label']).strip().lower().replace(' ','_').replace('-','_')
            if label != g['class']:
                raise ValueError(f'GT annotations changed since E04 on {stem}, id={g["gt_id"]}')
        all_candidates = candidates_by_page.get(stem)
        if not all_candidates:
            raise ValueError(f'No candidates found in E04 CSV for {stem}')
        meta = manifest.get(p, {})
        title = (f'HARVEST | Page {p:04d} | {meta.get("source_id", "unknown")} | '
                 f'Source page {meta.get("source_page", "?")}')
        for baseline, folder, baseline_caption, winner_caption in results:
            old_img = load_image(vis / ('E02_9labels_050' if baseline == 'e02' else 'E03_diagnostic025') / f'{stem}_res.png')
            if old_img.size != original.size:
                raise ValueError(f'{stem} {baseline} visualization size differs from original')
            missed = [g for g in gts if g['class'] in CLASSES and g[baseline] == 'no']
            winners = []
            for g in missed:
                eligible = []
                for c in all_candidates:
                    if c['class'] not in CLASSES and not args.include_excluded:
                        continue
                    if c['score'] >= threshold(c['class'], baseline):
                        continue  # Only candidates suppressed by this baseline's score cutoff.
                    iou = box_iou(g['box'], c['bbox'])
                    if iou >= args.min_iou:
                        eligible.append((c['score'], iou, c))
                if eligible:
                    geometrically_good = [t for t in eligible if t[1] >= 0.50]
                    if geometrically_good:
                        # First require a meaningful match, then highest confidence wins.
                        score, iou, best = max(geometrically_good, key=lambda e: (e[0], e[1]))
                        selection_basis = 'highest_score_iou_ge_0.50'
                    else:
                        # Partial fallback: highest IoU wins (not necessarily score).
                        score, iou, best = max(eligible, key=lambda e: (e[1], e[0]))
                        selection_basis = 'highest_iou_partial_fallback'
                    winners.append({'gt_id':g['gt_id'],'gt_class':g['class'], 'iou':iou, 'candidate':best})
                else:
                    best, iou = None, None
                    selection_basis = 'none'
                result_rows.append({
                    'page':stem,'baseline':baseline.upper(),'gt_id':g['gt_id'],'gt_class':g['class'],
                    'winner_candidate_id':best['candidate_id'] if best else '',
                    'winner_class':best['class'] if best else '',
                    'winner_score':round(best['score'],6) if best else '',
                    'winner_iou':round(iou,6) if iou is not None else '',
                    'winner_correct_class':('yes' if best['class']==g['class'] else 'no') if best else '',
                    'suppression_reason':('excluded_class' if best['class'] not in CLASSES else 'below_class_threshold') if best else '',
                    'baseline_threshold_of_winner_class':threshold(best['class'],baseline) if best else '',
                    'candidate_status':selection_basis if best else 'no_suppressed_candidate_above_min_iou',
                    'min_iou':args.min_iou,
                    'notes':'Candidate after E04 NMS; GT polygon IoU uses bounding rectangle',
                })
            e04_overlay = draw_winners(original, missed, winners)
            panel = build_panel([('Ground truth', gt_img), (baseline_caption, old_img),
                                 (winner_caption, e04_overlay)], title)
            dest = output_root / folder / f'{stem}.{args.format}'
            save_panel(panel, dest, args.format, args.quality)
            print(f'{stem} {baseline.upper()}: {len(missed)} misses, {len(winners)} winners -> {dest.name}', flush=True)
    review_path = output_root / 'E04_lowconf_winner_review.csv'
    cols = [
        'page','baseline','gt_id','gt_class','winner_candidate_id','winner_class',
        'winner_score','winner_iou','winner_correct_class','suppression_reason',
        'baseline_threshold_of_winner_class','candidate_status','min_iou','notes',
    ]
    with review_path.open('w', newline='', encoding='utf-8') as h:
        writer = csv.DictWriter(h, fieldnames=cols)
        writer.writeheader()
        writer.writerows(result_rows)
    print(f'Created {len(pages) * len(results)} comparison images; winner register: {review_path}')
    print('Legend: teal = winner class agrees with GT; orange = different class; '
          'gray dashed rectangle = baseline-missed GT. NO box = no eligible winner.')
    print('Source PNGs, annotations, E02/E03 overlays, and E04 candidate CSVs were not modified.')


if __name__ == '__main__':
    main()
