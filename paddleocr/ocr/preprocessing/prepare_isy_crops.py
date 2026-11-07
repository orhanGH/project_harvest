#!/usr/bin/env python3
"""Crop detected ISY tables from the exact images used in layout detection."""
import argparse
import csv
import json
import math
from pathlib import Path
from PIL import Image


def parse_pages(spec):
    values = set()
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            a, b = (int(t) for t in chunk.split('-', 1))
            if a > b:
                raise ValueError('Invalid page range: '+chunk)
            values.update(range(a, b+1))
        else:
            values.add(int(chunk))
    return sorted(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layout-dir', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--pages', default='44,45,47,48,49')
    p.add_argument('--min-confidence', type=float, default=0.5)
    p.add_argument('--padding', type=int, default=0)
    args = p.parse_args()
    selected = set(parse_pages(args.pages))
    src = args.layout_dir / 'tables.jsonl'
    if not src.exists():
        raise SystemExit(f'Missing detection index: {src}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    with src.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            page = int(rec['page_number'])
            if page not in selected:
                continue
            seen.add(page)
            image_path = args.layout_dir/'rendered'/f'page_{page:04d}.png'
            if not image_path.is_file():
                image_path = Path(rec['rendered_image'])
            if not image_path.is_file():
                print(f'SKIP page {page}: rendered page missing')
                continue
            with Image.open(image_path) as im:
                for index, entry in enumerate(rec.get('tables', []), 1):
                    conf = float(entry.get('confidence', 0))
                    if conf < args.min_confidence:
                        continue
                    x1,y1,x2,y2 = map(float, entry['bbox'])
                    box = (
                        max(0, math.floor(x1)-args.padding),
                        max(0, math.floor(y1)-args.padding),
                        min(im.width, math.ceil(x2)+args.padding),
                        min(im.height, math.ceil(y2)+args.padding),
                    )
                    if box[2] <= box[0] or box[3] <= box[1]:
                        print(f'SKIP invalid table bbox on page {page}')
                        continue
                    target_dir = args.output_dir/f'page_{page:04d}'
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target = target_dir/f'region_{index:03d}_table.png'
                    im.crop(box).save(target)
                    records.append({'page':page,'table_index':index,'confidence':conf,'bbox':repr(box),'crop':str(target)})
                    print(f'page {page} | table {index} | score={conf:.3f} | crop={target}')
    missing = sorted(selected-seen)
    if missing:
        print('Pages not in layout index:', missing)
    manifest = args.output_dir/'crop_manifest.csv'
    with manifest.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['page','table_index','confidence','bbox','crop']);writer.writeheader();writer.writerows(records)
    print(f'PREPARED {len(records)} crops on {len(set(r["page"] for r in records))} pages')
    print('Manifest:',manifest)
    if not records:
        raise SystemExit('No crops created. Check page selection and layout files.')

if __name__=='__main__':
    main()
