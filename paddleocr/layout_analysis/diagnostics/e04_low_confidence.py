#!/usr/bin/env python3
"""HARVEST E04 — low-confidence PaddleOCR layout candidates, aligned to LabelMe.

This is a DIAGNOSTIC of returned, postprocessed detections; not raw logits or
all class alternatives for one proposal. E04 retains layout_nms=True, the same
model/device and pre-rendered image inputs as E01–E03.
"""
import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def norm(label):
    return str(label).lower().strip().replace(' ', '_').replace('-', '_')


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersect(a, b):
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def overlap(a, b):
    i = intersect(a, b)
    u = area(a) + area(b) - i
    return (i / u if u > 0 else 0.), (i / area(a) if area(a) > 0 else 0.)


def load_boxes_json(path):
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    while isinstance(data, dict) and 'res' in data:
        data = data['res']
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f'Expected one page in {path}, got {len(data)}')
        data = data[0]
    if not isinstance(data, dict) or not isinstance(data.get('boxes'), list):
        raise ValueError(f'Unsupported saved prediction JSON: {path}')
    return data['boxes']


def load_gt(path, size):
    data = json.loads(path.read_text(encoding='utf-8'))
    expected = (data.get('imageWidth'), data.get('imageHeight'))
    if expected != size:
        raise ValueError(f'GT/image size mismatch on {path.name}: GT={expected}, input={size}. Stop and check source before comparing boxes.')
    shapes = []
    for i, s in enumerate(data.get('shapes', [])):
        points = s.get('points') or []
        if len(points) < 2:
            continue
        xs, ys = [float(p[0]) for p in points], [float(p[1]) for p in points]
        shapes.append({'gt_id': i, 'gt_class': norm(s['label']),
                       'gt_box': [min(xs), min(ys), max(xs), max(ys)],
                       'shape_type': s.get('shape_type', 'rectangle')})
    return shapes


def to_records(boxes):
    out = []
    for j, item in enumerate(boxes):
        coords = item.get('coordinate')
        if not coords or len(coords) != 4:
            raise ValueError(f'Unexpected candidate coordinates: {item}')
        b = [float(x) for x in coords]
        out.append({'id': j, 'label': norm(item['label']), 'score': float(item['score']), 'box': b,
                    'cls_id': item.get('cls_id', '')})
    return out


def reference_match(gt, raw, min_iou=0.5):
    if raw is None:
        return ''
    for r in to_records(raw):
        ov, _ = overlap(gt['gt_box'], r['box'])
        if ov >= min_iou and r['label'] == gt['gt_class']:
            return 'yes'
    return 'no'


def pick_font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_full_page(img, gts, recs, dest, show_floor):
    out = img.copy()
    dr = ImageDraw.Draw(out)
    width = max(2, round(min(img.size)/600))
    font = pick_font(max(15, min(35, img.width//120)))
    for gt in gts:
        dr.rectangle(gt['gt_box'], outline='#0aaa36', width=width+2)
    for r in sorted((r for r in recs if r['score'] >= show_floor), key=lambda r: r['score']):
        color = '#ff9900' if r['score'] < 0.25 else ('#e52a2a' if r['score'] < 0.5 else '#225bd9')
        dr.rectangle(r['box'], outline=color, width=width)
        xy = (int(r['box'][0]), max(0, int(r['box'][1])-24))
        dr.text(xy, f"#{r['id']} {r['label']} {r['score']:.3f}", fill=color, font=font,
                stroke_width=2, stroke_fill='white')
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest)


def draw_gt_crop(img, gt, ranked, dest, top_k):
    x1, y1, x2, y2 = gt['gt_box']
    pad_x = max(80, (x2-x1)*0.20)
    pad_y = max(70, (y2-y1)*0.25)
    bbox = (max(0,int(x1-pad_x)), max(0,int(y1-pad_y)),
            min(img.width,int(x2+pad_x)), min(img.height,int(y2+pad_y)))
    crop = img.crop(bbox).copy()
    draw = ImageDraw.Draw(crop)
    dx, dy = bbox[0], bbox[1]
    rect = (x1-dx,y1-dy,x2-dx,y2-dy)
    draw.rectangle(rect, outline='#0aaa36', width=5)
    font = pick_font(16)
    for j, item in enumerate(ranked[:top_k]):
        rec = item['candidate']
        coords = rec['box']
        xy = (coords[0]-dx, coords[1]-dy, coords[2]-dx, coords[3]-dy)
        color = '#9c30be' if rec['label'] == gt['gt_class'] else '#e52a2a'
        draw.rectangle(xy, outline=color, width=3)
        label = f"#{rec['id']} {rec['label']} {rec['score']:.3f} IoU={item['iou']:.2f}"
        ypos = max(0, min(crop.height-22, int(coords[1]-dy)+j*18))
        draw.text((max(0, int(coords[0]-dx)), ypos), label, fill=color, font=font,
                  stroke_width=2, stroke_fill='white')
    dest.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images', required=True, type=Path)
    ap.add_argument('--labels', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--e02-raw', type=Path, help='Existing E02 raw JSON directory for original detection status')
    ap.add_argument('--e03-raw', type=Path, help='Existing E03 raw JSON directory for original detection status')
    ap.add_argument('--pages', default='2,6', help='2,6 or all')
    ap.add_argument('--threshold', type=float, default=0.01)
    ap.add_argument('--device', default='gpu:0')
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--show-floor', type=float, default=0.05,
                    help='Full-page overlay floor, NOT model threshold')
    args = ap.parse_args()
    if not 0 <= args.threshold < 1:
        ap.error('--threshold must be between 0 (inclusive) and 1')
    if args.top_k < 1:
        ap.error('--top-k must be >=1')
    pages = list(range(1,75)) if args.pages.lower() == 'all' else sorted(set(int(x) for x in args.pages.split(',')))
    if not pages or min(pages) < 1 or max(pages) > 74:
        ap.error('Pages must be within 1..74')
    # Fail before GPU initialization if essential files are missing or GT dimensions differ.
    for page in pages:
        stem = f'page_{page:04d}'
        image_path = args.images / f'{stem}.png'
        label_path = args.labels / f'{stem}.json'
        for p in (image_path, label_path):
            if not p.is_file():
                raise FileNotFoundError(p)
        with Image.open(image_path) as im:
            load_gt(label_path, im.size)
    args.out.mkdir(parents=True, exist_ok=True)
    from paddleocr import LayoutDetection
    import paddleocr
    import paddle
    metadata = {'experiment': 'E04_postprocessed_low_confidence_candidates',
                'model': 'PP-DocLayout_plus-L', 'model_threshold': args.threshold,
                'device': args.device, 'layout_nms': True, 'batch_size': 1,
                'layout_merge_bboxes_mode': 'default', 'page_numbers': pages,
                'input_dir': str(args.images.resolve()), 'GT_dir': str(args.labels.resolve()),
                'E02_reference_dir': str(args.e02_raw.resolve()) if args.e02_raw else None,
                'E03_reference_dir': str(args.e03_raw.resolve()) if args.e03_raw else None,
                'paddleocr_version': getattr(paddleocr, '__version__', 'unknown'),
                'paddle_version': getattr(paddle, '__version__', 'unknown'),
                'python': sys.version, 'host_platform': platform.platform(),
                'warning': 'Only returned, NMS-filtered candidates; NOT raw per-proposal class scores. GT polygons use enclosing rectangles.'}
    (args.out/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print('Initializing PP-DocLayout_plus-L on',args.device,flush=True)
    model = LayoutDetection(model_name='PP-DocLayout_plus-L',device=args.device)
    cand_rows=[]
    gt_rows=[]
    rank_rows=[]
    for page in pages:
        stem = f'page_{page:04d}'
        image_path = args.images / f'{stem}.png'
        label_path = args.labels / f'{stem}.json'
        with Image.open(image_path) as im:
            img = im.convert('RGB')
        gts = load_gt(label_path,img.size)
        predictions = model.predict(input=str(image_path),batch_size=1,layout_nms=True,threshold=args.threshold)
        result = next(iter(predictions))
        raw_dir = args.out/'predictions'
        raw_dir.mkdir(exist_ok=True)
        result.save_to_json(save_path=str(raw_dir / f'{stem}.json'))
        boxes = load_boxes_json(raw_dir / f'{stem}.json')
        recs = to_records(boxes)
        for r in recs:
            cand_rows.append({'page': stem,'candidate_id':r['id'],'class':r['label'],'cls_id':r['cls_id'],
                              'score':r['score'],'bbox':json.dumps(r['box'])})
        old02 = load_boxes_json(args.e02_raw / f'{stem}.json') if args.e02_raw else None
        old03 = load_boxes_json(args.e03_raw / f'{stem}.json') if args.e03_raw else None
        for gt in gts:
            matched02 = reference_match(gt, old02)
            matched03 = reference_match(gt, old03)
            annotated=[]
            for r in recs:
                iou, coverage = overlap(gt['gt_box'],r['box'])
                if iou > 0:
                    annotated.append({'candidate':r,'iou':iou,'gt_coverage':coverage})
            by_score = sorted(annotated,key=lambda a:a['candidate']['score'],reverse=True)
            by_iou = sorted(annotated,key=lambda a:a['iou'],reverse=True)
            same = [a for a in by_score if a['candidate']['label']==gt['gt_class']]
            same_geom = [a for a in same if a['iou']>=0.5]
            highest = by_score[0] if by_score else None
            best_same = same[0] if same else None
            best_geom = max(same_geom,key=lambda a:a['candidate']['score']) if same_geom else None
            def describe(obj, prefix):
                return {prefix+'_class': obj['candidate']['label'] if obj else '',
                        prefix+'_score': round(obj['candidate']['score'],6) if obj else '',
                        prefix+'_iou': round(obj['iou'],5) if obj else '',
                        prefix+'_id': obj['candidate']['id'] if obj else ''}
            row = {'page':stem,'gt_id':gt['gt_id'],'gt_class':gt['gt_class'],
                   'gt_bbox':json.dumps(gt['gt_box']),'shape_type':gt['shape_type'],
                   'e02_matched_iou050':matched02,'e03_matched_iou050':matched03,
                   'overlap_candidates':len(annotated)}
            row.update(describe(highest,'top_score'))
            row.update(describe(best_same,'top_correct_class'))
            row.update(describe(best_geom,'top_correct_class_iou050'))
            gt_rows.append(row)
            for kind, seq in (('highest_score',by_score[:args.top_k]),
                              ('correct_class',same[:args.top_k]),
                              ('highest_iou',by_iou[:args.top_k])):
                for rank, obj in enumerate(seq, 1):
                    r = obj['candidate']
                    rank_rows.append({'page':stem,'gt_id':gt['gt_id'],'gt_class':gt['gt_class'],
                                      'ranking':kind,'rank':rank,'candidate_id':r['id'],
                                      'candidate_class':r['label'],'score':r['score'],
                                      'iou':round(obj['iou'],6),'gt_coverage':round(obj['gt_coverage'],6),
                                      'candidate_bbox':json.dumps(r['box'])})
            # Crops focus on GT regions missed in original E02. If reference raw
            # isn't supplied, generate all crops to avoid silently guessing.
            if matched02 != 'yes':
                crop_dest = args.out/'gt_crops'/stem/f"gt_{gt['gt_id']:03d}_{gt['gt_class']}.png"
                draw_gt_crop(img,gt,by_score,crop_dest,args.top_k)
        draw_full_page(img,gts,recs,args.out/'visualizations'/f'{stem}.png',args.show_floor)
        missed = sum(r['e02_matched_iou050']=='no' for r in gt_rows if r['page']==stem)
        print(f'{stem}: GT={len(gts)}, candidates={len(recs)}, E02 GT-unmatched={missed}',flush=True)

    def write_csv(filename, records, columns):
        with (args.out/filename).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns)
            w.writeheader();w.writerows(records)
    write_csv('all_candidates.csv',cand_rows,['page','candidate_id','class','cls_id','score','bbox'])
    write_csv('gt_candidate_summary.csv',gt_rows,['page','gt_id','gt_class','gt_bbox','shape_type',
              'e02_matched_iou050','e03_matched_iou050','overlap_candidates',
              'top_score_class','top_score_score','top_score_iou','top_score_id',
              'top_correct_class_class','top_correct_class_score','top_correct_class_iou','top_correct_class_id',
              'top_correct_class_iou050_class','top_correct_class_iou050_score',
              'top_correct_class_iou050_iou','top_correct_class_iou050_id'])
    write_csv('gt_candidate_rankings.csv',rank_rows,['page','gt_id','gt_class','ranking','rank',
              'candidate_id','candidate_class','score','iou','gt_coverage','candidate_bbox'])
    print(f'DONE: {args.out} | pages={len(pages)} candidates={len(cand_rows)} gt={len(gt_rows)}',flush=True)

if __name__ == '__main__':
    main()
