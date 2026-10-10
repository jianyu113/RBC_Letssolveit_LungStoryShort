#!/usr/bin/env python3
"""Organize RSNA-style annotations; optionally resize images and inspect boxes.

Metadata-only runs use Python's standard library. Images need requirements.txt.
No source files are modified. Output must be a new or empty directory.
"""
import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path, required=()):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(f'{path}: required columns: {required}')
        return list(reader)


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def geometry(w, h, size, mode):
    if not size:
        return w, h, 1., 1., 0, 0, w, h
    ow, oh = size
    if mode == 'stretch':
        rw, rh = ow, oh
    else:
        scale = min(ow / w, oh / h)
        rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    return ow, oh, rw / w, rh / h, (ow-rw)//2, (oh-rh)//2, rw, rh


def load_image(path):
    from PIL import Image
    if path.suffix.lower() == '.dcm':
        import numpy as np
        import pydicom
        ds = pydicom.dcmread(path)
        a = ds.pixel_array.astype(float)
        if a.ndim != 2:
            raise ValueError('Only single-frame grayscale DICOM supported')
        a = a * float(getattr(ds, 'RescaleSlope', 1)) + float(getattr(ds, 'RescaleIntercept', 0))
        lo, hi = float(a.min()), float(a.max())
        a = (a-lo) / (hi-lo) if hi > lo else np.zeros_like(a)
        if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
            a = 1-a
        return Image.fromarray((a*255).astype('uint8')).convert('RGB'), ds
    with Image.open(path) as im:
        if im.getexif().get(274, 1) != 1:
            raise ValueError('EXIF orientation needs explicit annotation-aware handling')
        return im.convert('RGB'), None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--images-dir', type=Path, help='Override image folder; files matched by manifest basename')
    p.add_argument('--labels-file', default='labels.csv')
    p.add_argument('--classes-file', default='detailed_class_info.csv')
    p.add_argument('--manifest-file', default='sample_ids.csv')
    p.add_argument('--source-size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'), help='Optional documented dimensions for missing images; never treated as measured')
    p.add_argument('--size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'))
    p.add_argument('--resize-mode', choices=['letterbox', 'stretch'], default='letterbox')
    p.add_argument('--visual-limit', type=int, default=100,
                   help='Maximum previews per class/box-count group; -1 for every image, 0 for none')
    args = p.parse_args()
    for size in (args.size, args.source_size):
        if size and min(size) <= 0:
            p.error('Dimensions must be positive')
    src, out = args.input.resolve(), args.output.resolve()
    if src == out or src in out.parents:
        p.error('Use an output directory outside the input folder')
    if out.exists() and any(out.iterdir()):
        p.error('Output must be empty or new; choose a fresh output folder')
    if not (src / args.labels_file).is_file():
        p.error('Input labels file is missing')
    raw = read_csv(src / args.labels_file, ['patientId','x','y','width','height','Target'])
    classes = read_csv(src / args.classes_file, ['patientId','class'])
    manifest = read_csv(src / args.manifest_file, ['image_id','path'])
    issues = []
    def issue(i, code, detail, severity='error'):
        issues.append(dict(image_id=i, severity=severity, issue=code, detail=detail))
    def keyed(rows, key, name):
        result = {}
        for r in rows:
            i = r[key].strip()
            if i in result:
                issue(i, 'duplicate_metadata_id', name)
            result[i] = r
        return result
    cm, mm = keyed(classes, 'patientId', 'classes'), keyed(manifest, 'image_id', 'manifest')
    for field in ['path', 'sha256', 'SOPInstanceUID']:
        owners = defaultdict(list)
        for i, r in mm.items():
            if r.get(field):
                owners[r[field]].append(i)
        for ids in owners.values():
            if len(ids) > 1:
                for i in ids:
                    issue(i, 'duplicate_manifest_' + field, 'Shared by image IDs: ' + ', '.join(ids))
    groups = defaultdict(list)
    for line, r in enumerate(raw, 2):
        groups[r['patientId'].strip()].append((line, r))
    out.mkdir(parents=True, exist_ok=True)
    labels, boxes, visuals = [], [], []
    matched_paths = set()
    visual_counts = Counter()
    actual_hashes = {}
    for index, i in enumerate(sorted(set(groups) | set(cm) | set(mm)), 1):
        rows, meta = groups[i], mm.get(i, {})
        if not i:
            issue(i, 'empty_id', 'Image identifier is blank')
        targets = {r['Target'].strip() for _, r in rows}
        target = next(iter(targets)) if len(targets) == 1 and targets <= {'0','1'} else ''
        if not target:
            issue(i, 'invalid_target', 'Missing, conflicting, or nonbinary target')
        if manifest and i not in mm:
            issue(i, 'missing_manifest_entry', 'No image manifest record')
        if classes and i not in cm:
            issue(i, 'missing_class', 'No detailed class record')
        cls = cm.get(i, {}).get('class', '') or meta.get('class', '')
        expected = {'Lung Opacity':'1', 'Normal':'0', 'No Lung Opacity / Not Normal':'0'}.get(cls)
        if expected is not None and target != expected:
            issue(i, 'class_target_mismatch', cls)
        for key, value in [('Target', target), ('class', cls)]:
            if meta.get(key, '') and meta[key] != value:
                issue(i, 'manifest_mismatch', key)
        rel = meta.get('path') or f'images/{i}.dcm'
        path = ((args.images_dir / Path(rel).name) if args.images_dir else src / rel).resolve()
        matched_paths.add(path)
        im, ds, measured = None, None, False
        w, h = args.source_size or ('', '')
        status = 'missing'
        if path.is_file():
            status = 'unreadable'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in actual_hashes:
                issue(i, 'duplicate_image_bytes', 'Same bytes as image ID ' + actual_hashes[digest])
            actual_hashes[digest] = i
            if meta.get('sha256') and digest != meta['sha256']:
                issue(i, 'checksum_mismatch', 'Image differs from manifest')
            try:
                im, ds = load_image(path)
                w, h = im.size
                measured, status = True, 'decoded'
                if ds is not None and meta.get('SOPInstanceUID') and str(ds.SOPInstanceUID) != meta['SOPInstanceUID']:
                    issue(i, 'dicom_id_mismatch', 'SOPInstanceUID differs from manifest')
            except Exception as e:
                issue(i, 'image_decode_failed', str(e))
        else:
            issue(i, 'missing_image', str(path))
        image_boxes = []
        seen = set()
        for line, r in rows:
            vals = [r[k].strip() for k in ['x','y','width','height']]
            if not any(vals):
                if r['Target'].strip() == '1':
                    issue(i, 'positive_without_box', f'Source row {line}')
                continue
            valid = True
            try:
                x, y, bw, bh = map(float, vals)
                if not all(math.isfinite(v) for v in [x,y,bw,bh]):
                    raise ValueError('Nonfinite coordinates')
                coords = (x, y, x+bw, y+bh)
                if x < 0 or y < 0 or bw <= 0 or bh <= 0 or (w and (x+bw > w or y+bh > h)):
                    valid = False
                    issue(i, 'invalid_box_bounds', f'Source row {line}')
                if coords in seen:
                    issue(i, 'duplicate_box', f'Source row {line}; retained')
                seen.add(coords)
            except ValueError:
                coords, valid = ('', '', '', ''), False
                issue(i, 'invalid_coordinates', f'Source row {line}: {vals}')
            if r['Target'].strip() != '1':
                issue(i, 'box_on_nonpositive', f'Source row {line}; retained')
            b = dict(image_id=i, box_id=f'{i}_{len(image_boxes)+1:03}', source_row=line,
                     class_label='Lung Opacity', source_x=r['x'], source_y=r['y'], source_width=r['width'], source_height=r['height'],
                     x_min=coords[0], y_min=coords[1], x_max=coords[2], y_max=coords[3],
                     coordinate_status='valid' if valid and measured else 'valid_assumed_dimensions' if valid and w else 'unchecked_dimensions' if valid else 'invalid',
                     processed_x_min='', processed_y_min='', processed_x_max='', processed_y_max='')
            image_boxes.append(b)
        if target == '0' and len(rows) > 1:
            issue(i, 'repeated_negative_rows', f'{len(rows)} source rows', 'warning')
        if meta.get('box_count') and meta['box_count'] != str(len(image_boxes)):
            issue(i, 'box_count_mismatch', f'Manifest={meta["box_count"]}; annotations={len(image_boxes)}')
        processed_path, ow, oh, sx, sy, px, py = '', '', '', '', '', '', ''
        if im is not None:
            from PIL import Image, ImageDraw
            ow, oh, sx, sy, px, py, rw, rh = geometry(w, h, args.size, args.resize_mode)
            processed = Image.new('RGB', (ow, oh))
            processed.paste(im.resize((rw,rh), Image.Resampling.BILINEAR), (px,py))
            visual_group = (cls, 'multiple' if len(image_boxes)>1 else 'single' if image_boxes else 'none')
            make_visual = args.visual_limit < 0 or visual_counts[visual_group] < args.visual_limit
            original_view, processed_view = (im.copy(), processed.copy()) if make_visual else (None, None)
            for b in image_boxes:
                if b['coordinate_status'] != 'valid':
                    continue
                c = [b[k] for k in ['x_min','y_min','x_max','y_max']]
                t = [c[0]*sx+px,c[1]*sy+py,c[2]*sx+px,c[3]*sy+py]
                assert all(abs(a-z) < 1e-6 for a,z in zip(c, [(t[0]-px)/sx,(t[1]-py)/sy,(t[2]-px)/sx,(t[3]-py)/sy]))
                for k,v in zip(['processed_x_min','processed_y_min','processed_x_max','processed_y_max'], t):
                    b[k] = v
                if make_visual:
                    ImageDraw.Draw(original_view).rectangle(c, outline='red', width=3)
                    ImageDraw.Draw(processed_view).rectangle(t, outline='red', width=3)
            (out/'processed_images').mkdir(exist_ok=True)
            (out/'visual_checks').mkdir(exist_ok=True)
            processed_path = f'processed_images/{index:05}.png'
            processed.save(out/processed_path)
            if make_visual:
                panel = Image.new('RGB', (1000,560), 'white')
                d = ImageDraw.Draw(panel)
                d.text((10,8), f'{i} | {cls} | Target={target} | boxes={len(image_boxes)}', fill='black')
                for view, xoff, title in [(original_view,0,'Original'),(processed_view,500,'Processed')]:
                    view.thumbnail((490,490))
                    panel.paste(view,(xoff+5,55))
                    d.text((xoff+10,32),title,fill='black')
                vp = f'visual_checks/{index:05}.jpg'
                panel.save(out/vp)
                visuals.append((i,vp))
                visual_counts[visual_group] += 1
        boxes.extend(image_boxes)
        labels.append(dict(image_id=i, image_path=str(path), nih_patient_id=meta.get('nih_patient_id',''),
                           target=target, class_label=cls, is_negative=str(target=='0').lower() if target else '',
                           box_count=len(image_boxes), width=w, height=h,
                           dimensions_source='decoded_image' if measured else 'user_assumption' if w else 'unknown',
                           image_status=status, processed_path=processed_path, processed_width=ow, processed_height=oh,
                           scale_x=sx, scale_y=sy, pad_left=px, pad_top=py,
                           check_status=''))
        if index % 1000 == 0:
            print(f'Processed {index} images', flush=True)
    image_root = args.images_dir or src/'images'
    if image_root.exists():
        for path in image_root.rglob('*'):
            if path.suffix.lower() in {'.dcm','.png','.jpg','.jpeg','.tif','.tiff'} and path.resolve() not in matched_paths:
                issue('', 'unmatched_image', str(path))
    affected = {r['image_id'] for r in issues}
    for row in labels:
        row['check_status'] = 'issues' if row['image_id'] in affected else 'automated_checks_passed_visual_review_pending'
    lf = 'image_id image_path nih_patient_id target class_label is_negative box_count width height dimensions_source image_status processed_path processed_width processed_height scale_x scale_y pad_left pad_top check_status'.split()
    bf = 'image_id box_id source_row class_label source_x source_y source_width source_height x_min y_min x_max y_max coordinate_status processed_x_min processed_y_min processed_x_max processed_y_max'.split()
    write_csv(out/'labels.csv', labels, lf)
    write_csv(out/'boxes.csv', boxes, bf)
    write_csv(out/'issue_summary.csv', issues, ['image_id','severity','issue','detail'])
    counts = Counter(r['issue'] for r in issues)
    summary = dict(images=len(labels), positives=sum(r['target']=='1' for r in labels), negatives=sum(r['target']=='0' for r in labels),
                   boxes=len(boxes), multiple_box_images=sum(r['box_count']>1 for r in labels),
                   decoded_images=sum(r['image_status']=='decoded' for r in labels), visual_previews=len(visuals), class_counts=dict(Counter(r['class_label'] for r in labels)), issues=dict(counts),
                   distinct_nih_patients=len({r['nih_patient_id'] for r in labels if r['nih_patient_id']}),
                   visual_alignment='manual review pending' if visuals else 'no previews generated; see decoded_images and visual_limit',
                   settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report = ['# Dataset checks', '', f'Images: {len(labels)}; boxes: {len(boxes)}; negatives: {summary["negatives"]}; multiple-box images: {summary["multiple_box_images"]}.', '',
              f'Decoded images: {summary["decoded_images"]}. Previews: {len(visuals)}. Visual alignment: {summary["visual_alignment"]}.', '',
              'Coordinates are floating-point pixel edges, origin top-left, exclusive maxima. Original and processed coordinates are separate. Missing images never become negatives.', '',
              'Issue counts:', *[f'- {k}: {v}' for k,v in counts.items()], '',
              'Source: input labels.csv, detailed_class_info.csv and sample_ids.csv. Dimensions for missing images, if supplied, are assumptions, not measurements.', '',
              'DICOM rendering uses per-image min/max scaling and MONOCHROME1 inversion; these previews are for annotation QA. No crop, rotation or clinical windowing is applied. Keep NIH patient IDs together in future train/test splits.']
    (out/'issue_summary.md').write_text('\n'.join(report)+'\n')
    gallery = '<!doctype html><meta charset="utf-8"><title>Annotation visual checks</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{max-width:100%}figure{margin:30px 0}</style><h1>Annotation visual checks</h1>'
    gallery += '<p>Original left; processed right. Red rectangles show boxes. Review object alignment manually; numerical transform checks do not prove semantic correctness.</p>'
    if not visuals:
        gallery += '<p><strong>No previews generated. Check decoded_images and visual_limit in summary.json.</strong></p>'
    for i,vp in visuals:
        gallery += f'<figure><figcaption>{html.escape(i)}</figcaption><img loading="lazy" src="{vp}"></figure>'
    (out/'visual_checks.html').write_text(gallery)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
