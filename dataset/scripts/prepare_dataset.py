#!/usr/bin/env python3
"""Prepare RSNA DICOM images, matching boxes, and fixed patient-level splits.

Reuse existing splits or create them with seed 42. Output must be new or empty.
"""
import argparse
import csv
import hashlib
import html
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw

from convert_rsna_annotations import convert_annotations
from image_processing import apply_intensity_processing, load_dicom_pixels, resize_and_pad


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


def png_image(pixels):
    """Encode normalized grayscale pixels as RGB with identical channels."""
    return Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype('uint8')).convert('RGB')


def load_fixed_splits(path, manifest, image_ids):
    """Require complete assignments and keep every patient's images together."""
    rows = read_csv(path, ['image_id', 'nih_patient_id', 'split'])
    assignments = {r['image_id']: r for r in rows}
    if len(assignments) != len(rows) or set(assignments) != image_ids:
        raise ValueError('splits.csv must contain exactly one row for every input image')
    patients = {}
    for image_id, row in assignments.items():
        patient = manifest[image_id]['nih_patient_id']
        if not patient or row['nih_patient_id'] != patient:
            raise ValueError(f'Patient ID mismatch in splits.csv: {image_id}')
        if row['split'] not in {'train', 'val', 'test'}:
            raise ValueError(f'Invalid split for {image_id}: {row["split"]}')
        if patients.setdefault(patient, row['split']) != row['split']:
            raise ValueError(f'Patient appears in multiple splits: {patient}')
    return assignments


def make_splits(manifest: pd.DataFrame) -> pd.DataFrame:
    """Keep each NIH patient's images together and stratify by patient target."""
    patient_targets = manifest.groupby("nih_patient_id", sort=True)["Target"].max()
    train_ids, remaining_ids = train_test_split(
        patient_targets.index,
        test_size=0.30,
        stratify=patient_targets,
        random_state=42,
    )
    val_ids, test_ids = train_test_split(
        remaining_ids,
        test_size=0.50,
        stratify=patient_targets.loc[remaining_ids],
        random_state=42,
    )

    patient_splits = {}
    for name, patient_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        patient_splits.update({patient_id: name for patient_id in patient_ids})

    splits = manifest[["image_id", "nih_patient_id"]].copy()
    splits["split"] = splits["nih_patient_id"].map(patient_splits)
    return splits.sort_values("image_id").reset_index(drop=True)


def prepare_splits(path, manifest, check_only=False):
    """Preserve saved splits; create them only when missing."""
    if check_only or not path.exists():
        frame = pd.DataFrame(manifest)
        frame['Target'] = frame['Target'].astype(int)
        csv_text = make_splits(frame).to_csv(index=False, lineterminator='\n')
        if check_only:
            if path.read_text(encoding='utf-8') != csv_text:
                raise ValueError('splits.csv does not match the fixed patient-level split.')
            print(f'Verified {path} against the manifest and seed 42.')
        else:
            with path.open('x', encoding='utf-8', newline='') as handle:
                handle.write(csv_text)
            print(f'Created {path}')
    else:
        print(f'Reusing {path}')


def save_review_examples(output, previews):
    """Combine the first preview from each class/box-count group."""
    if not previews:
        return
    sheet = Image.new('RGB', (1000, 560 * len(previews)), 'white')
    for index, path in enumerate(previews):
        with Image.open(output / path) as preview:
            sheet.paste(preview, (0, index * 560))
    sheet.save(output / 'review_examples.jpg')


def prepare_images(args):
    src, out = args.input.resolve(), args.output.resolve()
    raw = read_csv(src / 'labels.csv', ['patientId','x','y','width','height','Target'])
    classes = read_csv(src / 'detailed_class_info.csv', ['patientId','class'])
    manifest = read_csv(src / args.manifest_file, ['image_id','nih_patient_id','Target','path'])
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
    image_ids = set(groups) | set(cm) | set(mm)
    if set(mm) != image_ids:
        raise ValueError('The manifest must contain every input image')
    prepare_splits(src / args.splits_file, manifest)
    assignments = load_fixed_splits(src / args.splits_file, mm, image_ids)
    out.mkdir(parents=True, exist_ok=True)
    labels, boxes, visuals = [], [], []
    matched_paths = set()
    visual_counts = Counter()
    review_previews = {}
    actual_hashes = {}
    for index, i in enumerate(sorted(image_ids), 1):
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
        path = (src / rel).resolve()
        matched_paths.add(path)
        im, ds, measured = None, None, False
        processed, intensity, spatial = None, {}, {}
        w, h = '', ''
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
                ds, pixels = load_dicom_pixels(path)
                normalized, intensity = apply_intensity_processing(pixels, ds)
                tensor, spatial = resize_and_pad(normalized, args.target_size)
                im = png_image(normalized)
                processed = png_image(tensor[0].numpy())
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
                     coordinate_status='valid' if valid and measured else 'unchecked_dimensions' if valid else 'invalid',
                     processed_x_min='', processed_y_min='', processed_x_max='', processed_y_max='')
            image_boxes.append(b)
        if target == '0' and len(rows) > 1:
            issue(i, 'repeated_negative_rows', f'{len(rows)} source rows', 'warning')
        if meta.get('box_count') and meta['box_count'] != str(len(image_boxes)):
            issue(i, 'box_count_mismatch', f'Manifest={meta["box_count"]}; annotations={len(image_boxes)}')
        processed_path, ow, oh, sx, sy, px, py = '', '', '', '', '', '', ''
        if processed is not None:
            ow, oh = spatial['processed_width'], spatial['processed_height']
            sx, sy = spatial['scale_x'], spatial['scale_y']
            px, py = spatial['pad_left'], spatial['pad_top']
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
            processed_path = f'processed_images/{i}.png'
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
                review_previews.setdefault(visual_group, vp)
        boxes.extend(image_boxes)
        labels.append(dict(image_id=i, image_path=Path(os.path.relpath(path, out)).as_posix(),
                           nih_patient_id=meta['nih_patient_id'], split=assignments[i]['split'],
                           target=target, class_label=cls, is_negative=str(target=='0').lower() if target else '',
                           box_count=len(image_boxes), width=w, height=h,
                           dimensions_source='decoded_image' if measured else 'unknown',
                           image_status=status, processed_path=processed_path, processed_width=ow, processed_height=oh,
                           scale_x=sx, scale_y=sy, pad_left=px, pad_top=py,
                           rescale_slope=intensity.get('rescale_slope', ''), rescale_intercept=intensity.get('rescale_intercept', ''),
                           inverted=intensity.get('inverted', ''), clip_low=intensity.get('clip_low', ''), clip_high=intensity.get('clip_high', ''),
                           check_status=''))
        if index % 1000 == 0:
            print(f'Processed {index} images', flush=True)
    image_root = src/'images'
    if image_root.exists():
        for path in image_root.rglob('*'):
            if path.suffix.lower() in {'.dcm','.png','.jpg','.jpeg','.tif','.tiff'} and path.resolve() not in matched_paths:
                issue('', 'unmatched_image', str(path))
    affected = {r['image_id'] for r in issues}
    for row in labels:
        row['check_status'] = 'issues' if row['image_id'] in affected else 'automated_checks_passed_visual_review_pending'
    lf = 'image_id image_path nih_patient_id split target class_label is_negative box_count width height dimensions_source image_status processed_path processed_width processed_height scale_x scale_y pad_left pad_top rescale_slope rescale_intercept inverted clip_low clip_high check_status'.split()
    bf = 'image_id box_id source_row class_label source_x source_y source_width source_height x_min y_min x_max y_max coordinate_status processed_x_min processed_y_min processed_x_max processed_y_max'.split()
    write_csv(out/'labels.csv', labels, lf)
    write_csv(out/'boxes.csv', boxes, bf)
    write_csv(out/'issue_summary.csv', issues, ['image_id','severity','issue','detail'])
    counts = Counter(r['issue'] for r in issues)
    summary = dict(images=len(labels), positives=sum(r['target']=='1' for r in labels), negatives=sum(r['target']=='0' for r in labels),
                   boxes=len(boxes), multiple_box_images=sum(r['box_count']>1 for r in labels),
                   decoded_images=sum(r['image_status']=='decoded' for r in labels), visual_previews=len(visuals), class_counts=dict(Counter(r['class_label'] for r in labels)), issues=dict(counts),
                   distinct_nih_patients=len({r['nih_patient_id'] for r in labels if r['nih_patient_id']}),
                   split_counts=dict(Counter(r['split'] for r in labels)),
                   patient_split_counts={name:len({r['nih_patient_id'] for r in labels if r['split']==name}) for name in ['train','val','test']},
                   preprocessing=dict(target_size=args.target_size, normalization='per-image 0.5/99.5 percentile clipping',
                                      interpolation='bilinear, align_corners=False', padding='centered black',
                                      output='8-bit RGB PNG with identical channels'),
                   visual_alignment='manual review pending' if visuals else 'no previews generated; see decoded_images and visual_limit',
                   settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report = ['# Dataset checks', '', f'Images: {len(labels)}; boxes: {len(boxes)}; negatives: {summary["negatives"]}; multiple-box images: {summary["multiple_box_images"]}.', '',
              f'Decoded images: {summary["decoded_images"]}. Previews: {len(visuals)}. Visual alignment: {summary["visual_alignment"]}.', '',
              'Coordinates are floating-point pixel edges, origin top-left, exclusive maxima. Original and processed coordinates are separate. Missing images never become negatives.', '',
              'Issue counts:', *[f'- {k}: {v}' for k,v in counts.items()], '',
              f'Source: labels.csv, detailed_class_info.csv, {args.manifest_file}, and {args.splits_file}.', '',
              f'Fixed image splits: {summary["split_counts"]}. Patient splits: {summary["patient_split_counts"]}.', '',
              'DICOM preprocessing uses rescale slope/intercept, MONOCHROME1 inversion, and per-image 0.5/99.5 percentile clipping. Images are normalized to [0, 1], resized with bilinear interpolation, padded, and saved as rounded 8-bit RGB PNGs with identical channels. Box coordinates use the same spatial transform. No crop, rotation, or random augmentation is applied.', '',
              'Image paths in labels.csv are relative to this output directory. Review the previews before using the outputs for training.']
    (out/'issue_summary.md').write_text('\n'.join(report)+'\n')
    gallery = '<!doctype html><meta charset="utf-8"><title>Annotation visual checks</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{max-width:100%}figure{margin:30px 0}</style><h1>Annotation visual checks</h1>'
    gallery += '<p>Original dimensions left; processed right. Both views use the shared intensity normalization. Red rectangles show boxes. Review object alignment manually; numerical transform checks do not prove semantic correctness.</p>'
    if not visuals:
        gallery += '<p><strong>No previews generated. Check decoded_images and visual_limit in summary.json.</strong></p>'
    for i,vp in visuals:
        gallery += f'<figure><figcaption>{html.escape(i)}</figcaption><img loading="lazy" src="{vp}"></figure>'
    (out/'visual_checks.html').write_text(gallery)
    save_review_examples(out, list(review_previews.values()))
    print(json.dumps(summary,indent=2))
    if any(r['severity'] == 'error' for r in issues):
        raise SystemExit('Preparation found errors; inspect issue_summary.csv before using these outputs.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path, help='Metadata directory')
    p.add_argument('--output', type=Path, help='New or empty directory for prepared data')
    p.add_argument('--manifest-file', default='image_manifest.csv',
                   help='Manifest filename; use sample_ids.csv for the development sample')
    p.add_argument('--splits-file', default='splits.csv')
    p.add_argument('--target-size', type=int, default=512)
    p.add_argument('--visual-limit', type=int, default=100,
                   help='Previews per class/box-count group; -1 for all, 0 for none')
    p.add_argument('--check-splits', action='store_true',
                   help='Check saved splits against seed 42 without writing any files')
    source = p.add_argument_group('Optional official JSON conversion into a new metadata directory')
    source.add_argument('--annotations', type=Path)
    source.add_argument('--mappings', type=Path)
    source.add_argument('--images-dir', type=Path, help='Extracted DICOM root for JSON conversion')
    source.add_argument('--ids-csv', type=Path, help='Optional image ID selection for JSON conversion')
    args = p.parse_args()

    conversion = [args.annotations, args.mappings, args.images_dir]
    if any(conversion) and not all(conversion):
        p.error('JSON conversion requires --annotations, --mappings, and --images-dir together')
    if args.ids_csv and not all(conversion):
        p.error('--ids-csv requires JSON conversion')
    if args.check_splits:
        if any(conversion):
            p.error('--check-splits only reads existing metadata')
        manifest = read_csv(args.input / args.manifest_file, ['image_id', 'nih_patient_id', 'Target'])
        if not manifest:
            p.error('The manifest is missing or empty')
        prepare_splits(args.input / args.splits_file, manifest, check_only=True)
        return
    if args.output is None:
        p.error('--output is required for preparation')
    if args.target_size <= 0:
        p.error('--target-size must be positive')
    src, out = args.input.resolve(), args.output.resolve()
    if src == out or src in out.parents:
        p.error('Use an output directory outside the input folder')
    if out.exists() and any(out.iterdir()):
        p.error('Output must be empty or new; choose a fresh output folder')
    if all(conversion):
        if args.manifest_file != 'image_manifest.csv':
            p.error('JSON conversion writes image_manifest.csv; omit --manifest-file')
        convert_annotations(args.annotations, args.mappings, args.images_dir, src, args.ids_csv)
    for filename in ('labels.csv', 'detailed_class_info.csv', args.manifest_file):
        if not (src / filename).is_file():
            p.error(f'Input file is missing: {filename}')
    prepare_images(args)


if __name__ == '__main__':
    main()
