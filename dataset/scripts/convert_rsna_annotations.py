#!/usr/bin/env python3
"""Convert official adjudicated RSNA Calculated JSON annotations to CSV inputs."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def convert_annotations(annotations_path, mappings_path, images_dir, output, ids_csv=None):
    """Write CSV inputs from official JSON files into a new metadata directory."""
    if not images_dir.is_dir():
        raise ValueError('--images-dir must be an existing directory')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new or empty output directory')
    project = json.loads(annotations_path.read_text())
    mappings = json.loads(mappings_path.read_text())
    group, = [g for g in project['labelGroups'] if g['name'] == 'Calculated']
    names = {label['id']: label['name'] for label in group['labels']}
    if set(names.values()) != {'Normal', 'No Lung Opacity / Not Normal', 'Lung Opacity'}:
        raise ValueError('Unexpected Calculated class definitions')
    by_sop = defaultdict(list)
    for dataset in project['datasets']:
        for annotation in dataset['annotations']:
            if annotation['labelId'] in names:
                by_sop[annotation['SOPInstanceUID']].append(annotation)
    for key in ['subset_img_id', 'SOPInstanceUID']:
        if len({r[key] for r in mappings}) != len(mappings):
            raise ValueError(f'Duplicate mapping {key}')
    unknown = set(by_sop) - {r['SOPInstanceUID'] for r in mappings}
    if unknown:
        raise ValueError(f'{len(unknown)} annotated SOP identifiers lack mappings')
    selected = None
    if ids_csv:
        with ids_csv.open(newline='', encoding='utf-8-sig') as f:
            selected = {r['image_id'] for r in csv.DictReader(f)}
        if selected - {r['subset_img_id'] for r in mappings}:
            raise ValueError('Requested development IDs are absent from mappings')
    files = defaultdict(list)
    for path in sorted(images_dir.rglob('*')):
        if path.suffix.lower() == '.dcm':
            files[path.stem].append(path.resolve())
    labels, classes, manifest, excluded = [], [], [], []
    for m in sorted(mappings, key=lambda r: r['subset_img_id']):
        image_id, sop = m['subset_img_id'], m['SOPInstanceUID']
        if selected is not None and image_id not in selected:
            continue
        annotations = by_sop[sop]
        if not annotations:
            excluded.append({'image_id': image_id, 'reason': 'No Calculated annotation; not a negative'})
            continue
        class_names = {names[a['labelId']] for a in annotations}
        if len(class_names) != 1:
            raise ValueError(f'Conflicting Calculated classes for {image_id}')
        class_name = class_names.pop()
        target = int(class_name == 'Lung Opacity')
        candidates = set(files[image_id] + files[sop])
        if len(candidates) > 1:
            raise ValueError(f'Ambiguous image files for {image_id}: {sorted(map(str,candidates))}')
        path = next(iter(candidates)) if candidates else images_dir.resolve() / (image_id + '.dcm')
        ordered = sorted(annotations, key=lambda a: (a.get('annotationNumber') or 0, a['id']))
        if target:
            for a in ordered:
                box = a['data']
                if box is None:
                    raise ValueError(f'Positive annotation missing box: {a["id"]}')
                labels.append(dict(patientId=image_id, Target=1, **{k: box[k] for k in ['x','y','width','height']}))
        else:
            labels.append(dict(patientId=image_id, Target=0, x='', y='', width='', height=''))
        classes.append(dict(patientId=image_id, **{'class': class_name}))
        manifest.append(dict(image_id=image_id, nih_image_id=m['img_id'],
                             nih_patient_id=m['img_id'].split('_')[0], Target=target,
                             **{'class': class_name}, box_count=len(ordered) if target else 0,
                             path=str(path), SOPInstanceUID=sop))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'labels.csv', ['patientId','x','y','width','height','Target'], labels)
    write_csv(output/'detailed_class_info.csv', ['patientId','class'], classes)
    write_csv(output/'image_manifest.csv', ['image_id','nih_image_id','nih_patient_id','Target','class','box_count','path','SOPInstanceUID'], manifest)
    write_csv(output/'excluded_images.csv', ['image_id','reason'], excluded)
    summary = dict(images=len(manifest), boxes=sum(r['box_count'] for r in manifest),
                   classes=dict(Counter(r['class'] for r in manifest)), excluded_unlabelled=len(excluded),
                   missing_image_paths=sum(not Path(r['path']).is_file() for r in manifest),
                   label_group='Calculated', selection='full' if selected is None else str(ids_csv),
                   sources={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in [annotations_path, mappings_path]})
    (output/'conversion_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))

    return summary
