#!/usr/bin/env python3
"""Create one contact sheet from four representative pipeline preview pairs."""
import argparse
import csv
from pathlib import Path
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a new output filename')
    with (args.results/'labels.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    chosen = {}
    for index, row in enumerate(rows, 1):
        group = row['class_label'] if row['target']=='0' else 'multiple' if int(row['box_count'])>1 else 'single'
        preview = args.results/'visual_checks'/f'{index:05}.jpg'
        if group not in chosen and preview.exists():
            chosen[group] = preview
    if not chosen:
        p.error('No preview pairs found; run prepare_dataset.py with --visual-limit greater than 0')
    sheet = Image.new('RGB',(1000,560*len(chosen)),'white')
    for index, path in enumerate(chosen.values()):
        with Image.open(path) as im:
            sheet.paste(im,(0,index*560))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
