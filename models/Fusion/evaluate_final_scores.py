"""Reproduce the final fixed 80:20 metrics from saved F04 scores, without inference."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import random


OTHER = "No Lung Opacity / Not Normal"
METHODS = ("detector", "classifier", "fusion_80_20")
SPLITS = {"val": (4271, 959, 1835), "test": (4595, 1161, 1835)}
MODEL_HASHES = {
    "detector_checkpoint_sha256": "6f002c4685f2ccba831388c3f22432fbf3fc36cec5a40d0b73c0ce5fadd7650d",
    "classifier_checkpoint_sha256": "bc1aea04eefd65113e94df0335163593fe81d2989d0695c689895795b423115c",
}
DATA_HASHES = {
    "labels.csv": "a632c1e75e3873c898906893de69c5c80164a58e378e2c26868a01517787c915",
    "boxes.csv": "2f493b8204a3260f602725305e29b06fd5f7e291c619590d563bfa7426889a2d",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_file(path, manifest_path, manifest):
    relative = path.resolve().relative_to(manifest_path.resolve().parent).as_posix()
    actual = sha256(path)
    if manifest.get(relative) != actual:
        raise ValueError(f"Source manifest mismatch or missing entry: {relative}")
    return dict(path=str(path.resolve()), sha256=actual)


def read_inputs(scores_path, split, config_path, manifest_path, raw_path=None):
    manifest = json.loads(manifest_path.read_text())
    sources = {name: verified_file(path, manifest_path, manifest)
               for name, path in (("scores", scores_path), ("config", config_path))}
    if raw_path is not None:
        sources["raw_predictions"] = verified_file(raw_path, manifest_path, manifest)
    sources["manifest"] = dict(path=str(manifest_path.resolve()), sha256=sha256(manifest_path))
    config = json.loads(config_path.read_text())
    if config["source_versions"] != {"prepared": 358799531, "detector": 364103459, "classifier": 362276331}:
        raise ValueError("Use the saved F04 output with the final model and data versions.")
    if any(config["checkpoints"][key] != value for key, value in MODEL_HASHES.items()):
        raise ValueError("Source config does not identify the final checkpoints.")
    if config["data"]["metadata_sha256"] != DATA_HASHES or not config["data"]["patient_disjoint"]:
        raise ValueError("Source data hashes or patient-split checks differ from F04.")
    if config["alpha"] != 0.2 or any(config[key] for key in ("new_training", "parameter_fitting", "weight_search")):
        raise ValueError("Source config is not the frozen 80:20 evaluation.")
    if scores_path.resolve().relative_to(manifest_path.resolve().parent).as_posix() != f"{split}/image_scores.csv":
        raise ValueError("Select the requested split's image_scores.csv from the original output layout.")
    if raw_path is not None and raw_path.resolve().relative_to(manifest_path.resolve().parent).as_posix() != f"{split}/raw_detector_predictions.jsonl":
        raise ValueError("Select the same split's raw detector predictions.")
    with scores_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        row["target"] = int(row["target"])
        if row["class_label"] not in {"Normal", OTHER, "Lung Opacity"}:
            raise ValueError("Unexpected class label.")
        if row["target"] != int(row["class_label"] == "Lung Opacity"):
            raise ValueError("Image target and class label disagree.")
        if not row["image_id"] or not row["patient_id"]:
            raise ValueError("Image and patient IDs must be present.")
        for key in ("detector", "classifier"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError("Image scores must be finite and between zero and one.")
        row["fusion_80_20"] = 0.8 * row["detector"] + 0.2 * row["classifier"]
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image IDs.")
    counts = (len(rows), sum(row["target"] for row in rows), len({row["patient_id"] for row in rows}))
    if counts != SPLITS[split]:
        raise ValueError(f"Expected images/positives/patients {SPLITS[split]}, got {counts}.")
    return rows, sources


def ranked_rows(rows, method):
    return sorted(rows, key=lambda row: (-row[method], row["image_id"]))


def metrics(rows, method, split):
    positive = sum(row["target"] for row in rows)
    negative = len(rows) - positive
    if not positive or not negative:
        raise ValueError("AUROC requires positive and negative images.")
    ascending = sorted(rows, key=lambda row: row[method])
    rank_sum, start = 0.0, 0
    while start < len(rows):
        end = start + 1
        while end < len(rows) and ascending[end][method] == ascending[start][method]:
            end += 1
        rank_sum += (start + 1 + end) / 2 * sum(row["target"] for row in ascending[start:end])
        start = end
    budget = len(rows) // 5
    selected = ranked_rows(rows, method)[:budget]
    recalled = sum(row["target"] for row in selected)
    return dict(method=method, split=split, images=len(rows),
                patients=len({row["patient_id"] for row in rows}), positives=positive,
                review_count=budget, recalled_positives=recalled, recall_at20=recalled / positive,
                auroc=(rank_sum - positive * (positive + 1) / 2) / (positive * negative),
                top20_other_abnormal=sum(row["class_label"] == OTHER for row in selected),
                top20_normal=sum(row["class_label"] == "Normal" for row in selected),
                fp_other_abnormal_030=sum(row["class_label"] == OTHER and row[method] >= .3 for row in rows),
                fp_normal_030=sum(row["class_label"] == "Normal" and row[method] >= .3 for row in rows))


def percentile(values, q):
    values = sorted(values)
    position = q * (len(values) - 1)
    index = int(position)
    fraction = position - index
    return values[index] * (1 - fraction) + values[min(index + 1, len(values) - 1)] * fraction


def patient_bootstrap(rows, repeats=2000):
    """Resample patients jointly and recalculate floor(N/5), matching the validation protocol."""
    patients = sorted({row["patient_id"] for row in rows})
    patient_index = {patient: i for i, patient in enumerate(patients)}
    ordered = {method: [(patient_index[row["patient_id"]], row["target"])
                        for row in ranked_rows(rows, method)]
               for method in ("detector", "fusion_80_20")}
    rng = random.Random(42)
    differences = []
    for _ in range(repeats):
        counts = Counter(rng.choices(range(len(patients)), k=len(patients)))
        n = sum(counts[patient] for patient, _ in ordered["detector"])
        positives = sum(counts[patient] * target for patient, target in ordered["detector"])
        if not positives:
            raise ValueError("A bootstrap sample has no positive images.")
        recalls = {}
        for method, sequence in ordered.items():
            remaining, recalled = n // 5, 0
            for patient, target in sequence:
                take = min(remaining, counts[patient])
                recalled += take * target
                remaining -= take
                if not remaining:
                    break
            recalls[method] = recalled / positives
        differences.append(100 * (recalls["fusion_80_20"] - recalls["detector"]))
    return dict(method="Paired patient-cluster bootstrap; percentile interval", repeats=repeats, seed=42,
                patients=len(patients), comparison="fusion_80_20 minus detector",
                units="percentage points of Recall@20%",
                ci95_low=percentile(differences, .025), ci95_high=percentile(differences, .975),
                limitation="Conditional on the frozen models; does not estimate training-seed variation or selection optimism.")


def detection_metrics(raw_path, rows):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if importlib.metadata.version("pycocotools") != "2.0.11":
        raise ValueError("Use pycocotools==2.0.11 for the saved detector evaluation.")
    with raw_path.open() as source:
        raw = [json.loads(line) for line in source if line.strip()]
    scores = {row["image_id"]: row for row in rows}
    if len(raw) != len(rows) or {row["image_id"] for row in raw} != set(scores):
        raise ValueError("Raw predictions must contain every scored image exactly once.")
    if len({row["numeric_id"] for row in raw}) != len(raw):
        raise ValueError("Duplicate numeric IDs in raw predictions.")
    images, truth_boxes, predictions = [], [], []
    for row in raw:
        image_id = row["numeric_id"]
        reference = scores[row["image_id"]]
        if len(row["boxes"]) != len(row["box_scores"]) or len(row["boxes"]) != len(row["box_labels"]):
            raise ValueError("Raw detector arrays have different lengths.")
        if any(label != 1 for label in row["box_labels"]):
            raise ValueError("Raw detector labels must be Lung Opacity (1).")
        if any(not math.isfinite(score) or not 0 <= score <= 1 for score in row["box_scores"]):
            raise ValueError("Invalid raw box score.")
        if not math.isclose(max(row["box_scores"], default=0), reference["detector"], abs_tol=1e-12, rel_tol=0):
            raise ValueError("Cached box scores disagree with the image detector score.")
        if bool(row["gt_boxes"]) != bool(reference["target"]):
            raise ValueError("Cached ground-truth boxes disagree with the image target.")
        images.append(dict(id=image_id, width=512, height=512))
        for boxes, destination in ((row["gt_boxes"], truth_boxes), (row["boxes"], predictions)):
            for index, box in enumerate(boxes):
                if len(box) != 4 or any(not math.isfinite(value) for value in box):
                    raise ValueError("Invalid cached box coordinates.")
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= 512 and 0 <= y1 < y2 <= 512):
                    raise ValueError("Cached boxes must fit the prepared 512-pixel image.")
                annotation = dict(id=len(destination) + 1, image_id=image_id, category_id=1,
                                  bbox=[x1, y1, x2-x1, y2-y1], area=(x2-x1)*(y2-y1), iscrowd=0)
                if destination is predictions:
                    annotation["score"] = row["box_scores"][index]
                destination.append(annotation)
    with redirect_stdout(io.StringIO()):
        truth, prediction = COCO(), COCO()
        for target, annotations in ((truth, truth_boxes), (prediction, predictions)):
            target.dataset = dict(images=images, annotations=annotations,
                                  categories=[dict(id=1, name="Lung Opacity")])
            target.createIndex()
        evaluator = COCOeval(truth, prediction, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return dict(raw_detector_coco_map=float(evaluator.stats[0]), raw_detector_ap50=float(evaluator.stats[1]),
                raw_detector_ap75=float(evaluator.stats[2]), fusion_changes_boxes=False,
                fusion_changes_box_scores=False)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-predictions", type=Path, help="Optional same-split cached JSONL for raw detector AP.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output directory; existing results are preserved.")
    rows, sources = read_inputs(args.scores, args.split, args.source_config, args.source_manifest, args.raw_predictions)
    comparison = [metrics(rows, method, args.split) for method in METHODS]
    detection = detection_metrics(args.raw_predictions, rows) if args.raw_predictions is not None else None
    summary = dict(status="complete", sources=sources, code_sha256=sha256(__file__),
                   protocol=dict(formula="0.8 * detector + 0.2 * classifier", ranking="descending score, then image_id",
                                 review_budget="floor(N/5)", image_flag_threshold=0.3, new_inference=False,
                                 training=False, weight_search=False, manifest_scope="Input files checked against supplied F04 manifest; model and data identities checked in its config."),
                   comparison=comparison, patient_bootstrap=patient_bootstrap(rows), raw_detection=detection,
                   detection_note="Raw box metrics only; fusion does not alter boxes or box scores." if detection else
                   "Not recomputed: no raw detector cache supplied. Image flags are not false-positive box counts.")
    args.output.mkdir(parents=True)
    with (args.output / "comparison.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "sha256.json", {name: sha256(args.output / name) for name in ("comparison.csv", "summary.json")})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
