"""Evaluate the fixed 80:20 detector/classifier fusion; never train or tune."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
from functools import partial
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models import resnet18
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms import Normalize


ALPHA = 0.2
OTHER = "No Lung Opacity / Not Normal"
EXPECTED_DATA = {
    "labels.csv": "a632c1e75e3873c898906893de69c5c80164a58e378e2c26868a01517787c915",
    "boxes.csv": "2f493b8204a3260f602725305e29b06fd5f7e291c619590d563bfa7426889a2d",
}
DETECTOR_SHA = "6f002c4685f2ccba831388c3f22432fbf3fc36cec5a40d0b73c0ce5fadd7650d"
CLASSIFIER_RUN = "resnet18_full_training_20261025_221305_402389"
CLASSIFIER_REFERENCE_SEMANTIC_SHA = "de1b78d451fc937286e3584f5dee1d479afbe0c9641123fdf105833efca84749"
SOURCE_VERSIONS = {"prepared": 358799531, "detector": 364103459, "classifier": 362276331}
METHODS = ("detector", "classifier", "fusion_80_20")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as source:
        return list(csv.DictReader(source))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class PreparedImages(Dataset):
    def __init__(self, data_dir, labels, boxes, split):
        self.data_dir = data_dir
        self.records = [dict(row, numeric_id=i) for i, row in enumerate(labels)
                        if row["split"] == split]
        self.boxes = boxes

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        path = self.data_dir / row["processed_path"]
        encoded = path.read_bytes()
        with Image.open(io.BytesIO(encoded)) as source:
            pixels = np.array(source.convert("RGB"), dtype=np.uint8)
        if pixels.shape != (512, 512, 3):
            raise ValueError(f"Unexpected prepared image shape: {row['image_id']}")
        image = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255)
        metadata = dict(row, png_sha256=hashlib.sha256(encoded).hexdigest(),
                        gt_boxes=self.boxes.get(row["image_id"], []))
        return image, metadata


def collate(batch):
    images, metadata = zip(*batch)
    return list(images), list(metadata)


def validate_data(data_dir):
    actual_hashes = {name: sha256(data_dir / name) for name in EXPECTED_DATA}
    if actual_hashes != EXPECTED_DATA:
        raise ValueError(f"Prepared metadata differs from Crystal version 358799531: {actual_hashes}")
    labels = read_csv(data_dir / "labels.csv")
    if len({row["image_id"] for row in labels}) != len(labels):
        raise ValueError("Duplicate image IDs in prepared labels")
    counts = Counter(row["split"] for row in labels)
    if counts != {"train": 20818, "val": 4271, "test": 4595}:
        raise ValueError(f"Unexpected split counts: {counts}")
    patients = {}
    for row in labels:
        patient = row["nih_patient_id"]
        if not patient or patients.setdefault(patient, row["split"]) != row["split"]:
            raise ValueError("Patient split overlap or missing patient ID")
        if row["image_status"] != "decoded" or not row["processed_path"]:
            raise ValueError("Use fully prepared decoded images")
        if int(row["target"]) != int(row["class_label"] == "Lung Opacity"):
            raise ValueError("Image target and class differ")
    patient_counts = Counter(patients.values())
    if patient_counts != {"train": 8563, "val": 1835, "test": 1835}:
        raise ValueError(f"Unexpected patient counts: {patient_counts}")
    boxes = {}
    for row in read_csv(data_dir / "boxes.csv"):
        box = [float(row[key]) for key in ("processed_x_min", "processed_y_min",
                                           "processed_x_max", "processed_y_max")]
        if not (0 <= box[0] < box[2] <= 512 and 0 <= box[1] < box[3] <= 512):
            raise ValueError("Invalid prepared bounding box")
        boxes.setdefault(row["image_id"], []).append(box)
    for row in labels:
        if bool(boxes.get(row["image_id"])) != bool(int(row["target"])):
            raise ValueError("Prepared boxes and image target disagree")
    return labels, boxes, dict(metadata_sha256=actual_hashes, images_by_split=dict(counts),
                               patients_by_split=dict(patient_counts), patient_disjoint=True)


def load_models(detector_path, classifier_path, device):
    if sha256(detector_path) != DETECTOR_SHA:
        raise ValueError("Wrong detector checkpoint: require Baseline44 epoch 6")
    if classifier_path.name != "best.pt" or classifier_path.parent.name != CLASSIFIER_RUN:
        raise ValueError("Select best.pt from the original Crystal baseline run only")
    detector_checkpoint = torch.load(detector_path, map_location="cpu", weights_only=True)
    classifier_checkpoint = torch.load(classifier_path, map_location="cpu", weights_only=True)
    if float(detector_checkpoint["epoch"]) != 6:
        raise ValueError("Detector checkpoint is not epoch 6")
    if classifier_checkpoint["completed_epoch"] != 2:
        raise ValueError("Classifier best.pt is not the original baseline epoch 2")
    settings = classifier_checkpoint["settings"]
    expected = dict(labels_sha256=EXPECTED_DATA["labels.csv"], batch_size=16,
                    workers=2, seed=42, model="resnet18")
    if any(settings[key] != value for key, value in expected.items()):
        raise ValueError(f"Unexpected classifier training settings: {settings}")
    if classifier_checkpoint["history"][-1]["epoch"] != 2:
        raise ValueError("Classifier checkpoint history does not end at epoch 2")
    if not math.isclose(classifier_checkpoint["best_validation_loss"],
                        0.35849242885423693, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Classifier baseline validation loss differs from the selected run")
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50", weights=None, trainable_layers=3,
        norm_layer=partial(FrozenBatchNorm2d, eps=0.0))
    detector = FasterRCNN(backbone, num_classes=91, min_size=512, max_size=512)
    detector.roi_heads.box_predictor = FastRCNNPredictor(
        detector.roi_heads.box_predictor.cls_score.in_features, 2)
    detector.load_state_dict(detector_checkpoint["model"], strict=True)
    classifier = resnet18(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, 1)
    classifier.load_state_dict(classifier_checkpoint["model_state_dict"], strict=True)
    for model in (detector, classifier):
        model.requires_grad_(False)
        model.eval().to(device)
    if (detector.roi_heads.score_thresh, detector.roi_heads.nms_thresh,
            detector.roi_heads.detections_per_img) != (0.05, 0.5, 100):
        raise ValueError("Unexpected detector inference settings")
    return detector, classifier, dict(
        detector_checkpoint_sha256=sha256(detector_path),
        classifier_checkpoint_sha256=sha256(classifier_path),
        detector_state_sha256=state_sha256(detector),
        classifier_state_sha256=state_sha256(classifier),
        classifier_settings=settings,
        classifier_validation_loss=classifier_checkpoint["best_validation_loss"])


def ranking_metrics(rows, score_key, split):
    ranked = sorted(rows, key=lambda row: (-row[score_key], row["image_id"]))
    positive = sum(row["target"] for row in rows)
    negative = len(rows) - positive
    budget = len(rows) // 5
    ascending = sorted(rows, key=lambda row: row[score_key])
    rank_sum, start = 0.0, 0
    while start < len(rows):
        end = start + 1
        while end < len(rows) and ascending[end][score_key] == ascending[start][score_key]:
            end += 1
        rank_sum += (start + 1 + end) / 2 * sum(row["target"] for row in ascending[start:end])
        start = end
    selected = ranked[:budget]
    counts = Counter(row["class_label"] for row in selected)
    recalled = sum(row["target"] for row in selected)
    metrics = dict(method=score_key, split=split, images=len(rows), positives=positive,
                   review_count=budget, recalled_positives=recalled,
                   recall_at20=recalled / positive,
                   auroc=(rank_sum - positive * (positive + 1) / 2) / (positive * negative),
                   top20_other_abnormal=counts.get(OTHER, 0), top20_normal=counts.get("Normal", 0),
                   fp_other_abnormal_030=sum(row["class_label"] == OTHER and row[score_key] >= .3 for row in rows),
                   fp_normal_030=sum(row["class_label"] == "Normal" and row[score_key] >= .3 for row in rows))
    saved = [dict(row, rank=i + 1, selected_top20=i < budget) for i, row in enumerate(ranked)]
    return metrics, saved


def coco_metrics(raw):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images, truth_boxes, predictions = [], [], []
    for row in raw:
        image_id = row["numeric_id"]
        images.append(dict(id=image_id, width=512, height=512))
        for box in row["gt_boxes"]:
            x1, y1, x2, y2 = box
            truth_boxes.append(dict(id=len(truth_boxes) + 1, image_id=image_id, category_id=1,
                                    bbox=[x1, y1, x2-x1, y2-y1], area=(x2-x1)*(y2-y1), iscrowd=0))
        for box, score in zip(row["boxes"], row["box_scores"]):
            x1, y1, x2, y2 = box
            predictions.append(dict(id=len(predictions) + 1, image_id=image_id, category_id=1,
                                    bbox=[x1, y1, x2-x1, y2-y1], area=(x2-x1)*(y2-y1),
                                    iscrowd=0, score=score))
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
    return dict(raw_detector_coco_map=float(evaluator.stats[0]),
                raw_detector_ap50=float(evaluator.stats[1]),
                raw_detector_ap75=float(evaluator.stats[2]),
                fusion_changes_box_scores=False, fusion_changes_boxes=False)


@torch.inference_mode()
def evaluate_split(detector, classifier, dataset, device, output, workers):
    output.mkdir()
    detector.eval()
    classifier.eval()
    records, raw = {}, []
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=workers,
                        collate_fn=collate, pin_memory=True)
    for step, (images, metadata) in enumerate(loader, 1):
        outputs = detector([image.to(device) for image in images])
        for meta, prediction in zip(metadata, outputs):
            scores = prediction["scores"].cpu().tolist()
            row = dict(image_id=meta["image_id"], patient_id=meta["nih_patient_id"],
                       target=int(meta["target"]), class_label=meta["class_label"],
                       png_sha256=meta["png_sha256"], detector=max(scores, default=0.0))
            records[row["image_id"]] = row
            raw.append(dict(image_id=meta["image_id"], numeric_id=meta["numeric_id"],
                            gt_boxes=meta["gt_boxes"], boxes=prediction["boxes"].cpu().tolist(),
                            box_scores=scores, box_labels=prediction["labels"].cpu().tolist()))
        if step % 50 == 0 or len(records) == len(dataset):
            print(f"{output.name} detector {len(records)}/{len(dataset)}", flush=True)
    normalizer = Normalize((.485, .456, .406), (.229, .224, .225))
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=workers,
                        collate_fn=collate, pin_memory=True)
    seen = 0
    for images, metadata in loader:
        inputs = normalizer(torch.stack(images)).to(device)
        probabilities = classifier(inputs).squeeze(1).sigmoid().cpu().tolist()
        for meta, probability in zip(metadata, probabilities):
            row = records[meta["image_id"]]
            if row["png_sha256"] != meta["png_sha256"]:
                raise ValueError("Models did not read identical image bytes")
            row["classifier"] = probability
            row["fusion_80_20"] = (1 - ALPHA) * row["detector"] + ALPHA * probability
            seen += 1
        if seen % 800 == 0 or seen == len(dataset):
            print(f"{output.name} classifier {seen}/{len(dataset)}", flush=True)
    rows = list(records.values())
    if len(rows) != len(dataset) or seen != len(dataset):
        raise ValueError("Incomplete split evaluation")
    if any(not math.isfinite(row[name]) or not 0 <= row[name] <= 1 for row in rows for name in METHODS):
        raise ValueError("Invalid inference score")
    write_csv(output / "image_scores.csv", rows)
    with (output / "raw_detector_predictions.jsonl").open("w") as target:
        for row in raw:
            target.write(json.dumps(row, allow_nan=False) + "\n")
    summary = []
    for method in METHODS:
        metrics, ranked = ranking_metrics(rows, method, output.name)
        summary.append(metrics)
        write_csv(output / f"{method}_ranked.csv", ranked)
    detection = coco_metrics(raw)
    write_csv(output / "comparison.csv", summary)
    write_json(output / "raw_detection_metrics.json", detection)
    return rows, summary, raw, detection


def validate_reference(rows, summary, detection, reference_path):
    saved_rows = read_csv(reference_path)
    semantic_rows = [[row["image_id"], row["patient_id"], int(row["target"]), float(row["probability"])]
                     for row in sorted(saved_rows, key=lambda row: row["image_id"])]
    semantic_sha = hashlib.sha256(json.dumps(semantic_rows, separators=(",", ":")).encode()).hexdigest()
    if semantic_sha != CLASSIFIER_REFERENCE_SEMANTIC_SHA:
        raise ValueError("Original classifier reference scores differ from the verified baseline epoch-2 artifact")
    saved = {row["image_id"]: row for row in saved_rows}
    if len(saved) != 4271 or len(saved_rows) != len(saved) or set(saved) != {row["image_id"] for row in rows}:
        raise ValueError("Original classifier reference is not the full matching validation split")
    differences = []
    for row in rows:
        reference = saved[row["image_id"]]
        if int(reference["target"]) != row["target"] or reference["patient_id"] != row["patient_id"]:
            raise ValueError("Original classifier reference metadata mismatch")
        differences.append(abs(float(reference["probability"]) - row["classifier"]))
    if max(differences) > 1e-5:
        raise ValueError(f"Classifier validation probabilities differ: max_abs={max(differences)}")
    expected = {
        "detector": (591, 260, 3, .8938266952290278),
        "classifier": (568, 280, 6, .8808314820691844),
        "fusion_80_20": (596, 256, 2, .8968902540),
    }
    for metric in summary:
        recall_count, other, normal, auc = expected[metric["method"]]
        if (metric["images"], metric["positives"], metric["review_count"],
                metric["recalled_positives"], metric["top20_other_abnormal"], metric["top20_normal"]) != (
                4271, 959, 854, recall_count, other, normal):
            raise ValueError(f"Validation reference counts differ: {metric}")
        if abs(metric["auroc"] - auc) > 1e-6:
            raise ValueError(f"Validation AUROC differs: {metric}")
    detector = summary[0]
    if (detector["fp_other_abnormal_030"], detector["fp_normal_030"]) != (1099, 124):
        raise ValueError("Detector validation threshold false positives differ")
    if abs(detection["raw_detector_ap50"] - .4012761237282653) > 1e-5:
        raise ValueError("Detector validation AP50 differs")
    return dict(passed=True, validation_images=4271, reference_csv_sha256=sha256(reference_path),
                reference_csv_semantic_sha256=semantic_sha,
                classifier_max_abs_probability_difference=max(differences),
                classifier_probability_abs_tolerance=1e-5, auroc_abs_tolerance=1e-6,
                raw_detector_ap50_abs_tolerance=1e-5, exact_review_counts=True,
                test_inference_started=False)


def figures(data_dir, test_rows, summary, raw, selected_cases, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    names = ["Detector", "ResNet baseline e2", "Fixed 80:20"]
    colors = ["#0072B2", "#E69F00", "#009E73"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, key, title in zip(axes,
                             ["recall_at20", "top20_other_abnormal", "fp_other_abnormal_030"],
                             ["Recall@20%", "Other Abnormal in top 919", "Other image flags at score >= 0.30"]):
        values = [100 * row[key] if key == "recall_at20" else row[key] for row in summary]
        bars = ax.bar(names, values, color=colors)
        ax.bar_label(bars, labels=[f"{v:.2f}%" if key == "recall_at20" else str(v) for v in values], padding=3)
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fixed test evaluation: 4,595 images; no fitting or weight search")
    fig.savefig(output / "test_comparison.png", dpi=300)
    fig.savefig(output / "test_comparison.pdf")
    plt.close(fig)
    scores = {row["image_id"]: row for row in test_rows}
    predictions = {row["image_id"]: row for row in raw}
    ranks = {name: {row["image_id"]: row["rank"] for row in ranking_metrics(test_rows, name, "test")[1]}
             for name in METHODS}
    fig, axes = plt.subplots(3, 2, figsize=(11, 16), layout="constrained")
    for ax, metadata in zip(axes.flat, selected_cases):
        image_id = metadata["image_id"]
        with Image.open(data_dir / metadata["processed_path"]) as source:
            ax.imshow(source.convert("RGB"))
        for box in predictions[image_id]["gt_boxes"]:
            x1, y1, x2, y2 = box
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="#00cc44", linewidth=1.5))
        for box, score in zip(predictions[image_id]["boxes"], predictions[image_id]["box_scores"]):
            if score < .3:
                continue
            x1, y1, x2, y2 = box
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="#ff9900", linewidth=1.2))
            ax.text(x1, y1, f"{score:.2f}", color="#ff9900", fontsize=7,
                    bbox=dict(facecolor="black", alpha=.5, pad=.3))
        row = scores[image_id]
        ax.set_title(f"{metadata['class_label']} | {image_id[:13]}\n"
                     f"D {row['detector']:.3f} / C {row['classifier']:.3f} / fusion {row['fusion_80_20']:.3f}\n"
                     f"Rank D {ranks['detector'][image_id]} -> fusion {ranks['fusion_80_20'][image_id]} (budget 919)", fontsize=9)
        ax.axis("off")
    fig.suptitle("Seed-42 test cases selected before inference; green GT / orange unchanged detector boxes >= 0.30")
    fig.savefig(output / "test_fixed_cases.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory; inspect existing results before rerunning")
    if not torch.cuda.is_available():
        raise RuntimeError("Select an available Kaggle GPU for full inference")
    if (str(torch.__version__), str(torchvision.__version__), torch.version.cuda,
            importlib.metadata.version("pycocotools")) != ("2.11.0+cu128", "0.26.0+cu128", "12.8", "2.0.11"):
        raise RuntimeError("Use the historical PyTorch 2.11.0/cu128, torchvision 0.26.0/cu128 and pycocotools 2.0.11")
    if args.classifier_reference != args.classifier_checkpoint.parent / "best_validation_predictions.csv":
        raise ValueError("Use the original classifier checkpoint's sibling validation predictions")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    print("Checking pinned prepared metadata, patient splits and selected checkpoints.", flush=True)
    labels, boxes, data_report = validate_data(args.data_dir)
    detector, classifier, checkpoints = load_models(args.detector_checkpoint, args.classifier_checkpoint, device)
    print("Source and checkpoint checks PASS; both models frozen in eval mode.", flush=True)
    args.output.mkdir(parents=True)
    config = dict(alpha=ALPHA, formula="0.8 * detector_max_box_score + 0.2 * classifier_probability",
                  source_versions=SOURCE_VERSIONS, source_paths={key: str(value) for key, value in vars(args).items()
                                                               if isinstance(value, Path)},
                  detector_batch_size=8, classifier_batch_size=16, workers=args.workers,
                  seed=42, precision="float32, no autocast", detector_mode="frozen eval",
                  classifier_preprocessing="RGB 512 PNG; ToTensor; ImageNet mean/std; no resize or augmentation",
                  detector_preprocessing="RGB 512 PNG; tensor [0,1]; native detector normalization",
                  ranking="descending score, then ascending image_id; floor(N/5) review slots",
                  fixed_threshold=.3, new_training=False, parameter_fitting=False, weight_search=False,
                  detection="Fusion changes image ranking only, not boxes or individual box scores",
                  limitation="Test confirmation of a previously fixed rule; historical project test evaluation exists",
                  torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                  gpu=torch.cuda.get_device_name(), code_sha256=sha256(__file__),
                  data=data_report, checkpoints=checkpoints)
    write_json(args.output / "config.json", config)
    rng = random.Random(42)
    cases = []
    for label in ("Lung Opacity", OTHER, "Normal"):
        candidates = sorted([row for row in labels if row["split"] == "test" and row["class_label"] == label],
                            key=lambda row: row["image_id"])
        cases.extend(rng.sample(candidates, 2))
    write_json(args.output / "fixed_test_case_ids.json", [row["image_id"] for row in cases])
    validation = PreparedImages(args.data_dir, labels, boxes, "val")
    rows, summary, _, detection = evaluate_split(detector, classifier, validation, device,
                                                 args.output / "val", args.workers)
    reference = validate_reference(rows, summary, detection, args.classifier_reference)
    write_json(args.output / "validation_reference_passed.json", reference)
    print("Full validation reference PASS. Starting the fixed test evaluation.", flush=True)
    test = PreparedImages(args.data_dir, labels, boxes, "test")
    rows, summary, raw, detection = evaluate_split(detector, classifier, test, device,
                                                   args.output / "test", args.workers)
    if any(row["images"] != 4595 or row["review_count"] != 919 for row in summary):
        raise ValueError("Incomplete test evaluation or incorrect fixed review budget")
    if state_sha256(detector) != checkpoints["detector_state_sha256"] or state_sha256(classifier) != checkpoints["classifier_state_sha256"]:
        raise ValueError("Inference changed model parameters or buffers")
    figures(args.data_dir, rows, summary, raw, cases, args.output)
    write_json(args.output / "completed.json", dict(status="complete", test_evaluated=True,
               validation_reference_passed=True, frozen_models_unchanged=True, alpha=ALPHA,
               source_versions=SOURCE_VERSIONS, test_metrics=summary, raw_detection=detection))
    manifest = {str(path.relative_to(args.output)): sha256(path)
                for path in sorted(args.output.rglob("*")) if path.is_file()}
    write_json(args.output / "sha256.json", manifest)
    print(json.dumps(dict(test_metrics=summary, raw_detection=detection), indent=2), flush=True)
    print("COMPLETE: fixed test evaluation; no training or tuning performed.", flush=True)


if __name__ == "__main__":
    main()
