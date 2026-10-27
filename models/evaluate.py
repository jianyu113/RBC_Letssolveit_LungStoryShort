"""Evaluate ResNet-18 or Faster R-CNN on the complete validation split.

Inputs: prepared labels.csv and Crystal's best_validation_predictions.csv.
All validation images must have exactly one prediction. Test images are excluded.
Classification accepts the existing prediction CSV. Detection accepts JSON from
export_detection_predictions plus prepared boxes.csv. Box metrics use COCO bbox
AP, not the RSNA competition metric. Test evaluation is intentionally excluded.
Image metrics are not patient metrics; scores are not calibrated probabilities.
"""

import argparse
import hashlib
import json
import math
import io
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def validate_predictions(labels, predictions, score_column="probability"):
    """Match complete validation predictions against authoritative labels."""
    expected_columns = {"image_id", "nih_patient_id", "target", "split"}
    prediction_columns = {"image_id", "patient_id", "target", score_column}
    for frame, required, name in (
        (labels, expected_columns, "labels"),
        (predictions, prediction_columns, "predictions"),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"Missing values in required {name} columns.")
        if not frame["image_id"].is_unique:
            raise ValueError(f"Duplicate image IDs in {name}.")

    expected = labels.loc[labels["split"] == "val"].set_index("image_id")
    if set(expected["target"].unique()) != {0, 1}:
        raise ValueError("Validation labels must contain both binary classes.")
    if set(predictions["image_id"]) != set(expected.index):
        missing = set(expected.index) - set(predictions["image_id"])
        extra = set(predictions["image_id"]) - set(expected.index)
        raise ValueError(
            f"Validation image IDs differ: {len(missing)} missing, {len(extra)} extra."
        )

    matched = predictions.set_index("image_id")
    truth = expected.loc[matched.index]
    if not np.array_equal(matched["target"].to_numpy(), truth["target"].to_numpy()):
        raise ValueError("Prediction targets do not match prepared labels.")
    if not np.array_equal(
        matched["patient_id"].to_numpy(), truth["nih_patient_id"].to_numpy()
    ):
        raise ValueError("Prediction patient IDs do not match prepared labels.")
    train_patients = set(labels.loc[labels["split"] == "train", "nih_patient_id"])
    if train_patients & set(expected["nih_patient_id"]):
        raise ValueError("Patients overlap between training and validation.")

    scores = pd.to_numeric(matched[score_column], errors="raise").to_numpy(float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite values in [0, 1].")

    return pd.DataFrame({
        "image_id": matched.index.to_numpy(),
        "patient_id": truth["nih_patient_id"].to_numpy(),
        "target": truth["target"].to_numpy(dtype=int),
        "score": scores,
    })


def threshold_metrics(data, threshold):
    """Use score >= threshold; undefined precision is reported as zero."""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between zero and one.")
    targets = data["target"].to_numpy()
    predicted = data["score"].to_numpy() >= threshold
    tn, fp, fn, tp = confusion_matrix(targets, predicted, labels=[0, 1]).ravel()
    return {
        "decision_threshold": float(threshold),
        "accuracy": float((tp + tn) / len(data)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "flagged_images": int(tp + fp),
        "flagged_fraction": float((tp + fp) / len(data)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_predictions(data, threshold=0.5, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """Compute threshold and ranking metrics with a fixed tie-breaking rule."""
    ranked = data.sort_values(
        ["score", "image_id"], ascending=[False, True]
    ).reset_index(drop=True)
    total = len(ranked)
    positive_count = int(ranked["target"].sum())
    review_count = math.ceil(0.20 * total)
    top_positive_count = int(ranked.head(review_count)["target"].sum())

    metrics = {
        "split": "val",
        "evaluation_unit": "image",
        "validation_images": total,
        "validation_patients": int(data["patient_id"].nunique()),
        "positive_images": positive_count,
        "positive_prevalence": positive_count / total,
        **threshold_metrics(data, threshold),
        "auroc": float(roc_auc_score(data["target"], data["score"])),
        "average_precision": float(
            average_precision_score(data["target"], data["score"])
        ),
        "top20_review_count": review_count,
        "top20_actual_review_fraction": review_count / total,
        "top20_positive_count": top_positive_count,
        "top20_recall": top_positive_count / positive_count,
        "ranking_ties": "score descending, image_id ascending",
        "precision_zero_division": 0,
        "test_set_used": False,
    }

    reviewed = np.arange(total + 1)
    found = np.r_[0, ranked["target"].to_numpy().cumsum()]
    budget_curve = pd.DataFrame({
        "review_count": reviewed,
        "review_fraction": reviewed / total,
        "positive_found": found,
        "recall": found / positive_count,
    })

    precision, recall, pr_thresholds = precision_recall_curve(
        data["target"], data["score"]
    )
    pr_curve = pd.DataFrame({
        "threshold": np.r_[pr_thresholds, np.nan],
        "precision": precision,
        "recall": recall,
    })
    # The final PR endpoint has no associated threshold.
    threshold_table = pd.DataFrame([
        threshold_metrics(data, value)
        for value in sorted(set([threshold, *thresholds]))
    ])

    ranked.insert(0, "rank", np.arange(1, total + 1))
    ranked["prediction"] = (ranked["score"] >= threshold).astype(int)
    ranked["in_top20"] = ranked["rank"] <= review_count
    return metrics, ranked, threshold_table, budget_curve, pr_curve


def export_detection_predictions(model, loader, output_path, device):
    """Export this repo's RSNADetectionDataset outputs, including empty images.

    Use a complete validation loader, not a Subset. Numeric tensor IDs are
    mapped through dataset.records to the original image and patient IDs.
    This runs inference only; it does not load a checkpoint or train the model.
    """
    import torch

    records = loader.dataset.records
    if not records or any(row["split"] != "val" for row in records):
        raise ValueError("Export requires a validation RSNADetectionDataset.")
    lookup = {int(row["numeric_image_id"]): row for row in records}
    images = []
    model.eval()
    with torch.inference_mode():
        for batch, targets in loader:
            outputs = model([image.to(device) for image in batch])
            for target, output in zip(targets, outputs):
                row = lookup[int(target["image_id"].item())]
                images.append({
                    "image_id": str(row["image_id"]),
                    "patient_id": str(row["nih_patient_id"]),
                    "boxes": output["boxes"].detach().cpu().tolist(),
                    "scores": output["scores"].detach().cpu().tolist(),
                    "labels": output["labels"].detach().cpu().tolist(),
                })
    payload = {
        "split": "val", "box_format": "xyxy",
        "coordinate_space": "processed_pixels", "images": images,
        "postprocessing": {
            "score_threshold": float(model.roi_heads.score_thresh),
            "nms_threshold": float(model.roi_heads.nms_thresh),
            "detections_per_image": int(model.roi_heads.detections_per_img),
        },
    }
    output_path = Path(output_path)
    with output_path.open("x") as handle:
        json.dump(payload, handle, allow_nan=False)
    return output_path


def validate_xyxy(boxes, width, height, name):
    """Check pixel-edge boxes without silently clipping or changing them."""
    boxes = np.asarray(boxes, dtype=float)
    if boxes.shape == (0,):
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError(f"{name}: boxes must be finite Nx4 xyxy coordinates.")
    if ((boxes[:, :2] < 0).any()
            or (boxes[:, 2] > width).any() or (boxes[:, 3] > height).any()
            or (boxes[:, 2:] <= boxes[:, :2]).any()):
        raise ValueError(f"{name}: boxes must have positive area and fit the processed image.")
    return boxes


def detection_inputs(labels, boxes, payload):
    """Build authoritative COCO truth and aligned image-level predictions."""
    if (payload.get("split") != "val" or payload.get("box_format") != "xyxy"
            or payload.get("coordinate_space") != "processed_pixels"):
        raise ValueError("Detection JSON must specify val, xyxy, and processed_pixels.")
    records = payload["images"]
    if not records:
        raise ValueError("Include every validation image, even images with no detections.")
    expected = labels.loc[labels["split"] == "val"].set_index("image_id")
    image_rows = []
    for row in records:
        image_id = row["image_id"]
        if image_id not in expected.index:
            raise ValueError(f"Non-validation image in predictions: {image_id}")
        scores = np.asarray(row["scores"], dtype=float)
        categories = np.asarray(row["labels"])
        if (scores.ndim != 1 or categories.ndim != 1
                or len(scores) != len(row["boxes"]) or len(categories) != len(scores)):
            raise ValueError(f"{image_id}: boxes, scores, and labels must have equal lengths.")
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError(f"{image_id}: invalid detection scores.")
        if (categories != 1).any():
            raise ValueError("The only detection category is Lung Opacity (label 1).")
        image_rows.append({
            "image_id": image_id, "patient_id": row["patient_id"],
            "target": expected.loc[image_id, "target"],
            "probability": float(scores.max()) if len(scores) else 0.0,
        })
    # Reuse the same full-split, patient-ID, and leakage checks for both models.
    data = validate_predictions(labels, pd.DataFrame(image_rows))
    coordinates = ["processed_x_min", "processed_y_min", "processed_x_max", "processed_y_max"]
    required = {"image_id", *coordinates}
    if not required.issubset(boxes.columns):
        raise ValueError(f"Missing boxes.csv columns: {sorted(required - set(boxes.columns))}")
    if boxes["image_id"].isna().any() or not set(boxes["image_id"]).issubset(set(labels["image_id"])):
        raise ValueError("boxes.csv contains missing or unknown image IDs.")
    selected_boxes = boxes.loc[boxes["image_id"].isin(expected.index)]
    grouped = {key: group[coordinates].to_numpy(float)
               for key, group in selected_boxes.groupby("image_id")}
    records_by_id = {row["image_id"]: row for row in records}
    images, annotations, detections = [], [], []
    for numeric_id, image_id in enumerate(sorted(expected.index), start=1):
        truth = expected.loc[image_id]
        width, height = float(truth["processed_width"]), float(truth["processed_height"])
        if not all(math.isfinite(x) and x > 0 and x.is_integer() for x in (width, height)):
            raise ValueError(f"{image_id}: invalid processed image dimensions.")
        images.append({"id": numeric_id, "width": int(width), "height": int(height)})
        gt = validate_xyxy(grouped.get(image_id, []), width, height, image_id)
        if bool(len(gt)) != bool(truth["target"]):
            raise ValueError(f"{image_id}: ground-truth boxes disagree with the binary target.")
        row = records_by_id[image_id]
        predicted = validate_xyxy(row["boxes"], width, height, image_id)
        for target_list, items, scores in (
            (annotations, gt, [None] * len(gt)),
            (detections, predicted, row["scores"]),
        ):
            for (x1, y1, x2, y2), score in zip(items, scores):
                record = {
                    "id": len(target_list) + 1, "image_id": numeric_id, "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0,
                }
                if score is not None:
                    record["score"] = float(score)
                target_list.append(record)
    return data, images, annotations, detections


def evaluate_detection(images, annotations, detections):
    """Use official COCO matching, 101 recall points and maxDets=100."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with redirect_stdout(io.StringIO()):
        truth, prediction = COCO(), COCO()
        for coco, records in ((truth, annotations), (prediction, detections)):
            coco.dataset = {
                "images": images, "annotations": records,
                "categories": [{"id": 1, "name": "Lung Opacity"}],
            }
            coco.createIndex()
        evaluator = COCOeval(truth, prediction, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    names = ["map", "map_50", "map_75", "map_small", "map_medium", "map_large",
             "mar_1", "mar_10", "mar_100", "mar_small", "mar_medium", "mar_large"]
    metrics = {name: float(value) if value >= 0 else None
               for name, value in zip(names, evaluator.stats)}
    return {
        "split": "val", "evaluation_unit": "box", "metric_standard": "COCO bbox",
        "images": len(images), "ground_truth_boxes": len(annotations),
        "submitted_prediction_boxes": len(detections),
        "iou_thresholds": evaluator.params.iouThrs.tolist(),
        "max_detections": evaluator.params.maxDets,
        "area_units": "processed pixel squared",
        "undefined_metrics": "null means no ground-truth boxes in that area range",
        "test_set_used": False, **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--score-column", default="probability")
    parser.add_argument("--task", choices=["classification", "detection"], default="classification")
    parser.add_argument("--boxes", type=Path, help="Prepared boxes.csv; required for detection")
    args = parser.parse_args()
    if args.task == "detection" and args.boxes is None:
        parser.error("--boxes is required for detection")

    labels = pd.read_csv(args.labels, dtype={
        "image_id": "string", "nih_patient_id": "string", "split": "string",
    })
    detection_metrics = None
    if args.task == "classification":
        predictions = pd.read_csv(args.predictions, dtype={
            "image_id": "string", "patient_id": "string",
        })
        data = validate_predictions(labels, predictions, args.score_column)
    else:
        boxes = pd.read_csv(args.boxes, dtype={"image_id": "string"})
        payload = json.loads(args.predictions.read_text())
        data, images, annotations, detections = detection_inputs(labels, boxes, payload)
        detection_metrics = evaluate_detection(images, annotations, detections)
    metrics, ranked, thresholds, budget_curve, pr_curve = evaluate_predictions(
        data, args.threshold, args.thresholds
    )

    # A new directory preserves earlier evaluations.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    provenance = {
        "task": args.task,
        "labels": str(args.labels.resolve()),
        "predictions": str(args.predictions.resolve()),
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "score_column": args.score_column,
        "threshold": args.threshold,
        "thresholds": thresholds["decision_threshold"].tolist(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "note": "Validation analysis only; no threshold is automatically selected.",
    }
    if detection_metrics is not None:
        (args.output_dir / "detection_metrics.json").write_text(
            json.dumps(detection_metrics, indent=2, allow_nan=False) + "\n"
        )
        provenance.update({
            "boxes": str(args.boxes.resolve()),
            "boxes_sha256": hashlib.sha256(args.boxes.read_bytes()).hexdigest(),
            "pycocotools_version": version("pycocotools"),
            "score_column": None,
            "image_score": "Maximum returned lung-opacity score; zero for no detections",
            "box_format": "xyxy", "coordinate_space": "processed_pixels",
            "postprocessing": payload.get("postprocessing"),
            "box_score_filter": "None added by evaluator; exporter retains model postprocessing",
            "threshold_scope": "Image classification only; COCO AP uses ranked detections",
        })
    (args.output_dir / "settings.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["actual_negative", "actual_positive"],
        columns=["predicted_negative", "predicted_positive"],
    ).to_csv(args.output_dir / "confusion_matrix.csv", index_label="actual_class")
    thresholds.to_csv(args.output_dir / "threshold_analysis.csv", index=False)
    budget_curve.to_csv(args.output_dir / "recall_budget_curve.csv", index=False)
    pr_curve.to_csv(args.output_dir / "precision_recall_curve.csv", index=False)
    ranked.to_csv(args.output_dir / "ranked_predictions.csv", index=False)
    ranked.loc[ranked["target"] != ranked["prediction"]].to_csv(
        args.output_dir / "incorrect_predictions.csv", index=False
    )
    print(json.dumps(metrics, indent=2))
    if detection_metrics is not None:
        print(json.dumps(detection_metrics, indent=2))
    print(f"Saved evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
