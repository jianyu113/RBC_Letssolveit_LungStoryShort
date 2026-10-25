"""Evaluate image-level validation predictions for LungStoryShort.

Inputs: prepared labels.csv and Crystal's best_validation_predictions.csv.
All validation images must have exactly one prediction. Test images are excluded.
Metrics are image-level, not patient-level. No confidence intervals or box AP
are computed. Scores are model outputs, not calibrated disease probabilities.
"""

import argparse
import hashlib
import json
import math
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--score-column", default="probability")
    args = parser.parse_args()

    labels = pd.read_csv(args.labels, dtype={
        "image_id": "string", "nih_patient_id": "string", "split": "string",
    })
    predictions = pd.read_csv(args.predictions, dtype={
        "image_id": "string", "patient_id": "string",
    })
    data = validate_predictions(labels, predictions, args.score_column)
    metrics, ranked, thresholds, budget_curve, pr_curve = evaluate_predictions(
        data, args.threshold, args.thresholds
    )

    # A new directory preserves earlier evaluations.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    provenance = {
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
    print(f"Saved evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
