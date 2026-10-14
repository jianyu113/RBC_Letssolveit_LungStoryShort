"""Create or verify the fixed patient-level splits for the development sample."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "rsna_dev_100"


def make_splits(manifest: pd.DataFrame) -> pd.DataFrame:
    """Keep each NIH patient's images together and stratify by patient target."""
    patient_targets = manifest.groupby("nih_patient_id", sort=True)["Target"].max()
    train_ids, remaining_ids = train_test_split(
        patient_targets.index,
        test_size=0.30,
        stratify=patient_targets,
        random_state=SEED,
    )
    val_ids, test_ids = train_test_split(
        remaining_ids,
        test_size=0.50,
        stratify=patient_targets.loc[remaining_ids],
        random_state=SEED,
    )

    patient_splits = {}
    for name, patient_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        patient_splits.update({patient_id: name for patient_id in patient_ids})

    splits = manifest[["image_id", "nih_patient_id"]].copy()
    splits["split"] = splits["nih_patient_id"].map(patient_splits)
    return splits.sort_values("image_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Sample folder containing sample_ids.csv (default: dataset/rsna_dev_100).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the existing splits.csv without changing it.",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(
        args.data_dir / "sample_ids.csv",
        usecols=["image_id", "nih_patient_id", "Target"],
        dtype={"image_id": str, "nih_patient_id": str, "Target": int},
    )
    splits = make_splits(manifest)
    output = args.data_dir / "splits.csv"
    csv_text = splits.to_csv(index=False, lineterminator="\n")

    if args.check:
        if output.read_text(encoding="utf-8") != csv_text:
            raise ValueError("splits.csv does not match the fixed patient-level split.")
        print("Verified splits.csv against sample_ids.csv and seed 42.")
    else:
        with output.open("x", encoding="utf-8", newline="") as handle:
            handle.write(csv_text)
        print(f"Created {output}")

    summary = splits.merge(manifest, on=["image_id", "nih_patient_id"])
    counts = summary.groupby("split").agg(
        patients=("nih_patient_id", "nunique"),
        images=("image_id", "size"),
        positive=("Target", "sum"),
    )
    counts["negative"] = counts["images"] - counts["positive"]
    print(counts.loc[["train", "val", "test"]].to_string())


if __name__ == "__main__":
    main()
