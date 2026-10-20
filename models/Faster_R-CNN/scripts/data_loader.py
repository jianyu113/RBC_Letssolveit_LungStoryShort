"""Load prepared RSNA PNGs and detection targets using the saved splits."""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RSNADetectionDataset(Dataset):
    """One foreground class: lung opacity (1); background is reserved as 0."""

    def __init__(self, data_dir, split):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.data_dir = Path(data_dir)
        self.split = split

        with (self.data_dir / "labels.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "image_id", "nih_patient_id", "split",
                "processed_path", "image_status",
            }
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Use labels.csv from the prepared data directory")
            rows = list(reader)

        patient_splits = {}
        for row in rows:
            patient = row["nih_patient_id"]
            assigned = patient_splits.setdefault(patient, row["split"])
            if assigned != row["split"]:
                raise ValueError(f"Patient {patient} appears in multiple splits")
            if row["image_status"] != "decoded" or not row["processed_path"]:
                raise ValueError(f"Image {row['image_id']} has not been prepared")

        # Keep numeric IDs stable across train, validation, and test datasets.
        self.records = [
            dict(row, numeric_image_id=index)
            for index, row in enumerate(rows)
            if row["split"] == split
        ]
        if not self.records:
            raise ValueError(f"No images found for split {split}")

        self.boxes = {}
        coordinates = (
            "processed_x_min", "processed_y_min",
            "processed_x_max", "processed_y_max",
        )
        with (self.data_dir / "boxes.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                self.boxes.setdefault(row["image_id"], []).append(
                    [float(row[column]) for column in coordinates]
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(self.data_dir / row["processed_path"]) as image:
            pixels = np.array(image.convert("RGB"), dtype=np.uint8)
        image = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255)
        boxes = torch.tensor(
            self.boxes.get(row["image_id"], []), dtype=torch.float32
        ).reshape(-1, 4)
        target = {
            "boxes": boxes,
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor(row["numeric_image_id"], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    """Keep variable-length box tensors separate for each image."""
    images, targets = zip(*batch)
    return list(images), list(targets)
