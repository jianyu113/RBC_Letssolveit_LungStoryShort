"""Prepared-image loaders for the original, unaugmented ResNet-18 recipe."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REQUIRED_COLUMNS = {"image_id", "nih_patient_id", "target", "split", "processed_path"}


def create_classification_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_prepared_labels(prepared_dir):
    labels = pd.read_csv(
        Path(prepared_dir) / "labels.csv",
        dtype={"image_id": "string", "nih_patient_id": "string", "split": "string"},
    )
    missing = REQUIRED_COLUMNS - set(labels.columns)
    if missing:
        raise ValueError(f"Missing required label columns: {sorted(missing)}")
    return labels


class RSNAClassificationDataset(Dataset):
    def __init__(self, labels, prepared_dir, split, transform):
        if split not in {"train", "val"}:
            raise ValueError("Classification training only supports 'train' and 'val'.")
        self.rows = labels.loc[labels["split"] == split].reset_index(drop=True).copy()
        if self.rows.empty:
            raise ValueError(f"No images found for split '{split}'.")
        self.prepared_dir = Path(prepared_dir)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(self.prepared_dir / row["processed_path"]) as image_file:
            image = image_file.convert("RGB")
        return {
            "image": self.transform(image),
            "label": torch.tensor(float(row["target"]), dtype=torch.float32),
            "image_id": str(row["image_id"]),
            "patient_id": str(row["nih_patient_id"]),
        }


def create_classification_loaders(prepared_dir, batch_size=16, num_workers=2, seed=42):
    labels = load_prepared_labels(prepared_dir)
    transform = create_classification_transform()
    train_dataset = RSNAClassificationDataset(labels, prepared_dir, "train", transform)
    validation_dataset = RSNAClassificationDataset(labels, prepared_dir, "val", transform)
    train_generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, generator=train_generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, validation_loader
