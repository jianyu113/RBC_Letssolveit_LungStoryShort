from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn.functional as functional


CSV_FIELDS = [
    "image_id",
    "sop_instance_uid",
    "source_path",
    "readable",
    "original_height",
    "original_width",
    "pixel_dtype",
    "pixel_min",
    "pixel_max",
    "photometric_interpretation",
    "bits_stored",
    "pixel_representation",
    "rescale_slope",
    "rescale_intercept",
    "inverted",
    "clip_low",
    "clip_high",
    "processed_height",
    "processed_width",
    "scale_x",
    "scale_y",
    "pad_left",
    "pad_top",
    "issue",
]


def load_dicom_pixels(path: Path) -> tuple[pydicom.Dataset, np.ndarray]:
    """Read one DICOM file and return its dataset and raw pixel array."""
    dataset = pydicom.dcmread(path)
    pixels = dataset.pixel_array

    if pixels.ndim != 2:
        raise ValueError(f"Expected a 2D image, received shape {pixels.shape}")

    return dataset, pixels


def apply_intensity_processing(
    pixels: np.ndarray,
    dataset: pydicom.Dataset,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Convert DICOM pixels to a normalized float32 image."""
    image = pixels.astype(np.float32)

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    image = image * slope + intercept

    photometric = str(getattr(dataset, "PhotometricInterpretation", ""))
    inverted = photometric == "MONOCHROME1"

    if inverted:
        image = image.max() + image.min() - image

    clip_low = float(np.percentile(image, lower_percentile))
    clip_high = float(np.percentile(image, upper_percentile))

    if clip_high <= clip_low:
        raise ValueError("Image has no usable intensity range")

    image = np.clip(image, clip_low, clip_high)
    image = (image - clip_low) / (clip_high - clip_low)

    metadata = {
        "rescale_slope": slope,
        "rescale_intercept": intercept,
        "inverted": inverted,
        "clip_low": clip_low,
        "clip_high": clip_high,
    }

    return image.astype(np.float32), metadata


def resize_and_pad(
    image: np.ndarray,
    target_size: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Resize and pad to a square, returning scales for the rounded dimensions."""
    original_height, original_width = image.shape
    scale = min(
        target_size / original_height,
        target_size / original_width,
    )

    resized_height = max(1, round(original_height * scale))
    resized_width = max(1, round(original_width * scale))

    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    tensor = functional.interpolate(
        tensor,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )

    vertical_padding = target_size - resized_height
    horizontal_padding = target_size - resized_width

    pad_top = vertical_padding // 2
    pad_bottom = vertical_padding - pad_top
    pad_left = horizontal_padding // 2
    pad_right = horizontal_padding - pad_left

    tensor = functional.pad(
        tensor,
        (pad_left, pad_right, pad_top, pad_bottom),
        value=0.0,
    )

    metadata = {
        "processed_height": target_size,
        "processed_width": target_size,
        "scale_x": resized_width / original_width,
        "scale_y": resized_height / original_height,
        "pad_left": pad_left,
        "pad_top": pad_top,
    }

    return tensor.squeeze(0), metadata


def process_dicom(path: Path, target_size: int) -> dict[str, object]:
    """Read, inspect, and preprocess one DICOM image."""
    record = {field: "" for field in CSV_FIELDS}
    record["image_id"] = path.stem
    record["source_path"] = path.as_posix()
    record["readable"] = False
    record["issue"] = ""

    try:
        dataset, raw_pixels = load_dicom_pixels(path)

        record.update(
            {
                "sop_instance_uid": str(
                    getattr(dataset, "SOPInstanceUID", "")
                ),
                "original_height": int(raw_pixels.shape[0]),
                "original_width": int(raw_pixels.shape[1]),
                "pixel_dtype": str(raw_pixels.dtype),
                "pixel_min": float(raw_pixels.min()),
                "pixel_max": float(raw_pixels.max()),
                "photometric_interpretation": str(
                    getattr(dataset, "PhotometricInterpretation", "")
                ),
                "bits_stored": getattr(dataset, "BitsStored", ""),
                "pixel_representation": getattr(
                    dataset, "PixelRepresentation", ""
                ),
            }
        )

        normalized_image, intensity_metadata = apply_intensity_processing(
            raw_pixels,
            dataset,
        )
        processed_image, spatial_metadata = resize_and_pad(
            normalized_image,
            target_size,
        )

        if processed_image.shape != (1, target_size, target_size):
            raise ValueError(
                f"Unexpected processed shape: {tuple(processed_image.shape)}"
            )

        record.update(intensity_metadata)
        record.update(spatial_metadata)
        record["readable"] = True

    except Exception as error:
        record["issue"] = f"{type(error).__name__}: {error}"

    return record


def write_csv(records: list[dict[str, object]], output_path: Path) -> None:
    """Write image metadata to a CSV file."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def write_summary(
    records: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Write a short processing summary."""
    readable_records = [
        record for record in records if record["readable"] is True
    ]

    dimension_counts = Counter(
        (
            record["original_height"],
            record["original_width"],
        )
        for record in readable_records
    )

    summary = {
        "files_found": len(records),
        "readable_files": len(readable_records),
        "unreadable_files": len(records) - len(readable_records),
        "monochrome1_files": sum(
            record["photometric_interpretation"] == "MONOCHROME1"
            for record in readable_records
        ),
        "original_dimension_counts": {
            f"{height}x{width}": count
            for (height, width), count in sorted(dimension_counts.items())
        },
        "issues": [
            {
                "image_id": record["image_id"],
                "source_path": record["source_path"],
                "issue": record["issue"],
            }
            for record in records
            if record["readable"] is False
        ],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=512,
    )
    arguments = parser.parse_args()

    dicom_paths = sorted(arguments.input_dir.rglob("*.dcm"))

    if not dicom_paths:
        raise FileNotFoundError(
            f"No DICOM files found in {arguments.input_dir}"
        )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        process_dicom(path, arguments.target_size)
        for path in dicom_paths
    ]

    write_csv(records, arguments.output_dir / "images.csv")
    write_summary(records, arguments.output_dir / "processing_summary.json")

    readable_count = sum(
        record["readable"] is True for record in records
    )

    print(f"Files found: {len(records)}")
    print(f"Readable files: {readable_count}")
    print(f"Unreadable files: {len(records) - readable_count}")
    print(f"Output directory: {arguments.output_dir}")


if __name__ == "__main__":
    main()
