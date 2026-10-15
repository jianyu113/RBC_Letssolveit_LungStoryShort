"""Shared DICOM intensity processing and spatial transforms."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn.functional as functional


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
