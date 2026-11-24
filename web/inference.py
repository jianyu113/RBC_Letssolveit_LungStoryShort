"""Serve the frozen model pair used in the fixed 80:20 fusion evaluation."""

from __future__ import annotations

import base64
from functools import partial
import hashlib
import io
import json
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image
import pydicom
import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms import Normalize

from dataset.scripts.image_processing import apply_intensity_processing, resize_and_pad


MANIFEST = json.loads(Path(__file__).with_name("models.json").read_text())
FUSION = MANIFEST["fusion"]
MODEL_NAMES = {key: MANIFEST[key]["name"] for key in ("detector", "classifier")}
INPUT_SIZE = MANIFEST["input_size"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_image(content: bytes, input_kind: str) -> tuple[Image.Image, torch.Tensor]:
    """Return the exact prepared RGB preview and the detector's [0, 1] tensor."""
    if input_kind == "prepared_png":
        try:
            with Image.open(io.BytesIO(content)) as source:
                if (source.format, source.mode, source.size) != ("PNG", "RGB", (512, 512)):
                    raise ValueError("Use a prepared 512 × 512 RGB PNG from the data pipeline.")
                image = source.copy()
        except (OSError, Image.DecompressionBombError) as error:
            raise ValueError("This file is not a readable prepared PNG.") from error
    elif input_kind == "dicom":
        try:
            dataset = pydicom.dcmread(io.BytesIO(content))
            if str(getattr(dataset, "PhotometricInterpretation", "")) not in {"MONOCHROME1", "MONOCHROME2"}:
                raise ValueError("Use a single grayscale chest X-ray DICOM.")
            if int(getattr(dataset, "NumberOfFrames", 1)) != 1:
                raise ValueError("Multi-frame DICOM files are not supported.")
            if int(dataset.Rows) * int(dataset.Columns) > 16_777_216:
                raise ValueError("The DICOM image exceeds the 16 megapixel upload limit.")
            pixels = dataset.pixel_array
        except (pydicom.errors.InvalidDicomError, AttributeError, RuntimeError, NotImplementedError) as error:
            raise ValueError("This DICOM could not be decoded. Use an uncompressed grayscale chest X-ray.") from error
        if pixels.ndim != 2 or not np.isfinite(pixels).all():
            raise ValueError("Use one finite, two-dimensional grayscale DICOM image.")
        normalized, _ = apply_intensity_processing(pixels, dataset)
        resized, _ = resize_and_pad(normalized, INPUT_SIZE)
        # This is the same rounding and RGB conversion as prepare_dataset.png_image.
        encoded_pixels = np.rint(np.clip(resized.squeeze(0).numpy(), 0, 1) * 255).astype("uint8")
        image = Image.fromarray(encoded_pixels).convert("RGB")
    else:
        raise ValueError("Select DICOM or prepared PNG as the input format.")
    pixels = np.array(image, dtype=np.uint8)
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255)
    return image, tensor


class ModelService:
    """Load the two verified checkpoints once and serialize prediction requests."""

    def __init__(self, detector_path: str | Path, classifier_path: str | Path, device: str = "cpu"):
        paths = {"detector": Path(detector_path), "classifier": Path(classifier_path)}
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name} checkpoint: {path}")
            if sha256(path) != MANIFEST[name]["sha256"]:
                raise ValueError(f"Wrong or incomplete {name} checkpoint; its SHA256 does not match models.json.")
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Choose cpu or an available cuda device.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable. Choose cpu explicitly.")
        detector_checkpoint = torch.load(paths["detector"], map_location="cpu", weights_only=True)
        classifier_checkpoint = torch.load(paths["classifier"], map_location="cpu", weights_only=True)
        if float(detector_checkpoint["epoch"]) != MANIFEST["detector"]["epoch"]:
            raise ValueError("The detector must be Baseline44 epoch 6.")
        if classifier_checkpoint["completed_epoch"] != MANIFEST["classifier"]["epoch"]:
            raise ValueError("The classifier must be the original ResNet-18 baseline epoch 2.")
        if classifier_checkpoint["settings"]["labels_sha256"] != MANIFEST["prepared_labels_sha256"]:
            raise ValueError("The classifier was trained on a different prepared dataset.")
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50", weights=None, trainable_layers=3,
            norm_layer=partial(FrozenBatchNorm2d, eps=0.0),
        )
        self.detector = FasterRCNN(backbone, num_classes=91, min_size=512, max_size=512)
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(
            self.detector.roi_heads.box_predictor.cls_score.in_features, 2,
        )
        self.detector.load_state_dict(detector_checkpoint["model"], strict=True)
        self.classifier = resnet18(weights=None)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, 1)
        self.classifier.load_state_dict(classifier_checkpoint["model_state_dict"], strict=True)
        for model in (self.detector, self.classifier):
            model.requires_grad_(False).eval().float().to(self.device)
        roi = self.detector.roi_heads
        if (roi.score_thresh, roi.nms_thresh, roi.detections_per_img) != (0.05, 0.5, 100):
            raise ValueError("Detector postprocessing differs from the reference evaluation.")
        self.normalizer = Normalize((.485, .456, .406), (.229, .224, .225))
        self.lock = threading.Lock()
        self.models = MODEL_NAMES
        self.fusion = FUSION

    @torch.inference_mode()
    def predict(self, content: bytes, file_name: str, input_kind: str) -> dict:
        started = time.perf_counter()
        image, tensor = prepare_image(content, input_kind)
        with self.lock:
            prediction = self.detector([tensor.to(self.device)])[0]
            classifier_input = self.normalizer(tensor.unsqueeze(0)).to(self.device)
            classifier_score = self.classifier(classifier_input).sigmoid().item()
            coordinates = prediction["boxes"].cpu().tolist()
            scores = prediction["scores"].cpu().tolist()
        detector_score = max(scores, default=0.0)
        preview = io.BytesIO()
        image.save(preview, format="PNG")
        return {
            "file_name": Path(file_name).name,
            "preview_url": "data:image/png;base64," + base64.b64encode(preview.getvalue()).decode("ascii"),
            "width": INPUT_SIZE,
            "height": INPUT_SIZE,
            "detector_score": detector_score,
            "classifier_score": classifier_score,
            "fusion_score": FUSION["detector_weight"] * detector_score + FUSION["classifier_weight"] * classifier_score,
            "boxes": [{"xyxy": box, "score": score} for box, score in zip(coordinates, scores)],
            "elapsed_seconds": time.perf_counter() - started,
        }
