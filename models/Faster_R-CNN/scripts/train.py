"""Train or evaluate a Faster R-CNN baseline on the fixed RSNA splits."""

import argparse
from contextlib import redirect_stdout
import csv
from functools import partial
import hashlib
import io
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d
from tqdm import tqdm

from data_loader import RSNADetectionDataset, collate_fn


def build_model(pretrained=True):
    # Match torchvision's COCO_V1 architecture, including its FrozenBN epsilon.
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50", weights=None, trainable_layers=3,
        norm_layer=partial(FrozenBatchNorm2d, eps=0.0),
    )
    model = FasterRCNN(backbone, num_classes=91, min_size=512, max_size=512)
    if pretrained:
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, 2
    )
    return model


def train_one_epoch(model, loader, optimizer, scaler, device, amp):
    model.train()
    totals = {}
    seen = 0
    for images, targets in tqdm(loader, desc="Train"):
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [
            {key: value.to(device, non_blocking=True) for key, value in target.items()}
            for target in targets
        ]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=amp):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"Nonfinite training loss: {losses}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        for key, value in {"loss": loss, **losses}.items():
            totals[key] = totals.get(key, 0.0) + value.item() * len(images)
        seen += len(images)
    return {key: value / seen for key, value in totals.items()}


def coco_metrics(images, annotations, detections):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    categories = [{"id": 1, "name": "Lung Opacity"}]
    # An empty detection list is valid, including for an untrained model.
    with redirect_stdout(io.StringIO()):
        truth, prediction = COCO(), COCO()
        for coco, records in [(truth, annotations), (prediction, detections)]:
            coco.dataset = {
                "images": images, "annotations": records, "categories": categories,
            }
            coco.createIndex()
        evaluator = COCOeval(truth, prediction, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "map": float(evaluator.stats[0]),
        "map_50": float(evaluator.stats[1]),
        "map_75": float(evaluator.stats[2]),
        "mar_100": float(evaluator.stats[8]),
    }


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    images_info, annotations, detections = [], [], []
    for images, targets in tqdm(loader, desc="Evaluate"):
        # Use float32 for evaluation and the same settings on every run.
        outputs = model([image.to(device, non_blocking=True) for image in images])
        for image, target, output in zip(images, targets, outputs):
            image_id = target["image_id"].item()
            images_info.append({
                "id": image_id, "height": image.shape[1], "width": image.shape[2],
            })
            for box in target["boxes"].tolist():
                x1, y1, x2, y2 = box
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                })
            for box, score in zip(output["boxes"].cpu().tolist(), output["scores"].cpu().tolist()):
                x1, y1, x2, y2 = box
                detections.append({
                    "id": len(detections) + 1, "image_id": image_id, "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1), "iscrowd": 0, "score": score,
                })
    return coco_metrics(images_info, annotations, detections)


def make_loader(data_dir, split, batch_size, workers, device, generator, limit):
    dataset = RSNADetectionDataset(data_dir, split)
    if limit:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=split == "train",
        num_workers=workers, collate_fn=collate_fn,
        pin_memory=device.type == "cuda", generator=generator,
    )


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/rsna_full/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10, help="Total epochs, including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, help="Resume last.pt in the same output directory")
    parser.add_argument("--evaluate", type=Path, help="Evaluate a saved checkpoint without training")
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--limit-images", type=int, default=0, help="Smoke test only: limit each split")
    parser.add_argument("--no-pretrained", action="store_true", help="Offline smoke test only")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.workers < 0 or args.limit_images < 0:
        parser.error("Epochs and batch size must be positive; workers and limit must be nonnegative")
    if args.resume and args.evaluate:
        parser.error("Choose either --resume or --evaluate")
    if args.resume and args.resume.resolve() != (args.output / "last.pt").resolve():
        parser.error("Resume the output directory's last.pt")
    if not args.resume and args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new or empty output directory")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install CUDA-enabled PyTorch and select an available GPU.")
    amp = device.type == "cuda" and not args.no_amp
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    checkpoint_path = args.resume or args.evaluate
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if checkpoint_path else None
    fingerprints = {
        name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
        for name in ["labels.csv", "boxes.csv"]
    }
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if checkpoint:
        if fingerprints != checkpoint["data_sha256"]:
            raise ValueError("Prepared metadata differs from the checkpoint's dataset")
        if args.resume:
            for key in ["batch_size", "lr", "seed", "no_amp", "limit_images", "no_pretrained"]:
                if config[key] != checkpoint["config"][key]:
                    raise ValueError(f"Resume with the original --{key.replace('_', '-')} setting")
            if args.epochs <= checkpoint["epoch"]:
                raise ValueError("--epochs must exceed the last completed epoch")
    model = build_model(pretrained=checkpoint is None and not args.no_pretrained)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    model.to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; GPU: {torch.cuda.get_device_name() if device.type == 'cuda' else 'none'}", flush=True)

    if args.evaluate:
        loader = make_loader(
            args.data_dir, args.eval_split, args.batch_size, args.workers,
            device, generator, args.limit_images,
        )
        result = {
            "split": args.eval_split, "images": len(loader.dataset),
            "checkpoint": str(args.evaluate), "epoch": checkpoint["epoch"],
            **evaluate(model, loader, device),
        }
        (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return

    train_loader = make_loader(
        args.data_dir, "train", args.batch_size, args.workers, device, generator, args.limit_images,
    )
    val_loader = make_loader(
        args.data_dir, "val", args.batch_size, args.workers, device, None, args.limit_images,
    )
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr, momentum=0.9, weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best_map, history = 1, -1.0, []
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_map, history = checkpoint["best_map"], checkpoint["history"]
        torch.set_rng_state(checkpoint["torch_rng"])
        generator.set_state(checkpoint["loader_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    run_info = {
        "config": config, "data_sha256": fingerprints,
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "train_images": len(train_loader.dataset), "val_images": len(val_loader.dataset),
        "model": "Faster R-CNN ResNet-50-FPN", "image_size": 512,
        "selection_metric": "validation COCO mAP@[0.50:0.95]",
    }
    (args.output / "config.json").write_text(json.dumps(run_info, indent=2) + "\n")
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.monotonic()
        learning_rate = optimizer.param_groups[0]["lr"]
        losses = train_one_epoch(model, train_loader, optimizer, scaler, device, amp)
        metrics = evaluate(model, val_loader, device)
        scheduler.step()
        row = {"epoch": epoch, "lr": learning_rate, **losses, **metrics, "seconds": time.monotonic() - started}
        history.append(row)
        improved = metrics["map"] > best_map
        best_map = max(best_map, metrics["map"])
        state = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "best_map": best_map, "history": history, "config": config,
            "data_sha256": fingerprints, "torch_rng": torch.get_rng_state(),
            "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        save_checkpoint(args.output / "last.pt", state)
        if improved:
            save_checkpoint(args.output / "best.pt", state)
        with (args.output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row), flush=True)
    print(f"Finished. Best validation mAP: {best_map:.4f}. Checkpoints: {args.output}", flush=True)


if __name__ == "__main__":
    main()
