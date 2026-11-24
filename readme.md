# LungStoryShort

Prioritize chest X-rays for review, measured by **Recall@20%**. The final system uses **Faster R-CNN Baseline44 epoch 6 + original ResNet-18 baseline epoch 2**:
`ranking score = 0.8 × max detector score + 0.2 × classifier sigmoid score`.
Detector boxes and their scores stay unchanged; the display threshold is 0.30.

## Code and fixed inputs

| Component | Code | Adopted artifact |
|---|---|---|
| Data | `dataset/scripts/prepare_dataset.py` | [Crystal prepared 358799531](https://www.kaggle.com/code/crystalchencc/lungstoryshort-full-dataset-preparation/output?scriptVersionId=358799531), `rsna_full/prepared/` |
| Detector | `models/Faster_R-CNN/scripts/` | [Baseline44 364103459](https://www.kaggle.com/code/jiayun042/lungstoryshort-faster-r-cnn-controlled-experiments/output?scriptVersionId=364103459), `baseline_batch44/run/epoch_06.0.pt` |
| Classifier | `models/ResNet-18/scripts/` | [Baseline 362276331](https://www.kaggle.com/code/crystalchencc/lungstoryshort-resnet-18-full-training-week-3/output?scriptVersionId=362276331), `resnet18_full_training_20261025_221305_402389/best.pt` |
| Fusion | `models/Fusion/` | [Fixed evaluation 368426793](https://www.kaggle.com/code/jiayun042/lungstoryshort-fixed-80-20-fusion-test/output?scriptVersionId=368426793), `fixed_fusion_test_v1/` |
| Web | `web/run.sh` | Checkpoint hashes and fusion settings: [models.json](web/models.json) |

Use the saved prepared PNGs and unchanged CSVs: **20,818 train / 4,271 val / 4,595 test**, with no patient overlap. Downloaded data, weights and new outputs can go in ignored `artifacts/`. Final training entry points are in each model’s `scripts/` directory; historical experiment scripts, notebooks, reports and figures are excluded.

## Setup and verify

Run from the repository root. For CUDA training/full evaluation, install the cu128 wheels before other dependencies; omit that line on a Mac.

```bash
conda create -n lungstoryshort python=3.12.15 pip
conda activate lungstoryshort
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check

export LSS_PREPARED="$PWD/artifacts/prepared"
export LSS_DETECTOR="$PWD/artifacts/baseline44_epoch6.pt"
export LSS_CLASSIFIER="$PWD/artifacts/resnet18_baseline_epoch2.pt"
export LSS_SCORES="$PWD/artifacts/fixed_fusion_test_v1"
python models/Fusion/verify.py --detector "$LSS_DETECTOR" --classifier "$LSS_CLASSIFIER" --prepared-dir "$LSS_PREPARED"
```

## Train the baseline recipes

| Setting | Detector | Classifier |
|---|---|---|
| Epochs / batch / seed | 10 / 44 / 42 | 10 / 16 / 42 |
| Optimizer | SGD, LR 0.005, momentum 0.9, weight decay 0.0005 | Adam, LR 0.0001, weight decay 0 |
| Schedule / precision | StepLR(5, 0.1), AMP | Constant LR, FP32 |
| Input / augmentation | 512 RGB, no augmentation | Same PNG, ImageNet normalization, no augmentation |

```bash
bash models/Faster_R-CNN/scripts/run.sh "$LSS_PREPARED" models/Faster_R-CNN/runs/baseline44
bash models/ResNet-18/scripts/run.sh "$LSS_PREPARED" models/ResNet-18/runs/baseline
```

Use new output directories. Resume with `--resume <output>/last.pt`. Detector half-epoch snapshots are model-only; its mAP-selected `best.pt` is not the adopted epoch-6 snapshot. Classifier `best.pt` uses validation BCE (epoch 2 in the original run). Each scripts directory contains a source/recipe manifest. The classifier loader is reconstructed, not the recovered original file. Retraining does not guarantee identical weights or metrics; use the verified saved checkpoints for the adopted system.

## Evaluate saved scores

```bash
python models/Fusion/evaluate_final_scores.py   --scores "$LSS_SCORES/val/image_scores.csv" --split val   --source-config "$LSS_SCORES/config.json" --source-manifest "$LSS_SCORES/sha256.json"   --output artifacts/runs/final_val_scores
```

Change both `val` values to `test` for the fixed test report. Outputs include Recall@20%, AUROC, image-level threshold flags, shortlist composition and a patient-bootstrap comparison. Add `--raw-predictions "$LSS_SCORES/val/raw_detector_predictions.jsonl"` for cached box AP. For full GPU inference, use [evaluate_fixed_test.py](models/Fusion/evaluate_fixed_test.py) (`--help` lists its arguments), preserving the original classifier folder, `best.pt` and sibling `best_validation_predictions.csv`. Do not tune on test.

The fixed test fusion retrieved **675/1,161 positives (58.14%)**, versus **671 (57.80%)** for the detector alone. The confidence interval includes zero; see [the saved evaluation](https://www.kaggle.com/code/jiayun042/lungstoryshort-fixed-80-20-fusion-test/output?scriptVersionId=368426793).

## Open the web prototype

With the checkpoint variables above exported:

```bash
python -m pip install -r web/requirements.txt
bash web/run.sh
```

Open [localhost:8000](http://127.0.0.1:8000) and confirm **Both models ready**. CPU is the default. Upload prepared 512×512 RGB PNGs or single-frame grayscale DICOMs (20 files / 32 MB maximum); uploads stay in memory. For an HTTPS team link, set `LSS_ACCESS_PASSWORD` before starting and share the code separately. This research prototype is not a clinical diagnostic tool.
