# Shared validation evaluation

`evaluate.py` evaluates saved predictions for ResNet-18 and Faster R-CNN.
It never trains a model. It requires the entire fixed validation split and
rejects train/test prediction IDs. Test-set evaluation is deliberately deferred
until the final evaluation protocol is agreed.

## Dependencies

Both modes require NumPy, pandas and scikit-learn. Detection additionally uses
`pycocotools==2.0.11`, matching this repository's Faster R-CNN requirements.
The export helper additionally requires PyTorch and the existing detector code.
In Kaggle, if the COCO package is missing, install it in a code cell:

```python
%pip install pycocotools==2.0.11
```

## ResNet-18: existing classification workflow

The current Kaggle evaluation cells still work: classification is the default.
Use the full prepared `labels.csv` and the selected checkpoint's
`best_validation_predictions.csv` (image_id, patient_id, target, probability).
The script preserves patient IDs as strings, including leading zeros.

```bash
python evaluate.py --task classification \
  --labels /path/to/prepared/labels.csv \
  --predictions /path/to/best_validation_predictions.csv \
  --output-dir /path/to/new_classification_evaluation
```

It produces image-level accuracy, sensitivity, specificity, precision, confusion
counts, AUROC, classification average precision, and recall among the top 20%
of ranked images. Default threshold is 0.5; the threshold table also includes
0.1, 0.2, 0.3 and 0.4. No threshold is automatically selected.

## Faster R-CNN: export validation predictions once

Use the trained detector's selected `best.pt`, not the ResNet-18 checkpoint.
The existing detector evaluation returns summary metrics, not saved boxes;
those summary metrics alone cannot be reevaluated. Export boxes with a single
inference pass. No retraining is needed.

Run the following in a separate, fresh detector notebook/session. Set the four
paths to the actual attached inputs. `detector_scripts` must contain the repo's
Faster_R-CNN `train.py` and `data_loader.py`. Avoid importing the classification
`data_loader` into this session because the two files share a module name.

```python
from pathlib import Path
import hashlib
import runpy
import sys
import torch

shared_script = Path('/path/to/evaluate.py')
detector_scripts = Path('/path/to/Faster_R-CNN/scripts')
prepared_dir = Path('/path/to/prepared')
checkpoint_path = Path('/path/to/detector/best.pt')
predictions_path = Path('/kaggle/working/detection_predictions.json')

sys.path.insert(0, str(detector_scripts))
detector = runpy.run_path(str(detector_scripts / 'train.py'))
shared = runpy.run_path(str(shared_script))

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
metadata_hashes = {
    name: hashlib.sha256((prepared_dir / name).read_bytes()).hexdigest()
    for name in ['labels.csv', 'boxes.csv']
}
assert metadata_hashes == checkpoint['data_sha256'], 'Checkpoint dataset differs.'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = detector['build_model'](pretrained=False)
model.load_state_dict(checkpoint['model'])
model.to(device)
loader = detector['make_loader'](
    data_dir=prepared_dir, split='val', batch_size=4, workers=2,
    device=device, generator=None, limit=0,
)
shared['export_detection_predictions'](
    model, loader, predictions_path, device,
)
print('Saved predictions:', predictions_path)
print('Checkpoint:', checkpoint_path, 'Epoch:', checkpoint['epoch'])
```

The helper converts numeric loader IDs back to the original image UUIDs using
`dataset.records`. Every validation image must be exported, including images
with zero detections. It refuses to overwrite an existing prediction file.
Save the checkpoint identity with the export in your experiment log.

The exported JSON format is:

```json
{
  "split": "val",
  "box_format": "xyxy",
  "coordinate_space": "processed_pixels",
  "images": [
    {"image_id": "positive-image-uuid", "patient_id": "00000001",
     "boxes": [[10, 20, 100, 120]], "scores": [0.9], "labels": [1]},
    {"image_id": "negative-image-uuid", "patient_id": "00000002",
     "boxes": [], "scores": [], "labels": []}
  ]
}
```

This is a format example, not a dataset. Category 1 is lung opacity.
Coordinates must use processed image pixels, not normalized values or original
X-ray coordinates. Ground truth comes from `processed_x_min`, `processed_y_min`,
`processed_x_max`, `processed_y_max` in prepared `boxes.csv`. Image dimensions
come from `processed_width` and `processed_height` in `labels.csv`.

## Faster R-CNN: evaluate the saved export

This step requires no GPU and can run in the separate shared evaluation notebook.

```bash
python evaluate.py --task detection \
  --labels /path/to/prepared/labels.csv \
  --boxes /path/to/prepared/boxes.csv \
  --predictions /path/to/detection_predictions.json \
  --output-dir /path/to/new_detection_evaluation
```

For a Kaggle Python cell, use your resolved paths:

```python
import subprocess
import sys
from datetime import datetime, timezone

run_name = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
evaluation_dir = Path(f'/kaggle/working/detection_evaluation_{run_name}')
subprocess.run([
    sys.executable, str(shared_script), '--task', 'detection',
    '--labels', str(prepared_dir / 'labels.csv'),
    '--boxes', str(prepared_dir / 'boxes.csv'),
    '--predictions', str(predictions_path),
    '--output-dir', str(evaluation_dir),
], check=True)
```

### Metric definitions

- `detection_metrics.json`: COCO bbox mAP across IoU 0.50:0.05:0.95,
  AP50, AP75, area-specific AP, and average recall at 1/10/100 detections.
  There is one foreground category. COCO uses 101 recall points and up to
  100 detections per image for AP. `null` means an area group has no ground truth.
  Size groups refer to processed pixel area. This is **not** RSNA competition AP.
- `metrics.json`: the same image-level classification metrics as ResNet-18,
  using the maximum returned detection score per image (0 for no detections).
  This image-level score is an explicit aggregation choice, not a calibrated
  disease probability. It does not measure localization.
- `--threshold` and `--thresholds` affect image classification only. The evaluator
  does not remove low-score boxes before COCO AP. The model's own score filter,
  NMS and output cap still apply; the export helper records those settings.
- Negative images remain in detection evaluation; their predicted boxes count
  as false positives. Duplicate detections use COCO one-to-one matching.

## Saved outputs and comparison

Both modes save `metrics.json`, `settings.json`, `confusion_matrix.csv`,
`threshold_analysis.csv`, `recall_budget_curve.csv`, `precision_recall_curve.csv`,
`ranked_predictions.csv` and `incorrect_predictions.csv`.
Detection additionally saves `detection_metrics.json`.

Use a new output directory for each run. Keep the prediction input, selected
checkpoint, prepared metadata, script version and output folder together.
`settings.json` includes hashes of the script and prediction/label files;
detection also records the boxes hash and pycocotools version.

Compare the two models on the same complete validation split. Classification AP
and box AP are distinct quantities and must not be placed in the same comparison
column. Patient-level metrics and confidence intervals are not computed.

## Verification performed

Synthetic tests passed for perfect boxes; empty predictions; false positives on
negative images; IoU-specific AP; duplicate detections; malformed/missing/wrong
split inputs; both command-line modes; overwrite protection; and exporting with
the repository's actual RSNADetectionDataset using a controlled test model.
Existing classification outputs matched the original evaluator on synthetic
cases, including tied scores. Real trained Faster R-CNN predictions still need
to be exported and evaluated before reporting this team's detection results.

Metric reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
