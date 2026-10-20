#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd -- "$MODEL_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lungstoryshort-detection" ]]; then
    printf '%s\n' "Activate lungstoryshort-detection before running setup.sh." >&2
    exit 1
fi

# Keep pretrained weights inside the project, separate from training outputs.
export TORCH_HOME="$MODEL_DIR/weights"

python -m pip install --upgrade --force-reinstall \
    torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$MODEL_DIR/requirements.txt"
python -m pip check

# Download and load the pretrained model on CPU. This does not start training.
python - <<'PY'
import sys

import torch
import torchvision

sys.path.insert(0, "models/Faster_R-CNN/scripts")
from train import build_model

assert torch.version.cuda == "12.8", "Expected the CUDA 12.8 PyTorch build."
model = build_model(pretrained=True)
assert model.roi_heads.box_predictor.cls_score.out_features == 2
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("Pretrained weights:", torch.hub.get_dir())
print("Deployment complete: Faster R-CNN is ready. Training has not started.")
PY
