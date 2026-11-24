#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || $# -lt 2 ]]; then
    printf '%s\n' 'Usage: bash models/Faster_R-CNN/scripts/run.sh PREPARED_DIR NEW_OUTPUT_DIR [--resume OUTPUT_DIR/last.pt]'
    printf '%s\n' 'Train the historical Baseline44 recipe for 10 total epochs; the adopted snapshot is epoch_06.0.pt.'
    exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TORCH_HOME="${TORCH_HOME:-$SCRIPT_DIR/../../../artifacts/pretrained}"
export PYTHONDONTWRITEBYTECODE=1
PREPARED_DIR="$1"
OUTPUT_DIR="$2"
shift 2

exec "${PYTHON:-python}" -u "$SCRIPT_DIR/train.py" \
    --data-dir "$PREPARED_DIR" --output "$OUTPUT_DIR" \
    --epochs 10 --batch-size 44 --workers 2 --lr 0.005 \
    --seed 42 --device cuda --checkpoint-half-epoch "$@"
