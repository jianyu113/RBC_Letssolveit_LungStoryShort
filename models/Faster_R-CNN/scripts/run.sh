#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd -- "$MODEL_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
export TORCH_HOME="$MODEL_DIR/weights"

# Change these defaults here, or override them with train.py command-line options.
DATA_DIR="$PROJECT_DIR/dataset/rsna_full/prepared"
OUTPUT_DIR="$MODEL_DIR/runs/baseline"
EPOCHS=10
BATCH_SIZE=4
WORKERS=4

if [[ "${1:-}" != "--help" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    printf '%s\n' "Set CUDA_VISIBLE_DEVICES to the GPU assigned to you before running run.sh." >&2
    exit 1
fi

exec python -u "$SCRIPT_DIR/train.py" \
    --data-dir "$DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    "$@"
