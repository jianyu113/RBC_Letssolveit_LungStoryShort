#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || $# -lt 2 ]]; then
    printf '%s\n' 'Usage: bash models/ResNet-18/scripts/run.sh PREPARED_DIR NEW_OUTPUT_DIR [--resume OUTPUT_DIR/last.pt]'
    printf '%s\n' 'Train the original ResNet-18 baseline for 10 total epochs; select best.pt by validation BCE.'
    exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TORCH_HOME="${TORCH_HOME:-$SCRIPT_DIR/../../../artifacts/pretrained}"
export PYTHONDONTWRITEBYTECODE=1
PREPARED_DIR="$1"
OUTPUT_DIR="$2"
shift 2

exec "${PYTHON:-python}" -u "$SCRIPT_DIR/train_classifier.py" \
    --prepared-dir "$PREPARED_DIR" --output-dir "$OUTPUT_DIR" \
    --epochs 10 --batch-size 16 --workers 2 --learning-rate 0.0001 \
    --seed 42 --device cuda "$@"
