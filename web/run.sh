#!/usr/bin/env bash
set -euo pipefail
WEB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$WEB_DIR/.."
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export LSS_DEVICE="${LSS_DEVICE:-cpu}"
exec "${PYTHON:-python}" -B -m uvicorn web.app:app --host 127.0.0.1 --port "${PORT:-8000}" --workers 1
