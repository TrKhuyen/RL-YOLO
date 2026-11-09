#!/usr/bin/env bash
set -Eeuo pipefail
KB2_HYBRID_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$KB2_HYBRID_REPO/.venv-ubuntu/bin/python" ]]; then
  KB2_HYBRID_PYTHON="$KB2_HYBRID_REPO/.venv-ubuntu/bin/python"
elif [[ -x "$KB2_HYBRID_REPO/.venv/bin/python" ]]; then
  KB2_HYBRID_PYTHON="$KB2_HYBRID_REPO/.venv/bin/python"
else
  echo 'KB2 hybrid requires a Linux CUDA Python environment (.venv-ubuntu or .venv).' >&2
  exit 1
fi
export NO_ALBUMENTATIONS_UPDATE=1
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/kb2-matplotlib}"
cd "$KB2_HYBRID_REPO"
exec "$KB2_HYBRID_PYTHON" -u -m scripts.run_kb2_dpo --initialization tpe "$@"
