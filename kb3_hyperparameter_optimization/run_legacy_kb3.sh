#!/usr/bin/env bash
# Historical 10-epoch protocol, preserved only for reproducing old experiments.
set -Eeuo pipefail
KB3_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$KB3_DIR/.." && pwd)"
if [[ -x "$REPO_DIR/.venv/Scripts/python.exe" ]]; then
  PYTHON="$REPO_DIR/.venv/Scripts/python.exe"
elif [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
  PYTHON="$REPO_DIR/.venv/bin/python"
else
  echo "KB3 requires repository .venv Python." >&2
  exit 1
fi
export PYTHONIOENCODING=utf-8
export NO_ALBUMENTATIONS_UPDATE=1
cd "$REPO_DIR"
echo "KB3 LEGACY: no verified KB1 baseline or quality v2 protocol." >&2
exec "$PYTHON" -u -m kb3_hyperparameter_optimization.run_all "$@"
