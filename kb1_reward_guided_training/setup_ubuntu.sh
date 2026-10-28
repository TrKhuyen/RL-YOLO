#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p .kb1-tools
export YOLO_CONFIG_DIR="$ROOT/.kb1-tools/ultralytics"
export MPLCONFIGDIR="$ROOT/.kb1-tools/matplotlib"
if [[ ! -x .kb1-tools/uv ]]; then
  archive="$(mktemp -d)"
  curl -fL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz -o "$archive/uv.tar.gz"
  tar -xzf "$archive/uv.tar.gz" -C "$archive"
  cp "$archive/uv-x86_64-unknown-linux-gnu/uv" .kb1-tools/uv
fi
export UV_PYTHON_INSTALL_DIR="$ROOT/.kb1-tools/python"
export UV_CACHE_DIR="$ROOT/.kb1-tools/cache"
if [[ ! -x .venv-ubuntu/bin/python ]]; then
  .kb1-tools/uv venv --python 3.10.20 .venv-ubuntu
fi
.kb1-tools/uv pip sync --index-strategy unsafe-best-match --python .venv-ubuntu/bin/python kb1_reward_guided_training/requirements.ubuntu.lock.txt
.venv-ubuntu/bin/python - <<'PY'
import sys
sys.path.insert(0, 'kb1_reward_guided_training')
from run_provenance import dataset_manifest, runtime_data_config, verify_supervised
from train_rl import CHECKPOINTS
import torch
manifest = dataset_manifest()
for model, checkpoint in CHECKPOINTS.items():
    verify_supervised(model, checkpoint, manifest)
print('Dataset:', manifest['sha256'])
print('Native training config:', runtime_data_config())
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
