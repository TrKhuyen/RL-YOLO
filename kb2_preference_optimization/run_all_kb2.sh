#!/usr/bin/env bash
set -Eeuo pipefail

KB2_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$KB2_DIR/.." && pwd)"
DATA_ROOT="../pre-data/data/v2i_cleanned"
if [[ -x "$REPO_DIR/.venv/Scripts/python.exe" ]]; then
  PYTHON="$REPO_DIR/.venv/Scripts/python.exe"
elif [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
  PYTHON="$REPO_DIR/.venv/bin/python"
else
  echo "KB2: repository .venv Python is missing." >&2
  exit 1
fi
export PYTHONIOENCODING=utf-8
export NO_ALBUMENTATIONS_UPDATE=1

usage() {
  cat <<'EOF'
Usage: bash kb2_preference_optimization/run_all_kb2.sh [options]
  --model NAME           all (default): yolov8n, yolov8s, yolov11n, yolov11s;
                         or one of these model names
  --device DEVICE        cuda (default) or cpu
  --steps N              Updates per ablation variant (default: 1000)
  --batch-size N         Fine-tuning batch size (default: 4)
  --feedback-batch-size N  Feedback generation batch size (default: 8)
  --img-size N           Square image size (default: 640)
  --seed N               Training seed (default: 42)
  --pairwise-alpha X     Pairwise loss weight (default: 0.01)
  --check-only           Verify dataset, checkpoint provenance and device;
                         do not generate feedback or train
EOF
}

model=all
device=cuda
steps=1000
batch_size=4
feedback_batch_size=8
img_size=640
seed=42
pairwise_alpha=0.01
check_only=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--device|--steps|--batch-size|--feedback-batch-size|--img-size|--seed|--pairwise-alpha)
      if [[ $# -lt 2 ]]; then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --model) model="$2";;
        --device) device="$2";;
        --steps) steps="$2";;
        --batch-size) batch_size="$2";;
        --feedback-batch-size) feedback_batch_size="$2";;
        --img-size) img_size="$2";;
        --seed) seed="$2";;
        --pairwise-alpha) pairwise_alpha="$2";;
      esac
      shift 2;;
    --check-only) check_only=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done
case "$model" in
  all) models=(yolov8n yolov8s yolov11n yolov11s);;
  yolov8n|yolov8s|yolov11n|yolov11s) models=("$model");;
  yolov5s|dp_yolo) echo "$model lacks the supervised_loss adapter required by feedback training." >&2; exit 2;;
  *) echo "Unknown model: $model" >&2; exit 2;;
esac
for value in "$steps" "$batch_size" "$feedback_batch_size" "$img_size"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Steps, batch sizes and img-size must be positive integers." >&2
    exit 2
  fi
done
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "Seed must be a non-negative integer." >&2
  exit 2
fi
if [[ "$device" != cpu && "$device" != cuda ]]; then
  echo "Device must be cpu or cuda." >&2
  exit 2
fi

cd "$KB2_DIR"
trap 'echo "KB2 stopped at line $LINENO. Inspect the command above." >&2' ERR
printf '[KB2 preflight] models: %s | device: %s\n' "${models[*]}" "$device"
"$PYTHON" - "$device" "${models[@]}" <<'PY'
import sys
from pathlib import Path
import torch

repo = Path.cwd().parent
sys.path.insert(0, str(repo / 'kb1_reward_guided_training'))
from run_provenance import dataset_manifest, verify_supervised

device, *models = sys.argv[1:]
if device == 'cuda' and not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable; pass --device cpu or configure GPU PyTorch.')
manifest = dataset_manifest()
for model in models:
    checkpoint = repo / 'kb1_reward_guided_training' / 'checkpoint_based' / model / 'weights' / 'best.pt'
    verify_supervised(model, checkpoint, current_dataset=manifest)
    print(f'PASS {model}: KB1 checkpoint and clean dataset provenance verified', flush=True)
print('Dataset SHA256:', manifest['sha256'], flush=True)
PY
# Windows Python launched from WSL must receive relative paths. Passing
# /mnt/d/... directly would resolve to D:\mnt\d\... on Windows.
"$PYTHON" - "$DATA_ROOT" "${models[@]}" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
if not root.is_dir():
    raise SystemExit(f'KB2 data root missing: {root}')
for model in sys.argv[2:]:
    checkpoint = Path('../kb1_reward_guided_training/checkpoint_based') / model / 'weights' / 'best.pt'
    if not checkpoint.resolve().is_file():
        raise SystemExit(f'KB2 checkpoint path missing: {checkpoint.resolve()}')
print('PASS Windows Python path resolution:', root, flush=True)
PY

if [[ "$check_only" == 1 ]]; then
  echo 'KB2 preflight PASS; feedback generation and training were not started.'
  exit 0
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"
for model in "${models[@]}"; do
  checkpoint="../kb1_reward_guided_training/checkpoint_based/$model/weights/best.pt"
  feedback="feedback_data_clean/${model}_train_${run_id}.jsonl"
  output_dir="checkpoint_preference_optimization/ablation/$run_id/$model"
  printf '\n[KB2 1/2] Generate schema-2.1 train feedback: %s\n' "$model"
  "$PYTHON" -u generate_feedback.py \
    --model "$model" --checkpoint "$checkpoint" \
    --data-root "$DATA_ROOT" --device "$device" \
    --batch-size "$feedback_batch_size" --img-size "$img_size" \
    --output "$feedback"

  "$PYTHON" - "$feedback" "$DATA_ROOT" "$img_size" <<'PY'
import sys
from pathlib import Path
from dataloader import PestDataset
from feedback_dataset import load_feedback_records

feedback, root, img_size = sys.argv[1:]
records = load_feedback_records(feedback)
dataset = PestDataset(root, 'train', int(img_size))
expected = set(range(len(dataset)))
if set(records) != expected:
    raise SystemExit(f'Incomplete feedback: {len(records)}/{len(dataset)} images')
for index, record in records.items():
    if Path(record['image_path']).resolve() != dataset.img_paths[index].resolve():
        raise SystemExit(f'Feedback image order mismatch at image_id {index}')
print(f'PASS: {len(records)} train feedback records, schema 2.1', flush=True)
PY

  printf '\n[KB2 2/2] Baseline / sampling / hybrid / pairwise ablation: %s\n' "$model"
  eval_interval=250
  if (( steps < eval_interval )); then eval_interval=$steps; fi
  "$PYTHON" -u run_ablation.py \
    --model "$model" --checkpoint "$checkpoint" \
    --data-root "$DATA_ROOT" --feedback "$feedback" \
    --device "$device" --img-size "$img_size" \
    --steps "$steps" --batch-size "$batch_size" \
    --seed "$seed" --pairwise-alpha "$pairwise_alpha" \
    --eval-interval "$eval_interval" \
    --output-dir "$output_dir"
  echo "KB2 $model complete: $output_dir/ablation_results.json"
done
echo 'KB2 COMPLETE'
