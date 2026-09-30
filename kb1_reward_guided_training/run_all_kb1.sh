#!/usr/bin/env bash
set -Eeuo pipefail

KB1_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$KB1_DIR/.." && pwd)"
if [[ -x "$REPO_DIR/.venv/Scripts/python.exe" ]]; then
  PYTHON="$REPO_DIR/.venv/Scripts/python.exe"
elif [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
  PYTHON="$REPO_DIR/.venv/bin/python"
else
  echo "KB1: Python environment missing. Create the repository .venv first." >&2
  exit 1
fi
export PYTHONIOENCODING=utf-8
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != "--check-only" ) ]]; then
  echo "Usage: bash run_all_kb1.sh [--check-only]" >&2
  exit 2
fi
cd "$KB1_DIR"
trap 'echo "KB1 stopped at line $LINENO. Check the error above or the run console.log." >&2' ERR

printf '\n[KB1 1/4] Auditing the clean dataset\n'
"$PYTHON" audit_splits.py
"$PYTHON" - <<'PY'
import json
from pathlib import Path
from run_provenance import dataset_manifest
summary = json.loads(Path('results/dataset_audit_clean/summary.json').read_text(encoding='utf-8'))
if summary['status'] != 'no_known_source_overlap':
    raise SystemExit('Dataset audit found source overlap; training stopped.')
manifest = dataset_manifest()
print('Dataset SHA256:', manifest['sha256'], flush=True)
print('Split counts:', manifest['counts'], flush=True)
PY

if [[ "${1:-}" == "--check-only" ]]; then
  echo 'KB1 audit and dataset manifest PASS; training was not started.'
  exit 0
fi

printf '\n[KB1 2/4] Supervised training or verified resume\n'
for model in yolov5s yolov8n yolov8s yolov11n yolov11s dp_yolo; do
  checkpoint="checkpoint_based/$model/weights/best.pt"
  manifest="checkpoint_based/$model/supervised_manifest.json"
  if [[ -e "$checkpoint" || -e "$manifest" ]]; then
    "$PYTHON" - "$model" <<'PY'
import sys
from pathlib import Path
from run_provenance import verify_supervised
model = sys.argv[1]
checkpoint = Path('checkpoint_based') / model / 'weights' / 'best.pt'
verify_supervised(model, checkpoint)
print(f'SKIP supervised {model}: checkpoint and dataset provenance verified', flush=True)
PY
  else
    echo "TRAIN supervised $model"
    "$PYTHON" -u train_supervised.py --model "$model"
    "$PYTHON" - "$model" <<'PY'
import sys
from pathlib import Path
from run_provenance import verify_supervised
model = sys.argv[1]
verify_supervised(model, Path('checkpoint_based') / model / 'weights' / 'best.pt')
print(f'PASS supervised {model}', flush=True)
PY
  fi
done

printf '\n[KB1 3/4] Two-step preflight and full guided/native screening\n'
"$PYTHON" -u run_screening.py
"$PYTHON" - <<'PY'
import json
from run_screening import MODELS, MODES, OUTPUT
for model in MODELS:
    for mode in MODES:
        folder = OUTPUT / model / mode
        completed = folder / 'completed.json'
        best = folder / 'best.pt'
        if not completed.is_file() or not best.is_file():
            raise SystemExit(f'Incomplete screening run: {model}/{mode}')
        if json.loads(completed.read_text(encoding='utf-8')).get('status') != 'complete':
            raise SystemExit(f'Invalid screening status: {model}/{mode}')
print(f'PASS: {len(MODELS) * len(MODES)} screening runs completed', flush=True)
PY

printf '\n[KB1 4/4] Locked-checkpoint test evaluation\n'
export KB1_EVAL_STARTED_NS="$("$PYTHON" -c 'import time; print(time.time_ns())')"
"$PYTHON" -u evaluate.py --checkpoint-source screening --split test
"$PYTHON" - <<'PY'
import csv
import json
import math
import os
from pathlib import Path
from run_screening import MODELS, OUTPUT

root = Path('results/canonical_clean/screening/test')
full = root / 'results_full.csv'
deltas = root / 'results_delta.csv'
started = int(os.environ['KB1_EVAL_STARTED_NS'])
for path in (full, deltas):
    if not path.is_file() or path.stat().st_mtime_ns < started:
        raise SystemExit(f'Test evaluation did not produce a fresh result: {path}')
with full.open(newline='', encoding='utf-8-sig') as stream:
    rows = list(csv.DictReader(stream))
with deltas.open(newline='', encoding='utf-8-sig') as stream:
    delta_rows = list(csv.DictReader(stream))
labels = {'yolov5s': 'YOLOv5s', 'yolov8n': 'YOLOv8n', 'yolov8s': 'YOLOv8s',
          'yolov11n': 'YOLOv11n', 'yolov11s': 'YOLOv11s', 'dp_yolo': 'DP-YOLO'}
expected = {(label, stage) for label in labels.values()
            for stage in ('supervised', 'native_only', 'kb1b')}
actual = {(row['Model'], row['Stage']) for row in rows}
if len(rows) != len(expected) or actual != expected:
    raise SystemExit(f'Incomplete test table: {len(rows)}/{len(expected)} model-stage rows')
if len(delta_rows) != len(labels) or {row['Model'] for row in delta_rows} != set(labels.values()):
    raise SystemExit('Incomplete test delta table')
by_key = {(row['Model'], row['Stage']): row for row in rows}
for row in rows:
    if not math.isfinite(float(row['mAP50_95'])):
        raise SystemExit(f'Invalid test mAP50_95: {row}')
validation = {model: json.loads((OUTPUT / model / 'kb1b' / 'completed.json').read_text(encoding='utf-8'))['best']['mAP50_95']
              for model in MODELS}
selected = max(MODELS, key=lambda model: validation[model])
lines = [
    '# KB1 final clean-data report', '',
    'Primary metric: mAP50_95. Candidate set: best reward-guided checkpoint of each model.',
    'Checkpoint selection and the validation leader were locked before this test evaluation.',
    f'Validation leader: {labels[selected]} ({validation[selected] * 100:.3f}%).', '',
    '| Model | Supervised test | Guided test | Guided - supervised | Native-only test | Guided - native-only |',
    '|---|---:|---:|---:|---:|---:|',
]
for model in MODELS:
    label = labels[model]
    supervised = float(by_key[label, 'supervised']['mAP50_95']) * 100
    guided = float(by_key[label, 'kb1b']['mAP50_95']) * 100
    native = float(by_key[label, 'native_only']['mAP50_95']) * 100
    lines.append(f'| {label} | {supervised:.3f} | {guided:.3f} | {guided-supervised:+.3f} | {native:.3f} | {guided-native:+.3f} |')
test_leader = max(MODELS, key=lambda model: float(by_key[labels[model], 'kb1b']['mAP50_95']))
lines += ['', f'Highest guided test score: {labels[test_leader]}. This is descriptive; test did not select checkpoints or hyperparameters.',
          'The guided - native-only difference includes the reward and proxy terms together.', '',
          f'Full metrics: `{full.as_posix()}`; paired deltas: `{deltas.as_posix()}`.', '']
report = Path('docs/KB1_FINAL_CLEAN_REPORT.md')
report.write_text('\n'.join(lines), encoding='utf-8')
print('PASS: all 18 model-stage test rows verified', flush=True)
print('Final report:', report.resolve(), flush=True)
PY
printf '\nKB1 COMPLETE\n'
