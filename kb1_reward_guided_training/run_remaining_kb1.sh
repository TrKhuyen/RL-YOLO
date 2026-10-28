#!/usr/bin/env bash
# Ubuntu: bash kb1_reward_guided_training/run_remaining_kb1.sh --plan
# Run:    bash kb1_reward_guided_training/run_remaining_kb1.sh
# Resume: repeat the same command/output with unchanged source/config/runtime.
# Scope: existing supervised S0; continuation 43/44 for 7 models; fixed-budget
# ablations on v11n/v8s with 42/43/44. Does not collect a new holdout, retrain S0
# with new seeds, invent a new reward, or claim reward-only significance.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv-ubuntu/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo 'Missing Ubuntu environment. Run: bash kb1_reward_guided_training/setup_ubuntu.sh' >&2
  exit 1
fi
export YOLO_CONFIG_DIR="$ROOT/.kb1-tools/ultralytics"
export MPLCONFIGDIR="$ROOT/.kb1-tools/matplotlib"
export PYTHONUNBUFFERED=1
export KB1_REPO_ROOT="$ROOT"
"$PYTHON" - "$@" <<'PY'
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

repo = Path(os.environ['KB1_REPO_ROOT'])
kb = repo / 'kb1_reward_guided_training'
sys.path.insert(0, str(kb))
p = argparse.ArgumentParser(description='Run remaining KB1 experiments on existing data/S0; no historical overwrites.')
p.add_argument('--plan', action='store_true', help='Print commands only; no training or file writes')
p.add_argument('--output', type=Path, default=kb / 'checkpoint_reward_guide_trainning/remaining_seed43_44_ablation_v1')
p.add_argument('--ablation-steps', type=int, default=10000, help='Same final-step/LR budget for all ablation modes/seeds')
a = p.parse_args()
if a.ablation_steps < 2 or a.ablation_steps % 2:
    p.error('ablation-steps must be an even integer >= 2')
root = a.output.resolve()
models = ['yolov5s', 'yolov8n', 'yolov8s', 'yolov11n', 'yolov11s', 'yolo26n', 'dp_yolo']
abl_models = ['yolov11n', 'yolov8s']
abl_modes = ['native_only', 'proxy_only', 'reward_only', 'kb1b', 'kb1b_no_sp']
labels = dict(zip(models, ['YOLOv5s','YOLOv8n','YOLOv8s','YOLOv11n','YOLOv11s','YOLO26n','DP-YOLO']))
plan = {'continuation_models': models, 'continuation_seeds': [43,44],
        'continuation_modes': ['native_only','kb1b'], 'continuation_steps': 10000,
        'ablation_models': abl_models, 'ablation_seeds': [42,43,44],
        'ablation_modes': abl_modes, 'ablation_steps': a.ablation_steps,
        'selection_continuation': 'best', 'selection_ablation': 'last',
        'workers': 0, 'batch_size': 8,
        'existing_seed42_reference': str(kb / 'results/canonical_clean/screening/ubuntu_all_models_seed42_20261022_v1'),
        'limitations': ['Existing test is exploratory, not an independent new holdout.',
            'Continuation 43/44 is summarized separately from historical seed 42 (mixed training platforms).',
            'Fixed-budget ablations provide three continuation seeds on one S0 for two models.',
            'Independent supervised seeds, new holdout, error analysis and revised objectives remain separate work.']}
jobs = []
for seed in (43,44):
    for model in models:
        for mode in ('native_only','kb1b'):
            jobs.append(('continuation', seed, model, mode, 10000, False))
for seed in (42,43,44):
    for model in abl_models:
        for mode in abl_modes:
            jobs.append(('ablation', seed, model, mode, a.ablation_steps, True))
preflights = sorted({(job[2], job[3]) for job in jobs})

def command(seed, model, mode, output, steps, fixed=False, preflight=False):
    cmd = [sys.executable, '-u', str(kb/'run_verified.py'), '--model', model, '--mode', mode,
           '--seed', str(seed), '--workers', '0', '--batch-size', '8', '--steps', str(steps),
           '--eval-interval', '2' if preflight else '500', '--output', str(output)]
    if fixed:
        cmd.append('--fixed-budget')
    return cmd

print(json.dumps(plan, indent=2), flush=True)
print(f'Plan: {len(preflights)} preflights + {len(jobs)} full training runs. Output: {root}', flush=True)
if a.plan:
    import shlex
    for group, seed, model, mode, steps, fixed in jobs:
        print(shlex.join(command(seed, model, mode, root/group/f'seed{seed}', steps, fixed)))
    print('Then: freeze all checkpoint hashes, evaluate validation/test, aggregate paired deltas and mean/std.')
    raise SystemExit(0)

from run_provenance import dataset_manifest, runtime_manifest, verify_supervised, sha256
from train_rl import CHECKPOINTS
from evaluation_provenance import inspect_checkpoint
from run_verified import atomic_json
import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable. Run this script from an Ubuntu terminal with GPU access.')
root.mkdir(parents=True, exist_ok=True)
lock = (root/'pipeline.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
data = dataset_manifest()
plan.update(dataset=data, runtime=runtime_manifest(),
            sources={m:verify_supervised(m, CHECKPOINTS[m], data) for m in models})
files = [*kb.glob('*.py'), kb/'run_remaining_kb1.sh', *kb.glob('configs/*.yaml'),
         *kb.glob('adapters/*.py'), *kb.glob('models/dp_yolo/*.py'), *kb.glob('models/dp_yolo/*.yaml'),
         kb/'requirements.ubuntu.lock.txt']
plan['code_sha256'] = {str(f.relative_to(kb)):sha256(f) for f in files}
planfile = root/'plan.json'
if planfile.exists() and json.loads(planfile.read_text()) != plan:
    raise RuntimeError('Plan/source/config/runtime changed. Use a new --output; historical metadata is preserved.')
atomic_json(planfile, plan)
for f in files:
    dest = root/'source_snapshot'/f.relative_to(kb)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copyfile(f, dest)
(root/'logs').mkdir(exist_ok=True)

def assert_frozen_inputs():
    if any(sha256(kb/name) != digest for name, digest in plan['code_sha256'].items()):
        raise RuntimeError('Code/config changed while pipeline was running; stop and use a new output')
    if runtime_manifest() != plan['runtime'] or dataset_manifest() != plan['dataset']:
        raise RuntimeError('Runtime/dataset changed while pipeline was running')

def run(cmd, name):
    assert_frozen_inputs()
    log = root/'logs'/f'{name}.log'
    print(f'RUN {name}; log: {log}', flush=True)
    with log.open('a') as stream:
        subprocess.run(cmd, check=True, cwd=repo, stdout=stream, stderr=subprocess.STDOUT)

# Fail before any long training if CPU tests or provenance contracts fail.
run([sys.executable, '-m', 'unittest', 'discover', '-s', str(kb/'tests'), '-p', 'test_*.py'], 'unittest')
run([sys.executable, str(kb/'tests/test_fixes.py')], 'legacy_tests')
if (root/'logs/legacy_tests.log').read_text().rsplit('PASSED:', 1)[-1].splitlines()[0].strip() != '19/19':
    raise RuntimeError('Legacy checks did not report 19/19 PASS')
for model, mode in preflights:
    run(command(42, model, mode, root/'preflight', 2, preflight=True), f'preflight_{model}_{mode}')
for group, seed, model, mode, steps, fixed in jobs:
    output = root/group/f'seed{seed}'
    run(command(seed, model, mode, output, steps, fixed), f'{group}_{seed}_{model}_{mode}')
    inspect_checkpoint(model, output/model/mode/('last.pt' if fixed else 'best.pt'),
                       CHECKPOINTS[model], seed, mode)

# All selections are frozen before the first new test inference.
locked = {}
for group, seed, model, mode, steps, fixed in jobs:
    cp = root/group/f'seed{seed}'/model/mode/('last.pt' if fixed else 'best.pt')
    locked[str(cp)] = sha256(cp)
if (root/'locked_checkpoints.json').exists() and json.loads((root/'locked_checkpoints.json').read_text()) != locked:
    raise RuntimeError('Previously locked checkpoint selection changed')
atomic_json(root/'locked_checkpoints.json', locked)

all_rows = []
for group, seeds, selected_models, modes, selection in [
    ('continuation', (43,44), ['all'], ['native_only','kb1b'], 'best'),
    ('ablation', (42,43,44), abl_models, abl_modes, 'last')]:
    for seed in seeds:
        for model in selected_models:
            for split in ('val','test'):
                parent = root/'evaluation'/group/f'seed{seed}'/model/split
                # Completed evaluations are reused after hash checks. Failed attempts stay intact.
                completed = sorted(parent.glob('attempt_*/completed.json')) if parent.exists() else []
                output = None
                if completed:
                    output = completed[-1].parent
                    d = json.loads(completed[-1].read_text())
                    if d.get('status') != 'complete':
                        raise RuntimeError('Evaluation completion status is invalid')
                    for file, key in [('manifest.json','manifest_sha256'),('results_full.csv','results_sha256'),('results_delta.csv','delta_sha256')]:
                        if sha256(output/file) != d[key]:
                            raise RuntimeError(f'Evaluation artifact changed: {output/file}')
                    saved = json.loads((output/'manifest.json').read_text())
                    for record in saved['inputs']:
                        if sha256(record['checkpoint']) != record['checkpoint_sha256']:
                            raise RuntimeError('Evaluated checkpoint changed')
                if output is None:
                    parent.mkdir(parents=True, exist_ok=True)
                    index = 1
                    while (parent/f'attempt_{index:03d}').exists():
                        index += 1
                    output = parent/f'attempt_{index:03d}'
                    cmd = [sys.executable, '-u', str(kb/'evaluate.py'), '--model', model if model=='all' else labels[model],
                        '--seed', str(seed), '--split', split, '--workers', '0', '--selection', selection,
                        '--screening-root', str(root/group/f'seed{seed}'), '--output', str(output),
                        '--stages', 'supervised', *modes]
                    run(cmd, f'eval_{group}_{seed}_{model}_{split}')
                rows = list(csv.DictReader((output/'results_full.csv').open()))
                expected = (len(models) if model=='all' else 1)*(len(modes)+1)
                if len(rows) != expected:
                    raise RuntimeError(f'Incomplete evaluation: {output}')
                for row in rows:
                    all_rows.append({'experiment':group,'split':split,'run_seed':seed,
                                     'selection':selection,'evaluation_path':str(output), **row})

# Do not average historical seed 42 into the fresh 43/44 group silently.
paired = []
keys = sorted({(r['experiment'],r['split'],r['Model'],int(r['run_seed'])) for r in all_rows})
for group, split, model, seed in keys:
    stages = {r['Stage']:float(r['mAP50_95']) for r in all_rows
              if (r['experiment'],r['split'],r['Model'],int(r['run_seed']))==(group,split,model,seed)}
    for stage in stages:
        if stage != 'kb1b':
            paired.append({'experiment':group,'split':split,'Model':model,'seed':seed,
                           'comparison':'kb1b_minus_'+stage,'delta_pp':100*(stages['kb1b']-stages[stage])})
summary = []
for key in sorted({(r['experiment'],r['split'],r['Model'],r['comparison']) for r in paired}):
    values = [r['delta_pp'] for r in paired if (r['experiment'],r['split'],r['Model'],r['comparison'])==key]
    summary.append(dict(zip(('experiment','split','Model','comparison'),key),
                        n=len(values),mean_delta_pp=statistics.mean(values),std_delta_pp=statistics.stdev(values)))
for name, rows in [('results_all.csv',all_rows),('paired_deltas.csv',paired),('summary_mean_std.csv',summary)]:
    temp = root/(name+'.tmp')
    with temp.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(root/name)
for file, digest in locked.items():
    if sha256(file) != digest:
        raise RuntimeError('Locked checkpoint changed during evaluation')
assert_frozen_inputs()
atomic_json(root/'completed.json', {'status':'complete','training_runs':len(jobs),
    'plan_sha256':sha256(planfile),'locked_sha256':sha256(root/'locked_checkpoints.json'),
    'summary_sha256':sha256(root/'summary_mean_std.csv'),'limitations':plan['limitations']})
print('Completed scheduled KB1 experiments:',root,flush=True)
print('Next research requiring new data/design is listed under limitations in plan.json.',flush=True)
PY
