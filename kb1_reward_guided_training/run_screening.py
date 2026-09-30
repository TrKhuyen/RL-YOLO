'''Summarize only completed, verified KB1 runs and orchestrate seed42 screening.'''
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'checkpoint_reward_guide_trainning' / 'screening_seed42_v1'
PREFLIGHT = ROOT / 'checkpoint_reward_guide_trainning' / 'preflight_seed42_v1'
REPORT = ROOT / 'docs' / 'KB1_SCREENING_CLEAN_REPORT.md'
MODELS = ['yolov5s', 'yolov8n', 'yolov8s', 'yolov11n', 'yolov11s', 'dp_yolo']
MODES = ['native_only', 'kb1b']
METRICS = ['mAP50', 'mAP50_95', 'AR300', 'precision', 'operating_recall', 'f1', 'macro_f1']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, deltas, statuses = [], [], []
    for model in MODELS:
        runs = {}
        for mode in MODES:
            folder = OUTPUT / model / mode
            if (folder / 'completed.json').exists():
                runs[mode] = read(folder / 'completed.json')
                statuses.append((model, mode, 'complete', runs[mode]['steps_run'],
                                 runs[mode].get('optimizer_updates', 0),
                                 round(runs[mode].get('training_seconds_total', 0.0), 1)))
            elif (folder / 'status.json').exists():
                status = read(folder / 'status.json')
                statuses.append((model, mode, status['status'], status.get('step', 0), 0, 0.0))
            else:
                statuses.append((model, mode, 'pending', 0, 0, 0.0))
        if runs:
            initial = next(iter(runs.values()))['baseline']
            rows.append(dict(Model=model, Stage='supervised', seed=42, checkpoint_step=0, **initial))
        for mode, run in runs.items():
            rows.append(dict(Model=model, Stage=mode, seed=42,
                             checkpoint_step=run['best_step'], **run['best']))
        if len(runs) == 2:
            a, b, n = runs['kb1b']['baseline'], runs['kb1b']['best'], runs['native_only']['best']
            for key in METRICS:
                if abs(a[key] - runs['native_only']['baseline'][key]) > 1e-6:
                    raise RuntimeError('Baseline mismatch for ' + model)
            deltas.append(dict(Model=model, **{
                key + suffix: b[key] - ref[key]
                for suffix, ref in [('_kb1b_minus_supervised', a), ('_kb1b_minus_native_only', n)]
                for key in METRICS}))
    for name, data in [('results_full.csv', rows), ('results_delta.csv', deltas)]:
        if data:
            path = OUTPUT / name
            tmp = path.with_suffix('.tmp')
            with tmp.open('w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
            tmp.replace(path)
    completed = sum(s[2] == 'complete' for s in statuses)
    audit_path = ROOT / 'results/dataset_audit_clean/summary.json'
    audit = read(audit_path) if audit_path.exists() else None
    lines = [
        '# KB1 screening on clean dataset',
        '',
        'Updated: ' + datetime.now().isoformat(timespec='seconds'),
        f'Completed runs: {completed}/{len(MODELS) * len(MODES)}; comparable pairs: {len(deltas)}/{len(MODELS)}.',
        '',
        'Dataset: pre-data/data/v2i_cleanned. Stage 1: checkpoint_based/<model>/weights/best.pt.',
        'Stage 2: checkpoint_reward_guide_trainning/screening_seed42_v1/<model>/<mode>/best.pt.',
        'Only completed runs are included below. Validation selects checkpoints; test is reserved for final evaluation.',
        '',
        '## Dataset audit',
        '',
    ]
    if audit:
        lines.append(f"Status: {audit['status']}; train/valid/test images: {audit['counts']}.")
        for pair in ('train_valid', 'train_test', 'valid_test'):
            lines.append(f"- {pair}: {audit[pair]['source_groups']} overlapping source groups")
    else:
        lines.append('Pending audit on clean dataset.')
    lines += [
        '',
        '## Validation results',
        '',
        '| Model | Stage | Best step | mAP50 | mAP50-95 | AR300 |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append('| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} |'.format(
            row['Model'], row['Stage'], row['checkpoint_step'],
            row['mAP50'] * 100, row['mAP50_95'] * 100, row['AR300'] * 100))
    lines += [
        '',
        '| Model | Delta mAP50-95 vs supervised | Delta mAP50-95 vs native-only |',
        '|---|---:|---:|',
    ]
    for delta in deltas:
        lines.append('| {} | {:+.3f} | {:+.3f} |'.format(
            delta['Model'],
            delta['mAP50_95_kb1b_minus_supervised'] * 100,
            delta['mAP50_95_kb1b_minus_native_only'] * 100))
    guided = [row for row in rows if row['Stage'] == 'kb1b']
    lines += [
        '',
        '## Validation ranking (locked before test)',
        '',
        'Primary metric: mAP50_95. Candidate set: guided best checkpoint of each model.',
    ]
    if len(guided) == len(MODELS):
        winner = max(guided, key=lambda row: row['mAP50_95'])
        lines.append(f"Validation leader: {winner['Model']} ({winner['mAP50_95'] * 100:.3f}%).")
        lines.append('Evaluate all locked checkpoints on test once; do not reselect hyperparameters from test.')
    else:
        lines.append(f'Pending: {len(guided)}/{len(MODELS)} guided runs complete.')
    lines += [
        '',
        '## Run status',
        '',
        '| Model | Mode | Status | Recorded steps | Optimizer updates | Training seconds |',
        '|---|---|---|---:|---:|---:|',
    ]
    lines += ['| {} | {} | {} | {} | {} | {:.1f} |'.format(*item) for item in statuses]
    lines += [
        '',
        'A complete status requires completed.json in the new run directory.',
        'Results from the former leaked split are excluded.',
        '',
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT.with_suffix('.tmp')
    tmp.write_text('\n'.join(lines), encoding='utf-8')
    tmp.replace(REPORT)
    return completed


def main():
    if '--report-only' in sys.argv:
        summarize()
        return
    from audit_splits import audit
    if audit()['status'] == 'leakage_detected':
        summarize()
        raise RuntimeError('Dataset source leakage detected; see KB1_SCREENING_CLEAN_REPORT.md before training')
    for model in MODELS:
        for mode in MODES:
            folder = PREFLIGHT / model / mode
            if (folder / 'completed.json').exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', str(ROOT / 'run_verified.py'),
                       '--model', model, '--mode', mode, '--output', str(PREFLIGHT),
                       '--steps', '2', '--eval-interval', '2']
            with (folder / 'console.log').open('a', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=ROOT.parent,
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Preflight failed: {model}/{mode}; see {folder / "console.log"}')
    summarize()
    for model in MODELS:
        for mode in MODES:
            folder = OUTPUT / model / mode
            folder.mkdir(parents=True, exist_ok=True)
            print('START', model, mode, flush=True)
            command = [sys.executable, '-u', str(ROOT / 'run_verified.py'), '--model', model,
                       '--mode', mode, '--output', str(OUTPUT)]
            with (folder / 'console.log').open('a', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
            summarize()
            if result.returncode:
                raise RuntimeError(f'Run failed: {model}/{mode}; see console.log')
            print('DONE', model, mode, flush=True)
    print('SCREENING COMPLETE', summarize(), flush=True)


if __name__ == '__main__':
    main()
