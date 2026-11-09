"""Locked scheduled-LR trial; compare with the completed constant-LR DPO pipeline."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import traceback

from kb2_preference_optimization.artifacts import REPO, atomic_json, sha256
from kb2_preference_optimization.hybrid.bank import Bank
from kb2_preference_optimization.hybrid.source import source_provenance, assert_provenance


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lr-drop-step', type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.lr_drop_step < 15000:
        parser.error('--lr-drop-step must be between1 and14999')
    root = args.output.resolve()
    baseline = REPO/'kb2_preference_optimization/outputs/yolov8n_hybrid_lambda003_s42_20261106_trial2/run'
    older = REPO/'kb2_preference_optimization/outputs/yolov8n_hybrid_tpe2_s42_20261105_180813_658387'
    scheduled_baseline = REPO/'kb2_preference_optimization/outputs/yolov8n_hybrid_schedule_s42_20261106_trial3/run'
    provenance = source_provenance('yolov8n')
    protected = {}
    for folder in (baseline, older, scheduled_baseline):
        aggregate = read(folder/'completed.json')
        assert_provenance(read(folder/'source.json'), provenance)
        assert not aggregate['partial_bank'] and not aggregate['partial_evaluation']
        assert aggregate['arms']['dpo']['steps'] == 15000
        for which in ('best', 'last'):
            path = folder/'dpo'/f'{which}.pt'
            assert sha256(path) == aggregate['arms']['dpo'][f'{which}_sha256']
            protected[str(path)] = sha256(path)
    bank = Bank(older/'preferences')
    assert not bank.manifest['partial_bank']
    for path in (Path(provenance['source_checkpoint']), Path(provenance['shared_S0']['checkpoint']),
                 bank.root/'manifest.json', bank.root/'pairs.jsonl', bank.root/'contexts.jsonl'):
        protected[str(path)] = sha256(path)
    command = ['bash', str(REPO/'kb2_preference_optimization/run_hybrid_kb3_kb2.sh'),
               '--device', 'cuda', '--steps', '15000', '--eval-interval', '1000', '--seed', '42',
               '--dpo-weight', '0.03', '--beta', '0.1', '--lr', '0.000001', '--background-weight', '0.5',
               '--lr-drop-step', str(args.lr_drop_step), '--lr-drop-factor', '0.1',
               '--early-eval-interval', '250', '--early-eval-until', '2000',
               '--preferences', str(bank.root), '--output-root', str(root/'run')]
    root.mkdir(parents=True, exist_ok=False)
    plan = {'created_UTC': datetime.now(timezone.utc).isoformat(), 'baseline': str(baseline),
            'older_DPO_baseline': str(older), 'source': provenance, 'bank': str(bank.root),
            'scheduled_baseline': str(scheduled_baseline),
            'steps': 15000, 'seed': 42, 'lambda': .03, 'background_weight': .5,
            'initial_lr': 1e-6, 'lr_drop_after_update': args.lr_drop_step, 'lr_drop_factor': .1,
            'eval_interval': 1000, 'early_eval_interval': 250, 'early_eval_until': 2000,
            'selection': 'validation mAP50-95, min_delta=1e-4; TPE step0 eligible',
            'comparison': ['matched dense evaluation grid against the completed drop1000 pipeline',
                           'common historical 1000-update grid against constant LR'],
            'immutable_files': protected, 'command': command, 'test_evaluation': False,
            'runtime': {key: os.environ.get(key) for key in
                        ('CUDA_LAUNCH_BLOCKING', 'PYTHONFAULTHANDLER', 'YOLO_CONFIG_DIR',
                         'MPLCONFIGDIR', 'NO_ALBUMENTATIONS_UPDATE', 'OMP_NUM_THREADS',
                         'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
            'cpu_affinity': sorted(os.sched_getaffinity(0))}
    atomic_json(root/'plan.json', plan)
    atomic_json(root/'process.json', {'pid': os.getpid(), 'command': command})
    atomic_json(root/'status.json', {'stage': 'training'})
    print(json.dumps({'event': 'schedule_trial_start', 'output': str(root)}), flush=True)
    try:
        with (root/'console.log').open('x', buffering=1) as log:
            process = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f'Training exited with {process.returncode}; see console.log')
        aggregate = read(root/'run/completed.json')
        result = aggregate['arms']['dpo']
        assert list(aggregate['arms']) == ['dpo'] and result['steps'] == 15000
        assert not aggregate['partial_bank'] and not aggregate['partial_evaluation']
        assert result['reference_unchanged'] and result['frozen_state_unchanged']
        assert (root/'run/dpo/stream.jsonl').read_bytes() == (baseline/'dpo/stream.jsonl').read_bytes()
        assert all(sha256(path) == digest for path, digest in protected.items())
        atomic_json(root/'completed.json', {'plan': plan, 'result': aggregate, 'matched_stream': True,
                                           'protected_files_unchanged': True})
        atomic_json(root/'status.json', {'stage': 'completed', 'best_step': result['best_step']})
        print(json.dumps({'event': 'schedule_trial_completed', 'best_step': result['best_step'],
                          'best_mAP50_95': result['best_metrics']['mAP50_95']}), flush=True)
    except BaseException as error:
        atomic_json(root/'status.json', {'stage': 'failed', 'error': str(error),
                                         'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
