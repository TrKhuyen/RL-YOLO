"""Run one lambda=1 continuation matched to the selected lambda=.03 hybrid.

Never changes historical trainers, bank contents, defaults or selected checkpoint.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'kb2_preference_optimization/outputs/yolov8n_hybrid_schedule_s42_20261106_trial3'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f'Use a fresh experiment directory: {root}')
    if not root.is_relative_to(ROOT / 'kb2_preference_optimization/outputs'):
        raise ValueError('Experiment output must stay inside KB2 outputs')

    # Match the documented runtime controls as well as the model recipe.
    historical_plan = read(BASELINE / 'plan.json')
    runtime = historical_plan['runtime']
    for key in ('CUDA_LAUNCH_BLOCKING', 'PYTHONFAULTHANDLER', 'NO_ALBUMENTATIONS_UPDATE',
                'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        if runtime.get(key) is not None:
            os.environ[key] = str(runtime[key])
    os.environ['YOLO_CONFIG_DIR'] = '/tmp'
    os.environ['MPLCONFIGDIR'] = '/tmp/kb2-matplotlib'
    affinity = set(historical_plan['cpu_affinity'])
    if not affinity.issubset(os.sched_getaffinity(0)):
        raise RuntimeError(f'Baseline CPU affinity unavailable: {sorted(affinity)}')
    os.sched_setaffinity(0, affinity)

    from kb2_preference_optimization.hybrid.bank import Bank
    from kb2_preference_optimization.hybrid.source import source_provenance, assert_provenance

    provenance = source_provenance('yolov8n')
    baseline_manifest = read(BASELINE / 'run/dpo/manifest.json')
    baseline_done = read(BASELINE / 'run/dpo/completed.json')
    baseline_root_done = read(BASELINE / 'run/completed.json')
    assert_provenance(baseline_manifest['provenance'], provenance)
    cfg = baseline_manifest['config']
    assert cfg['dpo_weight'] == .03 and cfg['seed'] == 42 and cfg['steps'] == 15000
    assert not cfg['partial_bank'] and not cfg['eval_limit']
    assert baseline_done['reference_unchanged'] and baseline_done['frozen_state_unchanged']
    bank_path = Path(historical_plan['bank'])
    bank = Bank(bank_path)
    assert_provenance(bank.manifest['provenance'], provenance)
    assert not bank.manifest['partial_bank']

    protected_paths = [
        Path(provenance['source_checkpoint']),
        Path(provenance['shared_S0']['checkpoint']),
        bank_path/'manifest.json', bank_path/'pairs.jsonl', bank_path/'contexts.jsonl',
        BASELINE/'run/dpo/best.pt', BASELINE/'run/dpo/last.pt',
        BASELINE/'run/dpo/manifest.json', BASELINE/'run/dpo/metrics.jsonl',
        BASELINE/'run/dpo/stream.jsonl',
        ROOT/'docs/report_final/assets/kb2_hybrid_selected_20261108/selection.json',
    ]
    protected = {str(path): digest(path) for path in protected_paths}
    for which in ('best', 'last'):
        assert protected[str(BASELINE/f'run/dpo/{which}.pt')] == baseline_done[f'{which}_sha256']

    command = [
        'bash', str(ROOT/'kb2_preference_optimization/run_hybrid_kb3_kb2.sh'),
        '--device', 'cuda', '--steps', str(cfg['steps']), '--seed', str(cfg['seed']),
        '--dpo-weight', '1.0', '--beta', str(cfg['beta']),
        '--lr', str(cfg['lr']), '--background-weight', str(cfg['native_background_weight']),
        '--eval-interval', str(cfg['eval_interval']),
        '--lr-drop-step', str(cfg['lr_schedule']['drop_after_update']),
        '--lr-drop-factor', str(cfg['lr_schedule']['factor']),
        '--early-eval-interval', str(cfg['early_eval_interval']),
        '--early-eval-until', str(cfg['early_eval_until']),
        '--audit-pairs', str(cfg['audit_pairs']) if 'audit_pairs' in cfg else '32',
        '--preferences', str(bank_path), '--output-root', str(root/'run'),
    ]
    plan = dict(created_UTC=datetime.now(timezone.utc).isoformat(),
                experiment='matched hybrid lambda1 versus selected lambda003',
                baseline=str(BASELINE/'run'), bank=str(bank_path), source=provenance,
                changed_hyperparameter={'dpo_weight': {'baseline': .03, 'experiment': 1.}},
                baseline_config=cfg, command=command, immutable_files=protected,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                runtime={key:os.environ.get(key) for key in (*runtime.keys(),)},
                test_evaluation=False, selected_model_promotion=False)
    if args.check_only:
        print(json.dumps({'event':'lambda1_preflight_pass', 'output':str(root),
                          'source_trial':provenance['source_tpe']['trial'],
                          'bank_pairs':len(bank.records), 'command':command,
                          'protected_files':len(protected)}, ensure_ascii=False), flush=True)
        return

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no experiment directory or training was created')
    plan['gpu'] = torch.cuda.get_device_name(0)
    plan['torch_version'] = torch.__version__
    plan['torch_cuda_version'] = torch.version.cuda
    root.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (root/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

    write('plan.json',plan)
    write('process.json',{'pid':os.getpid(),'command':command})
    write('status.json',{'stage':'training','started_UTC':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'event':'lambda1_training_start','output':str(root),'gpu':plan['gpu'],
                      'steps':15000,'lambda':1.0,'baseline_lambda':.03}),flush=True)
    try:
        with (root/'console.log').open('x',buffering=1) as log:
            child = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            write('process.json',{'pid':os.getpid(),'training_pid':child.pid,'command':command})
            returncode = child.wait()
        if returncode:
            raise RuntimeError(f'Training exited with {returncode}; inspect {root/"console.log"}')
        aggregate = read(root/'run/completed.json')
        done = read(root/'run/dpo/completed.json')
        trained_manifest = read(root/'run/dpo/manifest.json')
        ignored = {'dpo_weight','effective_dpo_weight'}
        assert {k:v for k,v in trained_manifest['config'].items() if k not in ignored} == {
            k:v for k,v in cfg.items() if k not in ignored}
        assert trained_manifest['config']['dpo_weight'] == 1.
        assert done['steps'] == 15000 and done['reference_unchanged'] and done['frozen_state_unchanged']
        assert not aggregate['partial_bank'] and not aggregate['partial_evaluation']
        assert done['stream_sha256'] == baseline_done['stream_sha256']
        assert digest(root/'run/dpo/stream.jsonl') == protected[str(BASELINE/'run/dpo/stream.jsonl')]
        assert_provenance(trained_manifest['provenance'],provenance)
        assert all(digest(path)==sha for path,sha in protected.items())
        result = dict(plan=plan,result=aggregate,matched_stream=True,
                      protected_files_unchanged=True,config_only_lambda_differs=True)
        write('completed.json',result)
        write('status.json',{'stage':'completed','best_step':done['best_step'],
                             'finished_UTC':datetime.now(timezone.utc).isoformat()})
        best = aggregate['arms']['dpo']['best_metrics']
        print(json.dumps({'event':'lambda1_completed','best_step':done['best_step'],
                          'best_mAP50_95':best['mAP50_95'],
                          'baseline_best_mAP50_95':baseline_root_done['arms']['dpo']['best_metrics']['mAP50_95'],
                          'matched_stream':True,'protected_files_unchanged':True}),flush=True)
    except BaseException as error:
        write('status.json',{'stage':'failed','error':str(error),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    main()
