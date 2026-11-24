"""Locked three-seed scheduled native/DPO comparison, including checkpoint500.

Both arms execute the same trainer; native has exactly zero auxiliary weight.
No test evaluation. Each worker starts from TPE in a fresh process.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT/'kb2_preference_optimization/outputs/yolov8n_hybrid_tpe2_s42_20261105_180813_658387/preferences'
FIELDS = ('mAP50_95', 'mAP50', 'mAP75', 'precision', 'operating_recall', 'f1', 'APs', 'APm', 'APl', 'tp', 'fp', 'fn')


def read(path):
    return json.loads(path.read_text())


def configure():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '4'
    os.environ.update(NO_ALBUMENTATIONS_UPDATE='1', YOLO_CONFIG_DIR='/tmp',
                      MPLCONFIGDIR='/tmp/kb2-matplotlib', CUDA_LAUNCH_BLOCKING='1', PYTHONFAULTHANDLER='1')
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this locked campaign')


def verify_pair(root, seed):
    from kb2_preference_optimization.artifacts import sha256
    arms = {}
    for name in ('native', 'dpo'):
        path = root/f'seed{seed}'/name
        m, d = read(path/'manifest.json'), read(path/'completed.json')
        assert d['steps'] == 15000 and d['reference_unchanged'] and d['frozen_state_unchanged']
        assert not d['partial_evaluation'] and not d['partial_bank']
        rows = [json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
        vals = {r['step']: r['metrics'] for r in rows if r['event'] == 'validation'}
        assert list(vals) == m['config']['evaluation_steps']
        for step, digest in d['evaluation_checkpoint_sha256'].items():
            assert sha256(path/f'step_{int(step):06d}.pt') == digest
        for which in ('best', 'last'):
            assert sha256(path/f'{which}.pt') == d[f'{which}_sha256']
        best_score, best_step = vals[0]['mAP50_95'], 0
        for step, metrics in vals.items():
            if metrics['mAP50_95'] > best_score + 1e-4:
                best_score, best_step = metrics['mAP50_95'], step
        assert best_step == d['best_step']
        train = [r for r in rows if r['event'] == 'train']
        assert all(r['lr'] == (1e-6 if r['step'] <= 1000 else 1e-7) for r in train)
        if name == 'native':
            assert all(r['gradient']['weighted_dpo_grad_norm'] == 0 for r in train if r['gradient'])
        arms[name] = {'manifest': m, 'done': d, 'vals': vals, 'first_train': train[0]}
    n, d = arms['native'], arms['dpo']
    assert (root/f'seed{seed}/native/stream.jsonl').read_bytes() == (root/f'seed{seed}/dpo/stream.jsonl').read_bytes()
    assert n['done']['stream_sha256'] == d['done']['stream_sha256']
    for key in n['manifest']:
        if key != 'config':
            assert n['manifest'][key] == d['manifest'][key], key
    nc, dc = n['manifest']['config'], d['manifest']['config']
    assert {k for k in nc if nc[k] != dc[k]} == {'arm', 'effective_dpo_weight', 'dpo_weight'}
    assert nc['effective_dpo_weight'] == 0 and dc['effective_dpo_weight'] == .03
    for key in FIELDS:
        assert n['vals'][0][key] == d['vals'][0][key]
    assert n['first_train']['native_loss'] == d['first_train']['native_loss']
    assert n['first_train']['gradient']['native_grad_norm'] == d['first_train']['gradient']['native_grad_norm']
    records = []
    for contrast in ('step500', 'selected_best', 'step15000'):
        steps = {name: (500 if contrast == 'step500' else 15000 if contrast == 'step15000' else arm['done']['best_step']) for name, arm in arms.items()}
        row = {'seed': seed, 'contrast': contrast, 'native_step': steps['native'], 'dpo_step': steps['dpo']}
        for key in FIELDS:
            x, y = n['vals'][steps['native']][key], d['vals'][steps['dpo']][key]
            row.update({f'native_{key}': x, f'dpo_{key}': y, f'delta_{key}': (y-x)*(1 if key in ('tp','fp','fn') else 100)})
        records.append(row)
    return records


def write_review(root, records):
    from kb2_preference_optimization.artifacts import atomic_json
    summary = {}
    for contrast in ('step500', 'selected_best', 'step15000'):
        rows = [r for r in records if r['contrast'] == contrast]
        values = [r['delta_mAP50_95'] for r in rows]
        summary[contrast] = {'seeds': len(values), 'mean_delta_mAP_pp': statistics.mean(values),
                             'sample_sd_delta_mAP_pp': statistics.stdev(values) if len(values)>1 else None,
                             'positive_seeds': sum(v>0 for v in values), 'negative_seeds': sum(v<0 for v in values)}
    atomic_json(root/'paired_results.json', {'records': records, 'summary': summary,
                'scope': 'development validation; three continuation seeds from one TPE checkpoint; descriptive statistics, not independent generalization'})
    with (root/'paired_results.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',choices=('native','dpo'))
    parser.add_argument('--seed',type=int,choices=(42,43,44))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    configure()
    from kb2_preference_optimization.artifacts import atomic_json, sha256
    from kb2_preference_optimization.hybrid_schedule.train import train
    root=args.output.resolve()
    if not root.is_relative_to(ROOT/'kb2_preference_optimization/outputs'):
        raise ValueError('Output must be under KB2 outputs')
    if args.worker:
        if args.seed is None: parser.error('--seed is required for a worker')
        train(BANK, root, arm=args.worker, device='cuda', steps=2 if args.smoke else 15000,
              seed=args.seed, eval_interval=1000, eval_limit=2 if args.smoke else 0,
              audit_pairs=2 if args.smoke else 32, beta=.1, dpo_weight=0. if args.worker=='native' else .03,
              lr=1e-6, background_weight=.5, lr_drop_step=1000, lr_drop_factor=.1,
              early_eval_interval=250, early_eval_until=2000)
        return
    if args.smoke: parser.error('--smoke requires --worker')
    from kb2_preference_optimization.hybrid.source import source_provenance, assert_provenance
    from kb2_preference_optimization.hybrid.bank import Bank
    import torch
    import importlib.metadata
    source=source_provenance('yolov8n');bank=Bank(BANK)
    assert_provenance(bank.manifest['provenance'],source)
    assert not bank.manifest['partial_bank']
    protected={str(p):sha256(p) for p in [Path(source['source_checkpoint']),BANK/'manifest.json',BANK/'pairs.jsonl',BANK/'contexts.jsonl',Path(__file__).resolve(),ROOT/'kb2_preference_optimization/hybrid_schedule/train.py']}
    plan={'seeds':[42,43,44],'arms':{'native':0.,'dpo':.03},'updates':15000,
          'primary_contrast':'paired DPO-native mAP50_95 at fixed step500',
          'secondary_contrasts':['selected best under same grid/min_delta and step0 eligible','fixed step15000'],
          'selection_min_delta_fraction':1e-4,'bank':str(BANK),'source':source,'protected':protected,
          'lr':{'initial':1e-6,'drop_after':1000,'factor':.1},'eval':{'early_interval':250,'early_until':2000,'later_interval':1000},
          'all_six_runs_fresh':True,'test_evaluation':False,'stopping':'complete all six arms regardless of favorable/unfavorable result',
          'runtime':{'python':sys.version,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
                     'packages':{n:importlib.metadata.version(n) for n in ('ultralytics','torchmetrics','faster-coco-eval','albumentations')},
                     'env':{k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_LAUNCH_BLOCKING')},'cpu_affinity':sorted(os.sched_getaffinity(0))}}
    root.mkdir(parents=True,exist_ok=False);atomic_json(root/'plan.json',plan)
    records=[]
    try:
        for seed in plan['seeds']:
            for arm in ('native','dpo'):
                folder=root/f'seed{seed}'/arm
                cmd=[sys.executable,'-u','-m','scripts.run_kb2_paired_control','--output',str(folder),'--worker',arm,'--seed',str(seed)]
                atomic_json(root/'status.json',{'stage':'training','seed':seed,'arm':arm,'command':cmd})
                print(json.dumps({'event':'start_arm','seed':seed,'arm':arm}),flush=True)
                with (root/f'seed{seed}_{arm}.log').open('x') as log:
                    subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                assert all(sha256(Path(p))==h for p,h in protected.items())
                print(json.dumps({'event':'completed_arm','seed':seed,'arm':arm}),flush=True)
            records.extend(verify_pair(root,seed));write_review(root,records)
            print(json.dumps({'event':'pair_verified','seed':seed,'summary':read(root/'paired_results.json')['summary']}),flush=True)
        atomic_json(root/'completed.json',{'plan':plan,'paired_results':read(root/'paired_results.json'),'protected_unchanged':True})
        atomic_json(root/'status.json',{'stage':'completed','seeds':plan['seeds']})
    except BaseException as error:
        atomic_json(root/'status.json',{'stage':'failed','error':str(error),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    main()
