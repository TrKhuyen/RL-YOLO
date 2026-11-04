"""Predeclared 2x2 pilot: native background weight x conditional DPO."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

import torch

from ..artifacts import atomic_json, sha256, source_checkpoint, source_provenance
from ..conditional.train import train as original_train
from . import LOSS_VERSION, code_fingerprints
from .train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preferences', required=True, type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--background-weight', type=float, default=.5)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.background_weight < 1 or min(args.steps, args.eval_interval) < 1:
        parser.error('Require 0 <= background weight < 1 and positive budgets')
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    root = (args.output_root or Path('kb2_preference_optimization/outputs') /
            f'yolov8n_native_background_s{args.seed}_{stamp}').resolve()
    root.mkdir(parents=True, exist_ok=False)
    source_hash = sha256(source_checkpoint('yolov8n'))
    plan = {'experiment': LOSS_VERSION, 'background_weight': args.background_weight,
            'foreground_weight': 1., 'steps': args.steps, 'seed': args.seed,
            'eval_interval': args.eval_interval, 'validation_images': 479,
            'native_batch': 4, 'preference_batch': 2, 'beta': .1, 'dpo_weight': .01,
            'lr': 1e-6, 'weight_decay': 5e-4, 'clip_norm': 1.,
            'primary_confidence': .25, 'fp_tolerance': .05, 'ap_tolerance': .001,
            'selection': 'Unchanged V0 gates; S0 eligible; maximize macro recall',
            'arms': ['original_native', 'original_dpo', 'reduced_native', 'reduced_dpo'],
            'preferences': str(args.preferences.resolve()), 'device': args.device,
            'source_sha256': source_hash, 'provenance': source_provenance('yolov8n'),
            'code_sha256': code_fingerprints(),
            'scope': 'One prespecified alpha; matched single-seed pilot; no test or data edits'}
    atomic_json(root / 'plan.json', plan)
    print(json.dumps({'event': 'start', 'output_root': str(root), 'plan': plan}), flush=True)
    results = {}
    for label in plan['arms']:
        arm = label.rsplit('_', 1)[1]
        kwargs = dict(bank_path=args.preferences, output=root / label, arm=arm,
                      device=args.device, steps=args.steps, seed=args.seed,
                      eval_interval=args.eval_interval)
        result = (original_train(**kwargs) if label.startswith('original') else
                  train(**kwargs, background_weight=args.background_weight))
        results[label] = result
        atomic_json(root / 'progress.json', results)
        print(json.dumps({'event': 'arm_complete', 'label': label,
                          'best_step': result['best_step'], 'metrics': result['last_metrics']}), flush=True)
    if len({r['stream_sha256'] for r in results.values()}) != 1:
        raise ValueError('Input streams differ')
    if len({(root / label / 'stream.jsonl').read_bytes() for label in results}) != 1:
        raise ValueError('Actual input stream bytes differ')
    if sha256(source_checkpoint('yolov8n')) != source_hash or code_fingerprints() != plan['code_sha256']:
        raise ValueError('Source or experiment code changed')
    atomic_json(root / 'completed.json', {'plan': plan, 'arms': results, 'matched_stream': True})
    print(json.dumps({'event': 'completed', 'output_root': str(root), 'matched_stream': True}), flush=True)


if __name__ == '__main__':
    main()
