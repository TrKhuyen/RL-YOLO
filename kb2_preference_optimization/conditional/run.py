"""Fresh GPU experiment: audit bank, matched native and conditional DPO arms."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from ..artifacts import atomic_json, source_provenance
from . import VERSION
from .bank import generate
from .train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['yolov8n'], default='yolov8n')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--preferences', type=Path, help='Verified V0 bank; old whole-action banks are rejected')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--arms', nargs='+', choices=['native', 'dpo', 'bce'], default=['native', 'dpo'])
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--eval-limit', type=int, default=0, help='Partial validation for smoke checks only')
    parser.add_argument('--limit-images', type=int, default=0, help='Partial bank for smoke checks only')
    parser.add_argument('--audit-pairs', type=int, default=32)
    parser.add_argument('--beta', type=float, default=.1)
    parser.add_argument('--dpo-weight', type=float, default=.01)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--fp-tolerance', type=float, default=.05)
    parser.add_argument('--ap-tolerance', type=float, default=.001)
    args = parser.parse_args()
    if (min(args.steps, args.eval_interval, args.audit_pairs) < 1
            or min(args.eval_limit, args.limit_images) < 0 or len(set(args.arms)) != len(args.arms)):
        parser.error('Invalid budget, limits or repeated arms')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; activate the CUDA environment/device')
    if args.check_only:
        provenance = source_provenance(args.model)
        print(json.dumps({'objective_version': VERSION, 'device': str(device),
                          'source_sha256': provenance['source_sha256'],
                          'dataset_sha256': provenance['dataset']['sha256'],
                          'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}), flush=True)
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = (args.output_root or Path('kb2_preference_optimization/outputs') /
              f'yolov8n_conditional_s{args.seed}_{stamp}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'request.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    print(json.dumps({'event': 'start', 'objective_version': VERSION, 'output_root': str(output)}), flush=True)
    bank = args.preferences.resolve() if args.preferences else output / 'preferences'
    if args.preferences and args.limit_images:
        raise ValueError('--limit-images applies only to a new bank')
    if not args.preferences:
        manifest = generate(bank, args.device, args.limit_images)
        print(json.dumps({'event': 'bank_complete', 'audit': manifest['audit']['totals']}), flush=True)
    if args.audit_only:
        # Existing banks must also be verified, never accepted by path alone.
        from .bank import Bank
        checked = Bank(bank)
        atomic_json(output / 'audit_completed.json', checked.manifest['audit'])
        return
    results = []
    for arm in args.arms:
        results.append(train(bank, output / arm, arm, args.device, args.steps, args.seed,
                             args.eval_interval, args.eval_limit, args.beta, args.dpo_weight,
                             args.lr, args.fp_tolerance, args.ap_tolerance, args.audit_pairs))
    if len({r['stream_sha256'] for r in results}) != 1:
        raise ValueError('Control/DPO streams differ')
    comparison = {'objective_version': VERSION, 'output_root': str(output),
                  'matched_stream': True, 'partial_bank': any(r['partial_bank'] for r in results),
                  'partial_evaluation': bool(args.eval_limit),
                  'interpretation': 'Technical pilot; train margins do not establish validation/test improvement',
                  'arms': {r['arm']: {k: r[k] for k in ('steps', 'best_step', 'stream_sha256',
                            'best_sha256', 'last_sha256', 'last_metrics')} for r in results}}
    atomic_json(output / 'completed.json', comparison)
    print(json.dumps({'event': 'completed', 'output_root': str(output), 'matched_stream': True}), flush=True)


if __name__ == '__main__':
    main()
