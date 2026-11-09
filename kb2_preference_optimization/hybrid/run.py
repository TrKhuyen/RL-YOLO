"""Fresh TPE -> DPO GPU continuation with S0/TPE/hybrid validation comparisons."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import torch

from ..artifacts import atomic_json, sha256, source_checkpoint as supervised_checkpoint
from ..conditional.evaluation import evaluate
from ..conditional.policy import ConditionalAdapter, configure_numerics
from ..data import detection_loader
from . import EXPERIMENT_VERSION
from .bank import Bank, generate
from .source import assert_provenance, source_provenance
from .train import train

METRICS = ('mAP50_95', 'mAP50', 'mAP75', 'precision', 'operating_recall',
           'f1', 'fp', 'fn', 'APs', 'APm', 'APl')


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def deltas(metrics, baseline):
    return {k: metrics[k] - baseline[k] for k in METRICS if k in metrics and k in baseline}


def baseline_signature(metrics):
    # Wall time varies even when predictions and every quality metric match.
    return {k: v for k, v in metrics.items() if k != 'evaluation_wall_seconds'}


def execute(args, output, provenance):
    atomic_json(output / 'request.json', {k: str(v) if isinstance(v, Path) else v
                                        for k, v in vars(args).items()})
    atomic_json(output / 'source.json', provenance)
    print(json.dumps({'event': 'start', 'experiment_version': EXPERIMENT_VERSION,
                      'output_root': str(output), 'TPE_trial': provenance['source_tpe']['trial'],
                      'initialization_and_reference': provenance['source_checkpoint'],
                      'arms': args.arms, 'steps_per_arm': args.steps}), flush=True)
    bank_path = args.preferences.resolve() if args.preferences else output / 'preferences'
    if not args.preferences:
        manifest = generate(bank_path, args.device, args.limit_images)
        print(json.dumps({'event': 'bank_complete', 'audit': manifest['audit']['totals']}), flush=True)
    bank = Bank(bank_path)
    assert_provenance(provenance, bank.manifest['provenance'])
    atomic_json(output / 'bank_audit.json', bank.manifest['audit'])
    del bank
    if args.audit_only:
        return

    configure_numerics()
    baseline = ConditionalAdapter(supervised_checkpoint('yolov8n'), args.device)
    baseline.model.requires_grad_(False)
    s0_metrics, _ = evaluate(baseline, detection_loader('valid', 640, 8), image_limit=args.eval_limit)
    atomic_json(output / 'S0_validation.json', s0_metrics)
    del baseline
    if torch.device(args.device).type == 'cuda':
        torch.cuda.empty_cache()

    results, tpe_metrics = {}, None
    for arm in args.arms:
        arm_dir = output / arm
        result = train(bank_path, arm_dir, arm=arm, device=args.device, steps=args.steps,
                       seed=args.seed, eval_interval=args.eval_interval, eval_limit=args.eval_limit,
                       beta=args.beta, dpo_weight=args.dpo_weight, lr=args.lr,
                       audit_pairs=args.audit_pairs, background_weight=args.background_weight)
        source = json.loads((arm_dir / 'source_validation.json').read_text())
        if tpe_metrics is not None and baseline_signature(source) != baseline_signature(tpe_metrics):
            raise ValueError('TPE step-zero metrics differ between arms')
        tpe_metrics = source
        rows = [json.loads(line) for line in (arm_dir / 'metrics.jsonl').read_text().splitlines()]
        best = next(r['metrics'] for r in rows
                    if r['event'] == 'validation' and r['step'] == result['best_step'])
        results[arm] = {**result, 'best_checkpoint': str(arm_dir / 'best.pt'),
                        'last_checkpoint': str(arm_dir / 'last.pt'), 'best_metrics': best,
                        'best_delta_vs_TPE': deltas(best, source),
                        'best_delta_vs_S0': deltas(best, s0_metrics),
                        'last_delta_vs_TPE': deltas(result['last_metrics'], source),
                        'last_delta_vs_S0': deltas(result['last_metrics'], s0_metrics)}
    matched = len({r['stream_sha256'] for r in results.values()}) == 1
    if not matched:
        raise ValueError('Native/DPO streams differ')
    assert_provenance(provenance, source_provenance('yolov8n'))
    comparison = {'experiment_version': EXPERIMENT_VERSION, 'split': 'valid',
                  'output_root': str(output), 'source': provenance['source_tpe'],
                  'matched_stream': matched, 'partial_bank': any(r['partial_bank'] for r in results.values()),
                  'partial_evaluation': bool(args.eval_limit), 'S0': s0_metrics,
                  'TPE': tpe_metrics, 'arms': results,
                  'delta_units': 'AP/P/R/F1 fractions; multiply by 100 for percentage points; FP/FN counts',
                  'interpretation': 'TPE -> native+DPO continuation; train/test performance not guaranteed. '
                                    'A native-only arm is needed to isolate auxiliary DPO contribution.'}
    if 'native' in results and 'dpo' in results:
        comparison['DPO_minus_native_best'] = deltas(results['dpo']['best_metrics'],
                                                    results['native']['best_metrics'])
        comparison['DPO_minus_native_last'] = deltas(results['dpo']['last_metrics'],
                                                    results['native']['last_metrics'])
    atomic_json(output / 'comparison.json', comparison)
    atomic_json(output / 'completed.json', comparison)
    print(json.dumps({'event': 'completed', 'output_root': str(output),
                      'comparison': str(output / 'comparison.json'),
                      'checkpoints': {arm: row['best_checkpoint'] for arm, row in results.items()}}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--arms', nargs='+', choices=['dpo', 'native'], default=['dpo'])
    parser.add_argument('--dpo-weight', type=float, default=.1)
    parser.add_argument('--beta', type=float, default=.1)
    parser.add_argument('--background-weight', type=float, default=.5)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--preferences', type=Path, help='Verified TPE hybrid bank only; S0 banks rejected')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--limit-images', type=int, default=0, help='Partial bank for smoke checks only')
    parser.add_argument('--eval-limit', type=int, default=0, help='Partial validation for smoke checks only')
    parser.add_argument('--audit-pairs', type=int, default=32)
    args = parser.parse_args(argv)
    if (min(args.steps, args.eval_interval, args.audit_pairs) < 1 or args.seed < 0
            or min(args.limit_images, args.eval_limit) < 0
            or len(set(args.arms)) != len(args.arms)
            or not all(math.isfinite(v) and v > 0 for v in (args.beta, args.dpo_weight, args.lr))
            or not math.isfinite(args.background_weight) or not 0 <= args.background_weight <= 1
            or (args.preferences and args.limit_images)):
        parser.error('Invalid budgets/weights, repeated arms or partial-bank reuse')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; activate the CUDA environment/device')
    provenance = source_provenance('yolov8n')
    if args.check_only:
        print(json.dumps({'event': 'preflight_pass', 'experiment_version': EXPERIMENT_VERSION,
                          'device': str(device), 'source': provenance['source_tpe'],
                          'shared_S0': provenance['shared_S0'],
                          'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}), flush=True)
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    trial = provenance['source_tpe']['trial']
    output = (args.output_root or Path('kb2_preference_optimization/outputs') /
              f'yolov8n_hybrid_tpe{trial}_s{args.seed}_{stamp}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'run.log').open('x', buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            execute(args, output, provenance)


if __name__ == '__main__':
    main()
