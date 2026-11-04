"""Run the balanced recovery method for a longer budget in a fresh directory.

The recovery trainer's experimental/pilot flag remains true: a larger budget
does not by itself validate the method or authorize final test evaluation.
"""
import argparse
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--output-root', type=Path,
                        help='fresh directory; defaults to a unique Bangkok-time run name')
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error('steps must be positive')
    if args.output_root is None:
        stamp = datetime.now(ZoneInfo('Asia/Bangkok')).strftime('%Y%m%d_%H%M%S_%f')
        args.output_root = ROOT / 'kb2_preference_optimization/outputs' / f'yolov8n_recovery_long_{stamp}'
    args.output_root = args.output_root.resolve()
    if args.output_root.exists():
        parser.error(f'Output already exists; refusing to overwrite: {args.output_root}')
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    from kb2_preference_optimization.artifacts import atomic_json, sha256
    from kb2_preference_optimization.recovery.generate import generate
    from kb2_preference_optimization.recovery.train_pilot import train

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; check the GPU-enabled environment')
    print(f'New run: {args.output_root}', flush=True)
    print(f'Checkpoints: {args.output_root / "pilot_balanced"}', flush=True)
    generate(args.output_root, args.device)
    atomic_json(args.output_root / 'long_run.json', {
        'kind': 'long_budget_recovery_experiment', 'steps': args.steps, 'seed': 42,
        'device': args.device, 'arm': 'balanced', 'initialization': 'supervised_S0',
        'resume': False, 'experimental_pilot_flag_retained': True,
        'output': str(args.output_root), 'runner_sha256': sha256(Path(__file__)),
    })
    train(args.output_root, 'balanced', args.device, args.steps)
    print(f'Complete. Best: {args.output_root / "pilot_balanced/best.pt"}', flush=True)
    print(f'Last: {args.output_root / "pilot_balanced/last.pt"}', flush=True)


if __name__ == '__main__':
    main()
