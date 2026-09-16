"""Run controlled KB2 ablations with one shared training/validation protocol."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent

VARIANTS = {
    'baseline': {'sampling_strategy': 'shuffle', 'feedback_alpha': 0.0},
    'sampling': {'sampling_strategy': 'feedback', 'feedback_alpha': 0.0},
    'hybrid': {'sampling_strategy': 'feedback', 'feedback_alpha': 0.10},
}


def build_command(args, name):
    variant = VARIANTS[name]
    output = Path(args.output_dir).resolve() / f'{name}_last.pt'
    return [
        sys.executable, str(SCRIPT_DIR / 'train_feedback.py'),
        '--model', args.model, '--steps', str(args.steps),
        '--batch-size', str(args.batch_size), '--workers', str(args.workers),
        '--lr', str(args.lr), '--seed', str(args.seed), '--device', args.device,
        '--sampling-strategy', variant['sampling_strategy'],
        '--feedback-alpha', str(variant['feedback_alpha']),
        '--sampling-strength', str(args.sampling_strength),
        '--eval-interval', str(args.eval_interval),
        '--val-batch-size', str(args.val_batch_size),
        '--patience', str(args.patience), '--save-interval', '0',
        '--output', str(output),
    ]


def checkpoint_result(path, name):
    data = torch.load(path, map_location='cpu', weights_only=False)
    validation = data.get('training_state', {}).get('validation', {})
    if not validation:
        raise ValueError(f'Best checkpoint has no validation metrics: {path}')
    return {'variant': name, 'step': int(data['step']),
            'checkpoint': str(path), **validation}


def run(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.variants:
        print(f'\n[Ablation] {name}', flush=True)
        subprocess.run(build_command(args, name), check=True, cwd=SCRIPT_DIR.parent)
        best = output_dir / f'{name}_last_best.pt'
        results.append(checkpoint_result(best, name))
    report = {'protocol': {
        'steps': args.steps, 'batch_size': args.batch_size, 'lr': args.lr,
        'seed': args.seed, 'eval_interval': args.eval_interval,
        'selection_metric': 'mAP50-95', 'val_conf': .25, 'val_iou': .45,
    }, 'results': results}
    report_path = output_dir / 'ablation_results.json'
    temporary = report_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(report_path)
    print(f'\nAblation report: {report_path}')
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Controlled KB2 ablation')
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS),
                        default=list(VARIANTS))
    parser.add_argument('--model', default='yolov8n')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sampling-strength', type=float, default=1.0)
    parser.add_argument('--eval-interval', type=int, default=250)
    parser.add_argument('--val-batch-size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', default=str(SCRIPT_DIR / 'ablation'))
    return parser.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
