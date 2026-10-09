"""Run controlled KB2 ablations with one shared training/validation protocol."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from feedback_loss import DPO_PAIR_SOURCE, DPO_OBJECTIVE_VERSION

SCRIPT_DIR = Path(__file__).resolve().parent

VARIANTS = {
    'baseline': {'sampling_strategy': 'shuffle', 'feedback_alpha': 0.0},
    'sampling': {'sampling_strategy': 'feedback', 'feedback_alpha': 0.0},
    'hybrid': {'sampling_strategy': 'feedback', 'feedback_alpha': 0.10},
    'pairwise': {'sampling_strategy': 'shuffle', 'feedback_alpha': 0.0},
    'dpo': {'sampling_strategy': 'shuffle', 'feedback_alpha': 0.0},
}


def base_checkpoint_path(args):
    return (Path(args.checkpoint).resolve() if args.checkpoint else
            SCRIPT_DIR.parent / 'kb1_reward_guided_training' / 'checkpoint_based'
            / args.model / 'weights' / 'best.pt')


def build_command(args, name):
    variant = VARIANTS[name]
    output = Path(args.output_dir).resolve() / f'{name}_last.pt'
    return [
        sys.executable, str(SCRIPT_DIR / 'train_feedback.py'),
        '--model', args.model, '--steps', str(args.steps),
        '--checkpoint', str(base_checkpoint_path(args)),
        '--img-size', str(args.img_size),
        '--data-root', str(Path(args.data_root).resolve()),
        '--feedback', str(Path(args.feedback).resolve()) if args.feedback
        else str(SCRIPT_DIR / 'feedback_data_clean' / f'{args.model}_train.jsonl'),
        '--batch-size', str(args.batch_size), '--workers', str(args.workers),
        '--lr', str(args.lr), '--seed', str(args.seed), '--device', args.device,
        '--sampling-strategy', variant['sampling_strategy'],
        '--feedback-alpha', str(variant['feedback_alpha']),
        '--pairwise-alpha', str(args.pairwise_alpha if name == 'pairwise' else 0.0),
        '--pairwise-margin', str(args.pairwise_margin),
        '--dpo-alpha', str(args.dpo_alpha if name == 'dpo' else 0.0),
        '--dpo-beta', str(args.dpo_beta),
        '--sampling-strength', str(args.sampling_strength),
        '--eval-interval', str(args.eval_interval),
        '--val-batch-size', str(args.val_batch_size),
        '--val-conf', str(args.val_conf), '--val-iou', str(args.val_iou),
        '--patience', str(args.patience), '--save-interval', str(args.save_interval),
        '--output', str(output),
    ]


def checkpoint_result(path, name):
    data = torch.load(path, map_location='cpu', weights_only=False)
    validation = data.get('training_state', {}).get('validation', {})
    if not validation:
        raise ValueError(f'Best checkpoint has no validation metrics: {path}')
    state = data.get('training_state', {})
    config = data.get('config', {})
    return {'variant': name, 'step': int(data['step']),
            'checkpoint': str(path),
            'training_method': data.get('training_method'),
            'pairwise_pairs_seen': state.get('pairwise_pairs_seen', 0),
            'pairwise_active_steps': state.get('pairwise_active_steps', 0),
            'dpo_pairs_seen': state.get('dpo_pairs_seen', 0),
            'dpo_active_steps': state.get('dpo_active_steps', 0),
            'dpo_diagnostics': state.get('dpo_diagnostics', {}),
            'base_checkpoint_sha256': config.get('base_checkpoint_sha256'),
            'reference_checkpoint_sha256': config.get('reference_checkpoint_sha256'),
            'dpo_policy': config.get('dpo_policy'),
            'dpo_pair_source': config.get('dpo_pair_source'),
            'dpo_objective_version': config.get('dpo_objective_version'),
            **validation}


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
        'steps': args.steps, 'batch_size': args.batch_size, 'img_size': args.img_size,
        'lr': args.lr, 'checkpoint': str(base_checkpoint_path(args)),
        'seed': args.seed, 'eval_interval': args.eval_interval,
        'patience': args.patience, 'save_interval': args.save_interval,
        'pairwise_alpha': args.pairwise_alpha,
        'pairwise_margin': args.pairwise_margin,
        'dpo_alpha': args.dpo_alpha, 'dpo_beta': args.dpo_beta,
        'dpo_policy': 'gt_class_anchor_softmax_v1',
        'dpo_pair_source': DPO_PAIR_SOURCE,
        'dpo_objective_version': DPO_OBJECTIVE_VERSION,
        'selection_metric': 'mAP50-95', 'val_conf': args.val_conf,
        'val_iou': args.val_iou,
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
    parser.add_argument('--model', default='yolov8n',
                        choices=['yolov8n', 'yolov8s', 'yolov11n', 'yolov11s', 'yolo26n'])
    parser.add_argument('--checkpoint')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--data-root', default=str(SCRIPT_DIR.parent / 'pre-data/data/v2i_cleanned'))
    parser.add_argument('--feedback')
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sampling-strength', type=float, default=1.0)
    parser.add_argument('--pairwise-alpha', type=float, default=0.01)
    parser.add_argument('--pairwise-margin', type=float, default=0.1)
    parser.add_argument('--dpo-alpha', type=float, default=0.1)
    parser.add_argument('--dpo-beta', type=float, default=0.1)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--val-batch-size', type=int, default=8)
    parser.add_argument('--val-conf', type=float, default=.001)
    parser.add_argument('--val-iou', type=float, default=.45)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--save-interval', type=int, default=2000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', default=str(SCRIPT_DIR / 'checkpoint_preference_optimization' / 'ablation'))
    args = parser.parse_args(argv)
    if min(args.steps, args.eval_interval, args.patience) < 1 or args.save_interval < 0:
        parser.error('steps, eval_interval and patience must be positive; save_interval must be non-negative')
    for name in ('pairwise_alpha', 'pairwise_margin', 'dpo_alpha'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f'{name} must be finite and non-negative')
    if not math.isfinite(args.dpo_beta) or args.dpo_beta <= 0:
        parser.error('dpo_beta must be finite and positive')
    if 'pairwise' in args.variants and args.pairwise_alpha == 0:
        parser.error('pairwise variant requires positive --pairwise-alpha')
    if 'dpo' in args.variants and args.dpo_alpha == 0:
        parser.error('dpo variant requires positive --dpo-alpha')
    return args


if __name__ == '__main__':
    run(parse_args())
