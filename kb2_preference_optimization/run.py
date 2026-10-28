"""Single-method KB2 runner: preflight -> TRAIN preferences -> structured DPO."""
import argparse
from pathlib import Path

import torch

from .adapter import DFLAdapter, SUPPORTED_MODELS
from .artifacts import source_checkpoint, source_provenance
from .generate_preferences import generate, parse_args as preference_args
from .train_dpo import train, parse_args as training_args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=SUPPORTED_MODELS, default='yolov8n')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-root')
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--eval-limit', type=int, default=0)
    parser.add_argument('--pilot-images', type=int, default=0)
    parser.add_argument('--draws', type=int, default=4)
    parser.add_argument('--min-gap', type=float, default=.02)
    parser.add_argument('--class-temperature', type=float, default=1.)
    parser.add_argument('--box-temperature', type=float, default=1.)
    parser.add_argument('--background-logit', type=float, default=0.)
    parser.add_argument('--dpo-weight', type=float, default=.01)
    parser.add_argument('--beta', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu for a pilot')
    if args.check_only:
        record = source_provenance(args.model)
        adapter = DFLAdapter(source_checkpoint(args.model), args.device)
        print(f'PASS {args.model}: source={record["source_sha256"]}; '
              f'data={record["dataset"]["sha256"]}; DFL bins={adapter.head.reg_max}')
        return
    if not args.output_root:
        parser.error('--output-root is required unless --check-only')
    root = Path(args.output_root).resolve()
    if root.exists():
        raise FileExistsError(f'Use a fresh output root: {root}')
    shared = ['--model', args.model, '--device', args.device, '--seed', str(args.seed)]
    generate(preference_args(shared + [
        '--output', str(root / 'preferences'), '--image-size', str(args.image_size),
        '--draws', str(args.draws), '--limit', str(args.pilot_images), '--min-gap', str(args.min_gap),
        '--class-temperature', str(args.class_temperature), '--box-temperature', str(args.box_temperature),
        '--background-logit', str(args.background_logit)]))
    train(training_args(shared + [
        '--preferences', str(root / 'preferences'), '--output', str(root / 'training'),
        '--steps', str(args.steps), '--batch-size', str(args.batch_size),
        '--eval-interval', str(args.eval_interval), '--eval-limit', str(args.eval_limit),
        '--dpo-weight', str(args.dpo_weight), '--beta', str(args.beta)]))


if __name__ == '__main__':
    main()
