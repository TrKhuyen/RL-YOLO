"""Reload a hybrid state_dict using its TPE architecture and compare on validation."""
import argparse
from pathlib import Path

import torch

from ..artifacts import atomic_json, sha256, source_checkpoint as supervised_checkpoint
from ..conditional.evaluation import evaluate
from ..conditional.policy import ConditionalAdapter, configure_numerics
from ..data import detection_loader
from . import EXPERIMENT_VERSION
from .source import assert_provenance, source_checkpoint, source_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--eval-limit', type=int, default=0, help='Partial validation smoke only')
    args = parser.parse_args()
    if args.eval_limit < 0:
        parser.error('eval-limit must be nonnegative')
    if args.output.exists():
        raise FileExistsError('Use a fresh evaluation output')
    configure_numerics()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    manifest = checkpoint['manifest']
    if manifest['experiment_version'] != EXPERIMENT_VERSION:
        raise ValueError('Expected a hybrid checkpoint')
    assert_provenance(manifest['provenance'], source_provenance('yolov8n'))
    source = ConditionalAdapter(source_checkpoint('yolov8n'), args.device)
    policy = ConditionalAdapter(source_checkpoint('yolov8n'), args.device)
    policy.model.load_state_dict(checkpoint['state_dict'], strict=True)
    if policy.frozen_digest() != manifest['frozen_state_sha256']:
        raise ValueError('Checkpoint changed frozen state')
    loader = detection_loader('valid', 640, 8)
    tpe, cache = evaluate(source, loader, image_limit=args.eval_limit)
    metrics, _ = evaluate(policy, loader, cache, image_limit=args.eval_limit)
    del source, policy
    s0 = ConditionalAdapter(supervised_checkpoint('yolov8n'), args.device)
    s0_metrics, _ = evaluate(s0, loader, image_limit=args.eval_limit)
    atomic_json(args.output, {'experiment_version': EXPERIMENT_VERSION, 'split': 'valid',
                             'partial_evaluation': bool(args.eval_limit),
                             'checkpoint_sha256': sha256(args.checkpoint), 'step': checkpoint['step'],
                             'S0': s0_metrics, 'TPE': tpe, 'policy': metrics})


if __name__ == '__main__':
    main()
