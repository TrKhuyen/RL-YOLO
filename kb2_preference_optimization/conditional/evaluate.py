"""Validate a V0 checkpoint with ordinary inference; no test tuning entrypoint."""
import argparse
from pathlib import Path

import torch

from ..artifacts import assert_provenance, atomic_json, sha256, source_checkpoint, source_provenance
from ..data import detection_loader
from . import VERSION, code_fingerprints
from .evaluation import evaluate
from .policy import ConditionalAdapter, configure_numerics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh evaluation output')
    configure_numerics()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['objective_version'] != VERSION:
        raise ValueError('Checkpoint uses a different objective')
    manifest = checkpoint['manifest']
    assert_provenance(manifest['provenance'], source_provenance('yolov8n'))
    if manifest['conditional_code_sha256'] != code_fingerprints():
        raise ValueError('Conditional code changed')
    source = ConditionalAdapter(source_checkpoint('yolov8n'), args.device)
    policy = ConditionalAdapter(source_checkpoint('yolov8n'), args.device)
    policy.model.load_state_dict(checkpoint['state_dict'], strict=True)
    if policy.frozen_digest() != manifest['frozen_state_sha256']:
        raise ValueError('Checkpoint changed frozen model state')
    loader = detection_loader('valid', 640, 8)
    baseline, cache = evaluate(source, loader)
    metrics, _ = evaluate(policy, loader, cache)
    atomic_json(args.output, {'objective_version': VERSION, 'checkpoint_sha256': sha256(args.checkpoint),
                             'step': checkpoint['step'], 'split': 'valid', 'S0': baseline, 'policy': metrics})


if __name__ == '__main__':
    main()
