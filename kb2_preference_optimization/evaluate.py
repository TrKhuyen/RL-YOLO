"""Compare S0 and a completed structured-DPO checkpoint with identical inference."""
import argparse
import json
from pathlib import Path

import torch

from . import OBJECTIVE_VERSION
from .adapter import DFLAdapter, SUPPORTED_MODELS
from .artifacts import (assert_provenance, atomic_json, sha256,
                        source_checkpoint, source_provenance)
from .data import detection_loader
from .metrics import evaluate_adapter


def evaluate(args):
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    provenance = source_provenance(args.model)
    adapter = DFLAdapter(source_checkpoint(args.model), args.device)
    payload = None
    image_size = args.image_size
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).resolve()
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if payload.get('objective_version') != OBJECTIVE_VERSION:
            raise ValueError('Historical candidate-DPO checkpoints are incompatible')
        manifest_path = checkpoint.parent / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        completed = json.loads((checkpoint.parent / 'completed.json').read_text(encoding='utf-8'))
        if (manifest != payload['manifest'] or completed['status'] != 'complete'
                or completed['manifest_sha256'] != sha256(manifest_path)):
            raise ValueError('Run metadata is incomplete or mismatched')
        if checkpoint.name not in ('best.pt', 'last.pt'):
            raise ValueError('Only completed best.pt/last.pt are supported')
        if sha256(checkpoint) != completed[f'{checkpoint.stem}_sha256']:
            raise ValueError('Checkpoint hash mismatch')
        assert_provenance(manifest['provenance'], provenance)
        image_size = manifest['config']['image_size']
        if args.split == 'test' and (manifest['config']['pilot'] or args.limit):
            raise ValueError('Pilot runs/partial evaluation cannot be presented as final test')
    loader = detection_loader(args.split, image_size, args.batch_size)
    report = {'provenance': provenance, 'split': args.split, 'image_size': image_size,
              'supervised': evaluate_adapter(adapter, loader, args.device, args.limit)}
    if payload is not None:
        adapter.model.load_state_dict(payload['state_dict'], strict=True)
        report['dpo'] = evaluate_adapter(adapter, loader, args.device, args.limit)
        report['checkpoint'] = str(Path(args.checkpoint).resolve())
        report['checkpoint_sha256'] = sha256(args.checkpoint)
        report['selected_step'] = payload['step']
        report['pilot'] = payload['manifest']['config']['pilot']
        report['delta'] = {key: report['dpo'][key] - report['supervised'][key]
                           for key in ('mAP50_95', 'mAP50', 'mAP75', 'AR300', 'precision',
                                       'operating_recall', 'f1', 'fp_per_image', 'fn_rate')}
    atomic_json(destination, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=SUPPORTED_MODELS, default='yolov8n')
    parser.add_argument('--checkpoint', help='omit for supervised-only evaluation')
    parser.add_argument('--split', choices=('valid', 'test'), default='valid')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.limit < 0 or args.batch_size < 1:
        parser.error('limit must be nonnegative and batch size positive')
    return args


if __name__ == '__main__':
    evaluate(parse_args())
