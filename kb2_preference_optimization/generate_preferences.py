"""Generate offline chosen/rejected complete actions on TRAIN only."""
import argparse
import itertools
import json
from pathlib import Path

import torch

from .adapter import DFLAdapter, SUPPORTED_MODELS
from .artifacts import (append_jsonl, atomic_json, seed_everything, sha256,
                        source_checkpoint, source_provenance, tensor_digest)
from .data import DetectionDataset
from .policy import PolicyConfig, render_action
from .quality import QualityConfig, quality_score, select_pair
from .preview import save_preview


def generate(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a fresh preference directory: {output}')
    if args.draws < 2 or args.limit < 0:
        raise ValueError('draws >=2 and limit >=0 required')
    seed_everything(args.seed)
    policy = PolicyConfig(args.class_temperature, args.box_temperature,
                          args.background_logit)
    quality = QualityConfig(args.detection_weight, args.min_gap, args.f_beta)
    provenance = source_provenance(args.model)
    adapter = DFLAdapter(source_checkpoint(args.model), args.device)
    adapter.model.requires_grad_(False)
    before = tensor_digest(adapter.model)
    dataset = DetectionDataset('train', args.image_size)
    generator = torch.Generator(device=adapter.device).manual_seed(args.seed)
    output.mkdir(parents=True)
    records_path = output / 'pairs.jsonl'
    records_path.touch()
    summary = {'images_seen': 0, 'pairs_kept': 0, 'images_tied_or_identical': 0,
               'sampled_outputs': 0, 'empty_outputs': 0, 'pairwise_comparisons': 0,
               'distinct_output_comparisons': 0, 'classes_in_kept_gt': {}, 'quality_gaps': []}
    with torch.no_grad():
        for index in range(min(len(dataset), args.limit or len(dataset))):
            image, target = dataset[index]
            distributions, anchors, strides, reference_scores = adapter.distributions(image[None], policy)
            distribution = distributions[0]
            actions = [distribution.sample(generator) for _ in range(args.draws)]
            predictions = [render_action(distribution, action, anchors, strides,
                                         reference_scores[0], args.image_size) for action in actions]
            qualities = [quality_score(pred, target, quality) for pred in predictions]
            signatures = [sorted((int(c), *b.tolist()) for c, b in zip(p['labels'], p['boxes']))
                          for p in predictions]
            comparisons = list(itertools.combinations(range(args.draws), 2))
            summary['pairwise_comparisons'] += len(comparisons)
            summary['distinct_output_comparisons'] += sum(signatures[a] != signatures[b] for a, b in comparisons)
            summary['images_seen'] += 1
            summary['sampled_outputs'] += args.draws
            summary['empty_outputs'] += sum(not len(p['boxes']) for p in predictions)
            selected = select_pair(predictions, qualities, quality.min_gap)
            if selected is None:
                summary['images_tied_or_identical'] += 1
                continue
            chosen, rejected = selected
            record = {'image_id': index, 'image_path': target['image_path'],
                      'chosen': actions[chosen], 'rejected': actions[rejected],
                      'reference_chosen_logp': float(distribution.log_prob(actions[chosen])),
                      'reference_rejected_logp': float(distribution.log_prob(actions[rejected])),
                      'chosen_quality': qualities[chosen], 'rejected_quality': qualities[rejected],
                      'chosen_detection': {k: v.tolist() for k, v in predictions[chosen].items()},
                      'rejected_detection': {k: v.tolist() for k, v in predictions[rejected].items()}}
            append_jsonl(records_path, record)
            summary['pairs_kept'] += 1
            if summary['pairs_kept'] <= 8:
                save_preview(output / 'previews' / f'{index:05d}.png', image, target, record)
            summary['quality_gaps'].append(qualities[chosen]['score'] - qualities[rejected]['score'])
            for label in set(target['labels'].tolist()):
                key = str(label)
                summary['classes_in_kept_gt'][key] = summary['classes_in_kept_gt'].get(key, 0) + 1
            if index == 0 or (index + 1) % 100 == 0:
                print(f'preferences: {index + 1} images, {summary["pairs_kept"]} pairs', flush=True)
    if before != tensor_digest(adapter.model):
        raise RuntimeError('Frozen reference parameters/buffers changed')
    atomic_json(output / 'manifest.json', {
        'provenance': provenance, 'policy': policy.to_dict(), 'quality': quality.to_dict(),
        'image_size': args.image_size, 'split': 'train', 'seed': args.seed, 'draws': args.draws,
        'limit': args.limit, 'pairs_sha256': sha256(records_path),
        'reference_state_sha256': before, 'summary': summary,
        'status': 'complete' if summary['pairs_kept'] else 'no_usable_pairs',
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary['pairs_kept']:
        raise ValueError('No usable preferences: inspect sampling/quality diagnostics before training')
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=SUPPORTED_MODELS, default='yolov8n')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--draws', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='0 means full train; nonzero is a pilot')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--class-temperature', type=float, default=1.)
    parser.add_argument('--box-temperature', type=float, default=1.)
    parser.add_argument('--background-logit', type=float, default=0.)
    parser.add_argument('--detection-weight', type=float, default=.8)
    parser.add_argument('--min-gap', type=float, default=.02)
    parser.add_argument('--f-beta', type=float, default=2., help='F-beta in quality; 2 emphasizes recall, 1 replays F1')
    return parser.parse_args(argv)


if __name__ == '__main__':
    generate(parse_args())
