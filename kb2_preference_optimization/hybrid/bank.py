"""Versioned TPE bank; generation/judging/replay match conditional V0.

Kept separate so adding hybrid support does not invalidate historical banks.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou

from ..artifacts import (append_jsonl, atomic_json, sha256,
                         tensor_digest)
from ..data import DetectionDataset
from ..conditional import VERSION, code_fingerprints
from .source import source_checkpoint, source_provenance, assert_provenance
from ..conditional.pairs import PairConfig, build_pairs, judge, matches, render
from ..conditional.policy import ConditionalAdapter, configure_numerics


def probe_image(image_path):
    return int(hashlib.sha256(image_path.encode()).hexdigest()[:8], 16) % 10 == 0


def generate(output, device='cuda', limit_images=0):
    configure_numerics()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = source_provenance('yolov8n')
    fingerprints = code_fingerprints()
    dataset = DetectionDataset('train', 640)
    config = PairConfig()
    adapter = ConditionalAdapter(source_checkpoint('yolov8n'), device)
    adapter.model.requires_grad_(False)
    initial_digest = tensor_digest(adapter.model)
    groups, classes, density, sizes, overlap_bins = (defaultdict(Counter) for _ in range(5))
    totals, pairs = Counter(), []
    maximum = min(limit_images or len(dataset), len(dataset))
    # Empty banks still have hashable files and an explicit audit.
    for name in ('contexts.jsonl', 'pairs.jsonl'):
        (output / name).touch()
    with torch.no_grad():
        for image_id in range(maximum):
            image, target = dataset[image_id]
            logits, boxes = adapter.raw(image[None])
            context, records, diagnostic = build_pairs(boxes[0], logits[0], target, config)
            # Check renderer against actual ordinary S0, not just a second copy
            # of the preference renderer. This detects prefilter/NMS mismatches.
            ordinary = adapter.predict(image[None], .001, .6, 300)[0]
            actual_gt, actual_fp, _ = matches(ordinary, target, .25)
            baseline = render(context, config=config)
            op = ordinary['scores'] >= .25
            if (actual_gt != set(diagnostic['s0_gt']) or actual_fp != diagnostic['s0_fp']
                    or len(baseline['boxes']) != int(op.sum())
                    or not torch.allclose(baseline['boxes'], ordinary['boxes'][op], atol=1e-3, rtol=0)
                    or not torch.equal(baseline['labels'], ordinary['labels'][op])):
                raise ValueError(f'Canonical S0/context disagreement at image {image_id}')
            context.update(image_id=image_id, image_path=target['image_path'])
            append_jsonl(output / 'contexts.jsonl', context)
            n = len(target['labels'])
            d = '1-2' if n <= 2 else '3-5' if n <= 5 else '6+'
            heldout = probe_image(target['image_path'])
            rescued = {r['target_gt'] for r in records if r['group'] == 'recovery'}
            pair_gts = {r['target_gt'] for r in records if r['target_gt'] is not None}
            gt_iou = box_iou(target['boxes'], target['boxes'])
            gt_iou.fill_diagonal_(0)
            totals.update(images=1, gt=n, s0_fn=diagnostic['s0_fn'], s0_fp=actual_fp,
                          pairs=len(records), recovery_gt=len(rescued),
                          gate_rejected=sum(diagnostic['gate_rejected'].values()),
                          probe_images=int(heldout), pair_images=int(bool(records)),
                          no_pair_images=int(not records))
            density[d].update(images=1, gt=n, s0_fn=diagnostic['s0_fn'],
                              pairs=len(records), recovery_gt=len(rescued))
            for g, c in enumerate(target['labels'].tolist()):
                area = float((target['boxes'][g, 2:] - target['boxes'][g, :2]).prod())
                size = 'small' if area < 32**2 else 'medium' if area < 96**2 else 'large'
                overlap = 'iou>=.3' if float(gt_iou[g].max()) >= .3 else 'iou<.3'
                values = dict(gt=1, s0_fn=int(g not in actual_gt),
                              recovery_gt=int(g in rescued), pair_gt=int(g in pair_gts))
                classes[str(c)].update(values)
                sizes[size].update(values)
                overlap_bins[overlap].update(values)
            for record in records:
                record.update(pair_id=len(pairs), image_id=image_id,
                              image_path=target['image_path'], probe=heldout)
                groups[record['group']].update(pairs=1, train=int(not heldout), probe=int(heldout))
                append_jsonl(output / 'pairs.jsonl', record)
                pairs.append(record)
            if (image_id + 1) % 100 == 0:
                print(json.dumps({'event': 'pair_generation', **totals}), flush=True)
    if tensor_digest(adapter.model) != initial_digest:
        raise ValueError('Frozen S0 changed during generation')
    assert_provenance(provenance, source_provenance('yolov8n'))
    if fingerprints != code_fingerprints():
        raise ValueError('Conditional code changed during generation')
    # Missing classes are explicit, rather than disappearing from the report.
    for c in range(dataset.classes):
        classes[str(c)].update(gt=0, s0_fn=0, recovery_gt=0, pair_gt=0)
    manifest = {'objective_version': VERSION, 'provenance': provenance,
                'conditional_code_sha256': fingerprints, 'config': config.manifest(),
                'image_size': 640, 'reference_state_sha256': initial_digest,
                'pairs_sha256': sha256(output / 'pairs.jsonl'),
                'contexts_sha256': sha256(output / 'contexts.jsonl'),
                'partial_bank': maximum < len(dataset), 'processed_images': maximum,
                'probe_scope': 'train images excluded from DPO only; native may use them; not independent holdout',
                'judge_scope': 'keep bypasses operating confidence; scores/boxes remain S0; whole-image NMS then GT',
                'numerics': {'tf32': False, 'amp': False, 'generation_batch': 1},
                'audit': {'totals': dict(totals), 'groups': dict(groups), 'per_class': dict(classes),
                          'density': dict(density), 'size': dict(sizes), 'gt_overlap': dict(overlap_bins)}}
    atomic_json(output / 'manifest.json', manifest)
    return manifest


class Bank:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest['objective_version'] != VERSION:
            raise ValueError('Old preference objective cannot be reused')
        assert_provenance(self.manifest['provenance'], source_provenance('yolov8n'))
        if self.manifest['conditional_code_sha256'] != code_fingerprints():
            raise ValueError('Conditional implementation changed; regenerate the bank')
        for name in ('pairs', 'contexts'):
            if sha256(self.root / f'{name}.jsonl') != self.manifest[f'{name}_sha256']:
                raise ValueError(f'{name} file changed')
        self.records = [json.loads(line) for line in (self.root / 'pairs.jsonl').read_text().splitlines()]
        contexts = [json.loads(line) for line in (self.root / 'contexts.jsonl').read_text().splitlines()]
        self.contexts = {c['image_id']: c for c in contexts}
        self.dataset = DetectionDataset('train', self.manifest['image_size'])
        self.config = PairConfig(**self.manifest['config'])
        if not self.records:
            raise ValueError('No valid preference pairs; inspect bank audit')

    def replay(self, adapter, records):
        errors, count = [], 0
        with torch.no_grad():
            for r in records:
                image, target = self.dataset[r['image_id']]
                context = self.contexts[r['image_id']]
                if target['image_path'] != r['image_path'] or context['image_path'] != r['image_path']:
                    raise ValueError('Image identity mismatch')
                logits, boxes = adapter.raw(image[None])
                j, c = r['slot'], r['class_id']
                error = abs(float(logits[0, j, c]) - r['reference_logit'])
                if error > .002 or not torch.allclose(boxes[0, j].cpu(), torch.tensor(r['box']), atol=.002, rtol=0):
                    raise ValueError('Source slot/box/logit does not replay')
                valid, evidence = judge(context, target, j, r['chosen_keep'], r['target_gt'], self.config)
                if not valid or evidence != r['evidence']:
                    raise ValueError('GT judging does not replay')
                errors.append(error)
                count += 1
        return {'pairs': count, 'max_logit_error': max(errors, default=0.)}
