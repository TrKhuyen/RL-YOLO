"""Audit every pair in four train-only samples and retain max one pair/image."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from ..adapter import DFLAdapter
from ..artifacts import (append_jsonl, atomic_json, seed_everything, sha256,
                         source_checkpoint, source_provenance, tensor_digest)
from ..data import DetectionDataset
from ..policy import PolicyConfig, render_action
from ..quality import QualityConfig, quality_score, select_pair
from . import VERSION, code_fingerprints
from .selection import eligible_pairs, is_probe_image, matched_ids, pair_metadata


def generate(output, device='cuda'):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = source_provenance('yolov8n')
    extra_code = code_fingerprints()
    seed_everything(42)
    policy, quality = PolicyConfig(), QualityConfig()
    adapter = DFLAdapter(source_checkpoint('yolov8n'), device)
    adapter.model.requires_grad_(False)
    before = tensor_digest(adapter.model)
    generator = torch.Generator(device=device).manual_seed(42)
    dataset = DetectionDataset('train', 640)
    folders = {name: output / name for name in ('control', 'selected')}
    counts = {name: Counter() for name in folders}
    class_counts, totals = Counter(), Counter()
    for folder in folders.values():
        folder.mkdir()
        (folder / 'pairs.jsonl').touch()
    pool_path = output / 'all_candidates.jsonl'
    with torch.no_grad():
        for index in range(len(dataset)):
            image, target = dataset[index]
            distributions, anchors, strides, priorities = adapter.distributions(image[None], policy)
            dist = distributions[0]
            actions = [dist.sample(generator) for _ in range(4)]
            predictions = [render_action(dist, action, anchors, strides, priorities[0], 640)
                           for action in actions]
            qualities = [quality_score(prediction, target, quality) for prediction in predictions]
            matches = [matched_ids(prediction, target) for prediction in predictions]
            labels = target['labels'].tolist()
            recovery, maintenance = eligible_pairs(qualities, matches, labels, actions, quality.min_gap)
            totals['images'] += 1
            totals['unordered_comparisons'] += 6
            totals['eligible_recovery_pairs'] += len(recovery)
            totals['eligible_maintenance_pairs'] += len(maintenance)
            totals['images_with_recovery'] += bool(recovery)
            totals['images_with_maintenance'] += bool(maintenance)
            totals['gt'] += len(labels)
            totals['gt_never_covered'] += len(labels) - len(set().union(*matches))
            for c in set(labels):
                class_counts[(c, 'images_train')] += 1
                class_counts[(c, 'gt_train')] += labels.count(c)
                eligible = [r for r in recovery if c in r['rescued_classes']]
                class_counts[(c, 'images_with_eligible_recovery')] += bool(eligible)
                class_counts[(c, 'eligible_recovery_pairs')] += len(eligible)
            append_jsonl(pool_path, {
                'image_id': index, 'image_path': target['image_path'],
                'labels': labels, 'gt_boxes': target['boxes'].tolist(),
                'probe': is_probe_image(index),
                'candidates': [{'action': a, 'detection': {k: v.tolist() for k, v in p.items()},
                                'quality': q, 'matched_gt': sorted(m)}
                               for a, p, q, m in zip(actions, predictions, qualities, matches)],
                'eligible_recovery': recovery, 'eligible_maintenance': maintenance,
            })
            standard = select_pair(predictions, qualities, quality.min_gap)
            selections = {'control': None, 'selected': (recovery or maintenance or [None])[0]}
            if standard:
                a, b = standard
                meta = pair_metadata(a, b, qualities, matches, labels, actions)
                group = ('recovery' if meta['rescued_gt'] and not meta['lost_gt'] and meta['fp_delta'] <= 0
                         else 'maintenance' if matches[a] == matches[b] else 'other')
                selections['control'] = {'chosen_index': a, 'rejected_index': b, 'group': group, **meta}
            for name, selected in selections.items():
                if selected is None:
                    continue
                a, b = selected['chosen_index'], selected['rejected_index']
                record = {'image_id': index, 'image_path': target['image_path'],
                          'chosen': actions[a], 'rejected': actions[b],
                          'reference_chosen_logp': float(dist.log_prob(actions[a])),
                          'reference_rejected_logp': float(dist.log_prob(actions[b])),
                          'chosen_quality': qualities[a], 'rejected_quality': qualities[b],
                          'chosen_detection': {k: v.tolist() for k, v in predictions[a].items()},
                          'rejected_detection': {k: v.tolist() for k, v in predictions[b].items()},
                          'selection': selected, 'probe': is_probe_image(index)}
                append_jsonl(folders[name] / 'pairs.jsonl', record)
                counts[name]['pairs_kept'] += 1
                counts[name][selected['group']] += 1
                counts[name]['probe_pairs' if record['probe'] else 'training_pairs'] += 1
                if name == 'selected' and selected['group'] == 'recovery':
                    for c in selected['rescued_classes']:
                        class_counts[(c, 'selected_recovery_images')] += 1
                        class_counts[(c, 'probe_recovery_images' if record['probe'] else 'training_recovery_images')] += 1
            if (index + 1) % 100 == 0:
                print(f'audit {index+1}/{len(dataset)}: {totals["images_with_recovery"]} recovery images', flush=True)
    assert before == tensor_digest(adapter.model)
    assert extra_code == code_fingerprints()
    for name, folder in folders.items():
        atomic_json(folder / 'manifest.json', {
            'provenance': provenance, 'recovery_code_sha256': extra_code,
            'selection_version': VERSION, 'selector': name,
            'policy': policy.to_dict(), 'quality': quality.to_dict(), 'image_size': 640,
            'split': 'train', 'seed': 42, 'draws': 4, 'limit': 0,
            'pairs_sha256': sha256(folder / 'pairs.jsonl'), 'reference_state_sha256': before,
            'summary': dict(counts[name]), 'status': 'complete',
        })
    audit = {'version': VERSION, 'provenance': provenance, 'recovery_code_sha256': extra_code,
             'criteria': {'no_lost_gt': True, 'fp_must_not_increase': True,
                          'min_quality_gap': .02, 'match_iou': .5, 'max_pairs_per_image': 1,
                          'recovery_ranking': 'most rescued GT, then smallest action distance, then quality gap'},
             'totals': dict(totals), 'datasets': {k: dict(v) for k, v in counts.items()},
             'all_candidates_sha256': sha256(pool_path),
             'per_class': [{'class_id': c, **{f: class_counts[(c, f)] for f in (
                 'images_train', 'gt_train', 'images_with_eligible_recovery', 'eligible_recovery_pairs',
                 'selected_recovery_images', 'training_recovery_images', 'probe_recovery_images')}} for c in range(28)]}
    atomic_json(output / 'audit.json', audit)
    print(json.dumps({'totals': audit['totals'], 'datasets': audit['datasets']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    generate(args.output, args.device)
