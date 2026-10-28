"""Read-only KB2 class/FN audit; writes reports without changing training sources."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torchvision.ops import box_iou
import yaml

from kb1_reward_guided_training.canonical_eval import _operating_point_counts
from kb2_preference_optimization.adapter import DFLAdapter
from kb2_preference_optimization.artifacts import (
    assert_provenance, atomic_json, seed_everything, sha256, source_checkpoint,
    source_provenance, tensor_digest,
)
from kb2_preference_optimization.data import DetectionDataset, detection_loader
from kb2_preference_optimization.policy import PolicyConfig, render_action
from kb2_preference_optimization.quality import match_boxes

RUNS = {
    42: 'yolov8n_dpo_s42_20261026_135931',
    43: 'yolov8n_dpo_fixedpairs_s43_20261026_143830',
    44: 'yolov8n_dpo_fixedpairs_s44_20261026_145438',
}
OUTPUTS = ROOT / 'kb2_preference_optimization/outputs'
PREFS = OUTPUTS / RUNS[42] / 'preferences'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def operating_matches(prediction, target):
    """Expose GT identities using exactly the canonical greedy matching order."""
    keep = prediction['scores'] >= .25
    boxes, labels, scores = (prediction[k][keep] for k in ('boxes', 'labels', 'scores'))
    ious = box_iou(target['boxes'].float(), boxes.float()) if len(boxes) else None
    matched, counts = {}, Counter()
    for p in scores.argsort(descending=True).tolist():
        label = int(labels[p])
        eligible = [g for g in range(len(target['boxes']))
                    if g not in matched and int(target['labels'][g]) == label]
        if eligible:
            values = ious[eligible, p]
            position = int(values.argmax())
            if float(values[position]) >= .5:
                matched[eligible[position]] = {
                    'score': float(scores[p]), 'iou': float(values[position]),
                    'box': boxes[p].tolist(),
                }
                counts[(label, 'tp')] += 1
                continue
        counts[(label, 'fp')] += 1
    for g, label in enumerate(target['labels'].tolist()):
        if g not in matched:
            counts[(label, 'fn')] += 1
    reference = _operating_point_counts([prediction], [target], .25, .5)
    assert dict(reference) == {
        c: [counts[(c, kind)] for kind in ('tp', 'fp', 'fn')]
        for c in reference
    }, 'GT identity matching differs from canonical evaluator'
    return matched, counts


def validation_audit(destination, provenance, names, device):
    reports, checkpoints = {}, {}
    for seed, run in RUNS.items():
        directory = OUTPUTS / run
        reports[seed] = read_json(directory / 'validation_comparison.json')
        checkpoint = directory / ('training/best.pt' if seed == 42 else 'best.pt')
        completed = read_json(checkpoint.parent / 'completed.json')
        assert sha256(checkpoint) == completed['best_sha256']
        assert sha256(checkpoint) == reports[seed]['checkpoint_sha256']
        assert_provenance(reports[seed]['provenance'], provenance)
        checkpoints[seed] = checkpoint

    all_matches, totals, targets_by_image = {}, {}, {}
    with (destination / 'validation_matches.jsonl').open('w', encoding='utf-8') as stream:
        for key in ('s0', 42, 43, 44):
            adapter = DFLAdapter(source_checkpoint('yolov8n'), device)
            if key != 's0':
                payload = torch.load(checkpoints[key], map_location='cpu', weights_only=False)
                assert_provenance(payload['manifest']['provenance'], provenance)
                adapter.model.load_state_dict(payload['state_dict'], strict=True)
                del payload
            adapter.model.requires_grad_(False)
            before = tensor_digest(adapter.model)
            counts, matches_by_image = Counter(), {}
            with torch.no_grad():
                for batch_index, (images, targets) in enumerate(detection_loader('valid', 640, 8)):
                    predictions = adapter.predict(images.to(device), .001, .60, 300)
                    for prediction, target in zip(predictions, targets):
                        image_id = target['image_id']
                        matched, image_counts = operating_matches(prediction, target)
                        counts.update(image_counts)
                        matches_by_image[image_id] = set(matched)
                        targets_by_image[image_id] = target
                        stream.write(json.dumps({
                            'checkpoint': key, 'image_id': image_id,
                            'image_path': target['image_path'], 'matched_gt': matched,
                        }) + '\n')
                    if batch_index % 20 == 0:
                        print(f'validation {key}: {min((batch_index + 1) * 8, 479)}/479', flush=True)
            assert before == tensor_digest(adapter.model), 'Inference changed checkpoint state'
            reference = reports[42]['supervised'] if key == 's0' else reports[key]['dpo']
            for c, values in reference['per_class'].items():
                for kind in ('tp', 'fp', 'fn'):
                    assert counts[(int(c), kind)] == values[kind], (key, c, kind, counts, values)
            all_matches[key], totals[key] = matches_by_image, counts
            print(f'validation {key}: exact match to saved TP/FP/FN for all classes', flush=True)
            del adapter
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

    transitions = {seed: Counter() for seed in RUNS}
    with (destination / 'validation_gt_transitions.jsonl').open('w', encoding='utf-8') as stream:
        for image_id, target in sorted(targets_by_image.items()):
            for g, c in enumerate(target['labels'].tolist()):
                s0 = g in all_matches['s0'][image_id]
                row = {'image_id': image_id, 'image_path': target['image_path'],
                       'transformed_gt_row': g, 'class_id': c, 's0_detected': s0,
                       'gt_box': target['boxes'][g].tolist()}
                for seed in RUNS:
                    detected = g in all_matches[seed][image_id]
                    kind = ('retained' if detected else 'lost') if s0 else (
                        'recovered' if detected else 'persistent_miss')
                    transitions[seed][(c, kind)] += 1
                    row[f's{seed}'] = kind
                stream.write(json.dumps(row) + '\n')

    seed_rows, summary_rows = [], []
    for c, name in enumerate(names):
        support = totals['s0'][(c, 'tp')] + totals['s0'][(c, 'fn')]
        for seed in RUNS:
            count, transition = totals[seed], transitions[seed]
            assert count[(c, 'fn')] - totals['s0'][(c, 'fn')] == (
                transition[(c, 'lost')] - transition[(c, 'recovered')])
            seed_rows.append({
                'class_id': c, 'class_name': name, 'seed': seed, 'gt_valid': support,
                's0_tp': totals['s0'][(c, 'tp')], 's0_fn': totals['s0'][(c, 'fn')],
                's0_fp': totals['s0'][(c, 'fp')],
                's0_miss_pct': pct(totals['s0'][(c, 'fn')], support),
                'dpo_tp': count[(c, 'tp')], 'dpo_fn': count[(c, 'fn')], 'dpo_fp': count[(c, 'fp')],
                'dpo_miss_pct': pct(count[(c, 'fn')], support),
                'delta_miss_pp': pct(count[(c, 'fn')] - totals['s0'][(c, 'fn')], support),
                'recovered': transition[(c, 'recovered')], 'lost': transition[(c, 'lost')],
                'persistent_miss': transition[(c, 'persistent_miss')],
                'retained': transition[(c, 'retained')],
            })
        rows = [r for r in seed_rows if r['class_id'] == c]
        summary_rows.append({
            'class_id': c, 'class_name': name, 'gt_valid': support,
            's0_fn': totals['s0'][(c, 'fn')],
            's0_recall_pct': pct(totals['s0'][(c, 'tp')], support),
            's0_miss_pct': pct(totals['s0'][(c, 'fn')], support),
            'dpo_fn_mean': statistics.mean(r['dpo_fn'] for r in rows),
            'dpo_recall_mean_pct': 100 - statistics.mean(r['dpo_miss_pct'] for r in rows),
            'dpo_miss_mean_pct': statistics.mean(r['dpo_miss_pct'] for r in rows),
            'delta_miss_mean_pp': statistics.mean(r['delta_miss_pp'] for r in rows),
            'recovered_mean': statistics.mean(r['recovered'] for r in rows),
            'lost_mean': statistics.mean(r['lost'] for r in rows),
            'persistent_miss_mean': statistics.mean(r['persistent_miss'] for r in rows),
            's0_fp': totals['s0'][(c, 'fp')],
            'dpo_fp_mean': statistics.mean(r['dpo_fp'] for r in rows),
            'seeds_miss_increased': sum(r['delta_miss_pp'] > 0 for r in rows),
        })
    write_csv(destination / 'validation_by_class_seed.csv', seed_rows)
    write_csv(destination / 'validation_by_class.csv', summary_rows)
    return {'summary': summary_rows, 'by_seed': seed_rows,
            'checkpoint_sha256': {str(seed): sha256(path) for seed, path in checkpoints.items()}}


def matched_gt(prediction, target):
    return {g for _, g, _ in match_boxes(prediction, target, .5)}


def preference_audit(destination, provenance, names, device):
    manifest = read_json(PREFS / 'manifest.json')
    assert_provenance(manifest['provenance'], provenance)
    assert sha256(PREFS / 'pairs.jsonl') == manifest['pairs_sha256']
    records = {r['image_id']: r for r in (
        json.loads(line) for line in (PREFS / 'pairs.jsonl').read_text().splitlines())}
    dataset = DetectionDataset('train', manifest['image_size'])
    assert len(dataset) == manifest['summary']['images_seen']
    seed_everything(manifest['seed'])
    adapter = DFLAdapter(source_checkpoint('yolov8n'), device)
    adapter.model.requires_grad_(False)
    before = tensor_digest(adapter.model)
    assert before == manifest['reference_state_sha256']
    generator = torch.Generator(device=adapter.device).manual_seed(manifest['seed'])
    policy = PolicyConfig(**manifest['policy'])
    counts, checks = Counter(), Counter()
    with torch.no_grad(), (destination / 'train_gt_coverage.jsonl').open('w', encoding='utf-8') as stream:
        for index in range(len(dataset)):
            image, target = dataset[index]
            distributions, anchors, strides, scores = adapter.distributions(image[None], policy)
            distribution = distributions[0]
            actions = [distribution.sample(generator) for _ in range(manifest['draws'])]
            predictions = [render_action(distribution, action, anchors, strides, scores[0],
                                         manifest['image_size']) for action in actions]
            matches = [matched_gt(prediction, target) for prediction in predictions]
            union = set().union(*matches)
            record = records.get(index)
            chosen, rejected = set(), set()
            if record:
                assert record['image_path'] == target['image_path']
                for role in ('chosen', 'rejected'):
                    action_index = actions.index(record[role])
                    prediction = predictions[action_index]
                    for field in ('boxes', 'labels', 'scores'):
                        old = torch.tensor(record[f'{role}_detection'][field],
                                           dtype=prediction[field].dtype).reshape(prediction[field].shape)
                        assert torch.equal(prediction[field], old), (index, role, field)
                    if role == 'chosen':
                        chosen = matches[action_index]
                    else:
                        rejected = matches[action_index]
                    checks['replayed_stored_actions_and_outputs'] += 1
                assert len(chosen) == record['chosen_quality']['tp']
                assert len(rejected) == record['rejected_quality']['tp']
                checks['pairs'] += 1
                cq, rq = record['chosen_quality'], record['rejected_quality']
                checks['chosen_more_fn_fewer_fp'] += cq['fn'] > rq['fn'] and cq['fp'] < rq['fp']
                checks['chosen_same_tp_fewer_fp'] += cq['tp'] == rq['tp'] and cq['fp'] < rq['fp']
                checks['chosen_more_tp'] += cq['tp'] > rq['tp']
                checks['chosen_fewer_tp'] += cq['tp'] < rq['tp']
            labels = target['labels'].tolist()
            for c in set(labels):
                indices = {g for g, label in enumerate(labels) if label == c}
                counts[(c, 'images_train')] += 1
                counts[(c, 'images_paired' if record else 'images_unpaired')] += 1
                counts[(c, 'images_zero_union_coverage')] += not bool(indices & union)
                if record:
                    rescue, loss = len((chosen - rejected) & indices), len((rejected - chosen) & indices)
                    counts[(c, 'pairs_with_recovered_gt')] += rescue > 0
                    counts[(c, 'pairs_with_lost_gt')] += loss > 0
                    counts[(c, 'pairs_net_tp_better')] += len(chosen & indices) > len(rejected & indices)
                    counts[(c, 'pairs_net_tp_worse')] += len(chosen & indices) < len(rejected & indices)
            for g, c in enumerate(labels):
                counts[(c, 'gt_train')] += 1
                counts[(c, 'gt_covered_any_draw')] += g in union
                counts[(c, 'gt_paired' if record else 'gt_unpaired')] += 1
                if record:
                    counts[(c, 'chosen_tp')] += g in chosen
                    counts[(c, 'rejected_tp')] += g in rejected
                    counts[(c, 'gt_covered_saved_pair')] += g in (chosen | rejected)
                    kind = ('retained' if g in chosen else 'lost') if g in rejected else (
                        'recovered' if g in chosen else 'both_miss')
                    counts[(c, kind)] += 1
                stream.write(json.dumps({
                    'image_id': index, 'image_path': target['image_path'],
                    'transformed_gt_row': g, 'class_id': c, 'has_pair': bool(record),
                    'covered_draws': [g in match for match in matches],
                    'chosen_detected': (g in chosen) if record else None,
                    'rejected_detected': (g in rejected) if record else None,
                }) + '\n')
            if index % 100 == 0:
                print(f'train replay: {index + 1}/{len(dataset)}', flush=True)
    assert before == tensor_digest(adapter.model)
    assert checks['pairs'] == len(records) == manifest['summary']['pairs_kept']
    assert checks['replayed_stored_actions_and_outputs'] == 2 * len(records)
    rows = []
    fields = ('gt_train', 'images_train', 'images_paired', 'images_unpaired', 'gt_paired',
              'gt_unpaired', 'chosen_tp', 'rejected_tp', 'recovered', 'lost', 'both_miss',
              'gt_covered_saved_pair', 'gt_covered_any_draw', 'images_zero_union_coverage',
              'pairs_with_recovered_gt', 'pairs_with_lost_gt', 'pairs_net_tp_better', 'pairs_net_tp_worse')
    for c, name in enumerate(names):
        row = {'class_id': c, 'class_name': name, **{f: counts[(c, f)] for f in fields}}
        row.update({
            'chosen_recall_paired_pct': pct(row['chosen_tp'], row['gt_paired']),
            'rejected_recall_paired_pct': pct(row['rejected_tp'], row['gt_paired']),
            'pair_union_coverage_pct': pct(row['gt_covered_saved_pair'], row['gt_paired']),
            'all_draws_union_coverage_pct': pct(row['gt_covered_any_draw'], row['gt_train']),
            'all_draws_never_detected': row['gt_train'] - row['gt_covered_any_draw'],
        })
        assert row['chosen_tp'] - row['rejected_tp'] == row['recovered'] - row['lost']
        rows.append(row)
    write_csv(destination / 'train_preferences_by_class.csv', rows)
    return {'summary': rows, 'checks': dict(checks), 'draws': manifest['draws'],
            'pairs_sha256': manifest['pairs_sha256'], 'seed': manifest['seed']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = source_provenance('yolov8n')
    names = yaml.safe_load((ROOT / 'kb1_reward_guided_training/configs/pest.yaml').read_text())['names']
    result = {'scope': 'full valid and train only; no training or test evaluation',
              'provenance': provenance, 'device': args.device,
              'validation_protocol': {'conf': .001, 'operating_conf': .25, 'operating_iou': .5,
                                      'nms_iou': .60, 'max_det': 300, 'batch_size': 8,
                                      'matcher': 'canonical confidence-ordered greedy'},
              'preference_protocol': {'iou': .5, 'score_threshold': None,
                                      'matcher': 'class-aware max-cardinality then IoU',
                                      'replay': 'same seed, policy and four sampled actions per image'}}
    result['validation'] = validation_audit(args.output, provenance, names, args.device)
    atomic_json(args.output / 'validation_audit.json', result)
    result['preferences'] = preference_audit(args.output, provenance, names, args.device)
    final_provenance = source_provenance('yolov8n')
    assert_provenance(provenance, final_provenance)
    result['checks'] = {'validation_counts_match_all_saved_reports': True,
                        'stored_preference_actions_and_detections_reproduced_exactly': True,
                        'source_data_code_runtime_unchanged': True,
                        'audit_script_sha256': sha256(Path(__file__))}
    atomic_json(args.output / 'audit.json', result)
    print(json.dumps({'status': 'complete', 'output': str(args.output),
                      'checks': result['checks'], 'preference_checks': result['preferences']['checks']}), flush=True)


if __name__ == '__main__':
    main()
