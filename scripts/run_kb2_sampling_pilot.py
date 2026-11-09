"""Locked 2k DPO sampling pilot against the first 2k of the completed drop1k run."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np
import torch

from kb2_preference_optimization.artifacts import REPO, atomic_json, sha256
from kb2_preference_optimization.conditional.bank import sampling_probabilities as historical_sampler
from kb2_preference_optimization.conditional.pairs import judge
from kb2_preference_optimization.hybrid.bank import Bank
from kb2_preference_optimization.hybrid.source import assert_provenance, source_provenance
from kb2_preference_optimization.hybrid_sampling import code_fingerprints
from kb2_preference_optimization.hybrid_sampling.sampler import GROUP_MASS, sampling_probabilities
from kb2_preference_optimization.hybrid_sampling.train import train


def read(path):
    return json.loads(Path(path).read_text())


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def selected(rows):
    best = rows[0]
    for row in rows[1:]:
        if row['metrics']['mAP50_95'] > best['metrics']['mAP50_95'] + 1e-4:
            best = row
    return best


def exposure(records, indices):
    groups, classes, images, pairs = Counter(), Counter(), Counter(), Counter()
    for i in indices:
        record = records[i]
        groups[record['group']] += 1
        classes[str(record['class_id'])] += 1
        images[str(record['image_id'])] += 1
        pairs[str(record['pair_id'])] += 1
    return {'group': dict(groups), 'class': dict(classes), 'image': dict(images),
            'pair': dict(pairs), 'decisions': len(indices), 'unique_images': len(images),
            'unique_pairs': len(pairs)}


def audit(bank):
    records = [r for r in bank.records if not r['probe']]
    old_weights, old_config = historical_sampler(records)
    weights, config = sampling_probabilities(records)
    equal, _ = sampling_probabilities(records, group_mass={k: .25 for k in GROUP_MASS})
    assert torch.allclose(equal, old_weights, atol=1e-15, rtol=1e-14)
    report = {'split': 'train only', 'bank_totals': bank.manifest['audit']['totals'],
              'bank_groups': bank.manifest['audit']['groups'],
              'train_records': len(records), 'probe_records_excluded': len(bank.records)-len(records),
              'old_sampler': old_config, 'new_sampler': config,
              'equal_group_mode_reproduces_historical_sampler': True,
              'groups': {}, 'per_class': []}
    for name, weight in (('old', old_weights), ('new', weights)):
        indices = torch.multinomial(weight, 4000, replacement=True,
                                    generator=torch.Generator().manual_seed(10042)).tolist()
        report[f'{name}_planned_exposure'] = exposure(records, indices)
    preserved = set()
    replayed, cached_targets = 0, {}
    for group in GROUP_MASS:
        subset = [r for r in records if r['group'] == group]
        scores = np.array([1/(1+math.exp(-r['reference_logit'])) for r in subset])
        report['groups'][group] = {
            'pairs': len(subset), 'images': len({r['image_id'] for r in subset}),
            'classes': len({r['class_id'] for r in subset}),
            'score_quantiles': dict(zip(('min', 'p25', 'median', 'p75', 'max'),
                                        map(float, np.quantile(scores, [0, .25, .5, .75, 1])))),
            'score_025_to_035': int(((scores >= .25) & (scores < .35)).sum()),
            'score_035_to_050': int(((scores >= .35) & (scores < .5)).sum())}
        if group == 'preservation':
            for r in subset:
                if r['image_id'] not in cached_targets:
                    cached_targets[r['image_id']] = bank.dataset[r['image_id']][1]
                valid, evidence = judge(bank.contexts[r['image_id']], cached_targets[r['image_id']],
                                        r['slot'], r['chosen_keep'], r['target_gt'], bank.config)
                assert valid and evidence == r['evidence']
                assert .25 <= 1/(1+math.exp(-r['reference_logit']))
                preserved.add((r['image_id'], r['target_gt']))
                replayed += 1
    report['preservation_full_GT_judge_replay_pairs'] = replayed
    report['preservation_unique_GT'] = len(preserved)
    for class_id in range(bank.dataset.classes):
        row = {'class_id': class_id}
        for group in GROUP_MASS:
            row[group] = sum(r['class_id'] == class_id and r['group'] == group for r in records)
        for label, config_ in (('old', old_config), ('new', config)):
            row[f'{label}_expected_probability'] = config_['expected_class_probability'].get(str(class_id), 0.)
        report['per_class'].append(row)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError('Use a fresh output directory')
    baseline = REPO/'kb2_preference_optimization/outputs/yolov8n_hybrid_schedule_s42_20261106_trial3/run'
    earlier = REPO/'kb2_preference_optimization/outputs/yolov8n_hybrid_drop500_s42_20261106_trial1/run'
    provenance = source_provenance('yolov8n')
    bank = Bank(REPO/'kb2_preference_optimization/outputs/yolov8n_hybrid_tpe2_s42_20261105_180813_658387/preferences')
    assert not bank.manifest['partial_bank']
    assert_provenance(bank.manifest['provenance'], provenance)
    protected = {}
    for module in ('conditional', 'conditional_bg', 'conditional_strength', 'hybrid', 'hybrid_schedule'):
        for path in (REPO/'kb2_preference_optimization'/module).glob('*.py'):
            protected[str(path)] = sha256(path)
    for folder in (baseline, earlier):
        done = read(folder/'completed.json')
        assert done['arms']['dpo']['steps'] == 15000
        for which in ('best', 'last'):
            path = folder/'dpo'/f'{which}.pt'
            assert sha256(path) == done['arms']['dpo'][f'{which}_sha256']
            protected[str(path)] = sha256(path)
    for path in (Path(provenance['source_checkpoint']), Path(provenance['shared_S0']['checkpoint']),
                 bank.root/'manifest.json', bank.root/'pairs.jsonl', bank.root/'contexts.jsonl'):
        protected[str(path)] = sha256(path)
    report = audit(bank)
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root/'bank_sampling_audit.json', report)
    plan = {'created_UTC': datetime.now(timezone.utc).isoformat(), 'baseline': str(baseline),
            'source': provenance, 'bank': str(bank.root), 'steps': 2000, 'seed': 42,
            'lambda': .03, 'beta': .1, 'background_weight': .5, 'initial_lr': 1e-6,
            'lr_drop_after_update': 1000, 'lr_drop_factor': .1, 'eval_interval': 1000,
            'early_eval_interval': 250, 'early_eval_until': 2000,
            'requested_group_probability': GROUP_MASS,
            'selection': 'validation mAP50-95; min_delta1e-4; TPE step0 eligible',
            'sampling_code_sha256': code_fingerprints(), 'immutable_files': protected,
            'comparison': 'same 0..2000 dense grid of completed drop1000; only sampler intervention',
            'native_exposure_note': 'uniform 2 images fixed; 2 preference-native images change with pair sampling',
            'test_evaluation': False, 'partial_bank': False, 'partial_validation': False,
            'runtime': {key: os.environ.get(key) for key in
                        ('CUDA_LAUNCH_BLOCKING', 'PYTHONFAULTHANDLER', 'YOLO_CONFIG_DIR',
                         'MPLCONFIGDIR', 'NO_ALBUMENTATIONS_UPDATE', 'OMP_NUM_THREADS',
                         'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
            'cpu_affinity': sorted(os.sched_getaffinity(0))}
    atomic_json(root/'plan.json', plan)
    atomic_json(root/'process.json', {'pid': os.getpid()})
    print(json.dumps({'event': 'sampling_audit', 'output': str(root),
                      'expected_groups': report['new_sampler']['expected_group_probability'],
                      'preservation': report['groups']['preservation']}), flush=True)
    if args.audit_only:
        atomic_json(root/'status.json', {'stage': 'audit_only'})
        return
    atomic_json(root/'status.json', {'stage': 'training'})
    try:
        result = train(bank.root, root/'run/dpo', steps=2000, seed=42, device='cuda',
                       eval_interval=1000, beta=.1, dpo_weight=.03, lr=1e-6, background_weight=.5,
                       lr_drop_step=1000, lr_drop_factor=.1, early_eval_interval=250, early_eval_until=2000)
        assert result['reference_unchanged'] and result['frozen_state_unchanged']
        old_stream = events(baseline/'dpo/stream.jsonl')[:2000]
        new_stream = events(root/'run/dpo/stream.jsonl')
        assert len(new_stream) == 2000
        assert all(a['native_image_ids'][:2] == b['native_image_ids'][:2]
                   for a, b in zip(old_stream, new_stream))
        assert any(a['pair_ids'] != b['pair_ids'] for a, b in zip(old_stream, new_stream))
        assert all(sha256(path) == digest for path, digest in protected.items())
        current_vals = [r for r in events(root/'run/dpo/metrics.jsonl') if r['event'] == 'validation']
        old_vals = [r for r in events(baseline/'dpo/metrics.jsonl')
                    if r['event'] == 'validation' and r['step'] <= 2000]
        assert [r['step'] for r in current_vals] == [r['step'] for r in old_vals]
        assert result['best_step'] == selected(current_vals)['step']
        completion = {'plan': plan, 'result': result, 'uniform_native_stream_matched': True,
                      'preference_stream_changed': True, 'protected_files_unchanged': True,
                      'pilot_best': selected(current_vals), 'baseline_2k_best': selected(old_vals),
                      'baseline_15k_best_step': read(baseline/'dpo/completed.json')['best_step']}
        atomic_json(root/'completed.json', completion)
        atomic_json(root/'status.json', {'stage': 'completed', 'best_step': result['best_step']})
        print(json.dumps({'event': 'sampling_pilot_completed', 'best_step': result['best_step'],
                          'best_mAP50_95': completion['pilot_best']['metrics']['mAP50_95'],
                          'baseline_best_mAP50_95': completion['baseline_2k_best']['metrics']['mAP50_95']}), flush=True)
    except BaseException as error:
        atomic_json(root/'status.json', {'stage': 'failed', 'error': str(error),
                                         'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
