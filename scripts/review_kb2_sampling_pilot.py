"""Verify sampling-only intervention and reload selected pilot checkpoints."""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml

from kb2_preference_optimization.artifacts import atomic_json, sha256, source_checkpoint
from kb2_preference_optimization.conditional.evaluation import evaluate
from kb2_preference_optimization.conditional.policy import ConditionalAdapter, configure_numerics
from kb2_preference_optimization.data import detection_loader
from kb2_preference_optimization.hybrid.source import source_provenance, assert_provenance
from kb2_preference_optimization.hybrid_sampling import code_fingerprints
from kb2_preference_optimization.hybrid_schedule.schedule import learning_rate_for_update
from scripts.audit_kb2_conditional_gradients import write_csv
from scripts.review_kb2_hybrid_bg_comparison import FIELDS
from scripts.review_kb2_strength_multiseed import state_equal
from scripts.review_kb2_strength_pilot import numeric_metrics
from scripts.run_kb2_sampling_pilot import read, events, selected


def review(root, output, device):
    root, output = root.resolve(), output.resolve()
    done = read(root/'completed.json')
    plan, result = done['plan'], done['result']
    assert done['uniform_native_stream_matched'] and done['preference_stream_changed']
    assert done['protected_files_unchanged'] and plan['steps'] == 2000 and plan['seed'] == 42
    assert result['frozen_state_unchanged'] and result['reference_unchanged']
    assert not result['partial_evaluation'] and not result['partial_bank']
    assert plan['sampling_code_sha256'] == code_fingerprints()
    provenance = source_provenance('yolov8n')
    assert_provenance(plan['source'], provenance)
    baseline = Path(plan['baseline'])
    folders = {'old_sampler': baseline/'dpo', 'preservation_sampler': root/'run/dpo'}
    manifests = {label: read(folder/'manifest.json') for label, folder in folders.items()}
    configs = {label: value['config'] for label, value in manifests.items()}
    permitted = {'steps', 'training_variant', 'evaluation_steps', 'sampler', 'pilot'}
    old_config, new_config = configs.values()
    assert {k: v for k, v in old_config.items() if k not in permitted} == {
        k: v for k, v in new_config.items() if k not in permitted}
    manifest_diff = {'training_variant', 'scheduled_code_sha256', 'sampling_code_sha256', 'config'}
    a, b = manifests.values()
    assert {k: v for k, v in a.items() if k not in manifest_diff} == {
        k: v for k, v in b.items() if k not in manifest_diff}
    assert old_config['evaluation_steps'][:9] == new_config['evaluation_steps'] == list(range(0, 2001, 250))
    histories = {label: [r for r in events(folder/'metrics.jsonl')
                         if r['event'] == 'validation' and r['step'] <= 2000]
                 for label, folder in folders.items()}
    streams = {label: events(folder/'stream.jsonl')[:2000] for label, folder in folders.items()}
    assert all(a['native_image_ids'][:2] == b['native_image_ids'][:2]
               for a, b in zip(*streams.values()))
    assert all(len(value) == 2000 for value in streams.values())
    protected = dict(plan['immutable_files'])
    states, exposure_rows = {}, []
    for label, folder in folders.items():
        completed = read(folder/'completed.json')
        assert_provenance(manifests[label]['provenance'], provenance)
        vals = histories[label]
        assert [r['step'] for r in vals] == list(range(0, 2001, 250))
        best = selected(vals)
        if label == 'preservation_sampler':
            assert best['step'] == completed['best_step']
        for row in vals:
            path = folder/f"step_{row['step']:06d}.pt"
            assert sha256(path) == completed['evaluation_checkpoint_sha256'][str(row['step'])]
            checkpoint = torch.load(path, weights_only=False, map_location='cpu')
            rate = 1e-6 if row['step'] == 0 else learning_rate_for_update(1e-6, row['step'], 1000, .1)
            assert checkpoint['optimizer']['param_groups'][0]['lr'] == rate
        for which, row in (('best_2k', best), ('step2000', vals[-1])):
            path = folder/f"step_{row['step']:06d}.pt"
            states[f'{label}_{which}'] = (path, row['metrics'], manifests[label]['frozen_state_sha256'])
            protected[str(path)] = sha256(path)
        if label == 'preservation_sampler':
            for which, step in (('best', best['step']), ('last', 2000)):
                path = folder/f'{which}.pt'
                assert sha256(path) == completed[f'{which}_sha256']
                checkpoint = torch.load(path, weights_only=False, map_location='cpu')
                saved = torch.load(folder/f'step_{step:06d}.pt', weights_only=False, map_location='cpu')
                assert state_equal(checkpoint['state_dict'], saved['state_dict'])
                assert state_equal(checkpoint['optimizer'], saved['optimizer'])
    old_zero = torch.load(folders['old_sampler']/'step_000000.pt', map_location='cpu', weights_only=False)
    new_zero = torch.load(folders['preservation_sampler']/'step_000000.pt', map_location='cpu', weights_only=False)
    assert state_equal(old_zero['state_dict'], new_zero['state_dict'])
    assert state_equal(old_zero['optimizer'], new_zero['optimizer'])
    del old_zero, new_zero
    records = [r for r in events(Path(plan['bank'])/'pairs.jsonl') if not r['probe']]
    by_id = {r['pair_id']: r for r in records}
    audit = read(root/'bank_sampling_audit.json')
    for label, stream in streams.items():
        groups, classes, images, pairs = Counter(), Counter(), Counter(), Counter()
        for row in stream:
            for pid in row['pair_ids']:
                r = by_id[pid]
                groups[r['group']] += 1
                classes[str(r['class_id'])] += 1
                images[str(r['image_id'])] += 1
                pairs[str(pid)] += 1
        planned = audit[('old' if label == 'old_sampler' else 'new')+'_planned_exposure']
        assert planned['group'] == groups and planned['pair'] == pairs
        for group, count in groups.items():
            sampler = configs[label]['sampler']
            exposure_rows.append({'model': label, 'group': group, 'actual_decisions': count,
                                  'actual_fraction': count/4000,
                                  'expected_fraction': sampler['expected_group_probability'][group],
                                  'requested_fraction': sampler.get('requested_group_probability', {}).get(group, .25),
                                  'unique_pairs_all_groups': len(pairs), 'unique_images_all_groups': len(images)})
    assert all(sha256(path) == digest for path, digest in protected.items())
    output.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    source = ConditionalAdapter(provenance['source_checkpoint'], device)
    loader = detection_loader('valid', 640, 8)
    tpe, predictions = evaluate(source, loader)
    assert numeric_metrics(tpe) == numeric_metrics(histories['old_sampler'][0]['metrics'])
    metrics = {'TPE': tpe}
    atomic_json(output/'TPE.json', tpe)
    for label, (path, logged, frozen) in states.items():
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        source.model.load_state_dict(checkpoint['state_dict'], strict=True)
        assert source.frozen_digest() == frozen
        measured, _ = evaluate(source, loader, predictions)
        assert numeric_metrics(measured) == numeric_metrics(logged), label
        assert measured['num_images'] == 479 and measured['num_gt'] == 1887
        metrics[label] = measured
        atomic_json(output/f'{label}.json', measured)
        print({'event': 'reloaded', 'model': label, 'mAP50_95': measured['mAP50_95'],
               'fp': measured['fp'], 'fn': measured['fn']}, flush=True)
    del source, predictions
    source = ConditionalAdapter(source_checkpoint('yolov8n'), device)
    s0, _ = evaluate(source, loader)
    metrics['S0'] = s0
    atomic_json(output/'S0.json', s0)
    summaries = [{'model': label, **{k: m[k] for k in FIELDS}} for label, m in metrics.items()]
    deltas = [{'model': f'preservation_sampler_{which}', 'baseline': f'old_sampler_{which}',
               **{k: metrics[f'preservation_sampler_{which}'][k]-metrics[f'old_sampler_{which}'][k] for k in FIELDS}}
              for which in ('best_2k', 'step2000')]
    transitions = [{'model': label, **{k: m['source_gt_transitions'].get(k, 0)
                                      for k in ('recovered', 'lost', 'net_recovery')}}
                   for label, m in metrics.items() if label != 'S0']
    names = yaml.safe_load(Path('kb1_reward_guided_training/configs/pest.yaml').read_text())['names']
    classes = [{'model': label, 'class_id': int(cid), 'class_name': names[int(cid)], **values}
               for label, m in metrics.items() for cid, values in m['per_class'].items()]
    curves = [{'model': label, 'step': r['step'], **{k: r['metrics'][k] for k in FIELDS}}
              for label, rows in histories.items() for r in rows]
    probe_rows = [{'model': label, 'step': r['step'], 'scope': scope, 'group': group, **values}
                  for label, rows in histories.items() for r in rows for scope, probe in r['probes'].items()
                  for group, values in probe['groups'].items()]
    for name, rows in [('summary', summaries), ('deltas', deltas), ('gt_transitions', transitions),
                       ('classes', classes), ('validation', curves), ('exposure', exposure_rows), ('probes', probe_rows)]:
        write_csv(output/f'{name}.csv', rows)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, key, scale in zip(axes.flat, ('mAP50_95', 'precision', 'operating_recall', 'f1', 'fp', 'fn'),
                              (100, 100, 100, 100, 1, 1)):
        for label, rows in histories.items():
            ax.plot([r['step'] for r in rows], [r['metrics'][key]*scale for r in rows], marker='o', label=label)
        ax.axhline(tpe[key]*scale, color='black', linestyle=':', label='TPE')
        ax.set_title(key + (' (%)' if scale == 100 else ''))
        ax.set_xlabel('Updates')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('KB2 preference sampling | seed42 | full validation 479 images | same LR schedule')
    fig.tight_layout()
    fig.savefig(output/'learning_curves.png', dpi=160)
    fig.savefig(output/'learning_curves.svg')
    plt.close(fig)
    assert all(sha256(path) == digest for path, digest in protected.items())
    review_result = {'root': str(root), 'plan': plan, 'summaries': summaries, 'deltas': deltas,
                     'GT_transitions': transitions, 'exposure': exposure_rows,
                     'hashed_eval_checkpoints': 18, 'fresh_full_validation_metric_sets': len(metrics),
                     'best_steps': {label: selected(rows)['step'] for label, rows in histories.items()},
                     'raw_max': {label: {'step': max(rows, key=lambda r: r['metrics']['mAP50_95'])['step'],
                                         'mAP50_95': max(r['metrics']['mAP50_95'] for r in rows)}
                                 for label, rows in histories.items()},
                     'initial_weights_optimizer_identical': True, 'uniform_native_stream_matched': True,
                     'sampling_intervention_only': True, 'protected_files_unchanged': True,
                     'source_and_frozen_states_unchanged': True,
                     'scope': '2k seed42 pilot on development validation; no test or confirmation claims',
                     'native_exposure_note': plan['native_exposure_note']}
    atomic_json(output/'review.json', review_result)
    print({'event': 'review_completed', 'best_steps': review_result['best_steps'], 'deltas': deltas}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    review(args.root, args.output, args.device)
