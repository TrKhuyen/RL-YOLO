"""Verify scheduled DPO and distinguish optimization from denser checkpoint selection."""
import argparse
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
from kb2_preference_optimization.hybrid_schedule import code_fingerprints as schedule_fingerprints
from kb2_preference_optimization.hybrid_schedule.schedule import evaluation_steps, learning_rate_for_update
from scripts.audit_kb2_conditional_gradients import write_csv
from scripts.review_kb2_hybrid_bg_comparison import read, events, FIELDS
from scripts.review_kb2_strength_multiseed import state_equal
from scripts.review_kb2_strength_pilot import numeric_metrics


def selected(rows):
    best = rows[0]
    for row in rows[1:]:
        if row['metrics']['mAP50_95'] > best['metrics']['mAP50_95'] + 1e-4:
            best = row
    return best


def review(root, output, device, reuse_evaluations=False):
    root, output = root.resolve(), output.resolve()
    done = read(root/'completed.json')
    plan = done['plan']
    assert done['matched_stream'] and done['protected_files_unchanged']
    folders = {'constant_lambda003': Path(plan['baseline']),
               'constant_lambda010': Path(plan['older_DPO_baseline']),
               'scheduled_lambda003': root/'run'}
    provenance = source_provenance('yolov8n')
    assert_provenance(plan['source'], provenance)
    protected = dict(plan['immutable_files'])
    stream = (Path(plan['baseline'])/'dpo/stream.jsonl').read_bytes()
    histories, configs, fixed, states, aggregates, best_steps, common_steps = {}, {}, {}, {}, {}, {}, {}
    hashes = 0
    excluded = {'effective_dpo_weight', 'dpo_weight', 'training_variant', 'lr_schedule',
                'evaluation_steps', 'early_eval_interval', 'early_eval_until', 'pilot'}
    for label, folder in folders.items():
        result = read(folder/'dpo/completed.json')
        manifest = read(folder/'dpo/manifest.json')
        aggregate = read(folder/'completed.json')
        assert_provenance(manifest['provenance'], provenance)
        assert list(aggregate['arms']) == ['dpo'] and result['steps'] == 15000
        assert result['reference_unchanged'] and result['frozen_state_unchanged']
        assert not aggregate['partial_bank'] and not aggregate['partial_evaluation']
        assert (folder/'dpo/stream.jsonl').read_bytes() == stream
        config = manifest['config']
        assert config['effective_dpo_weight'] == (.1 if label == 'constant_lambda010' else .03)
        assert config['seed'] == 42 and config['eval_limit'] == 0
        configs[label] = {k: v for k, v in config.items() if k not in excluded}
        fixed[label] = {k: v for k, v in manifest.items()
                        if k not in ('config', 'training_variant', 'scheduled_code_sha256')}
        vals = [r for r in events(folder/'dpo/metrics.jsonl') if r['event'] == 'validation']
        grid = evaluation_steps(15000, 1000, 250, 2000) if label.startswith('scheduled') else list(range(0, 15001, 1000))
        assert [r['step'] for r in vals] == grid
        assert selected(vals)['step'] == result['best_step']
        best_steps[label] = result['best_step']
        common_steps[label] = selected([r for r in vals if r['step'] % 1000 == 0])['step']
        assert len(result['evaluation_checkpoint_sha256']) == len(grid)
        for step, digest in result['evaluation_checkpoint_sha256'].items():
            assert sha256(folder/'dpo'/f'step_{int(step):06d}.pt') == digest
            hashes += 1
        for which, step in (('best', result['best_step']), ('last', 15000)):
            path = folder/'dpo'/f'{which}.pt'
            assert sha256(path) == result[f'{which}_sha256']
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            saved = torch.load(folder/'dpo'/f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
            assert state_equal(checkpoint['state_dict'], saved['state_dict'])
            assert state_equal(checkpoint['optimizer'], saved['optimizer'])
            states[f'{label}_{which}'] = (path, checkpoint['metrics'], manifest['frozen_state_sha256'])
            protected[str(path)] = result[f'{which}_sha256']
        if label.startswith('scheduled'):
            assert manifest['scheduled_code_sha256'] == schedule_fingerprints()
            assert config['lr_schedule'] == {'drop_after_update': 1000, 'factor': .1,
                                             'initial_lr': 1e-6, 'updates_are_one_based': True}
            for step in grid:
                checkpoint = torch.load(folder/'dpo'/f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
                expected = 1e-6 if step == 0 else learning_rate_for_update(1e-6, step, 1000, .1)
                assert checkpoint['optimizer']['param_groups'][0]['lr'] == expected
            common = common_steps[label]
            checkpoint = torch.load(folder/'dpo'/f'step_{common:06d}.pt', map_location='cpu', weights_only=False)
            states['scheduled_common_grid_best'] = (folder/'dpo'/f'step_{common:06d}.pt',
                                                     checkpoint['metrics'], manifest['frozen_state_sha256'])
        histories[label], aggregates[label] = vals, aggregate
    assert hashes == 54
    assert all(config == configs['constant_lambda003'] for config in configs.values())
    assert all(manifest == fixed['constant_lambda003'] for manifest in fixed.values())
    first = torch.load(root/'run/dpo/step_001000.pt', map_location='cpu', weights_only=False)
    old = torch.load(Path(plan['baseline'])/'dpo/step_001000.pt', map_location='cpu', weights_only=False)
    assert state_equal(first['state_dict'], old['state_dict'])
    assert state_equal(first['optimizer'], old['optimizer'])
    assert numeric_metrics(first['metrics']) == numeric_metrics(old['metrics'])
    output.mkdir(parents=True, exist_ok=reuse_evaluations)
    print({'event': 'artifact_checks_pass', 'hashes': hashes, 'first_1000_weights_optimizer_match': True}, flush=True)
    configure_numerics()
    loader = detection_loader('valid', 640, 8)
    adapter = ConditionalAdapter(provenance['source_checkpoint'], 'cpu' if reuse_evaluations else device)
    if reuse_evaluations:
        tpe, predictions = read(output/'TPE.json'), None
    else:
        tpe, predictions = evaluate(adapter, loader)
    assert numeric_metrics(tpe) == numeric_metrics(histories['scheduled_lambda003'][0]['metrics'])
    metrics = {'TPE': tpe}
    atomic_json(output/'TPE.json', tpe)
    for label, (path, logged, frozen) in states.items():
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        adapter.model.load_state_dict(checkpoint['state_dict'], strict=True)
        assert adapter.frozen_digest() == frozen
        if reuse_evaluations:
            measured = read(output/f'{label}.json')
        else:
            measured, _ = evaluate(adapter, loader, predictions)
        assert numeric_metrics(measured) == numeric_metrics(logged), label
        assert measured['num_images'] == 479 and measured['num_gt'] == 1887
        metrics[label] = measured
        atomic_json(output/f'{label}.json', measured)
        print({'event': 'reloaded', 'model': label, 'mAP50_95': measured['mAP50_95']}, flush=True)
    del adapter, predictions
    if reuse_evaluations:
        s0 = read(output/'S0.json')
    else:
        adapter = ConditionalAdapter(source_checkpoint('yolov8n'), device)
        s0, _ = evaluate(adapter, loader)
    assert numeric_metrics(s0) == numeric_metrics(aggregates['scheduled_lambda003']['S0'])
    metrics['S0'] = s0
    atomic_json(output/'S0.json', s0)
    summary = [{'model': label, **{k: m[k] for k in FIELDS}} for label, m in metrics.items()]
    deltas = []
    for which in ('best', 'last'):
        model = f'scheduled_lambda003_{which}'
        for reference in ('S0', 'TPE', f'constant_lambda003_{which}', f'constant_lambda010_{which}'):
            deltas.append({'model': model, 'baseline': reference,
                           **{k: metrics[model][k]-metrics[reference][k] for k in FIELDS}})
    reference = 'constant_lambda003_best'
    deltas.append({'model': 'scheduled_common_grid_best', 'baseline': reference,
                   **{k: metrics['scheduled_common_grid_best'][k]-metrics[reference][k] for k in FIELDS}})
    write_csv(output/'summary.csv', summary)
    write_csv(output/'deltas.csv', deltas)
    write_csv(output/'validation.csv', [{'model': label, 'step': r['step'],
                                        **{k: r['metrics'][k] for k in FIELDS}}
                                       for label, vals in histories.items() for r in vals])
    names = yaml.safe_load(Path('kb1_reward_guided_training/configs/pest.yaml').read_text())['names']
    write_csv(output/'classes.csv', [{'model': label, 'class_id': int(cid), 'class_name': names[int(cid)], **m}
                                    for label, values in metrics.items() for cid, m in values['per_class'].items()])
    transitions = [{'model': label, **{k: values['source_gt_transitions'].get(k, 0)
                                      for k in ('recovered', 'lost', 'net_recovery')}}
                   for label, values in metrics.items() if 'source_gt_transitions' in values]
    write_csv(output/'gt_transitions.csv', transitions)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title in zip(axes, ('mAP50_95', 'precision', 'operating_recall'),
                              ('mAP50-95 (%)', 'Precision @ 0.25 (%)', 'Recall @ 0.25 (%)')):
        ax.axhline(tpe[key]*100, color='#555555', linestyle=':', label='TPE')
        for label, vals in histories.items():
            ax.plot([r['step'] for r in vals], [r['metrics'][key]*100 for r in vals], label=label)
        ax.axvline(1000, color='#aaaaaa', linestyle='--', linewidth=.8)
        ax.set(title=title, xlabel='Updates')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output/'learning_curves.png', dpi=170)
    fig.savefig(output/'learning_curves.svg')
    plt.close(fig)
    assert all(sha256(path) == digest for path, digest in protected.items())
    atomic_json(output/'review.json', {'campaign': str(root), 'split': 'valid', 'images': 479, 'GT': 1887,
                                      'matched_stream': True, 'matched_other_hyperparameters': True,
                                      'first_1000_weights_optimizer_match': True,
                                      'protected_files_unchanged': True, 'hashed_eval_checkpoints': hashes,
                                      'reused_completed_evaluations': reuse_evaluations,
                                      'fresh_metrics_match': list(metrics), 'best_steps': best_steps,
                                      'common_grid_best_steps': common_steps, 'summary': summary, 'deltas': deltas,
                                      'limits': ['One continuation seed; development validation',
                                                 'Dense-grid best and common-grid best are separate comparisons']})
    print({'event': 'review_completed', 'best_steps': best_steps, 'common_steps': common_steps}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--reuse-completed-evaluations', action='store_true',
                        help='Finish reporting from existing metric JSONs; recheck all against logs and checkpoint hashes')
    args = parser.parse_args()
    review(args.campaign, args.output, args.device, args.reuse_completed_evaluations)
