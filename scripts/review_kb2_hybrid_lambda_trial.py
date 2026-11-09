"""Verify a lambda=.03 trial against matched native and lambda=.1 continuations."""
import argparse
from collections import defaultdict
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
from scripts.audit_kb2_conditional_gradients import write_csv
from scripts.review_kb2_hybrid_bg_comparison import read, events, FIELDS
from scripts.review_kb2_strength_multiseed import state_equal
from scripts.review_kb2_strength_pilot import numeric_metrics


def review(root, output, device):
    root, output = root.resolve(), output.resolve()
    done = read(root/'completed.json')
    plan = done['plan']
    assert done['matched_stream'] and done['protected_files_unchanged']
    folders = {'dpo_lambda010': (Path(plan['baseline']), 'dpo', .1),
               'native_lambda000': (Path(plan['native_control']), 'native', 0.),
               'dpo_lambda003': (root/'run', 'dpo', .03)}
    provenance = source_provenance('yolov8n')
    assert_provenance(plan['source'], provenance)
    protected = dict(plan['immutable_files'])
    stream = (Path(plan['baseline'])/'dpo/stream.jsonl').read_bytes()
    histories, states, configs, aggregates, best_steps, fixed_manifests = {}, {}, {}, {}, {}, {}
    for name, (folder, arm, weight) in folders.items():
        aggregate = read(folder/'completed.json')
        result = read(folder/arm/'completed.json')
        manifest = read(folder/arm/'manifest.json')
        assert_provenance(manifest['provenance'], provenance)
        assert result['steps'] == 15000 and not aggregate['partial_bank'] and not aggregate['partial_evaluation']
        assert result['reference_unchanged'] and result['frozen_state_unchanged']
        assert (folder/arm/'stream.jsonl').read_bytes() == stream
        assert manifest['bank_manifest_sha256'] == sha256(Path(plan['bank'])/'manifest.json')
        config = manifest['config']
        fixed_manifests[name] = {k: v for k, v in manifest.items() if k != 'config'}
        assert config['effective_dpo_weight'] == weight and config['native_background_weight'] == .5
        assert config['seed'] == 42 and config['eval_limit'] == 0 and config['steps'] == 15000
        configs[name] = {k: v for k, v in config.items()
                         if k not in ('arm', 'dpo_weight', 'effective_dpo_weight')}
        vals = [r for r in events(folder/arm/'metrics.jsonl') if r['event'] == 'validation']
        assert [r['step'] for r in vals] == list(range(0, 15001, 1000))
        selected, score = 0, vals[0]['metrics']['mAP50_95']
        for row in vals[1:]:
            if row['metrics']['mAP50_95'] > score + 1e-4:
                selected, score = row['step'], row['metrics']['mAP50_95']
        assert selected == result['best_step']
        assert len(result['evaluation_checkpoint_sha256']) == 16
        for step, digest in result['evaluation_checkpoint_sha256'].items():
            assert sha256(folder/arm/f'step_{int(step):06d}.pt') == digest
        for which, step in (('best', selected), ('last', 15000)):
            path = folder/arm/f'{which}.pt'
            assert sha256(path) == result[f'{which}_sha256']
            state = torch.load(path, map_location='cpu', weights_only=False)
            saved = torch.load(folder/arm/f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
            assert state_equal(state['state_dict'], saved['state_dict'])
            assert state_equal(state['optimizer'], saved['optimizer'])
            states[f'{name}_{which}'] = (path, state['metrics'], manifest['frozen_state_sha256'])
            protected[str(path)] = result[f'{which}_sha256']
        histories[name], aggregates[name], best_steps[name] = vals, aggregate, selected
    assert all(config == configs['dpo_lambda010'] for config in configs.values())
    assert all(manifest == fixed_manifests['dpo_lambda010'] for manifest in fixed_manifests.values())
    assert all(numeric_metrics(vals[0]['metrics']) == numeric_metrics(histories['dpo_lambda010'][0]['metrics'])
               for vals in histories.values())
    output.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    loader = detection_loader('valid', 640, 8)
    adapter = ConditionalAdapter(provenance['source_checkpoint'], device)
    tpe, reference_predictions = evaluate(adapter, loader)
    assert numeric_metrics(tpe) == numeric_metrics(histories['dpo_lambda010'][0]['metrics'])
    metrics = {'TPE': tpe}
    atomic_json(output/'TPE.json', tpe)
    for label, (path, logged, frozen) in states.items():
        state = torch.load(path, map_location='cpu', weights_only=False)
        adapter.model.load_state_dict(state['state_dict'], strict=True)
        assert adapter.frozen_digest() == frozen
        measured, _ = evaluate(adapter, loader, reference_predictions)
        assert numeric_metrics(measured) == numeric_metrics(logged), label
        assert measured['num_images'] == 479 and measured['num_gt'] == 1887
        metrics[label] = measured
        atomic_json(output/f'{label}.json', measured)
        print({'event': 'reloaded', 'label': label, 'mAP50_95': measured['mAP50_95']}, flush=True)
    del adapter, reference_predictions
    adapter = ConditionalAdapter(source_checkpoint('yolov8n'), device)
    s0, _ = evaluate(adapter, loader)
    assert numeric_metrics(s0) == numeric_metrics(aggregates['dpo_lambda003']['S0'])
    metrics['S0'] = s0
    atomic_json(output/'S0.json', s0)
    summary = [{'model': name, **{k: m[k] for k in FIELDS}} for name, m in metrics.items()]
    deltas = []
    for which in ('best', 'last'):
        label = f'dpo_lambda003_{which}'
        for reference in ('TPE', 'S0', f'dpo_lambda010_{which}', f'native_lambda000_{which}'):
            deltas.append({'model': label, 'baseline': reference,
                           **{k: metrics[label][k]-metrics[reference][k] for k in FIELDS}})
    write_csv(output/'summary.csv', summary)
    write_csv(output/'deltas.csv', deltas)
    write_csv(output/'validation.csv', [{'model': label, 'step': row['step'],
                                        **{k: row['metrics'][k] for k in FIELDS}}
                                       for label, vals in histories.items() for row in vals])
    names = yaml.safe_load(Path('kb1_reward_guided_training/configs/pest.yaml').read_text())['names']
    write_csv(output/'classes.csv', [{'model': label, 'class_id': int(cid), 'class_name': names[int(cid)],
                                     **values} for label, m in metrics.items()
                                    for cid, values in m['per_class'].items()])
    probes = []
    for label, vals in histories.items():
        for row in vals:
            for scope, diagnostic in row['probes'].items():
                grouped = defaultdict(list)
                for record in diagnostic['records']:
                    grouped[record['group']].append(record)
                for group, records in grouped.items():
                    probes.append({'model': label, 'step': row['step'], 'scope': scope, 'group': group,
                                   'n': len(records), 'positive_margin': sum(r['margin'] > 0 for r in records),
                                   'mean_score_delta': sum(r['score']-r['s0_score'] for r in records)/len(records),
                                   'target_detected': sum(r['ordinary_target_detected'] is True for r in records)})
    write_csv(output/'probes.csv', probes)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title in zip(axes, ('mAP50_95', 'precision', 'operating_recall'),
                              ('mAP50-95 (%)', 'Precision @ 0.25 (%)', 'Recall @ 0.25 (%)')):
        ax.axhline(tpe[key]*100, color='#555555', linestyle=':', label='TPE')
        for label, vals in histories.items():
            ax.plot([r['step'] for r in vals], [r['metrics'][key]*100 for r in vals], label=label)
        ax.set(title=title, xlabel='Updates')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output/'learning_curves.png', dpi=170)
    fig.savefig(output/'learning_curves.svg')
    plt.close(fig)
    assert all(sha256(path) == digest for path, digest in protected.items())
    atomic_json(output/'review.json', {'campaign': str(root), 'split': 'valid', 'images': 479, 'GT': 1887,
                                      'matched_stream': True, 'matched_other_hyperparameters': True,
                                      'protected_files_unchanged': True, 'hashed_eval_checkpoints': 48,
                                      'fresh_metrics_match': list(metrics), 'best_steps': best_steps,
                                      'summary': summary, 'deltas': deltas,
                                      'limits': ['One continuation seed from one TPE initialization',
                                                 'Development validation only; no untouched test evaluation']})
    print({'event': 'review_completed', 'output': str(output), 'best_steps': best_steps}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    review(args.campaign, args.output, args.device)
