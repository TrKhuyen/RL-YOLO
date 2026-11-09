"""Compare matched dense-eval DPO runs dropping LR after500 versus1000 updates."""
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
from kb2_preference_optimization.hybrid_schedule import code_fingerprints
from kb2_preference_optimization.hybrid_schedule.schedule import evaluation_steps, learning_rate_for_update
from scripts.audit_kb2_conditional_gradients import write_csv
from scripts.review_kb2_hybrid_bg_comparison import read, events, FIELDS
from scripts.review_kb2_hybrid_schedule_trial import selected
from scripts.review_kb2_strength_multiseed import state_equal
from scripts.review_kb2_strength_pilot import numeric_metrics


def review(root, output, device):
    root, output = root.resolve(), output.resolve()
    done = read(root/'completed.json')
    plan = done['plan']
    assert done['matched_stream'] and done['protected_files_unchanged']
    assert plan['lr_drop_after_update'] == 500 and plan['seed'] == 42 and plan['steps'] == 15000
    provenance = source_provenance('yolov8n')
    assert_provenance(plan['source'], provenance)
    folders = {'drop1000': Path(plan['scheduled_baseline']), 'drop500': root/'run'}
    histories, configs, fixed, states, best_steps, common_steps = {}, {}, {}, {}, {}, {}
    protected = dict(plan['immutable_files'])
    grid = evaluation_steps(15000, 1000, 250, 2000)
    source_stream = (folders['drop1000']/'dpo/stream.jsonl').read_bytes()
    for label, folder in folders.items():
        drop = 1000 if label == 'drop1000' else 500
        aggregate = read(folder/'completed.json')
        result = read(folder/'dpo/completed.json')
        manifest = read(folder/'dpo/manifest.json')
        assert_provenance(manifest['provenance'], provenance)
        assert list(aggregate['arms']) == ['dpo'] and result['steps'] == 15000
        assert not aggregate['partial_bank'] and not aggregate['partial_evaluation']
        assert result['reference_unchanged'] and result['frozen_state_unchanged']
        assert (folder/'dpo/stream.jsonl').read_bytes() == source_stream
        assert manifest['scheduled_code_sha256'] == code_fingerprints()
        config = manifest['config']
        assert config['lr_schedule'] == {'drop_after_update': drop, 'factor': .1,
                                         'initial_lr': 1e-6, 'updates_are_one_based': True}
        assert config['effective_dpo_weight'] == .03 and config['native_background_weight'] == .5
        assert config['seed'] == 42 and config['eval_limit'] == 0 and config['evaluation_steps'] == grid
        configs[label] = {k: v for k, v in config.items() if k != 'lr_schedule'}
        fixed[label] = {k: v for k, v in manifest.items() if k != 'config'}
        vals = [r for r in events(folder/'dpo/metrics.jsonl') if r['event'] == 'validation']
        assert [r['step'] for r in vals] == grid
        assert selected(vals)['step'] == result['best_step']
        best_steps[label] = result['best_step']
        common_steps[label] = selected([r for r in vals if r['step'] % 1000 == 0])['step']
        assert len(result['evaluation_checkpoint_sha256']) == len(grid)
        for step, digest in result['evaluation_checkpoint_sha256'].items():
            path = folder/'dpo'/f'step_{int(step):06d}.pt'
            assert sha256(path) == digest
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            rate = 1e-6 if int(step) == 0 else learning_rate_for_update(1e-6, int(step), drop, .1)
            assert checkpoint['optimizer']['param_groups'][0]['lr'] == rate
        for which, step in (('best', result['best_step']), ('last', 15000)):
            path = folder/'dpo'/f'{which}.pt'
            assert sha256(path) == result[f'{which}_sha256']
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            saved = torch.load(folder/'dpo'/f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
            assert state_equal(checkpoint['state_dict'], saved['state_dict'])
            assert state_equal(checkpoint['optimizer'], saved['optimizer'])
            states[f'{label}_{which}'] = (path, checkpoint['metrics'], manifest['frozen_state_sha256'])
            protected[str(path)] = result[f'{which}_sha256']
        if label == 'drop500':
            step = common_steps[label]
            path = folder/'dpo'/f'step_{step:06d}.pt'
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            states['drop500_common_grid_best'] = (path, checkpoint['metrics'], manifest['frozen_state_sha256'])
        histories[label] = vals
    assert configs['drop1000'] == configs['drop500']
    assert fixed['drop1000'] == fixed['drop500']
    first = torch.load(root/'run/dpo/step_000500.pt', map_location='cpu', weights_only=False)
    old = torch.load(folders['drop1000']/'dpo/step_000500.pt', map_location='cpu', weights_only=False)
    assert state_equal(first['state_dict'], old['state_dict'])
    assert state_equal(first['optimizer'], old['optimizer'])
    assert numeric_metrics(first['metrics']) == numeric_metrics(old['metrics'])
    output.mkdir(parents=True, exist_ok=False)
    print({'event': 'artifact_checks_pass', 'hashed_eval_checkpoints': 44,
           'only_changed_config': 'lr_schedule.drop_after_update', 'first_500_match': True}, flush=True)
    configure_numerics()
    loader = detection_loader('valid', 640, 8)
    adapter = ConditionalAdapter(provenance['source_checkpoint'], device)
    tpe, predictions = evaluate(adapter, loader)
    assert numeric_metrics(tpe) == numeric_metrics(histories['drop500'][0]['metrics'])
    metrics = {'TPE': tpe}
    atomic_json(output/'TPE.json', tpe)
    for label, (path, logged, frozen) in states.items():
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        adapter.model.load_state_dict(checkpoint['state_dict'], strict=True)
        assert adapter.frozen_digest() == frozen
        measured, _ = evaluate(adapter, loader, predictions)
        assert numeric_metrics(measured) == numeric_metrics(logged), label
        assert measured['num_images'] == 479 and measured['num_gt'] == 1887
        metrics[label] = measured
        atomic_json(output/f'{label}.json', measured)
        print({'event': 'reloaded', 'model': label, 'mAP50_95': measured['mAP50_95']}, flush=True)
    del adapter, predictions
    adapter = ConditionalAdapter(source_checkpoint('yolov8n'), device)
    s0, _ = evaluate(adapter, loader)
    assert numeric_metrics(s0) == numeric_metrics(done['result']['S0'])
    metrics['S0'] = s0
    atomic_json(output/'S0.json', s0)
    summary = [{'model': label, **{k: m[k] for k in FIELDS}} for label, m in metrics.items()]
    deltas = []
    for which in ('best', 'last'):
        model = f'drop500_{which}'
        for reference in ('S0', 'TPE', f'drop1000_{which}'):
            deltas.append({'model': model, 'baseline': reference,
                           **{k: metrics[model][k]-metrics[reference][k] for k in FIELDS}})
    write_csv(output/'summary.csv', summary)
    write_csv(output/'deltas.csv', deltas)
    write_csv(output/'validation.csv', [{'model': label, 'step': r['step'],
                                        **{k: r['metrics'][k] for k in FIELDS}}
                                       for label, vals in histories.items() for r in vals])
    old_rows = {r['step']: r['metrics'] for r in histories['drop1000']}
    write_csv(output/'paired_steps.csv', [{'step': r['step'],
                                          **{k: r['metrics'][k]-old_rows[r['step']][k] for k in FIELDS}}
                                         for r in histories['drop500']])
    names = yaml.safe_load(Path('kb1_reward_guided_training/configs/pest.yaml').read_text())['names']
    write_csv(output/'classes.csv', [{'model': label, 'class_id': int(cid), 'class_name': names[int(cid)], **m}
                                    for label, values in metrics.items() for cid, m in values['per_class'].items()])
    write_csv(output/'gt_transitions.csv', [{'model': label,
                                           **{k: values['source_gt_transitions'].get(k, 0)
                                              for k in ('recovered', 'lost', 'net_recovery')}}
                                          for label, values in metrics.items()])
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
    best_identical = state_equal(torch.load(states['drop500_best'][0], map_location='cpu', weights_only=False)['state_dict'],
                                 torch.load(states['drop1000_best'][0], map_location='cpu', weights_only=False)['state_dict'])
    atomic_json(output/'review.json', {'campaign': str(root), 'split': 'valid', 'images': 479, 'GT': 1887,
                                      'matched_stream': True, 'only_changed_config': 'lr_schedule.drop_after_update',
                                      'first_500_weights_optimizer_match': True,
                                      'protected_files_unchanged': True, 'hashed_eval_checkpoints': 44,
                                      'best_state_dicts_identical': best_identical,
                                      'fresh_metrics_match': list(metrics), 'best_steps': best_steps,
                                      'common_grid_best_steps': common_steps, 'summary': summary, 'deltas': deltas,
                                      'limits': ['One continuation seed; development validation only']})
    print({'event': 'review_completed', 'best_steps': best_steps, 'common_steps': common_steps,
           'best_state_dicts_identical': best_identical}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    review(args.campaign, args.output, args.device)
