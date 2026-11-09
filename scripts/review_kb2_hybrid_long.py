"""Verify hybrid artifacts and compare fresh validation with the prior S0-DPO run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml

from kb2_preference_optimization.artifacts import atomic_json, sha256, source_checkpoint as s0_checkpoint, tensor_digest
from kb2_preference_optimization.conditional.evaluation import evaluate
from kb2_preference_optimization.conditional.policy import ConditionalAdapter, configure_numerics
from kb2_preference_optimization.data import detection_loader
from kb2_preference_optimization.hybrid.bank import Bank
from kb2_preference_optimization.hybrid.source import assert_provenance, source_provenance
from scripts.audit_kb2_conditional_gradients import read, write_csv
from scripts.review_kb2_conditional_long import events, exact_fp_point
from scripts.review_kb2_strength_multiseed import reconstruct_stream, state_equal
from scripts.review_kb2_strength_pilot import numeric_metrics

FIELDS = ('mAP50_95', 'mAP50', 'mAP75', 'precision', 'operating_recall', 'f1',
          'macro_f1', 'macro_recall', 'AR300', 'APs', 'APm', 'APl', 'tp', 'fp', 'fn')


def review(root, previous, output, device, matched_fp=False):
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Fresh full checkpoint validation requires CUDA')
    root, previous, output = (Path(p).resolve() for p in (root, previous, output))
    done, comparison = read(root/'dpo/completed.json'), read(root/'comparison.json')
    manifest = read(root/'dpo/manifest.json')
    assert read(root/'completed.json') == comparison
    assert done == {k: comparison['arms']['dpo'][k] for k in done}
    assert done['steps'] == 15000 and not done['partial_bank'] and not done['partial_evaluation']
    assert done['reference_unchanged'] and done['frozen_state_unchanged']
    current = source_provenance('yolov8n')
    assert_provenance(manifest['provenance'], current)
    assert_provenance(read(root/'source.json'), current)
    bank = Bank(root/'preferences')
    assert sha256(bank.root/'manifest.json') == manifest['bank_manifest_sha256']
    assert bank.manifest['pairs_sha256'] == manifest['pairs_sha256']
    config = manifest['config']
    for key, value in {'seed': 42, 'effective_dpo_weight': .1, 'native_background_weight': .5,
                       'beta': .1, 'lr': 1e-6, 'eval_limit': 0, 'selection_min_delta': 1e-4}.items():
        assert config[key] == value, key
    vals = [r for r in events(root/'dpo/metrics.jsonl') if r['event'] == 'validation']
    assert [r['step'] for r in vals] == list(range(0, 15001, 1000))
    selected, score = 0, vals[0]['metrics']['mAP50_95']
    for row in vals[1:]:
        if row['metrics']['mAP50_95'] > score + 1e-4:
            selected, score = row['step'], row['metrics']['mAP50_95']
    assert selected == done['best_step'] == done['selection']['best_AP_step']
    for row in vals:
        assert row['metrics']['num_images'] == 479 and row['metrics']['num_gt'] == 1887
    protected = {current['source_checkpoint']: current['source_sha256'],
                 str(s0_checkpoint('yolov8n')): current['shared_S0']['sha256']}
    assert set(done['evaluation_checkpoint_sha256']) == {str(row['step']) for row in vals}
    for step, digest in done['evaluation_checkpoint_sha256'].items():
        path = root/'dpo'/f'step_{int(step):06d}.pt'
        assert sha256(path) == digest
        protected[str(path)] = digest
    for name, step in [('best', selected), ('last', 15000)]:
        path = root/'dpo'/f'{name}.pt'
        assert sha256(path) == done[f'{name}_sha256']
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        saved = torch.load(root/'dpo'/f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
        assert checkpoint['step'] == step
        assert state_equal(checkpoint['state_dict'], saved['state_dict'])
        assert state_equal(checkpoint['optimizer'], saved['optimizer'])
        protected[str(path)] = done[f'{name}_sha256']
    assert reconstruct_stream(bank, config['seed'], events(root/'dpo/stream.jsonl')) == done['stream_sha256']
    print(json.dumps({'event': 'artifact_integrity_pass', 'best_step': selected,
                      'hashed_evaluation_checkpoints': len(vals), 'stream_steps': 15000}), flush=True)

    old_done = read(previous/'completed.json')
    old_manifest = read(previous/'manifest.json')
    assert old_manifest['provenance']['source_sha256'] == current['shared_S0']['sha256']
    assert old_manifest['provenance']['dataset']['sha256'] == current['dataset']['sha256']
    assert old_manifest['trainable_names'] == manifest['trainable_names']
    for key in ('seed', 'steps', 'input', 'native_batch', 'preference_batch', 'beta', 'dpo_weight',
                'lr', 'weight_decay', 'clip_norm', 'eval_interval', 'eval_limit', 'native_background_weight'):
        assert old_manifest['config'][key] == config[key], key
    old_best = torch.load(previous/'best.pt', map_location='cpu', weights_only=False)
    assert sha256(previous/'best.pt') == old_done['best_sha256']
    assert old_best['step'] == old_done['best_step']
    protected[str(previous/'best.pt')] = old_done['best_sha256']
    output.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    loader = detection_loader('valid', 640, 8)
    targets = [target for _, batch in loader for target in batch]
    metrics, budget_rows = {}, []

    def measure(label, adapter, logged, reference=None):
        current_metrics, predictions = evaluate(adapter, loader, reference)
        assert numeric_metrics(current_metrics) == numeric_metrics(logged), label
        assert current_metrics['num_images'] == 479 and current_metrics['num_gt'] == 1887
        metrics[label] = current_metrics
        atomic_json(output/f'{label}.json', current_metrics)
        for budget in (sorted({comparison['S0']['fp'], comparison['TPE']['fp']}) if matched_fp else []):
            budget_rows.append({'model': label, **exact_fp_point(predictions, targets, budget)})
        print(json.dumps({'event': 'fresh_validation_match', 'model': label,
                          **{k: current_metrics[k] for k in ('mAP50_95', 'fp', 'fn')}}), flush=True)
        return predictions

    source = ConditionalAdapter(s0_checkpoint('yolov8n'), device)
    original_s0_digest = tensor_digest(source.model)
    s0_predictions = measure('S0', source, comparison['S0'])
    source.model.load_state_dict(old_best['state_dict'], strict=True)
    assert source.frozen_digest() == old_manifest['frozen_state_sha256']
    measure('old_KB2_s42_best', source, old_best['metrics'], s0_predictions)
    del source, s0_predictions, old_best
    source = ConditionalAdapter(current['source_checkpoint'], device)
    assert tensor_digest(source.model) == manifest['reference_state_sha256']
    tpe_predictions = measure('TPE', source, comparison['TPE'])
    for name in ('best', 'last'):
        checkpoint = torch.load(root/'dpo'/f'{name}.pt', map_location='cpu', weights_only=False)
        source.model.load_state_dict(checkpoint['state_dict'], strict=True)
        assert source.frozen_digest() == manifest['frozen_state_sha256']
        assert all(torch.isfinite(t).all() for t in checkpoint['state_dict'].values())
        measure(f'hybrid_{name}', source, checkpoint['metrics'], tpe_predictions)

    summary = [{'model': name, **{k: m[k] for k in FIELDS}} for name, m in metrics.items()]
    deltas = []
    for name in ('hybrid_best', 'hybrid_last'):
        for baseline in ('S0', 'TPE', 'old_KB2_s42_best'):
            deltas.append({'model': name, 'baseline': baseline,
                           **{k: metrics[name][k]-metrics[baseline][k] for k in FIELDS}})
    names = yaml.safe_load(Path('kb1_reward_guided_training/configs/pest.yaml').read_text())['names']
    classes = []
    for cid, values in metrics['hybrid_best']['per_class'].items():
        classes.append({'class_id': int(cid), 'class_name': names[int(cid)], 'support': values['gt_support'],
                        **{name: m['per_class'][cid]['AP50_95'] for name, m in metrics.items()},
                        **{f'delta_vs_{name}': values['AP50_95']-metrics[name]['per_class'][cid]['AP50_95']
                           for name in ('S0', 'TPE', 'old_KB2_s42_best')}})
    curves = [{'step': row['step'], **{k: row['metrics'][k] for k in FIELDS}} for row in vals]
    write_csv(output/'summary.csv', summary)
    write_csv(output/'deltas.csv', deltas)
    write_csv(output/'classes.csv', classes)
    write_csv(output/'validation_curve.csv', curves)
    write_csv(output/'matched_FP.csv', budget_rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title in zip(axes, ('mAP50_95', 'precision', 'operating_recall'),
                              ('mAP50-95 (%)', 'Precision @ 0.25 (%)', 'Recall @ 0.25 (%)')):
        ax.plot([r['step'] for r in curves], [r[key]*100 for r in curves], color='#ba4a00', label='Hybrid')
        for label, color in [('S0', '#666666'), ('TPE', '#176b3a'), ('old_KB2_s42_best', '#225daa')]:
            ax.axhline(metrics[label][key]*100, color=color, linestyle='--', linewidth=1, label=label)
        ax.axvline(selected, color='#ba4a00', linestyle=':', linewidth=1)
        ax.set(title=title, xlabel='DPO updates')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output/'learning_curve.png', dpi=170)
    fig.savefig(output/'learning_curve.svg')
    plt.close(fig)
    assert all(sha256(path) == digest for path, digest in protected.items())
    class_counts = {baseline: {'improved': sum(c[f'delta_vs_{baseline}'] > 0 for c in classes),
                               'decreased': sum(c[f'delta_vs_{baseline}'] < 0 for c in classes)}
                    for baseline in ('S0', 'TPE', 'old_KB2_s42_best')}
    report = {'run': str(root), 'previous': str(previous), 'split': 'valid', 'n_images': 479, 'n_GT': 1887,
              'steps': done['steps'], 'best_step': selected, 'unit': 'fractions except TP/FP/FN counts',
              'checks': {'source_and_runtime_provenance': True, 'bank_hashes': True,
                         'evaluation_checkpoint_hashes': len(vals), 'reconstructed_stream_steps': 15000,
                         'best_last_weights_and_optimizer_match_selected_steps': True,
                         'fresh_full_validation_matches': list(metrics), 'frozen_state_unchanged': True,
                         'source_and_checkpoints_unchanged': True, 'S0_state_sha256': original_s0_digest},
              'summary': summary, 'deltas': deltas, 'class_counts': class_counts,
              'matched_FP': budget_rows, 'bank_audit': bank.manifest['audit'],
              'train_wall_seconds_including_periodic_eval': done['wall_seconds'],
              'limits': ['One continuation seed; validation used for development; no fresh test',
                         'Hybrid has no TPE-initialized native-only control',
                         'Different initial model and regenerated bank from old KB2; not an objective-only ablation']}
    atomic_json(output/'review.json', report)
    print(json.dumps({'event': 'review_complete', 'output': str(output), 'class_counts': class_counts}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous', type=Path, default=Path('kb2_preference_optimization/outputs/yolov8n_strength_long_3seeds_20261103_095853_370249/s42/dpo'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--matched-fp', action='store_true', help='Also sweep validation thresholds (slower)')
    args = parser.parse_args()
    review(args.run, args.previous, args.output, args.device, args.matched_fp)
