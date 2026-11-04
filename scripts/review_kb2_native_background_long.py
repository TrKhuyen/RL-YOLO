"""Verify completed background-0.5 confirmation and audit ordinary validation/FP."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml

from kb2_preference_optimization.artifacts import (
    assert_provenance, atomic_json, sha256, source_checkpoint, source_provenance, tensor_digest,
)
from kb2_preference_optimization.conditional import code_fingerprints as conditional_fingerprints
from kb2_preference_optimization.conditional.bank import Bank
from kb2_preference_optimization.conditional.evaluation import evaluate, eligible
from kb2_preference_optimization.conditional.pairs import matches
from kb2_preference_optimization.conditional.policy import ConditionalAdapter, configure_numerics
from kb2_preference_optimization.conditional_bg import code_fingerprints
from kb2_preference_optimization.data import detection_loader
from scripts.audit_kb2_conditional_gradients import read, write_csv
from scripts.review_kb2_conditional_long import equal_metrics, events, exact_fp_point


def review(root, output, device):
    root, output = Path(root).resolve(), Path(output).resolve()
    completed = read(root / 'completed.json')
    plan = completed['plan']
    assert completed['matched_stream'] and completed['old_checkpoints_unchanged']
    assert plan['steps_per_arm'] == 15000 and plan['background_weight'] == .5
    assert code_fingerprints() == plan['code_sha256']
    assert sha256(Path('scripts/run_kb2_native_background_long.py')) == plan['runner_sha256']
    assert_provenance(plan['provenance'], source_provenance('yolov8n'))
    bank = Bank(plan['bank_path'])
    immutable = plan['immutable_checkpoints']
    assert all(sha256(Path(path)) == digest for path, digest in immutable.items())
    pilot = Path(plan['pilot_root'])
    old_root = root.parent / 'yolov8n_conditional_long_s42_20261030_200209_427399'
    previous_review = read('docs/report_final/assets/kb2_conditional_long_20261031/review.json')
    output.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    loader = detection_loader('valid', 640, 8)
    targets = [target for _, batch in loader for target in batch]
    adapter = ConditionalAdapter(source_checkpoint('yolov8n'), device)
    source_digest, frozen = tensor_digest(adapter.model), adapter.frozen_digest()
    baseline, source_predictions = evaluate(adapter, loader)
    assert baseline['num_images'] == 479 and baseline['num_gt'] == 1887
    atomic_json(output / 's0_validation.json', baseline)
    metrics, predictions = {'s0': baseline}, {'s0': source_predictions}
    validations, checks, streams, diagnostics = {}, {}, {}, {}
    for arm in ('dpo', 'native'):
        folder = root / arm
        result = completed['arms'][arm]
        assert result['steps'] == 15000 and not result['partial_bank'] and not result['partial_evaluation']
        assert result['reference_unchanged'] and result['frozen_state_unchanged']
        manifest = read(folder / 'manifest.json')
        assert manifest['conditional_code_sha256'] == conditional_fingerprints()
        assert manifest['native_background_code_sha256'] == code_fingerprints()
        assert manifest['bank_manifest_sha256'] == sha256(bank.root / 'manifest.json')
        streams[arm] = (folder / 'stream.jsonl').read_bytes()
        entries = events(folder / 'stream.jsonl')
        assert len(entries) == 15000
        digest = hashlib.sha256()
        for step, row in enumerate(entries, 1):
            assert row['step'] == step
            digest.update(json.dumps({k: v for k, v in row.items() if k != 'step'}, sort_keys=True).encode())
        assert digest.hexdigest() == result['stream_sha256']
        # Original 15k stream was independently reconstructed in its review.
        assert streams[arm] == (old_root / arm / 'stream.jsonl').read_bytes()
        assert streams[arm].splitlines()[:2000] == (pilot / f'reduced_{arm}/stream.jsonl').read_bytes().splitlines()
        rows = events(folder / 'metrics.jsonl')
        vals = [r for r in rows if r['event'] == 'validation']
        validations[arm] = vals
        assert [r['step'] for r in vals] == list(range(0, 15001, 1000))
        assert all(r['metrics']['num_images'] == 479 and r['metrics']['num_gt'] == 1887 for r in vals)
        assert equal_metrics(vals[0]['metrics'], baseline)
        old_vals = [r for r in events(pilot / f'reduced_{arm}/metrics.jsonl') if r['event'] == 'validation']
        assert all(equal_metrics(a['metrics'], b['metrics']) and a['probes'] == b['probes']
                   for a, b in zip(vals[:3], old_vals))
        for name in ('last', 'best'):
            assert sha256(folder / f'{name}.pt') == result[f'{name}_sha256']
        last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        assert last['step'] == 15000
        adapter.model.load_state_dict(last['state_dict'], strict=True)
        assert adapter.frozen_digest() == frozen
        current, prediction = evaluate(adapter, loader, source_predictions)
        assert equal_metrics(current, vals[-1]['metrics']) and equal_metrics(current, result['last_metrics'])
        metrics[arm], predictions[arm] = current, prediction
        atomic_json(output / f'{arm}_validation.json', current)
        best = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
        assert best['step'] == result['best_step'] == 0
        adapter.model.load_state_dict(best['state_dict'], strict=True)
        assert tensor_digest(adapter.model) == source_digest
        assert not any(r['eligible'] for r in vals[1:])
        ratios = [r['gradient']['weighted_dpo_native_norm_ratio'] for r in rows if r.get('gradient')]
        train_rows = [r for r in rows if r['event'] == 'train']
        diagnostics[arm] = {'mean_logged_weighted_dpo_native_gradient_ratio': statistics.mean(ratios),
                            'gradient_diagnostic_samples': len(ratios),
                            'max_logged_raw_box_delta': max(r['raw_box_max_abs_delta'] for r in train_rows),
                            'probes_at_last': {k: v['groups'] for k, v in vals[-1]['probes'].items()}}
        checks[arm] = {'full_metrics_reproduced': True, 'best_weights_equal_S0': True,
                       'frozen_state_unchanged': True, 'reference_unchanged': True,
                       'pilot_prefix_stream_metrics_and_probes_match': True,
                       'all_15000_stream_bytes_equal_previous_verified_run': True,
                       'last_sha256': result['last_sha256'], 'best_sha256': result['best_sha256']}
        print(json.dumps({'event': 'fresh_validation', 'arm': arm,
                          **{k: current[k] for k in ('mAP50_95', 'operating_recall', 'fp', 'fn')}}), flush=True)
        del last, best
    assert streams['dpo'] == streams['native']
    assert completed['arms']['dpo']['actual_exposure'] == completed['arms']['native']['actual_exposure']
    assert all(sha256(Path(path)) == digest for path, digest in immutable.items())
    exact = {label: exact_fp_point(ps, targets, baseline['fp']) for label, ps in predictions.items()}
    atomic_json(output / 'exact_fp_budget.json', exact)
    write_csv(output / 'exact_fp_budget.csv', [{'model': k, **v} for k, v in exact.items()])
    images, ledger, matched_ledger = [], [], []
    for index, target in enumerate(targets):
        row = {'image_id': target['image_id'], 'image_path': target['image_path'],
               'gt': len(target['labels']), 'density': '6+' if len(target['labels']) >= 6 else
               ('3-5' if len(target['labels']) >= 3 else '1-2')}
        found, matched = {}, {}
        for label, ps in predictions.items():
            found[label], fp, _ = matches(ps[index], target, .25)
            matched[label], _, _ = matches(ps[index], target, exact[label]['confidence'])
            row.update({f'{label}_tp': len(found[label]), f'{label}_fp': fp,
                        f'{label}_fn': len(target['labels'])-len(found[label])})
        images.append(row)
        for g, c in enumerate(target['labels'].tolist()):
            identity = {'image_id': target['image_id'], 'gt_id': g, 'class_id': c}
            ledger.append({**identity, **{label: g in ids for label, ids in found.items()}})
            matched_ledger.append({**identity, **{label: g in ids for label, ids in matched.items()}})
    for label, value in metrics.items():
        for key in ('tp', 'fp', 'fn'):
            assert sum(r[f'{label}_{key}'] for r in images) == value[key]
        assert sum(int(r[label]) for r in matched_ledger) == exact[label]['tp']
    write_csv(output / 'images.csv', images)
    write_csv(output / 'gt_ledger.csv', ledger)
    write_csv(output / 'matched_fp_gt_ledger.csv', matched_ledger)
    names = yaml.safe_load(Path('kb1_reward_guided_training/configs/pest.yaml').read_text())['names']
    classes = []
    for c, base in baseline['per_class'].items():
        row = {'class_id': int(c), 'class_name': names[int(c)], 'gt_support': base['gt_support']}
        for label, m in metrics.items():
            row.update({f'{label}_{k}': m['per_class'][c][k] for k in ('tp', 'fp', 'fn', 'AP50_95')})
        classes.append(row)
    write_csv(output / 'classes.csv', classes)
    keys = ('mAP50_95', 'operating_recall', 'macro_recall', 'precision', 'tp', 'fp', 'fn', 'fp_per_image')
    summary = [{'model': label, **{k: m[k] for k in keys}, 'eligible': eligible(m, baseline)}
               for label, m in metrics.items()]
    write_csv(output / 'summary.csv', summary)
    curves = []
    history = {}
    for arm in ('native', 'dpo'):
        history[arm] = [r for r in events(old_root / arm / 'metrics.jsonl') if r['event'] == 'validation']
        for bg, rows in ((.5, validations[arm]), (1., history[arm])):
            for row in rows:
                curves.append({'arm': arm, 'background_weight': bg, 'step': row['step'],
                               'eligible': row['eligible'], **{k: row['metrics'][k] for k in keys}})
    write_csv(output / 'validation.csv', curves)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title, scale) in zip(axes.flat, [
            ('operating_recall', 'Recall @ 0.25 (%)', 100), ('fp', 'FP @ 0.25', 1),
            ('mAP50_95', 'mAP50-95 (%)', 100), ('macro_recall', 'Macro recall @ 0.25 (%)', 100)]):
        ax.axhline(baseline[key]*scale, color='black', linestyle=':', label='S0')
        for bg, collection, color in ((.5, validations, '#c65719'), (1., history, '#2878ad')):
            for arm, rows in collection.items():
                ax.plot([r['step'] for r in rows], [r['metrics'][key]*scale for r in rows],
                        color=color, linestyle='-' if arm == 'dpo' else '--',
                        label=f'background {bg} / {arm}', linewidth=1.3)
        ax.axvline(2000, color='#999999', linestyle=':', linewidth=.8)
        ax.set_title(title)
        ax.set_xlabel('Updates per arm')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Native background 0.5 vs 1.0 | 15k | seed 42 | 479 validation images')
    fig.tight_layout()
    fig.savefig(output / 'learning_curves.svg')
    fig.savefig(output / 'learning_curves.png', dpi=160)
    plt.close(fig)
    differences = {name: sum(r['dpo'] != r['native'] for r in rows)
                   for name, rows in [('primary_GT_id_differences', ledger), ('matched_FP_GT_id_differences', matched_ledger)]}
    artifact = {'root': str(root), 'plan': plan, 'checks': checks, 'summary': summary,
                'diagnostics': diagnostics, 'exact_FP_budget': exact,
                'previous_background1_long_exact_FP_budget': previous_review['exact_FP_budget'],
                'old_checkpoints_unchanged': True, 'stream_sha256': completed['arms']['dpo']['stream_sha256'],
                'source_gt_transitions': {k: v['source_gt_transitions'] for k, v in metrics.items()},
                'GT_id_comparison': differences, 'review_script_sha256': sha256(Path(__file__)),
                'scope': 'Full validation, no test or training; matched-FP sweep is exploratory'}
    atomic_json(output / 'review.json', artifact)
    print(json.dumps({'event': 'review_completed', 'summary': summary, 'exact_FP_budget': exact,
                      'GT_id_comparison': differences}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    review(args.root, args.output, args.device)
