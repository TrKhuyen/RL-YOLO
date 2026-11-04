"""Fresh ordinary validation and FP audit of the prespecified background pilot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml

from kb2_preference_optimization.artifacts import (
    atomic_json, sha256, source_checkpoint, tensor_digest,
)
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
    assert completed['matched_stream']
    assert plan['steps'] == 2000 and plan['background_weight'] == .5
    assert code_fingerprints() == plan['code_sha256']
    output.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    valid = detection_loader('valid', 640, 8)
    targets = [t for _, batch_targets in valid for t in batch_targets]
    adapter = ConditionalAdapter(source_checkpoint('yolov8n'), device)
    source_digest = tensor_digest(adapter.model)
    frozen = adapter.frozen_digest()
    source_hash = sha256(source_checkpoint('yolov8n'))
    assert source_hash == plan['source_sha256']
    baseline, predictions = evaluate(adapter, valid)
    assert baseline['num_images'] == 479 and baseline['num_gt'] == 1887
    atomic_json(output / 's0_validation.json', baseline)
    prediction_sets, metrics = {'s0': predictions}, {'s0': baseline}
    verification, validations, checkpoints, selected_metrics = {}, {}, {}, {}
    streams = []
    for label, result in completed['arms'].items():
        folder = root / label
        streams.append((folder / 'stream.jsonl').read_bytes())
        assert len(streams[-1].splitlines()) == plan['steps']
        assert sha256(folder / 'last.pt') == result['last_sha256']
        assert sha256(folder / 'best.pt') == result['best_sha256']
        rows = [r for r in events(folder / 'metrics.jsonl') if r['event'] == 'validation']
        validations[label] = rows
        assert [r['step'] for r in rows] == [0, 1000, 2000]
        assert equal_metrics(baseline, rows[0]['metrics'])
        checkpoint = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        assert checkpoint['step'] == plan['steps']
        adapter.model.load_state_dict(checkpoint['state_dict'], strict=True)
        assert adapter.frozen_digest() == frozen
        current, predictions = evaluate(adapter, valid, prediction_sets['s0'])
        assert equal_metrics(current, result['last_metrics'])
        assert equal_metrics(current, rows[-1]['metrics'])
        metrics[label], prediction_sets[label] = current, predictions
        atomic_json(output / f'{label}_validation.json', current)
        best = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
        assert best['step'] == result['best_step']
        adapter.model.load_state_dict(best['state_dict'], strict=True)
        assert adapter.frozen_digest() == frozen
        chosen = next(r for r in rows if r['step'] == best['step'])
        assert chosen['eligible'] and chosen['selected_best']
        if best['step'] == 0:
            assert tensor_digest(adapter.model) == source_digest
            selected_metrics[label] = baseline
        elif best['step'] == checkpoint['step']:
            assert all(torch.equal(best['state_dict'][name], value)
                       for name, value in checkpoint['state_dict'].items())
            selected_metrics[label] = current
        else:
            selected_metrics[label], _ = evaluate(adapter, valid, prediction_sets['s0'])
            assert equal_metrics(selected_metrics[label], chosen['metrics'])
        atomic_json(output / f'{label}_best_validation.json', selected_metrics[label])
        verification[label] = {'best_step': best['step'], 'last_step': checkpoint['step'],
                               'full_metrics_reproduced': True, 'frozen_state_unchanged': True,
                               'reference_unchanged': result['reference_unchanged'],
                               'best_sha256': result['best_sha256'], 'last_sha256': result['last_sha256']}
        checkpoints[label] = {'last': str(folder / 'last.pt'), 'best': str(folder / 'best.pt')}
        print(json.dumps({'event': 'fresh_validation', 'label': label,
                          **{k: current[k] for k in ('mAP50_95', 'operating_recall', 'fp', 'fn')}}), flush=True)
        del checkpoint, best
    assert len(set(streams)) == 1
    assert sha256(source_checkpoint('yolov8n')) == source_hash
    # Historical controls must replay the original pilot, excluding wall time.
    old = root.parent / 'yolov8n_conditional_s42_20261030_180851_313339'
    historical = {}
    for arm in ('native', 'dpo'):
        previous = [r for r in events(old / arm / 'metrics.jsonl') if r['event'] == 'validation']
        current = validations['original_' + arm]
        assert (old / arm / 'stream.jsonl').read_bytes() == streams[0]
        assert all(equal_metrics(a['metrics'], b['metrics']) and a['probes'] == b['probes']
                   for a, b in zip(previous, current))
        historical[arm] = 'All 0/1000/2000 metrics/probes and stream bytes exactly reproduced'
    keys = ('mAP50_95', 'mAP50', 'mAP75', 'operating_recall', 'macro_recall', 'precision',
            'f1', 'tp', 'fp', 'fn', 'fp_per_image', 'num_images', 'num_gt')
    summary = []
    for label, value in metrics.items():
        summary.append({'model': label, **{k: value[k] for k in keys},
                        'eligible': eligible(value, baseline),
                        'best_step': 0 if label == 's0' else verification[label]['best_step']})
    write_csv(output / 'summary.csv', summary)
    class_rows, image_rows, ledger = [], [], []
    names = yaml.safe_load(Path('kb1_reward_guided_training/configs/pest.yaml').read_text())['names']
    for c, base in baseline['per_class'].items():
        row = {'class_id': int(c), 'class_name': names[int(c)], 'gt_support': base['gt_support']}
        for label, value in metrics.items():
            for k in ('tp', 'fp', 'fn', 'AP50_95'):
                row[f'{label}_{k}'] = value['per_class'][c][k]
        class_rows.append(row)
    for i, target in enumerate(targets):
        row = {'image_id': target['image_id'], 'image_path': target['image_path'],
               'gt': len(target['labels']), 'density': '6+' if len(target['labels']) >= 6 else
               ('3-5' if len(target['labels']) >= 3 else '1-2')}
        found_sets = {}
        for label, ps in prediction_sets.items():
            found, fp, _ = matches(ps[i], target, .25)
            found_sets[label] = found
            row.update({f'{label}_tp': len(found), f'{label}_fp': fp,
                        f'{label}_fn': len(target['labels'])-len(found)})
        image_rows.append(row)
        for g, c in enumerate(target['labels'].tolist()):
            ledger.append({'image_id': target['image_id'], 'gt_id': g, 'class_id': c,
                           **{label: g in found for label, found in found_sets.items()}})
    for label, value in metrics.items():
        for key in ('tp', 'fp', 'fn'):
            assert sum(row[f'{label}_{key}'] for row in image_rows) == value[key]
        assert sum(int(row[label]) for row in ledger) == value['tp']
    write_csv(output / 'classes.csv', class_rows)
    write_csv(output / 'images.csv', image_rows)
    write_csv(output / 'gt_ledger.csv', ledger)
    exact = {label: exact_fp_point(ps, targets, baseline['fp']) for label, ps in prediction_sets.items()}
    atomic_json(output / 'exact_fp_budget.json', exact)
    write_csv(output / 'exact_fp_budget.csv', [{'model': label, **value} for label, value in exact.items()])
    curves = []
    for label, rows in validations.items():
        for r in rows:
            curves.append({'model': label, 'step': r['step'], 'eligible': r['eligible'],
                           **{k: r['metrics'][k] for k in keys}})
    write_csv(output / 'validation.csv', curves)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title, factor) in zip(axes.flat, [
            ('operating_recall', 'Recall @ 0.25 (%)', 100), ('fp', 'FP @ 0.25', 1),
            ('mAP50_95', 'mAP50-95 (%)', 100), ('macro_recall', 'Macro recall @ 0.25 (%)', 100)]):
        ax.axhline(baseline[key]*factor, color='black', linestyle=':', label='S0')
        for label, rows in validations.items():
            ax.plot([r['step'] for r in rows], [r['metrics'][key]*factor for r in rows],
                    color='#2878ad' if label.startswith('original') else '#c65719',
                    linestyle='--' if label.endswith('native') else '-', marker='o', label=label)
        ax.set_title(title)
        ax.set_xlabel('Updates')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Native background 1.0 vs 0.5 | seed 42 | 479 validation images')
    fig.tight_layout()
    fig.savefig(output / 'learning_curves.svg')
    fig.savefig(output / 'learning_curves.png', dpi=160)
    plt.close(fig)
    artifact = {'run_root': str(root), 'plan': plan, 'verification': verification,
                'historical_control_reproduction': historical,
                'stream_sha256': completed['arms']['original_dpo']['stream_sha256'],
                'matched_stream': True, 'source_checkpoint_unchanged': True,
                'summary': summary, 'selected_metrics': selected_metrics,
                'exact_fp_budget': exact, 'checkpoints': checkpoints,
                'source_transitions': {label: m['source_gt_transitions'] for label, m in metrics.items()},
                'interpretation': 'Single-seed validation pilot; threshold .25 locked; no test or data changes'}
    atomic_json(output / 'review.json', artifact)
    print(json.dumps({'event': 'review_completed', 'output': str(output), 'summary': summary}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    review(args.root, args.output, args.device)
