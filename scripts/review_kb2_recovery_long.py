"""Read-only validation review of a completed KB2 recovery long run."""
import argparse
from collections import Counter
import contextlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from kb1_reward_guided_training.canonical_eval import _operating_point_counts, _summarize_counts
from kb2_preference_optimization.adapter import DFLAdapter
from kb2_preference_optimization.artifacts import assert_provenance, atomic_json, sha256, source_checkpoint, source_provenance
from kb2_preference_optimization.data import detection_loader
from kb2_preference_optimization.evaluate import evaluate, parse_args
from kb2_preference_optimization.recovery import code_fingerprints
from scripts.audit_kb2_class_fn import operating_matches, write_csv


def read(path):
    return json.loads(path.read_text())


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def validation_metrics_equal(a, b):
    # Timing is deliberately excluded; every numerical result must agree.
    return {k: v for k, v in a.items() if k != 'evaluation_wall_seconds'} == {
        k: v for k, v in b.items() if k != 'evaluation_wall_seconds'}


def compact(name, metrics, step=None):
    return {'model': name, 'step': step, **{k: metrics[k] for k in (
        'mAP50_95', 'mAP50', 'mAP75', 'AR300', 'precision', 'operating_recall', 'f1', 'tp', 'fp', 'fn')},
        'macro_recall': statistics.mean(v['tp']/v['gt_support'] for v in metrics['per_class'].values()
                                        if v['gt_support'])}


def review(root, output, device):
    if not torch.cuda.is_available() and device.startswith('cuda'):
        raise RuntimeError('CUDA is required for this review')
    output.mkdir(parents=True, exist_ok=False)
    folder = root/'pilot_balanced'
    done, manifest = read(folder/'completed.json'), read(folder/'manifest.json')
    long_run, audit = read(root/'long_run.json'), read(root/'audit.json')
    assert done['status'] == 'complete' and done['steps'] == long_run['steps'] == 15000
    assert done['reference_immutable'] and long_run['device'] == 'cuda'
    provenance = source_provenance('yolov8n')
    for saved in (manifest, audit, read(root/'selected/manifest.json')):
        assert_provenance(saved['provenance'], provenance)
        assert saved['recovery_code_sha256'] == code_fingerprints()
    assert sha256(root/'all_candidates.jsonl') == audit['all_candidates_sha256']
    assert sha256(root/'selected/pairs.jsonl') == manifest['config']['pairs_sha256']
    assert sha256(folder/'manifest.json') == done['manifest_sha256']
    for stem in ('best', 'last'):
        assert sha256(folder/f'{stem}.pt') == done[f'{stem}_sha256']
    rows = events(folder/'metrics.jsonl')
    validations = [r for r in rows if r['event'] == 'validation']
    assert [r['step'] for r in validations] == list(range(0, 15001, 1000))
    by_step = {r['step']: r for r in validations}
    pilot_root = ROOT/'kb2_preference_optimization/outputs/yolov8n_recovery_pilot_20261029'
    pilot = [r for r in events(pilot_root/'pilot_balanced/metrics.jsonl') if r['event'] == 'validation']
    for r in pilot:
        assert validation_metrics_equal(r['metrics'], by_step[r['step']]['metrics'])
        assert r['probes'] == by_step[r['step']]['probes']
    assert sha256(pilot_root/'selected/pairs.jsonl') == sha256(root/'selected/pairs.jsonl')
    reports = {}
    for stem in ('best', 'last'):
        with (output/f'{stem}_evaluation.log').open('w') as stream, contextlib.redirect_stdout(stream):
            reports[stem] = evaluate(parse_args([
                '--checkpoint', str(folder/f'{stem}.pt'), '--device', device,
                '--split', 'valid', '--output', str(output/f'{stem}_evaluation.json')]))
        report = reports[stem]
        assert validation_metrics_equal(report['supervised'], by_step[0]['metrics'])
        assert validation_metrics_equal(report['dpo'], by_step[report['selected_step']]['metrics'])
        print(f'Fresh GPU {stem} validation agrees with training log', flush=True)

    # Cache ordinary predictions once per checkpoint for paired GT transitions and
    # an exploratory threshold sweep; the canonical .25 result remains primary.
    models = {'S0': None, 'recovery_best': folder/'best.pt', 'recovery_last': folder/'last.pt'}
    matches, target_labels, sweep = {}, {}, []
    thresholds = (.05, .10, .15, .20, .25, .30, .50)
    for name, checkpoint in models.items():
        adapter = DFLAdapter(source_checkpoint('yolov8n'), device)
        if checkpoint:
            adapter.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict'])
        counts = {conf: Counter() for conf in thresholds}
        matches[name] = {}
        with torch.no_grad():
            for images, targets in detection_loader('valid', 640, 8):
                preds = adapter.predict(images.to(device), .001, .60, 300)
                for pred, target in zip(preds, targets):
                    image_id = target['image_id']
                    target_labels[image_id] = target['labels'].tolist()
                    matched, _ = operating_matches(pred, target)
                    matches[name][image_id] = set(matched)
                for conf in thresholds:
                    batch_counts = _operating_point_counts(preds, targets, conf, .5)
                    for c, values in batch_counts.items():
                        for k, value in enumerate(values):
                            counts[conf][c, k] += value
        for conf in thresholds:
            per_class = {c: [counts[conf][c, k] for k in range(3)] for c in range(28)}
            precision, recall, f1, _ = _summarize_counts(per_class)
            sweep.append({'model': name, 'conf': conf, 'precision': precision, 'recall': recall, 'f1': f1,
                          **{key: sum(v[k] for v in per_class.values()) for k, key in enumerate(('tp', 'fp', 'fn'))}})
        expected = by_step[0]['metrics'] if name == 'S0' else reports[name.removeprefix('recovery_')]['dpo']
        actual = next(r for r in sweep if r['model'] == name and r['conf'] == .25)
        assert all(actual[k] == expected[k] for k in ('tp', 'fp', 'fn'))
        print(f'GT identities / threshold sweep complete: {name}', flush=True)

    transitions, transition_records = {}, []
    for name in ('recovery_best', 'recovery_last'):
        counts = Counter()
        for image_id, labels in target_labels.items():
            for g, c in enumerate(labels):
                old, new = g in matches['S0'][image_id], g in matches[name][image_id]
                kind = ('retained' if new else 'lost') if old else ('recovered' if new else 'persistent_miss')
                counts[c, kind] += 1
                transition_records.append({'model': name, 'image_id': image_id, 'gt_index': g,
                                           'class_id': c, 'transition': kind})
        transitions[name] = {kind: sum(counts[c, kind] for c in range(28)) for kind in (
            'retained', 'lost', 'recovered', 'persistent_miss')}
        transitions[name]['per_class'] = {
            str(c): {kind: counts[c, kind] for kind in ('retained', 'lost', 'recovered', 'persistent_miss')}
            for c in range(28)}
        metrics = reports[name.removeprefix('recovery_')]['dpo']
        assert metrics['fn'] - by_step[0]['metrics']['fn'] == transitions[name]['lost'] - transitions[name]['recovered']
    with (output/'gt_transitions.jsonl').open('w') as stream:
        for r in transition_records:
            stream.write(json.dumps(r)+'\n')

    names = yaml.safe_load((ROOT/'kb1_reward_guided_training/configs/pest.yaml').read_text())['names']
    f2_folder = ROOT/'kb2_preference_optimization/outputs/yolov8n_f2_20261028/s42'
    f2_done = read(f2_folder/'completed.json')
    f2 = read(f2_folder/'validation_comparison.json')
    assert sha256(f2_folder/'best.pt') == f2_done['best_sha256'] == f2['checkpoint_sha256']
    assert sha256(f2_folder/'last.pt') == f2_done['last_sha256']
    assert sha256(f2_folder/'manifest.json') == f2_done['manifest_sha256']
    assert_provenance(f2['provenance'], provenance)
    summary = [compact('S0', reports['best']['supervised'], 0),
               compact('F2_s42_best', f2['dpo'], f2['selected_step']),
               compact('recovery_best', reports['best']['dpo'], done['best_step']),
               compact('recovery_last', reports['last']['dpo'], done['steps'])]
    exposure = read(folder/'exposure.json')
    assert exposure['native_sequence_sha256'] == done['native_sequence_sha256']
    classes = []
    for c, name in enumerate(names):
        values = {model: metric['per_class'][str(c)] for model, metric in (
            ('S0', reports['best']['supervised']), ('F2', f2['dpo']),
            ('best', reports['best']['dpo']), ('last', reports['last']['dpo']))}
        row = {'class_id': c, 'class_name': name, 'gt': values['S0']['gt_support'],
               'recovery_class_draws': exposure['classes'].get(str(c), 0)}
        for model, metric in values.items():
            row.update({f'{model}_{k}': metric[k] for k in ('tp', 'fp', 'fn', 'AP50_95', 'AR300')})
            row[f'{model}_recall'] = metric['tp']/metric['gt_support']
        row['best_FN_delta_S0'] = values['best']['fn'] - values['S0']['fn']
        row['best_FN_delta_F2'] = values['best']['fn'] - values['F2']['fn']
        row.update(transitions['recovery_best']['per_class'][str(c)])
        classes.append(row)
    trajectory = [compact('recovery', r['metrics'], r['step']) for r in validations]
    f2_by_step = {r['step']: r for r in events(f2_folder/'metrics.jsonl') if r['event'] == 'validation'}
    fixed_steps = []
    for step in sorted(f2_by_step.keys() & by_step.keys()):
        old, new = f2_by_step[step]['metrics'], by_step[step]['metrics']
        fixed_steps.append({'step': step, 'F2_AP': old['mAP50_95'], 'recovery_AP': new['mAP50_95'],
                            'F2_FN': old['fn'], 'recovery_FN': new['fn'],
                            'F2_FP': old['fp'], 'recovery_FP': new['fp']})
    probe_rows = []
    for r in validations:
        for cohort, values in r['probes'].items():
            probe_rows.append({'step': r['step'], 'cohort': cohort, **{k: values[k] for k in (
                'pairs', 'relative_margin', 'dpo_loss', 'relative_win_rate', 'rescued_gt',
                'rescued_detected', 'rescued_recall', 'rescued_mean_true_class_score')}})
    train = [r for r in rows if r['event'] == 'train']
    diagnostics = {
        'logged_updates': len(train),
        'mean_weighted_dpo_native_norm_ratio': statistics.mean(r['weighted_dpo_native_norm_ratio'] for r in train),
        'fraction_negative_gradient_cosine': statistics.mean(r['gradient_cosine'] < 0 for r in train),
        'recovery_draw_fraction': exposure['groups']['recovery']/sum(exposure['groups'].values()),
        'first_update_margin_max_abs': train[0]['relative_margin_max_abs'],
        'last_logged_train': train[-1],
    }
    checks = {'completed_15000_cuda_updates': True, 'reference_immutable_reported': done['reference_immutable'],
              'source_data_training_code_pool_pairs_checkpoint_manifest_hashes_verified': True,
              'fresh_best_last_gpu_validation_all_metrics_equal_logs': True,
              'first_2000_steps_reproduce_prior_balanced_pilot': True,
              'old_S0_and_F2_s42_checkpoint_hashes_unchanged': True,
              'identity_matcher_agrees_with_canonical_counts': True,
              'test_set_not_evaluated': True}
    result = {'scope': 'valid only, seed42; historical F2 comparison is not an isolated ablation',
              'run': str(root), 'completed': done, 'manifest': manifest, 'summary': summary,
              'classes': classes, 'transitions': transitions, 'diagnostics': diagnostics,
              'probes': probe_rows, 'trajectory': trajectory, 'fixed_steps': fixed_steps,
              'threshold_sweep_exploratory': sweep, 'checks': checks}
    for filename, data in (('summary.csv', summary), ('classes.csv', classes), ('trajectory.csv', trajectory),
                           ('fixed_steps.csv', fixed_steps), ('probes.csv', probe_rows), ('thresholds.csv', sweep)):
        write_csv(output/filename, data)
    atomic_json(output/'review.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for field, ax, title, scale in zip(('mAP50_95', 'operating_recall', 'fn'), axes,
                                      ('Validation mAP50-95 (%)', 'Recall at conf=.25 (%)', 'Missed GT (FN)'), (100, 100, 1)):
        ax.plot([r['step'] for r in trajectory], [scale*r[field] for r in trajectory], label='Recovery s42')
        ax.axhline(scale*trajectory[0][field], color='gray', linestyle='--', label='S0')
        ax.set(title=title, xlabel='Updates')
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output/'validation.svg')
    plt.close(fig)
    print(json.dumps({'summary': summary, 'checks': checks, 'diagnostics': diagnostics,
                      'transitions': {k: {f:v for f,v in r.items() if f!='per_class'} for k,r in transitions.items()}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    review(args.root.resolve(), args.output.resolve(), args.device)
