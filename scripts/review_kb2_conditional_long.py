"""Verify long V0 runs, reload checkpoints, and compare ordinary validation."""
import argparse
from collections import Counter
import copy
import itertools
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import yaml

from kb2_preference_optimization.artifacts import (
    assert_provenance, atomic_json, sha256, source_checkpoint, source_provenance, tensor_digest,
)
from kb2_preference_optimization.conditional import VERSION, code_fingerprints
from kb2_preference_optimization.conditional.bank import Bank, sampling_probabilities
from kb2_preference_optimization.conditional.evaluation import evaluate
from kb2_preference_optimization.conditional.pairs import matches
from kb2_preference_optimization.conditional.policy import ConditionalAdapter, configure_numerics
from kb2_preference_optimization.data import detection_loader
from scripts.audit_kb2_conditional_gradients import read, write_csv


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def equal_metrics(first, second):
    return {k: v for k, v in first.items() if k != 'evaluation_wall_seconds'} == {
        k: v for k, v in second.items() if k != 'evaluation_wall_seconds'}


def exact_fp_point(predictions, targets, budget):
    """Discrete global threshold, inclusive ties, canonical greedy matching.

    After fixed NMS/cap, lower-score predictions cannot change earlier greedy
    assignments. Sweep whole tied-score groups, then verify at the chosen cutoff.
    """
    detections = []
    for prediction, target in zip(predictions, targets):
        _, _, assignment = matches(prediction, target, .001)
        detections.extend((float(score), int(i in assignment))
                          for i, score in enumerate(prediction['scores']) if float(score) >= .001)
    detections.sort(key=lambda entry: entry[0], reverse=True)
    tp = fp = 0
    chosen = {'confidence': 1.000001, 'tp': 0, 'fp': 0}
    for score, group in itertools.groupby(detections, key=lambda entry: entry[0]):
        flags = [entry[1] for entry in group]
        tp += sum(flags)
        fp += len(flags)-sum(flags)
        if fp > budget:
            break
        chosen = {'confidence': score, 'tp': tp, 'fp': fp}
    actual_tp = actual_fp = 0
    for prediction, target in zip(predictions, targets):
        found, false_positives, _ = matches(prediction, target, chosen['confidence'])
        actual_tp += len(found)
        actual_fp += false_positives
    if (chosen['tp'], chosen['fp']) != (actual_tp, actual_fp):
        raise ValueError('Exact threshold sweep disagrees with canonical matching')
    support = sum(len(t['labels']) for t in targets)
    return {**chosen, 'fn': support-actual_tp, 'recall': actual_tp/support,
            'precision': actual_tp/max(actual_tp+actual_fp, 1), 'FP_budget': budget,
            'scope': 'Exploratory validation threshold; inclusive score ties; no interpolation or test tuning'}


def figure(runs, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    specs = [('operating_recall', 'Recall @ confidence 0.25 (%)', 100),
             ('fn', 'False negatives @ confidence 0.25', 1),
             ('mAP50_95', 'mAP50-95 (%)', 100),
             ('fp', 'False positives @ confidence 0.25', 1)]
    for ax, (key, title, factor) in zip(axes.flat, specs):
        baseline = runs['dpo']['validations'][0]['metrics'][key]*factor
        ax.axhline(baseline, color='black', linestyle='--', linewidth=1, label='S0')
        for arm, color, style, marker in (('native', '#2467a8', '--', 'o'), ('dpo', '#d95c17', '-', '.')):
            vals = runs[arm]['validations']
            ax.plot([v['step'] for v in vals], [v['metrics'][key]*factor for v in vals],
                    color=color, linestyle=style, marker=marker, markersize=4,
                    linewidth=1.4, label='Native only' if arm == 'native' else 'Native + DPO')
        ax.axvline(2000, color='#888888', linestyle=':', linewidth=1)
        ax.set_title(title)
        ax.set_xlabel('Updates per arm')
        ax.grid(alpha=.22)
        ax.legend(fontsize=8)
    fig.suptitle('Conditional V0 | seed 42 | 479 validation images | matched input streams')
    fig.tight_layout()
    fig.savefig(output/'learning_curves.svg')
    fig.savefig(output/'learning_curves.png', dpi=180)
    plt.close(fig)


def review(root, output, device):
    configure_numerics()
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    completed = read(root/'completed.json')
    if completed['objective_version'] != VERSION or not completed['matched_stream']:
        raise ValueError('Completed matched V0 run required')
    if completed['partial_bank'] or completed['partial_evaluation']:
        raise ValueError('Full bank/validation required')
    manifest = read(root/'dpo/manifest.json')
    bank = Bank(manifest['bank_path'])
    provenance = source_provenance('yolov8n')
    assert_provenance(manifest['provenance'], provenance)
    if sha256(bank.root/'manifest.json') != manifest['bank_manifest_sha256']:
        raise ValueError('Bank changed')
    config = manifest['config']
    if config['steps'] != 15000 or config['eval_interval'] != 1000 or config['eval_limit']:
        raise ValueError('Unexpected long-run budget')
    native_bytes = (root/'native/stream.jsonl').read_bytes()
    if native_bytes != (root/'dpo/stream.jsonl').read_bytes():
        raise ValueError('Native/DPO streams differ')
    stream = events(root/'dpo/stream.jsonl')
    if len(stream) != config['steps']:
        raise ValueError('Incomplete input stream')
    records = [r for r in bank.records if not r['probe']]
    weights, _ = sampling_probabilities(records)
    plan = torch.multinomial(weights, 2*config['steps'], replacement=True,
                             generator=torch.Generator().manual_seed(config['seed']+10000)).reshape(-1, 2)
    native_loader = DataLoader(list(range(len(bank.dataset))), batch_size=2, shuffle=True,
                              drop_last=True, generator=torch.Generator().manual_seed(config['seed']+20000))
    iterator = iter(native_loader)
    import hashlib
    digest = hashlib.sha256()
    exposure = {k: Counter() for k in ('group', 'class', 'image', 'pair', 'target')}
    for step, entry in enumerate(stream, 1):
        try:
            uniform = next(iterator).tolist()
        except StopIteration:
            iterator = iter(native_loader)
            uniform = next(iterator).tolist()
        selected = [records[i] for i in plan[step-1].tolist()]
        if (entry['step'] != step or entry['pair_ids'] != [r['pair_id'] for r in selected]
                or entry['native_image_ids'] != uniform+[r['image_id'] for r in selected]):
            raise ValueError('Input stream does not replay')
        digest.update(json.dumps({k: v for k, v in entry.items() if k != 'step'}, sort_keys=True).encode())
        for r in selected:
            for key, value in [('group', r['group']), ('class', r['class_id']), ('image', r['image_id']),
                               ('pair', r['pair_id']), ('target', f"{r['image_id']}:{r['target_gt'] if r['target_gt'] is not None else 'slot'+str(r['slot'])}")]:
                exposure[key][str(value)] += 1
    pilot_root = root.parent/'yolov8n_conditional_s42_20261030_180851_313339'
    if native_bytes.splitlines()[:2000] != (pilot_root/'native/stream.jsonl').read_bytes().splitlines():
        raise ValueError('Pilot stream prefix differs')
    source = ConditionalAdapter(source_checkpoint('yolov8n'), device)
    initial = {k: v.detach().cpu().clone() for k, v in source.model.state_dict().items()}
    if tensor_digest(source.model) != bank.manifest['reference_state_sha256']:
        raise ValueError('S0/reference differs from bank')
    runs, hashes = {}, {}
    for arm in ('native', 'dpo'):
        folder = root/arm
        done, saved = read(folder/'completed.json'), read(folder/'manifest.json')
        assert_provenance(saved['provenance'], provenance)
        if saved['conditional_code_sha256'] != code_fingerprints() or done['steps'] != 15000:
            raise ValueError('Code/budget mismatch')
        if done['stream_sha256'] != digest.hexdigest() or done['actual_exposure'] != {k: dict(v) for k, v in exposure.items()}:
            raise ValueError('Exposure mismatch')
        if not done['reference_unchanged'] or not done['frozen_state_unchanged']:
            raise ValueError('Immutable state check failed during training')
        for stem in ('best', 'last'):
            path = folder/f'{stem}.pt'
            hashes[str(path)] = sha256(path)
            if hashes[str(path)] != done[f'{stem}_sha256']:
                raise ValueError('Checkpoint hash mismatch')
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if payload['manifest'] != saved or payload['objective_version'] != VERSION:
                raise ValueError('Checkpoint metadata mismatch')
            if stem == 'best' and (payload['step'] != 0 or not all(torch.equal(v, initial[k]) for k, v in payload['state_dict'].items())):
                raise ValueError('Best is expected to be exactly S0')
        all_events = events(folder/'metrics.jsonl')
        validations = [e for e in all_events if e['event'] == 'validation']
        if [v['step'] for v in validations] != list(range(0, 15001, 1000)):
            raise ValueError('Validation points missing')
        for old in [v for v in events(pilot_root/arm/'metrics.jsonl') if v['event'] == 'validation']:
            new = next(v for v in validations if v['step'] == old['step'])
            if not equal_metrics(old['metrics'], new['metrics']) or old['probes'] != new['probes']:
                raise ValueError('Pilot validation prefix differs')
        gradient_rows = [e['gradient'] for e in all_events if e.get('gradient')]
        runs[arm] = {'manifest': saved, 'completed': done, 'validations': validations,
                     'gradient_logged_batches': len(gradient_rows),
                     'mean_weighted_DPO_native_norm_ratio': statistics.mean(e['weighted_dpo_native_norm_ratio'] for e in gradient_rows)}
    valid = detection_loader('valid', 640, 8)
    baseline, source_predictions = evaluate(source, valid)
    if not equal_metrics(baseline, runs['dpo']['validations'][0]['metrics']):
        raise ValueError('Reloaded S0 validation differs from log')
    print('Reloaded S0: full validation agrees with log', flush=True)
    fresh, predictions = {'S0': baseline}, {'S0': source_predictions}
    for arm in ('native', 'dpo'):
        adapter = ConditionalAdapter(source_checkpoint('yolov8n'), device)
        payload = torch.load(root/arm/'last.pt', map_location='cpu', weights_only=False)
        adapter.model.load_state_dict(payload['state_dict'], strict=True)
        if adapter.frozen_digest() != runs[arm]['manifest']['frozen_state_sha256']:
            raise ValueError('Frozen state differs from original')
        fresh[arm], predictions[arm] = evaluate(adapter, valid, source_predictions)
        if not equal_metrics(fresh[arm], runs[arm]['validations'][-1]['metrics']):
            raise ValueError(f'Reloaded {arm} validation differs from log')
        print(f'Reloaded {arm} last: full validation agrees with log', flush=True)
    targets = [target for _, batch_targets in valid for target in batch_targets]
    budget = baseline['fp']
    exact = {arm: exact_fp_point(values, targets, budget) for arm, values in predictions.items()}
    atomic_json(output/'exact_FP_budget.json', exact)
    ledger = []
    for i, target in enumerate(targets):
        found = {arm: matches(values[i], target, .25)[0] for arm, values in predictions.items()}
        for g, c in enumerate(target['labels'].tolist()):
            ledger.append({'image_id': target['image_id'], 'gt_row': g, 'class_id': c,
                           **{arm+'_detected': g in values for arm, values in found.items()}})
    write_csv(output/'gt_transitions.csv', ledger)
    names = yaml.safe_load(Path('kb1_reward_guided_training/configs/pest.yaml').read_text())['names']
    class_rows = []
    for c, name in enumerate(names):
        row = {'class_id': c, 'class_name': name, 'support': baseline['per_class'][str(c)]['gt_support']}
        for arm, metrics in fresh.items():
            row.update({arm+'_'+key: metrics['per_class'][str(c)][key] for key in ('tp', 'fp', 'fn', 'AP50_95')})
        class_rows.append(row)
    write_csv(output/'classes.csv', class_rows)
    validation_rows, probes = [], []
    for arm, result in runs.items():
        for v in result['validations']:
            metrics = v['metrics']
            validation_rows.append({'arm': arm, 'step': v['step'], 'eligible': v['eligible'],
                                    **{k: metrics[k] for k in ('mAP50_95', 'precision', 'operating_recall', 'macro_recall', 'fp', 'fn')}})
            for cohort, value in v['probes'].items():
                for group, values in value['groups'].items():
                    probes.append({'arm': arm, 'step': v['step'], 'cohort': cohort, 'group': group, **values})
    write_csv(output/'validation.csv', validation_rows)
    write_csv(output/'probes.csv', probes)
    figure(runs, output)
    for name, expected in hashes.items():
        if sha256(name) != expected:
            raise ValueError('Original checkpoint changed during review')
    result = {'objective_version': VERSION, 'run': str(root), 'scope': 'Matched seed42, 15000 updates/arm; full validation, no test',
              'runs': runs, 'fresh_validation': fresh, 'exact_FP_budget': exact, 'classes': class_rows,
              'stream_sha256': digest.hexdigest(), 'checkpoint_sha256': hashes,
              'DPO_vs_native_GT_differences': {
                  'recovered': sum(r['dpo_detected'] and not r['native_detected'] for r in ledger),
                  'lost': sum(r['native_detected'] and not r['dpo_detected'] for r in ledger)},
              'checks': {'checkpoints_verified_unchanged': True, 'best_exactly_S0': True,
                         'full_stream_and_exposure_reproduced': True, 'pilot_prefix_exact': True,
                         'fresh_GPU_validation_matches_all_three_logs': True, 'frozen_state_verified': True},
              'review_script_sha256': sha256(Path(__file__))}
    atomic_json(output/'review.json', result)
    print(json.dumps({'event': 'review_complete', 'output': str(output), 'checks': result['checks'],
                      'exact_FP_budget': exact, 'DPO_vs_native_GT_differences': result['DPO_vs_native_GT_differences']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    review(args.run, args.output, args.device)
