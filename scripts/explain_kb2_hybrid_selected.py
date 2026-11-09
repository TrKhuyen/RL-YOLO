"""Freeze the selected hybrid result and derive auditable mechanism evidence.

Read-only with respect to models/data/training code. No new training or eval.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import torch
import yaml

from kb2_preference_optimization.artifacts import REPO, atomic_json, sha256


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh analysis directory')
    run = REPO/'kb2_preference_optimization/outputs/yolov8n_hybrid_schedule_s42_20261106_trial3/run/dpo'
    source = REPO/'docs/report_final/assets/kb2_hybrid_schedule_20261106'
    verified = read(source/'review.json')
    assert verified['matched_stream'] and verified['matched_other_hyperparameters']
    assert verified['first_1000_weights_optimizer_match'] and verified['protected_files_unchanged']
    done, manifest = read(run/'completed.json'), read(run/'manifest.json')
    path = run/'best.pt'
    digest = sha256(path)
    assert digest == done['best_sha256'] and done['best_step'] == 500
    state = torch.load(path, weights_only=False, map_location='cpu')
    assert state['step'] == 500 and state['manifest'] == manifest
    assert state['optimizer']['param_groups'][0]['lr'] == 1e-6
    assert manifest['config']['dpo_weight'] == .03 and manifest['config']['native_background_weight'] == .5
    assert done['frozen_state_unchanged'] and done['reference_unchanged']
    metrics = {label: read(source/f'{filename}.json') for label, filename in
               [('S0', 'S0'), ('TPE', 'TPE'), ('selected', 'scheduled_lambda003_best'),
                ('lambda003_1k_grid', 'constant_lambda003_best'),
                ('lambda010_1k_grid', 'constant_lambda010_best')]}
    q = metrics['selected']
    descriptive = ('evaluation_wall_seconds', 'checkpoint_selection')
    assert {k: v for k, v in state['metrics'].items() if k not in descriptive} == {
        k: v for k, v in q.items() if k not in descriptive}
    fresh = read(REPO/'docs/report_final/assets/kb2_hybrid_drop500_20261106/drop1000_best.json')
    assert {k: v for k, v in fresh.items() if k not in ('evaluation_wall_seconds', 'checkpoint_selection')} == {
        k: v for k, v in q.items() if k not in ('evaluation_wall_seconds', 'checkpoint_selection')}
    names = yaml.safe_load((REPO/'kb1_reward_guided_training/configs/pest.yaml').read_text())['names']
    classes, class_summary = [], {}
    for base in ('TPE', 'S0'):
        contributions = []
        for cid, row in q['per_class'].items():
            old = metrics[base]['per_class'][cid]
            delta = 100*(row['AP50_95']-old['AP50_95'])
            contributions.append(delta/28)
            classes.append({'baseline': base, 'class_id': int(cid), 'class_name': names[int(cid)],
                            'GT': row['gt_support'], 'source_AP_percent': 100*old['AP50_95'],
                            'selected_AP_percent': 100*row['AP50_95'], 'delta_AP_pp': delta,
                            'contribution_mAP_pp': delta/28, 'delta_TP': row['tp']-old['tp'],
                            'delta_FP': row['fp']-old['fp'], 'delta_FN': row['fn']-old['fn']})
        measured = 100*(q['mAP50_95']-metrics[base]['mAP50_95'])
        assert math.isclose(sum(contributions), measured, abs_tol=1e-5)
        class_summary[base] = {'AP_increased': sum(v > 0 for v in contributions),
                               'AP_decreased': sum(v < 0 for v in contributions),
                               'AP_equal': sum(v == 0 for v in contributions),
                               'sum_mAP_contribution_pp': sum(contributions)}
    stage = {'KB3_TPE_vs_S0_mAP_pp': 100*(metrics['TPE']['mAP50_95']-metrics['S0']['mAP50_95']),
             'KB2_vs_TPE_mAP_pp': 100*(q['mAP50_95']-metrics['TPE']['mAP50_95']),
             'hybrid_vs_S0_mAP_pp': 100*(q['mAP50_95']-metrics['S0']['mAP50_95'])}
    t, f, g = metrics['TPE']['tp'], metrics['TPE']['fp'], metrics['TPE']['num_gt']
    d, removed_fp = t-q['tp'], f-q['fp']
    assert g == 1887 and q['fn'] == g-q['tp']
    assert math.isclose(q['precision'], q['tp']/(q['tp']+q['fp']), abs_tol=1e-12)
    assert math.isclose(q['f1'], 2*q['tp']/(q['tp']+q['fp']+g), abs_tol=1e-12)
    algebra = {'TP_removed': d, 'FP_removed': removed_fp,
               'precision_condition_T_f': t*removed_fp, 'precision_condition_F_d': f*d,
               'F1_condition_T_f': t*removed_fp, 'F1_condition_d_F_plus_G': d*(f+g),
               'precision_increases': t*removed_fp > f*d,
               'F1_increases': t*removed_fp > d*(f+g)}
    first = {}
    for label, folder in [('lambda003', 'yolov8n_hybrid_lambda003_s42_20261106_trial2/run'),
                          ('lambda010', 'yolov8n_hybrid_tpe2_s42_20261105_180813_658387')]:
        logs = REPO/'kb2_preference_optimization/outputs'/folder/'dpo/metrics.jsonl'
        row = next(r for r in map(json.loads, logs.read_text().splitlines()) if r['event'] == 'train')
        assert row['step'] == 1
        first[label] = row
    assert first['lambda003']['native_loss'] == first['lambda010']['native_loss']
    ratio = first['lambda003']['gradient']['weighted_dpo_grad_norm']/first['lambda010']['gradient']['weighted_dpo_grad_norm']
    assert math.isclose(ratio, .3, rel_tol=1e-5)
    protected = {str(path): digest, str(source/'review.json'): sha256(source/'review.json'),
                 str(run/'manifest.json'): sha256(run/'manifest.json')}
    for value in (Path(manifest['provenance']['source_checkpoint']),
                  Path(manifest['provenance']['shared_S0']['checkpoint'])):
        protected[str(value)] = sha256(value)
    selection = {'decision_date_local': '2026-11-08', 'timezone': 'Asia/Bangkok',
                 'user_selected': True, 'method': 'hybrid KB3 TPE -> KB2 DPO pipeline',
                 'checkpoint': str(path), 'checkpoint_sha256': digest,
                 'trained_budget_updates': 15000, 'selected_update': 500, 'seed': 42,
                 'lambda': .03, 'beta': .1, 'native_background_weight': .5,
                 'native_foreground_weight': 1., 'initialization_and_reference': manifest['provenance'],
                 'config': manifest['config'], 'metrics': q,
                 'scope': 'selected on development validation; one continuation seed; not independent confirmation',
                 'source_report': str(source/'review.json')}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'selection.json', selection)
    atomic_json(output/'mechanism_evidence.json', {
        'selection': selection, 'stage_gain': stage, 'class_summary': class_summary,
        'precision_F1_algebra': algebra, 'same_initial_batch_gradient': first,
        'lambda003_over_lambda010_weighted_gradient_ratio_step1': ratio,
        'verified_against_two_completed_checkpoint_reloads': True,
        'protected_files': protected, 'no_new_training_or_evaluation': True})
    write_csv(output/'class_contributions.csv', sorted(classes, key=lambda r:(r['baseline'], -r['contribution_mAP_pp'])))
    fields = ('mAP50_95', 'mAP50', 'mAP75', 'precision', 'operating_recall', 'f1',
              'macro_f1', 'macro_recall', 'AR300', 'APs', 'APm', 'APl', 'tp', 'fp', 'fn')
    write_csv(output/'metrics.csv', [{'model': label, **{k: value[k] for k in fields}}
                                    for label, value in metrics.items()])
    assert all(sha256(name) == value for name, value in protected.items())
    print(json.dumps({'selected_step': 500, 'checkpoint_sha256': digest, 'stage_gain': stage,
                      'class_summary': class_summary, 'precision_F1_algebra': algebra,
                      'step1_gradient_ratio': ratio, 'output': str(output)}))


if __name__ == '__main__':
    main()
