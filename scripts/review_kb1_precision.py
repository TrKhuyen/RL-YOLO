"""Summarize the completed KB1 precision audit into small report assets."""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/kb1-precision-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'kb1_reward_guided_training/checkpoint_reward_guide_trainning/remaining_seed43_44_ablation_v1'


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audit, output = args.audit.resolve(), args.output.resolve()
    done = json.loads((audit/'completed.json').read_text())
    if done['status'] != 'complete' or not done['validation_inference']:
        raise ValueError('A completed fresh validation audit is required')
    fresh = read_csv(audit/'validation_precision.csv')
    source = read_csv(RUN/'results_all.csv')
    if len(fresh) != 10:
        raise ValueError('Expected S0 and three seeds of S1/P/G')
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for stage in ('supervised', 'native_only', 'proxy_only', 'kb1b'):
        rows = [v for v in fresh if v['stage'] == stage]
        if stage != 'supervised' and sorted(int(v['seed']) for v in rows) != [42, 43, 44]:
            raise ValueError('Incomplete paired seeds')
        item = {'stage': stage, 'n': len(rows)}
        for field in ('mAP50_95', 'precision_at_025', 'recall_at_025', 'matched_recall_precision',
                      'matched_recall_recall', 'matched_recall_TP', 'matched_recall_FP', 'matched_recall_FN'):
            factor = 100 if field in ('mAP50_95', 'precision_at_025', 'recall_at_025',
                                      'matched_recall_precision', 'matched_recall_recall') else 1
            m, s = mean_sd([float(v[field])*factor for v in rows])
            item[field+'_mean'] = m
            item[field+'_sample_sd'] = s
        summaries.append(item)
    write_csv(output/'validation_summary.csv', summaries)
    write_csv(output/'validation_thresholds.csv', fresh)
    write_csv(output/'saved_metrics.csv', read_csv(audit/'saved_metrics.csv'))
    write_csv(output/'reconstructed_counts.csv', read_csv(audit/'reconstructed_counts.csv'))

    class_changes = []
    for model in ('YOLOv8s', 'YOLOv11n'):
        for cid in range(28):
            aps = {}
            for stage in ('supervised', 'kb1b'):
                values = []
                for row in source:
                    if row['experiment']=='continuation' and row['split']=='test' and row['Model']==model and row['Stage']==stage:
                        ids = ast.literal_eval(row['classes'])
                        values.append(ast.literal_eval(row['map_per_class'])[ids.index(cid)]*100)
                aps[stage] = statistics.mean(values)
            class_changes.append({'Model': model, 'class_id': cid, 'S0_AP_pct': aps['supervised'],
                                  'G_AP_pct': aps['kb1b'], 'delta_class_AP_pp': aps['kb1b']-aps['supervised'],
                                  'contribution_to_mAP_pp': (aps['kb1b']-aps['supervised'])/28})
    write_csv(output/'class_contributions.csv', class_changes)

    # Localize FP changes using fresh validation predictions, not test class AP.
    sys.path.insert(0, str(ROOT/'kb1_reward_guided_training'))
    import torch
    torch.set_num_threads(4)
    from canonical_eval import _operating_point_counts
    per_class = []
    for row in fresh:
        stage, seed = row['stage'], int(row['seed'])
        cache = torch.load(audit/f'validation_predictions_{stage}_seed{seed}.pt',
                           map_location='cpu', weights_only=False)
        for policy, threshold in [('fixed_025', .25), ('matched_recall', float(row['matched_recall_threshold']))]:
            counts = _operating_point_counts(cache['predictions'], cache['targets'], threshold)
            for cid in range(28):
                tp, fp, fn = counts[cid]
                per_class.append({'stage': stage, 'seed': seed, 'policy': policy,
                                  'class_id': cid, 'TP': tp, 'FP': fp, 'FN': fn,
                                  'precision': tp/max(tp+fp, 1), 'recall': tp/max(tp+fn, 1)})
    write_csv(output/'validation_counts_per_class.csv', per_class)

    # A raw PR plot from one predefined seed; no pooling independent seeds or test tuning.
    points = read_csv(audit/'validation_pr_curve.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'kb1-precision-20261104', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    labels = {'supervised': ('S0', '#777777'), 'native_only': ('S1', '#3474a8'),
              'proxy_only': ('P', '#238b45'), 'kb1b': ('G', '#c45d19')}
    for stage, (label, color) in labels.items():
        seed = 0 if stage == 'supervised' else 43
        curve = [p for p in points if p['stage']==stage and int(p['seed'])==seed]
        ax.plot([float(p['recall'])*100 for p in curve], [float(p['precision'])*100 for p in curve],
                color=color, label=label, linewidth=1.7)
        fixed = next(v for v in fresh if v['stage']==stage and int(v['seed'])==seed)
        ax.scatter(float(fixed['recall_at_025'])*100, float(fixed['precision_at_025'])*100,
                   marker='o', color=color, s=35)
        ax.scatter(float(fixed['matched_recall_recall'])*100, float(fixed['matched_recall_precision'])*100,
                   marker='x', color=color, s=55)
    baseline = next(v for v in fresh if v['stage']=='supervised')
    ax.axvline(float(baseline['recall_at_025'])*100, color='#666666', linestyle=':', linewidth=1)
    ax.set(xlim=(45, 65), ylim=(40, 65), xlabel='Validation recall at IoU 0.50 (%)',
           ylabel='Validation precision at IoU 0.50 (%)',
           title='YOLOv8s: score thresholds vs. precision–recall (seed 43)')
    ax.grid(alpha=.18)
    ax.legend(frameon=False)
    fig.text(.5, .01, 'Circles: confidence 0.25. Crosses: selected under recall ≥ S0 recall. Validation only.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(output/'validation_precision_recall.svg', metadata={'Date': None})
    fig.savefig(output/'validation_precision_recall.png', dpi=130)
    plt.close(fig)

    plan = {
        'phase_1_completed': 'Validation-only threshold calibration; no training or test re-inference.',
        'next_training_objective': 'Improve precision subject to validation recall >= S0 recall; retain canonical AP as a guard.',
        'comparison': 'S1/P/G at identical 10000 micro-steps, last checkpoint, seeds 42/43/44.',
        'reference_recall': float(baseline['recall_at_025']),
        'reference_TP': int(baseline['reference_S0_TP']),
        'reference_S0_mAP50_95': float(baseline['mAP50_95']),
        'acceptance_for_claiming_a_new_weight_variant_is_better':
            'Paired precision gain at recall>=S0 versus recalibrated P and G, with canonical AP not below the corresponding starting baseline; otherwise report a trade-off.',
        'hypothesis_to_test': 'Clipping the combined confidence coefficient to >=0 prevents reward from reversing TP/FP proxy directions.',
        'variant_definition': 'Replace (0.25 + 0.10*A) with max(0, 0.25 + 0.10*A); keep native/SP, FN definition and all other settings fixed.',
        'hypothesis_status': 'Not implemented or trained; absence of current advantage/gradient diagnostics prevents claiming this causes observed errors.',
        'required_before_variant': ['Measure fraction of A<-2.5 and gradient directions on fixed train batches.',
                                    'Keep validation-only threshold selection and compare to recalibrated P/G, not just confidence=.25.',
                                    'Lock candidate configuration before another independent evaluation.'],
        'test_scope': 'Existing test was previously inspected and remains exploratory; do not tune new variants on it.',
        'threshold_policies': fresh,
    }
    (output/'next_experiment.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False)+'\n')
    files = [audit/'completed.json', audit/'validation_precision.csv', audit/'validation_manifest.json',
             audit/'validation_pr_curve.csv', RUN/'results_all.csv', Path(__file__)]
    manifest = {'date': '2026-11-04', 'fresh_validation_inference': True, 'test_inference': False,
                'raw_audit': str(audit.relative_to(ROOT)), 'validation_images_per_checkpoint': 479,
                'checkpoints': len(fresh),
                'historical_cuda_parity_failures': sum(v['historical_parity_pass']=='False' for v in fresh),
                'max_backend_AP_difference_pp': max(abs(float(v['backend_delta_mAP_pp'])) for v in fresh if v['backend_delta_mAP_pp']),
                'sources_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (output/'evidence.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
