"""Summarize completed F2 seeds against S0 and archived F1 validation reports."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml

from kb2_preference_optimization.artifacts import (
    assert_provenance, atomic_json, sha256, source_provenance,
)

OLD_RUNS = {
    42: 'yolov8n_dpo_s42_20261026_135931',
    43: 'yolov8n_dpo_fixedpairs_s43_20261026_143830',
    44: 'yolov8n_dpo_fixedpairs_s44_20261026_145438',
}
METRICS = ('mAP50_95', 'mAP50', 'mAP75', 'AR300', 'precision',
           'operating_recall', 'f1', 'operating_f2', 'fp', 'fn', 'matched_mean_iou')


def read(path):
    return json.loads(path.read_text())


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def with_f2(metrics):
    tp, fp, fn = (metrics[k] for k in ('tp', 'fp', 'fn'))
    metrics['operating_f2'] = 5 * tp / max(5 * tp + 4 * fn + fp, 1)
    return metrics


def plots(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    classes = sorted(result['classes'], key=lambda r: r['f2_minus_f1_miss_pp'])
    fig, axis = plt.subplots(figsize=(10, 10))
    changes = [r['f2_minus_f1_miss_pp'] for r in classes]
    axis.barh(range(len(classes)), changes,
              color=['#21936b' if x < 0 else '#c85050' if x > 0 else '#999999' for x in changes])
    axis.set_yticks(range(len(classes)), [f'{r["class_name"]} (GT={r["gt_valid"]})' for r in classes], fontsize=9)
    axis.invert_yaxis()
    axis.axvline(0, color='#444444', linewidth=.8)
    axis.set_xlabel('Change in miss rate: F2 - F1 (percentage points); negative is better')
    axis.set_title('Validation FN by class: mean of best checkpoints, seeds 42/43/44')
    axis.grid(axis='x', alpha=.2)
    for y, value in enumerate(changes):
        axis.annotate(f'{value:+.2f}', (value, y), xytext=(4 if value >= 0 else -4, 0),
                      textcoords='offset points', ha='left' if value >= 0 else 'right', va='center', fontsize=8)
    axis.margins(x=.2)
    fig.tight_layout()
    fig.savefig(output / 'class_miss_delta.svg')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {42: '#2678b8', 43: '#cf7c21', 44: '#488b55'}
    for axis, key, title in zip(axes, ('mAP50_95', 'operating_recall'),
                                ('Validation mAP50-95', 'Operating recall (conf .25, IoU .5)')):
        for seed in (42, 43, 44):
            for recipe in ('f1', 'f2'):
                values = result['trajectories'][str(seed) if recipe == 'f2' else f'f1_s{seed}']
                axis.plot([r['step']/1000 for r in values], [100*r[key] for r in values],
                          color=colors[seed], linestyle='-' if recipe == 'f2' else '--',
                          label=f'{recipe.upper()} s{seed}', linewidth=1.3)
        axis.set_title(title)
        axis.set_xlabel('Updates (thousands)')
        axis.set_ylabel('%')
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output / 'validation_trajectories.svg')
    plt.close(fig)


def summarize(root, output):
    assert read(root / 'pipeline_completed.json')['status'] == 'complete'
    output.mkdir(parents=True, exist_ok=False)
    names = yaml.safe_load((ROOT / 'kb1_reward_guided_training/configs/pest.yaml').read_text())['names']
    experiment = read(root / 'experiment.json')
    current_provenance = source_provenance('yolov8n')
    assert_provenance(experiment['provenance'], current_provenance)
    runs, per_seed, trajectories = {}, [], {}
    for seed in (42, 43, 44):
        old_folder = ROOT / 'kb2_preference_optimization/outputs' / OLD_RUNS[seed]
        old = read(old_folder / 'validation_comparison.json')
        new_folder = root / f's{seed}'
        new = read(new_folder / 'validation_comparison.json')
        for report in (old, new):
            for kind in ('supervised', 'dpo'):
                with_f2(report[kind])
        completed = read(new_folder / 'completed.json')
        assert completed['reference_immutable'] and not completed['pilot']
        assert sha256(new_folder / 'best.pt') == completed['best_sha256'] == new['checkpoint_sha256']
        assert new['split'] == old['split'] == 'valid'
        for key in ('source_sha256', 'objective_version', 'transform_version'):
            assert new['provenance'][key] == old['provenance'][key]
        assert new['provenance']['dataset']['sha256'] == old['provenance']['dataset']['sha256']
        assert new['supervised']['protocol'] == old['supervised']['protocol']
        assert new['provenance']['runtime'] == old['provenance']['runtime']
        old_training = old_folder / ('training' if seed == 42 else '')
        old_config = read(old_training / 'manifest.json')['config']
        new_config = read(new_folder / 'manifest.json')['config']
        for key in old_config:
            if key not in ('preferences', 'output', 'preference_manifest_sha256', 'pairs_sha256'):
                assert old_config[key] == new_config[key], (seed, key)
        for key in METRICS:
            assert new['supervised'][key] == old['supervised'][key], (seed, key)
        validations = [r for r in (json.loads(line) for line in (new_folder / 'metrics.jsonl').read_text().splitlines())
                       if r['event'] == 'validation']
        for r in validations:
            with_f2(r['metrics'])
        best = [r for r in validations if r['step'] == new['selected_step']][0]
        for key in METRICS:
            assert new['dpo'][key] == best['metrics'][key], (seed, key)
        runs[str(seed)] = {'f1_report': str(old_folder / 'validation_comparison.json'),
                           'f2_report': str(new_folder / 'validation_comparison.json'),
                           'f1': old, 'f2': new, 'completed': completed}
        for recipe, report, step in (('f1', old, old['selected_step']), ('f2', new, new['selected_step'])):
            per_seed.append({'recipe': recipe, 'seed': seed, 'best_step': step,
                             **{key: report['dpo'][key] for key in METRICS}})
        trajectories[str(seed)] = [{
            'step': r['step'], **{k: r['metrics'][k] for k in METRICS}
        } for r in validations]
        old_validations = [r for r in (json.loads(line) for line in (old_training / 'metrics.jsonl').read_text().splitlines())
                           if r['event'] == 'validation']
        trajectories[f'f1_s{seed}'] = [{
            'step': r['step'], **{k: with_f2(r['metrics'])[k] for k in METRICS}
        } for r in old_validations]
    s0 = runs['42']['f1']['supervised']
    summary = {}
    for key in METRICS:
        a = [runs[str(s)]['f1']['dpo'][key] for s in (42, 43, 44)]
        b = [runs[str(s)]['f2']['dpo'][key] for s in (42, 43, 44)]
        differences = [y - x for x, y in zip(a, b)]
        summary[key] = {'s0': s0[key], 'f1_mean': statistics.mean(a), 'f1_sd': statistics.stdev(a),
                        'f2_mean': statistics.mean(b), 'f2_sd': statistics.stdev(b),
                        'f2_minus_f1_mean': statistics.mean(differences),
                        'f2_minus_f1_sd': statistics.stdev(differences)}
    class_rows, class_seed_rows = [], []
    for c, name in enumerate(names):
        baseline = s0['per_class'][str(c)]
        support = baseline['gt_support']
        row = {'class_id': c, 'class_name': name, 'gt_valid': support,
               's0_fn': baseline['fn'], 's0_miss_pct': 100 * baseline['fn'] / support}
        for recipe in ('f1', 'f2'):
            values = [runs[str(s)][recipe]['dpo']['per_class'][str(c)] for s in (42, 43, 44)]
            fn = statistics.mean(v['fn'] for v in values)
            row.update({f'{recipe}_fn_mean': fn, f'{recipe}_miss_pct': 100 * fn / support,
                        f'{recipe}_fp_mean': statistics.mean(v['fp'] for v in values),
                        f'{recipe}_ap_mean': statistics.mean(v['AP50_95'] for v in values)})
            for seed, value in zip((42, 43, 44), values):
                class_seed_rows.append({'class_id': c, 'class_name': name, 'seed': seed,
                                        'recipe': recipe, 'gt_valid': support, **value})
        row['f2_minus_f1_miss_pp'] = row['f2_miss_pct'] - row['f1_miss_pct']
        row['f2_minus_s0_miss_pp'] = row['f2_miss_pct'] - row['s0_miss_pct']
        row['f2_minus_f1_ap_pp'] = 100 * (row['f2_ap_mean'] - row['f1_ap_mean'])
        row['contribution_to_map_delta_pp'] = row['f2_minus_f1_ap_pp'] / len(names)
        class_rows.append(row)
    csv_write(output / 'per_seed.csv', per_seed)
    csv_write(output / 'classes.csv', class_rows)
    csv_write(output / 'classes_by_seed.csv', class_seed_rows)
    csv_write(output / 'summary.csv', [{'metric': k, **v} for k, v in summary.items()])
    macro = {recipe: statistics.mean(
        1 - r[f'{recipe}_miss_pct'] / 100 for r in class_rows) for recipe in ('f1', 'f2')}
    macro['s0'] = statistics.mean(1 - r['s0_miss_pct'] / 100 for r in class_rows)
    common_steps = set.intersection(*[
        {r['step'] for r in values} for values in trajectories.values()])
    fixed_steps = []
    for step in sorted(common_steps):
        row = {'step': step}
        for key in ('mAP50_95', 'operating_recall', 'operating_f2', 'fp', 'fn'):
            for recipe in ('f1', 'f2'):
                row[f'{recipe}_{key}'] = statistics.mean(
                    next(r[key] for r in trajectories[str(seed) if recipe == 'f2' else f'f1_s{seed}']
                         if r['step'] == step) for seed in (42, 43, 44))
        fixed_steps.append(row)
    csv_write(output / 'fixed_steps.csv', fixed_steps)
    result = {'experiment': str(root), 'seeds': [42, 43, 44], 'summary': summary,
              'macro_recall': macro, 'classes': class_rows, 'per_seed': per_seed,
              'runs': runs, 'trajectories': trajectories,
              'fixed_steps': fixed_steps,
              'preferences': read(root / 'preference_comparison.json'),
              'checks': {'s0_metrics_identical': True, 'best_metrics_replayed_exactly': True,
                         'checkpoint_hashes_verified': True, 'all_reference_immutable': True,
                         'source_data_code_unchanged_after_all_training': True,
                         'same_data_source_runtime_and_inference': True},
              'changed_training_package_sources': experiment['changed_training_package_sources']}
    atomic_json(output / 'comparison.json', result)
    plots(result, output)
    print(json.dumps({'summary': summary, 'macro_recall': macro,
                      'classes_fn_lower_than_f1': sum(r['f2_minus_f1_miss_pp'] < 0 for r in class_rows),
                      'classes_fn_higher_than_f1': sum(r['f2_minus_f1_miss_pp'] > 0 for r in class_rows)}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.experiment.resolve(), args.output.resolve())
