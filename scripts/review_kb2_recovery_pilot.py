"""Verify and summarize the controlled recovery pilots; no test evaluation."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from kb2_preference_optimization.artifacts import (
    assert_provenance, atomic_json, sha256, source_provenance,
)
from kb2_preference_optimization.recovery import code_fingerprints


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def review(root, output):
    output.mkdir(parents=True, exist_ok=False)
    complete = read(root/'pilot_completed.json')
    assert complete['native_sequence_identical'] and complete['all_reference_immutable']
    provenance = source_provenance('yolov8n')
    audit = read(root/'audit.json')
    assert_provenance(audit['provenance'], provenance)
    assert audit['recovery_code_sha256'] == code_fingerprints()
    assert sha256(root/'all_candidates.jsonl') == audit['all_candidates_sha256']
    names = yaml.safe_load((ROOT/'kb1_reward_guided_training/configs/pest.yaml').read_text())['names']
    results, summary, probe_rows, validation_rows = {}, [], [], []
    for arm in ('control', 'selector', 'balanced'):
        folder = root/f'pilot_{arm}'
        done, manifest = read(folder/'completed.json'), read(folder/'manifest.json')
        assert done['reference_immutable'] and done['pilot'] and done['steps'] == 2000
        assert manifest['recovery_code_sha256'] == code_fingerprints()
        for stem in ('best', 'last'):
            assert sha256(folder/f'{stem}.pt') == done[f'{stem}_sha256']
        assert sha256(folder/'manifest.json') == done['manifest_sha256']
        records = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        validations = [r for r in records if r['event'] == 'validation']
        initial, final = validations[0], validations[-1]
        assert initial['step'] == 0 and final['step'] == 2000
        best = next(r for r in validations if r['step'] == done['best_step'])
        exposure = read(folder/'exposure.json')
        metrics = final['metrics']
        row = {'arm': arm, 'step': 2000, 'best_step': done['best_step'],
               **{k: metrics[k] for k in ('mAP50_95', 'operating_recall', 'precision', 'fp', 'fn')},
               'macro_recall': sum(v['tp']/v['gt_support'] for v in metrics['per_class'].values()
                                   if v['gt_support'])/28,
               'best_map': best['metrics']['mAP50_95'], 'best_fn': best['metrics']['fn'],
               'dpo_recovery_draws': exposure['groups'].get('recovery', 0),
               'dpo_draws': sum(exposure['groups'].values()),
               'unique_dpo_images': exposure['unique_dpo_images']}
        summary.append(row)
        for validation in validations:
            validation_rows.append({'arm': arm, 'step': validation['step'],
                                    **{k: validation['metrics'][k] for k in (
                                        'mAP50_95', 'operating_recall', 'precision', 'fp', 'fn')}})
            for cohort, values in validation['probes'].items():
                probe_rows.append({'arm': arm, 'step': validation['step'], 'cohort': cohort,
                                   **{k: values[k] for k in ('pairs', 'relative_margin', 'dpo_loss',
                                                             'relative_win_rate', 'rescued_gt',
                                                             'rescued_detected', 'rescued_recall',
                                                             'rescued_mean_true_class_score', 'tp', 'fp', 'fn')}})
        results[arm] = {'manifest': manifest, 'completed': done, 'validations': validations,
                        'exposure': exposure}
    assert len({r['completed']['native_sequence_sha256'] for r in results.values()}) == 1
    first = results['control']['validations'][0]
    for values in results.values():
        initial = values['validations'][0]
        for k in ('mAP50_95', 'operating_recall', 'fp', 'fn'):
            assert initial['metrics'][k] == first['metrics'][k]
        assert initial['probes'] == first['probes']
    class_rows = []
    for c, name in enumerate(names):
        row = {'class_id': c, 'class_name': name, **audit['per_class'][c]}
        for arm, values in results.items():
            metric = values['validations'][-1]['metrics']['per_class'][str(c)]
            row[f'{arm}_fn'] = metric['fn']
            row[f'{arm}_miss_pct'] = 100*metric['fn']/metric['gt_support']
            row[f'{arm}_class_exposure'] = values['exposure']['classes'].get(str(c), 0)
        class_rows.append(row)
    write_csv(output/'summary.csv', summary)
    write_csv(output/'probes.csv', probe_rows)
    write_csv(output/'classes.csv', class_rows)
    write_csv(output/'validation.csv', validation_rows)
    result = {'scope': 'seed42, 2000 updates, pilot only; no long training or test',
              'audit': audit, 'summary': summary, 'probes': probe_rows,
              'classes': class_rows, 'runs': results,
              'checks': {'identical_native_image_sequence': True,
                         'same_S0_and_initial_probes': True, 'all_reference_immutable': True,
                         'checkpoint_manifest_pool_hashes_verified': True,
                         'base_F2_and_experimental_code_data_unchanged': True}}
    atomic_json(output/'review.json', result)
    print(json.dumps({'summary': summary, 'probes': [r for r in probe_rows if r['step'] in (0, 2000)],
                      'checks': result['checks']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review(args.root.resolve(), args.output.resolve())
