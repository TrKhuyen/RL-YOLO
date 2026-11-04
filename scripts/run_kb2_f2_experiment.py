"""Reproducible F2-only scorer experiment against the completed F1 recipe."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from kb2_preference_optimization.artifacts import atomic_json, sha256, source_provenance
from kb2_preference_optimization.data import DetectionDataset
from kb2_preference_optimization.quality import QualityConfig, match_boxes, quality_score

OLD = ROOT / 'kb2_preference_optimization/outputs/yolov8n_dpo_s42_20261026_135931'
PYTHON = str(ROOT / '.venv-ubuntu/bin/python')


def read(path):
    return json.loads(path.read_text())


def run_stage(root, label, arguments):
    command = [PYTHON, '-m', *arguments]
    print(f'STAGE {label}: {json.dumps(command)}', flush=True)
    with (root / f'{label}.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in child.stdout:
            log.write(line)
            log.flush()
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if line.startswith('preferences:') or 'Error' in line or 'Traceback' in line:
                    print(f'{label}: {line.rstrip()}', flush=True)
                continue
            if not isinstance(record, dict):
                continue
            if record.get('event') == 'validation':
                metrics = record['metrics']
                print(json.dumps({'stage': label, 'step': record['step'], 'best': record['best'],
                                  **{k: metrics[k] for k in ('mAP50_95', 'operating_recall',
                                                            'precision', 'fp', 'fn')}}), flush=True)
            elif record.get('event') == 'train' and (record['step'] == 1 or record['step'] % 1000 == 0):
                print(json.dumps({'stage': label, **{k: record[k] for k in (
                    'step', 'native_loss', 'dpo_loss', 'relative_margin_max_abs')}}), flush=True)
        code = child.wait()
    if code:
        raise RuntimeError(f'{label} failed ({code}); see {root / (label + ".log")}')


def audit_preferences(root):
    old_manifest = read(OLD / 'preferences/manifest.json')
    new_manifest = read(root / 'preferences/manifest.json')
    old = {r['image_id']: r for r in (
        json.loads(line) for line in (OLD / 'preferences/pairs.jsonl').read_text().splitlines())}
    new = {r['image_id']: r for r in (
        json.loads(line) for line in (root / 'preferences/pairs.jsonl').read_text().splitlines())}
    assert sha256(OLD / 'preferences/pairs.jsonl') == old_manifest['pairs_sha256']
    assert sha256(root / 'preferences/pairs.jsonl') == new_manifest['pairs_sha256']
    assert all(new_manifest[k] == old_manifest[k] for k in (
        'policy', 'image_size', 'split', 'seed', 'draws', 'limit', 'reference_state_sha256'))
    assert new_manifest['quality']['f_beta'] == 2
    dataset = DetectionDataset('train', new_manifest['image_size'])
    counts = {name: Counter() for name in ('f1', 'f2')}
    by_class = {name: Counter() for name in ('f1', 'f2')}
    for index in sorted(set(old) | set(new)):
        _, target = dataset[index]
        labels = target['labels'].tolist()
        for name, records, beta in (('f1', old, 1.), ('f2', new, 2.)):
            record = records.get(index)
            if record is None:
                continue
            sides = []
            for side in ('chosen', 'rejected'):
                saved = record[f'{side}_detection']
                pred = {'boxes': torch.tensor(saved['boxes'], dtype=torch.float32).reshape(-1, 4),
                        'labels': torch.tensor(saved['labels'], dtype=torch.long)}
                score = quality_score(pred, target, QualityConfig(f_beta=beta))
                assert abs(score['score'] - record[f'{side}_quality']['score']) < 1e-7
                sides.append({g for _, g, _ in match_boxes(pred, target, .5)})
            chosen, rejected = sides
            cq, rq = record['chosen_quality'], record['rejected_quality']
            counter = counts[name]
            counter['pairs'] += 1
            counter['chosen_more_tp'] += cq['tp'] > rq['tp']
            counter['chosen_fewer_tp'] += cq['tp'] < rq['tp']
            counter['chosen_same_tp_fewer_fp'] += cq['tp'] == rq['tp'] and cq['fp'] < rq['fp']
            counter['chosen_more_fn_fewer_fp'] += cq['fn'] > rq['fn'] and cq['fp'] < rq['fp']
            counter['pairs_with_rescued_gt'] += bool(chosen - rejected)
            counter['pairs_with_lost_gt'] += bool(rejected - chosen)
            for c in set(labels):
                indices = {g for g, label in enumerate(labels) if label == c}
                cc = by_class[name]
                cc[(c, 'pairs')] += 1
                cc[(c, 'rescue_pairs')] += bool((chosen - rejected) & indices)
                cc[(c, 'loss_pairs')] += bool((rejected - chosen) & indices)
                cc[(c, 'recovered_gt')] += len((chosen - rejected) & indices)
                cc[(c, 'lost_gt')] += len((rejected - chosen) & indices)
                cc[(c, 'gt')] += len(indices)
        if index % 500 == 0:
            print(f'preference audit: image {index}', flush=True)
    common = set(old) & set(new)
    payload = {
        'old_pairs_sha256': old_manifest['pairs_sha256'],
        'new_pairs_sha256': new_manifest['pairs_sha256'],
        'common_images': len(common), 'added_images': sorted(set(new) - set(old)),
        'dropped_images': sorted(set(old) - set(new)),
        'unchanged_pairs': sum(old[i]['chosen'] == new[i]['chosen']
                               and old[i]['rejected'] == new[i]['rejected'] for i in common),
        'unchanged_chosen': sum(old[i]['chosen'] == new[i]['chosen'] for i in common),
        'summary': {key: dict(value) for key, value in counts.items()},
        'by_class': {key: [{
            'class_id': c, **{field: value[(c, field)] for field in (
                'pairs', 'rescue_pairs', 'loss_pairs', 'recovered_gt', 'lost_gt', 'gt')}
        } for c in range(28)] for key, value in by_class.items()},
        'quality_scores_replayed_against_train_gt': True,
        'same_source_policy_seed_draws_renderer': True,
    }
    atomic_json(root / 'preference_comparison.json', payload)
    print(json.dumps({'preference_comparison': payload['summary'],
                      'common_images': len(common), 'unchanged_pairs': payload['unchanged_pairs'],
                      'added': len(payload['added_images']), 'dropped': len(payload['dropped_images'])}), flush=True)


def main():
    root = Path(sys.argv[1]).resolve()
    continuing = '--continue' in sys.argv[2:]
    root.mkdir(parents=True, exist_ok=continuing)
    provenance = source_provenance('yolov8n')
    old_manifest = read(OLD / 'training/manifest.json')
    old_provenance = old_manifest['provenance']
    assert provenance['source_sha256'] == old_provenance['source_sha256']
    assert provenance['dataset']['sha256'] == old_provenance['dataset']['sha256']
    assert provenance['runtime'] == old_provenance['runtime']
    changed_sources = [key for key, value in provenance['code_sha256'].items()
                       if value != old_provenance['code_sha256'][key]]
    assert set(changed_sources) == {'quality.py', 'generate_preferences.py', 'run.py'}, changed_sources
    experiment = {
        'started_utc': datetime.now(timezone.utc).isoformat(), 'provenance': provenance,
        'change': 'F1 -> F2 only in whole-output preference quality',
        'quality': QualityConfig().to_dict(), 'seeds': [42, 43, 44],
        'generation_seed': 42, 'shared_fixed_pairs': True,
        'selection': 'same validation mAP50_95; step0 eligible; patience5',
        'old_manifest': old_manifest, 'changed_training_package_sources': changed_sources,
        'test_evaluation': False,
    }
    if continuing:
        assert read(root / 'experiment.json')['provenance'] == provenance
    else:
        atomic_json(root / 'experiment.json', experiment)
    if not (root / 'preferences/manifest.json').exists():
        run_stage(root, 'generate', ['kb2_preference_optimization.generate_preferences',
            '--model', 'yolov8n', '--device', 'cuda', '--seed', '42', '--draws', '4',
            '--f-beta', '2', '--output', str(root / 'preferences')])
    assert read(root / 'preferences/manifest.json')['status'] == 'complete'
    if not (root / 'preference_comparison.json').exists():
        audit_preferences(root)
    for seed in (42, 43, 44):
        folder = root / f's{seed}'
        arguments = ['kb2_preference_optimization.train_dpo', '--model', 'yolov8n',
                     '--device', 'cuda', '--preferences', str(root / 'preferences'),
                     '--output', str(folder), '--seed', str(seed)]
        defaults = old_manifest['config']
        keys = ('steps', 'batch_size', 'val_batch_size', 'lr', 'weight_decay', 'dpo_weight',
                'beta', 'grad_clip', 'eval_interval', 'patience', 'log_interval',
                'gradient_interval', 'audit_images', 'eval_limit')
        for key in keys:
            arguments += ['--' + key.replace('_', '-'), str(defaults[key])]
        if not (folder / 'completed.json').exists():
            run_stage(root, f'train_s{seed}', arguments)
        manifest = read(folder / 'manifest.json')
        assert all(manifest['config'][key] == defaults[key] for key in keys)
        if not (folder / 'validation_comparison.json').exists():
            run_stage(root, f'evaluate_s{seed}', ['kb2_preference_optimization.evaluate',
                '--model', 'yolov8n', '--device', 'cuda', '--checkpoint', str(folder / 'best.pt'),
                '--split', 'valid', '--output', str(folder / 'validation_comparison.json')])
        print(json.dumps({'seed_complete': seed, **read(folder / 'completed.json')}), flush=True)
        atomic_json(root / 'progress.json', {'completed_seeds': list(range(42, seed + 1))})
    atomic_json(root / 'pipeline_completed.json', {
        'status': 'complete', 'seeds': [42, 43, 44], 'test_evaluated': False,
        'finished_utc': datetime.now(timezone.utc).isoformat()})
    print('F2_EXPERIMENT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
