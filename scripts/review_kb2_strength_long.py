"""Reload the 15k lambda-.1 run and compare matched native/.01 controls."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from kb2_preference_optimization.artifacts import (
    assert_provenance, atomic_json, sha256, source_checkpoint, source_provenance, tensor_digest,
)
from kb2_preference_optimization.conditional import code_fingerprints as conditional_fingerprints
from kb2_preference_optimization.conditional.bank import Bank
from kb2_preference_optimization.conditional.evaluation import evaluate
from kb2_preference_optimization.conditional.policy import ConditionalAdapter, configure_numerics
from kb2_preference_optimization.conditional_bg import code_fingerprints as background_fingerprints
from kb2_preference_optimization.conditional_strength import code_fingerprints
from kb2_preference_optimization.data import detection_loader
from scripts.audit_kb2_conditional_gradients import read, write_csv
from scripts.review_kb2_conditional_long import events, exact_fp_point
from scripts.review_kb2_strength_pilot import numeric_metrics


FIELDS = ('mAP50_95', 'mAP50', 'mAP75', 'AR300', 'APs', 'APm', 'APl',
          'precision', 'operating_recall', 'macro_recall', 'f1', 'macro_f1',
          'tp', 'fp', 'fn', 'matched_mean_iou', 'gt_normalized_iou')


def validations(folder):
    return [r for r in events(folder/'metrics.jsonl') if r['event'] == 'validation']


def review(root, control, pilot, output, device):
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Full checkpoint verification requires CUDA')
    root, control, pilot, output = (Path(p).resolve() for p in (root, control, pilot, output))
    done, manifest = read(root/'completed.json'), read(root/'manifest.json')
    assert done['steps'] == 15000 and done['arm'] == 'dpo'
    assert done['reference_unchanged'] and done['frozen_state_unchanged']
    assert not done['partial_bank'] and not done['partial_evaluation']
    assert manifest['strength_code_sha256'] == code_fingerprints()
    assert manifest['conditional_code_sha256'] == conditional_fingerprints()
    assert manifest['native_background_code_sha256'] == background_fingerprints()
    assert_provenance(manifest['provenance'], source_provenance('yolov8n'))
    bank = Bank(manifest['bank_path'])
    assert sha256(bank.root/'manifest.json') == manifest['bank_manifest_sha256']
    config = manifest['config']
    assert config['effective_dpo_weight'] == .1 and config['native_background_weight'] == .5
    assert config['eval_interval'] == 1000 and config['eval_limit'] == 0
    own = validations(root)
    assert [r['step'] for r in own] == list(range(0, 15001, 1000))
    for name in ('best', 'last'):
        assert sha256(root/f'{name}.pt') == done[f'{name}_sha256']
    for step, digest in done['evaluation_checkpoint_sha256'].items():
        assert sha256(root/f'step_{int(step):06d}.pt') == digest
    control_done = read(control/'completed.json')
    assert control_done['matched_stream'] and control_done['old_checkpoints_unchanged']
    assert control_done['plan']['code_sha256'] == background_fingerprints()
    assert all(sha256(Path(p)) == h for p, h in control_done['plan']['immutable_checkpoints'].items())
    stream = (root/'stream.jsonl').read_bytes()
    stream_rows = events(root/'stream.jsonl')
    assert [r['step'] for r in stream_rows] == list(range(1, 15001))
    digest = hashlib.sha256()
    for row in stream_rows:
        digest.update(json.dumps({k: v for k, v in row.items() if k != 'step'}, sort_keys=True).encode())
    assert digest.hexdigest() == done['stream_sha256']
    histories = {'lambda010': own}
    immutable = {str(source_checkpoint('yolov8n')): sha256(source_checkpoint('yolov8n'))}
    for label, arm in (('native', 'native'), ('lambda001', 'dpo')):
        folder = control/arm
        previous = read(folder/'manifest.json')
        assert_provenance(previous['provenance'], manifest['provenance'])
        assert previous['bank_manifest_sha256'] == manifest['bank_manifest_sha256']
        assert previous['trainable_names'] == manifest['trainable_names']
        for key in ('steps', 'seed', 'input', 'native_batch', 'preference_batch', 'beta', 'lr',
                    'weight_decay', 'clip_norm', 'eval_interval', 'eval_limit',
                    'native_background_weight', 'precision'):
            assert previous['config'][key] == config[key], key
        assert previous['conditional_code_sha256'] == conditional_fingerprints()
        assert previous['native_background_code_sha256'] == background_fingerprints()
        assert stream == (folder/'stream.jsonl').read_bytes()
        assert done['stream_sha256'] == control_done['arms'][arm]['stream_sha256']
        histories[label] = validations(folder)
        assert [r['step'] for r in histories[label]] == list(range(0, 15001, 1000))
        for name in ('best', 'last'):
            path = folder/f'{name}.pt'
            assert sha256(path) == control_done['arms'][arm][f'{name}_sha256']
            immutable[str(path)] = sha256(path)
    pilot_vals = validations(pilot/'lambda010')
    assert len(pilot_vals) == 3
    for a, b in zip(own[:3], pilot_vals):
        assert numeric_metrics(a['metrics']) == numeric_metrics(b['metrics'])
        assert a['probes'] == b['probes']
    assert stream.splitlines()[:2000] == (pilot/'lambda010/stream.jsonl').read_bytes().splitlines()
    old_train = [r for r in events(pilot/'lambda010/metrics.jsonl') if r['event'] == 'train']
    new_train = [r for r in events(root/'metrics.jsonl') if r['event'] == 'train' and r['step'] <= 2000]
    assert old_train == new_train
    output.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    loader = detection_loader('valid', 640, 8)
    targets = [target for _, batch in loader for target in batch]
    adapter = ConditionalAdapter(source_checkpoint('yolov8n'), device)
    source_digest, frozen = tensor_digest(adapter.model), adapter.frozen_digest()
    baseline, source_predictions = evaluate(adapter, loader)
    atomic_json(output/'s0.json', baseline)
    assert all(numeric_metrics(vals[0]['metrics']) == numeric_metrics(baseline) for vals in histories.values())
    exact = [{'arm': 'S0', 'step': 0, **exact_fp_point(source_predictions, targets, baseline['fp'])}]
    checks, summary, classes, curves = [], [], [], []
    tasks = [('lambda010', r['step'], root/f"step_{r['step']:06d}.pt", r['metrics']) for r in own]
    tasks += [('native', 1000, pilot/'native/step_001000.pt', histories['native'][1]['metrics']),
              ('native', 15000, control/'native/last.pt', histories['native'][-1]['metrics']),
              ('lambda001', 15000, control/'dpo/last.pt', histories['lambda001'][-1]['metrics'])]
    for label, step, path, logged in tasks:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['step'] == step
        adapter.model.load_state_dict(checkpoint['state_dict'], strict=True)
        assert adapter.frozen_digest() == frozen
        if step == 0:
            assert tensor_digest(adapter.model) == source_digest
        else:
            assert tensor_digest(adapter.model) != source_digest
        current, predictions = evaluate(adapter, loader, source_predictions)
        assert numeric_metrics(current) == numeric_metrics(logged), (label, step)
        assert current['num_images'] == 479 and current['num_gt'] == 1887
        atomic_json(output/f'{label}_step{step}.json', current)
        exact.append({'arm': label, 'step': step, **exact_fp_point(predictions, targets, baseline['fp'])})
        checks.append({'arm': label, 'step': step, 'path': str(path), 'sha256': sha256(path),
                       'full_metrics_match': True, 'frozen_state_unchanged': True})
        print(json.dumps({'event': 'verified', 'arm': label, 'step': step,
                          **{k: current[k] for k in ('mAP50_95', 'fp', 'fn')}}), flush=True)
        del checkpoint, predictions
    best = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
    selected = torch.load(root/f"step_{done['best_step']:06d}.pt", map_location='cpu', weights_only=False)
    assert best['step'] == selected['step'] == done['best_step']
    assert all(torch.equal(v, selected['state_dict'][k]) for k, v in best['state_dict'].items())
    last = torch.load(root/'last.pt', map_location='cpu', weights_only=False)
    saved_last = torch.load(root/'step_015000.pt', map_location='cpu', weights_only=False)
    assert last['step'] == 15000
    assert all(torch.equal(v, saved_last['state_dict'][k]) for k, v in last['state_dict'].items())
    gradient = [r['gradient']['weighted_dpo_native_norm_ratio']
                for r in events(root/'metrics.jsonl') if r.get('gradient')]
    for label, vals in histories.items():
        for row in vals:
            m = row['metrics']
            summary.append({'arm': label, 'step': row['step'], **{k: m[k] for k in FIELDS},
                            **{'delta_S0_'+k: m[k]-baseline[k] for k in FIELDS}})
            curves.append({'arm': label, 'step': row['step'], **{k: m[k] for k in FIELDS}})
            for cid, c in m['per_class'].items():
                classes.append({'arm': label, 'step': row['step'], 'class_id': cid, **c,
                                **{'delta_S0_'+k: c[k]-baseline['per_class'][cid][k]
                                   for k in ('AP50_95', 'tp', 'fp', 'fn')}})
    write_csv(output/'summary.csv', summary)
    write_csv(output/'classes.csv', classes)
    write_csv(output/'exact_fp_budget.csv', exact)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title, scale) in zip(axes.flat, [
            ('mAP50_95', 'mAP50-95 (%)', 100), ('mAP75', 'AP75 (%)', 100),
            ('precision', 'Precision @ .25 (%)', 100), ('operating_recall', 'Recall @ .25 (%)', 100),
            ('fp', 'FP @ .25', 1), ('fn', 'FN @ .25', 1)]):
        ax.axhline(baseline[key]*scale, color='black', linestyle=':', label='S0')
        for label, vals in histories.items():
            ax.plot([r['step'] for r in vals], [r['metrics'][key]*scale for r in vals],
                    marker='.', label=label)
        ax.set_title(title)
        ax.set_xlabel('Updates')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('DPO strength long run | background .5 | seed 42 | 479 validation images')
    fig.tight_layout()
    fig.savefig(output/'learning_curves.svg')
    fig.savefig(output/'learning_curves.png', dpi=160)
    plt.close(fig)
    assert all(sha256(Path(p)) == digest for p, digest in immutable.items())
    result = {'run': str(root), 'control': str(control), 'pilot': str(pilot),
              'best_step': done['best_step'], 'checkpoint_checks': checks,
              'best_equals_selected_step': True, 'last_equals_step15000': True,
              'all_15000_stream_bytes_equal_controls': True, 'pilot_2000_prefix_reproduced': True,
              'provenance_code_bank_and_immutable_checkpoints_verified': True,
              'summary': summary, 'exact_fp_budget': exact,
              'gradient': {'mean_weighted_dpo_native_norm_ratio': statistics.mean(gradient),
                           'samples': len(gradient)},
              'scope': 'Single seed; full development validation; no test; matched-FP exploratory',
              'reviewer_sha256': sha256(Path(__file__))}
    atomic_json(output/'review.json', result)
    print(json.dumps({'event': 'review_completed', 'best_step': done['best_step'], 'output': str(output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'control', 'pilot', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    review(args.root, args.control, args.pilot, args.output, args.device)
