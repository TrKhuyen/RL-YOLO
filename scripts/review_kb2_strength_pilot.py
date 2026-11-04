"""Reload all evaluation weights from the fixed lambda pilot; compare full metrics."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from kb2_preference_optimization.artifacts import atomic_json, sha256, source_checkpoint, tensor_digest
from kb2_preference_optimization.conditional.evaluation import evaluate
from kb2_preference_optimization.conditional.policy import ConditionalAdapter, configure_numerics
from kb2_preference_optimization.conditional_strength import code_fingerprints
from kb2_preference_optimization.conditional_strength.selection import dominates
from kb2_preference_optimization.data import detection_loader
from scripts.audit_kb2_conditional_gradients import read, write_csv
from scripts.review_kb2_conditional_long import events, exact_fp_point


def numeric_metrics(m):
    return {k: v for k, v in m.items() if k not in ('evaluation_wall_seconds', 'checkpoint_selection')}


def review(root, output, device):
    root, output = Path(root).resolve(), Path(output).resolve()
    done = read(root/'completed.json')
    plan = done['plan']
    assert done['matched_stream'] and done['old_checkpoints_unchanged']
    assert plan['code_sha256'] == code_fingerprints()
    output.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    loader = detection_loader('valid', 640, 8)
    targets = [t for _, batch in loader for t in batch]
    adapter = ConditionalAdapter(source_checkpoint('yolov8n'), device)
    initial_digest, frozen = tensor_digest(adapter.model), adapter.frozen_digest()
    baseline, source_predictions = evaluate(adapter, loader)
    atomic_json(output/'s0.json', baseline)
    fields = ('mAP50_95', 'mAP50', 'mAP75', 'AR300', 'APs', 'APm', 'APl', 'precision',
              'operating_recall', 'macro_recall', 'f1', 'tp', 'fp', 'fn', 'matched_mean_iou')
    rows, exact_rows, checks, histories, models = [], [], {}, {}, []
    baseline_row = {'arm': 'S0', 'lambda': 0., 'step': 0, 'best_AP': True,
                    **{k: baseline[k] for k in fields},
                    **{'delta_'+k: 0 for k in fields}}
    rows.append(baseline_row)
    models.append({'arm': 'S0', 'step': 0, 'metrics': baseline})
    source_exact = exact_fp_point(source_predictions, targets, baseline['fp'])
    exact_rows.append({'arm': 'S0', 'lambda': 0., 'step': 0, **source_exact})
    curves, class_rows, probe_rows, gradient = [], [], [], {}
    for label, result in done['arms'].items():
        folder = root/label
        assert result['steps'] == 2000 and result['reference_unchanged'] and result['frozen_state_unchanged']
        assert sha256(folder/'best.pt') == result['best_sha256']
        assert sha256(folder/'last.pt') == result['last_sha256']
        vals = [r for r in events(folder/'metrics.jsonl') if r['event'] == 'validation']
        histories[label] = vals
        assert [r['step'] for r in vals] == [0, 1000, 2000]
        assert numeric_metrics(vals[0]['metrics']) == numeric_metrics(baseline)
        for val in vals:
            curves.append({'arm': label, 'lambda': plan['arms'][label], 'step': val['step'],
                           **{k: val['metrics'][k] for k in fields}})
            for scope, p in val['probes'].items():
                for group, data in p['groups'].items():
                    probe_rows.append({'arm': label, 'step': val['step'], 'scope': scope, 'group': group, **data})
            path = folder/f"step_{val['step']:06d}.pt"
            assert sha256(path) == result['evaluation_checkpoint_sha256'][str(val['step'])]
            if not val['step']:
                continue
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            assert checkpoint['step'] == val['step']
            adapter.model.load_state_dict(checkpoint['state_dict'], strict=True)
            assert adapter.frozen_digest() == frozen
            current, predictions = evaluate(adapter, loader, source_predictions)
            assert numeric_metrics(current) == numeric_metrics(val['metrics'])
            assert current['num_images'] == 479 and current['num_gt'] == 1887
            atomic_json(output/f"{label}_step{val['step']}.json", current)
            row = {'arm': label, 'lambda': plan['arms'][label], 'step': val['step'],
                   'best_AP': val['step'] == result['best_step'], **{k: current[k] for k in fields},
                   **{'delta_'+k: current[k]-baseline[k] if current[k] is not None else None for k in fields}}
            rows.append(row)
            models.append({'arm': label, 'step': val['step'], 'metrics': current})
            exact_rows.append({'arm': label, 'lambda': plan['arms'][label], 'step': val['step'],
                               **exact_fp_point(predictions, targets, baseline['fp'])})
            for c, data in current['per_class'].items():
                class_rows.append({'arm': label, 'step': val['step'], 'class_id': c, **data,
                                   **{'delta_'+k: data[k]-baseline['per_class'][c][k]
                                      for k in ('tp', 'fp', 'fn', 'AP50_95')}})
            print(json.dumps({'event': 'fresh_validation', 'arm': label, 'step': val['step'],
                              **{k: current[k] for k in ('mAP50_95', 'precision', 'operating_recall', 'fp', 'fn')}}), flush=True)
            del checkpoint, predictions
        best = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        selected = torch.load(folder/f"step_{result['best_step']:06d}.pt", map_location='cpu', weights_only=False)
        assert best['step'] == result['best_step']
        assert all(torch.equal(value, selected['state_dict'][name]) for name, value in best['state_dict'].items())
        checks[label] = {'all_eval_checkpoints_verified': True, 'all_fresh_full_metrics_match': True,
                         'best_AP_step': result['best_step'], 'frozen_state_unchanged': True,
                         'reference_unchanged': True, 'pareto_steps': result['selection']['pareto_steps']}
        logs = events(folder/'metrics.jsonl')
        norms = [r['gradient']['weighted_dpo_native_norm_ratio'] for r in logs if r.get('gradient')]
        gradient[label] = {'mean_weighted_dpo_native_gradient_norm': statistics.mean(norms), 'samples': len(norms)}
        del best, selected
    assert len({(root/label/'stream.jsonl').read_bytes() for label in done['arms']}) == 1
    assert all(sha256(Path(path)) == digest for path, digest in plan['immutable_checkpoints'].items())
    pareto = [{'arm': r['arm'], 'step': r['step']} for r in models
              if not any(dominates(s['metrics'], r['metrics']) for s in models)]
    write_csv(output/'summary.csv', rows)
    write_csv(output/'validation.csv', curves)
    write_csv(output/'exact_fp_budget.csv', exact_rows)
    write_csv(output/'classes.csv', class_rows)
    write_csv(output/'probes.csv', probe_rows)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title, scale) in zip(axes.flat, [
            ('mAP50_95', 'mAP50-95 (%)', 100), ('mAP75', 'AP75 (%)', 100),
            ('precision', 'Precision @ .25 (%)', 100), ('operating_recall', 'Recall @ .25 (%)', 100),
            ('fp', 'FP @ .25', 1), ('fn', 'FN @ .25', 1)]):
        ax.axhline(baseline[key]*scale, color='black', linestyle=':', label='S0')
        for label, vals in histories.items():
            ax.plot([r['step'] for r in vals], [r['metrics'][key]*scale for r in vals],
                    marker='o', label=f"lambda {plan['arms'][label]}")
        ax.set_title(title)
        ax.set_xlabel('Updates')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('DPO strength | background .5 | seed 42 | 479 validation images')
    fig.tight_layout()
    fig.savefig(output/'learning_curves.svg')
    fig.savefig(output/'learning_curves.png', dpi=160)
    plt.close(fig)
    result = {'run': str(root), 'plan': plan, 'checks': checks, 'summary': rows,
              'exact_FP_budget': exact_rows, 'global_Pareto_AP_P_R': pareto,
              'gradient': gradient, 'step_audit': read(root/'step_audit.json'),
              'scope': 'Single-seed full-validation pilot; all eval weights reloaded; matched-FP exploratory; no test'}
    atomic_json(output/'review.json', result)
    print(json.dumps({'event': 'review_completed', 'Pareto': pareto, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    review(args.root, args.output, args.device)
