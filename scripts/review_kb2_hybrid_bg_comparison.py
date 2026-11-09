"""Verify the matched background comparison and reload selected states on validation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml

from kb2_preference_optimization.artifacts import atomic_json, sha256, source_checkpoint as s0_checkpoint
from kb2_preference_optimization.conditional.evaluation import evaluate
from kb2_preference_optimization.conditional.policy import ConditionalAdapter, configure_numerics
from kb2_preference_optimization.data import detection_loader
from kb2_preference_optimization.hybrid.source import source_provenance, assert_provenance
from kb2_preference_optimization.hybrid.bank import Bank
from scripts.audit_kb2_conditional_gradients import write_csv
from scripts.review_kb2_strength_multiseed import state_equal
from scripts.review_kb2_strength_pilot import numeric_metrics

FIELDS = ('mAP50_95', 'mAP50', 'mAP75', 'precision', 'operating_recall', 'f1', 'macro_f1',
          'macro_recall', 'AR300', 'APs', 'APm', 'APl', 'tp', 'fp', 'fn')


def read(path):
    return json.loads(Path(path).read_text())


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def review(root, output, device):
    root, output = Path(root).resolve(), Path(output).resolve()
    done = read(root/'completed.json')
    plan = done['plan']
    assert done['matched_stream'] and done['protected_files_unchanged']
    assert plan['steps_per_arm'] == 15000 and plan['seed'] == 42
    baseline = Path(plan['baseline'])
    folders = {'dpo_bg050': (baseline, 'dpo'), 'native_bg050': (root/'native_bg050', 'native'),
               'dpo_bg025': (root/'dpo_bg025', 'dpo')}
    provenance = source_provenance('yolov8n')
    assert_provenance(plan['source'], provenance)
    bank = Bank(plan['bank'])
    assert not bank.manifest['partial_bank']
    histories, summaries, states, protected = {}, {}, {}, dict(plan['immutable_files'])
    reference_stream = (baseline/'dpo/stream.jsonl').read_bytes()
    for name, (folder, arm) in folders.items():
        aggregate = read(folder/'completed.json')
        result = read(folder/arm/'completed.json')
        manifest = read(folder/arm/'manifest.json')
        assert_provenance(manifest['provenance'], provenance)
        assert result['steps'] == 15000 and not aggregate['partial_bank'] and not aggregate['partial_evaluation']
        assert result['reference_unchanged'] and result['frozen_state_unchanged']
        assert (folder/arm/'stream.jsonl').read_bytes() == reference_stream
        assert manifest['bank_manifest_sha256'] == sha256(bank.root/'manifest.json')
        config = manifest['config']
        assert config['effective_dpo_weight'] == (0. if arm == 'native' else .1)
        assert config['native_background_weight'] == (.25 if name == 'dpo_bg025' else .5)
        for k, v in {'steps': 15000, 'seed': 42, 'beta': .1, 'lr': 1e-6, 'weight_decay': 5e-4,
                     'clip_norm': 1., 'eval_interval': 1000, 'eval_limit': 0, 'native_foreground_weight': 1.}.items():
            assert config[k] == v, (name, k)
        vals = [r for r in events(folder/arm/'metrics.jsonl') if r['event'] == 'validation']
        assert [r['step'] for r in vals] == list(range(0, 15001, 1000))
        selected, score = 0, vals[0]['metrics']['mAP50_95']
        for row in vals[1:]:
            if row['metrics']['mAP50_95'] > score + 1e-4:
                selected, score = row['step'], row['metrics']['mAP50_95']
        assert selected == result['best_step']
        assert len(result['evaluation_checkpoint_sha256']) == 16
        for step, digest in result['evaluation_checkpoint_sha256'].items():
            assert sha256(folder/arm/f'step_{int(step):06d}.pt') == digest
        for which, step in [('best', selected), ('last', 15000)]:
            path = folder/arm/f'{which}.pt'
            assert sha256(path) == result[f'{which}_sha256']
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            saved = torch.load(folder/arm/f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
            assert state_equal(checkpoint['state_dict'], saved['state_dict'])
            assert state_equal(checkpoint['optimizer'], saved['optimizer'])
            states[f'{name}_{which}'] = (path, checkpoint['metrics'], manifest['frozen_state_sha256'])
            protected[str(path)] = result[f'{which}_sha256']
        histories[name] = vals
        summaries[name] = aggregate
    assert all(numeric_metrics(v[0]['metrics']) == numeric_metrics(histories['dpo_bg050'][0]['metrics'])
               for v in histories.values())
    output.mkdir(parents=True, exist_ok=False)
    print(json.dumps({'event': 'artifact_checks_pass', 'matched_stream': True,
                      'evaluation_checkpoint_hashes': 48}), flush=True)
    configure_numerics()
    loader = detection_loader('valid', 640, 8)
    source = ConditionalAdapter(provenance['source_checkpoint'], device)
    tpe, predictions = evaluate(source, loader)
    assert numeric_metrics(tpe) == numeric_metrics(histories['dpo_bg050'][0]['metrics'])
    metrics = {'TPE': tpe}
    atomic_json(output/'TPE.json', tpe)
    for label, (path, logged, frozen) in states.items():
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        source.model.load_state_dict(checkpoint['state_dict'], strict=True)
        assert source.frozen_digest() == frozen
        measured, _ = evaluate(source, loader, predictions)
        assert numeric_metrics(measured) == numeric_metrics(logged), label
        assert measured['num_images'] == 479 and measured['num_gt'] == 1887
        metrics[label] = measured
        atomic_json(output/f'{label}.json', measured)
        print(json.dumps({'event': 'reloaded', 'label': label, 'mAP50_95': measured['mAP50_95'],
                          'fp': measured['fp'], 'fn': measured['fn']}), flush=True)
    del source, predictions
    source = ConditionalAdapter(s0_checkpoint('yolov8n'), device)
    s0, _ = evaluate(source, loader)
    assert numeric_metrics(s0) == numeric_metrics(summaries['dpo_bg050']['S0'])
    metrics['S0'] = s0
    atomic_json(output/'S0.json', s0)
    summary = [{'model': label, **{k: m[k] for k in FIELDS}} for label, m in metrics.items()]
    deltas = []
    for name in ('native_bg050', 'dpo_bg025'):
        for which in ('best', 'last'):
            for baseline_label in ('TPE', f'dpo_bg050_{which}', 'S0'):
                label = f'{name}_{which}'
                deltas.append({'model': label, 'baseline': baseline_label,
                               **{k: metrics[label][k]-metrics[baseline_label][k] for k in FIELDS}})
    write_csv(output/'summary.csv', summary)
    write_csv(output/'deltas.csv', deltas)
    curves = [{'model': label, 'step': r['step'], **{k: r['metrics'][k] for k in FIELDS}}
              for label, vals in histories.items() for r in vals]
    write_csv(output/'validation.csv', curves)
    names = yaml.safe_load(Path('kb1_reward_guided_training/configs/pest.yaml').read_text())['names']
    classes = [{'model': label, 'class_id': int(cid), 'class_name': names[int(cid)], **values}
               for label, m in metrics.items() for cid, values in m['per_class'].items()]
    write_csv(output/'classes.csv', classes)
    probes = []
    for label, vals in histories.items():
        for row in vals:
            for scope, diagnostic in row['probes'].items():
                by_group = defaultdict(list)
                for record in diagnostic['records']:
                    by_group[record['group']].append(record)
                for group, records in by_group.items():
                    probes.append({'model': label, 'step': row['step'], 'scope': scope, 'group': group,
                                   'n': len(records), 'positive_margin': sum(r['margin'] > 0 for r in records),
                                   'mean_score_delta': sum(r['score']-r['s0_score'] for r in records)/len(records),
                                   'target_detected': sum(r['ordinary_target_detected'] is True for r in records)})
    write_csv(output/'probes.csv', probes)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    colors = {'dpo_bg050': '#8d4d00', 'native_bg050': '#2467a8', 'dpo_bg025': '#23833b'}
    for ax, key, title in zip(axes, ('mAP50_95', 'precision', 'operating_recall'),
                              ('mAP50-95 (%)', 'Precision @ 0.25 (%)', 'Recall @ 0.25 (%)')):
        ax.axhline(tpe[key]*100, color='#555555', linestyle=':', label='TPE')
        for label, vals in histories.items():
            ax.plot([r['step'] for r in vals], [r['metrics'][key]*100 for r in vals],
                    color=colors[label], label=label)
        ax.set(title=title, xlabel='Updates')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output/'learning_curves.png', dpi=170)
    fig.savefig(output/'learning_curves.svg')
    plt.close(fig)
    assert all(sha256(p) == h for p, h in protected.items())
    result = {'campaign': str(root), 'split': 'valid', 'images': 479, 'GT': 1887,
              'matched_stream': True, 'protected_files_unchanged': True,
              'hashed_eval_checkpoints': 48, 'fresh_metrics_match': list(metrics),
              'best_steps': {k: read(folder/arm/'completed.json')['best_step']
                             for k, (folder, arm) in folders.items()},
              'summary': summary, 'deltas': deltas,
              'limits': ['One continuation seed and development validation only',
                         'No native/bg=.25 arm: DPO contribution at bg=.25 is not isolated']}
    atomic_json(output/'review.json', result)
    print(json.dumps({'event': 'review_completed', 'output': str(output),
                      'best_steps': result['best_steps']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    review(args.campaign, args.output, args.device)
