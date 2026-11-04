"""Verify the six matched strength runs and summarize paired three-seed results."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from kb2_preference_optimization.artifacts import (
    assert_provenance, atomic_json, sha256, source_checkpoint, source_provenance, tensor_digest,
)
from kb2_preference_optimization.conditional import code_fingerprints as conditional_fingerprints
from kb2_preference_optimization.conditional.bank import Bank, sampling_probabilities
from kb2_preference_optimization.conditional.evaluation import evaluate
from kb2_preference_optimization.conditional.policy import ConditionalAdapter, configure_numerics
from kb2_preference_optimization.conditional_bg import code_fingerprints as background_fingerprints
from kb2_preference_optimization.conditional_strength import code_fingerprints
from kb2_preference_optimization.data import detection_loader
from scripts.audit_kb2_conditional_gradients import read, write_csv
from scripts.review_kb2_conditional_long import events, exact_fp_point
from scripts.review_kb2_strength_long import FIELDS, validations
from scripts.review_kb2_strength_pilot import numeric_metrics


SEEDS = (42, 43, 44)
ARMS = ('native', 'dpo')


def state_equal(first, second):
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(state_equal(first[k], second[k]) for k in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(state_equal(a, b) for a, b in zip(first, second))
    return first == second


def describe(values):
    return {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values),
            'min': min(values), 'max': max(values), 'n': len(values)}


def reconstruct_stream(bank, seed, stream):
    records = [r for r in bank.records if not r['probe']]
    weights, _ = sampling_probabilities(records)
    pairs = torch.multinomial(weights, 30000, replacement=True,
                              generator=torch.Generator().manual_seed(seed+10000)).reshape(-1, 2)
    loader = DataLoader(list(range(len(bank.dataset))), batch_size=2, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed+20000))
    iterator = iter(loader)
    digest = hashlib.sha256()
    assert len(stream) == 15000
    for step, entry in enumerate(stream, 1):
        try:
            uniform = next(iterator).tolist()
        except StopIteration:
            iterator = iter(loader)
            uniform = next(iterator).tolist()
        selected = [records[i] for i in pairs[step-1].tolist()]
        assert entry['step'] == step
        assert entry['pair_ids'] == [r['pair_id'] for r in selected]
        assert entry['native_image_ids'] == uniform+[r['image_id'] for r in selected]
        digest.update(json.dumps({k: v for k, v in entry.items() if k != 'step'}, sort_keys=True).encode())
    return digest.hexdigest()


def review(root, output, device):
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Fresh full validation requires CUDA')
    root, output = Path(root).resolve(), Path(output).resolve()
    first_manifest = read(root/'s42/native/manifest.json')
    provenance = source_provenance('yolov8n')
    assert_provenance(first_manifest['provenance'], provenance)
    bank = Bank(first_manifest['bank_path'])
    bank_hash = sha256(bank.root/'manifest.json')
    manifests, completions, histories, immutable = {}, {}, {}, {}
    immutable[str(source_checkpoint('yolov8n'))] = sha256(source_checkpoint('yolov8n'))
    expected = {'steps': 15000, 'input': 640, 'native_batch': 4, 'preference_batch': 2,
                'beta': .1, 'dpo_weight': .1, 'lr': 1e-6, 'weight_decay': 5e-4, 'clip_norm': 1.,
                'eval_interval': 1000, 'eval_limit': 0, 'native_background_weight': .5,
                'selection_min_delta': 1e-4, 'precision': 'float32_tf32_off_same_batch_reference'}
    for seed in SEEDS:
        for arm in ARMS:
            folder = root/f's{seed}'/arm
            m, done = read(folder/'manifest.json'), read(folder/'completed.json')
            assert_provenance(m['provenance'], provenance)
            assert m['bank_manifest_sha256'] == bank_hash
            assert m['conditional_code_sha256'] == conditional_fingerprints()
            assert m['native_background_code_sha256'] == background_fingerprints()
            assert m['strength_code_sha256'] == code_fingerprints()
            assert m['trainable_names'] == first_manifest['trainable_names']
            assert m['config']['seed'] == seed and m['config']['arm'] == arm
            assert m['config']['effective_dpo_weight'] == (0. if arm == 'native' else .1)
            assert all(m['config'][k] == v for k, v in expected.items())
            assert done['steps'] == 15000 and not done['partial_bank'] and not done['partial_evaluation']
            assert done['reference_unchanged'] and done['frozen_state_unchanged']
            vals = validations(folder)
            assert [r['step'] for r in vals] == list(range(0, 15001, 1000))
            assert all(r['metrics']['num_images'] == 479 and r['metrics']['num_gt'] == 1887 for r in vals)
            score, selected = vals[0]['metrics']['mAP50_95'], 0
            for v in vals[1:]:
                if v['metrics']['mAP50_95'] > score+1e-4:
                    score, selected = v['metrics']['mAP50_95'], v['step']
            assert done['best_step'] == done['selection']['best_AP_step'] == selected
            assert done['selection']['all_saved_steps'] == [r['step'] for r in vals]
            for name in ('best', 'last'):
                path = folder/f'{name}.pt'
                assert sha256(path) == done[f'{name}_sha256']
                immutable[str(path)] = done[f'{name}_sha256']
            assert set(done['evaluation_checkpoint_sha256']) == {str(r['step']) for r in vals}
            for step, digest in done['evaluation_checkpoint_sha256'].items():
                path = folder/f'step_{int(step):06d}.pt'
                assert sha256(path) == digest
                immutable[str(path)] = digest
            manifests[seed, arm], completions[seed, arm], histories[seed, arm] = m, done, vals
        assert (root/f's{seed}/native/stream.jsonl').read_bytes() == (root/f's{seed}/dpo/stream.jsonl').read_bytes()
        digest = reconstruct_stream(bank, seed, events(root/f's{seed}/native/stream.jsonl'))
        assert digest == completions[seed, 'native']['stream_sha256'] == completions[seed, 'dpo']['stream_sha256']
        print(json.dumps({'event': 'preflight_seed_verified', 'seed': seed}), flush=True)
    assert len({completions[seed, 'native']['stream_sha256'] for seed in SEEDS}) == 3
    output.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    loader = detection_loader('valid', 640, 8)
    targets = [t for _, batch in loader for t in batch]
    adapter = ConditionalAdapter(source_checkpoint('yolov8n'), device)
    source_digest, frozen = tensor_digest(adapter.model), adapter.frozen_digest()
    baseline, predictions = evaluate(adapter, loader)
    atomic_json(output/'s0.json', baseline)
    source_predictions = predictions
    exact_rows = [{'seed': 'S0', 'arm': 'S0', 'step': 0, 'mode': 'source',
                   **exact_fp_point(predictions, targets, baseline['fp'])}]
    rows, classes, checks, curves, probes = [], [], [], [], []
    for seed in SEEDS:
        for arm in ARMS:
            folder, done, vals = root/f's{seed}'/arm, completions[seed, arm], histories[seed, arm]
            assert numeric_metrics(vals[0]['metrics']) == numeric_metrics(baseline)
            initial = torch.load(folder/'step_000000.pt', map_location='cpu', weights_only=False)
            adapter.model.load_state_dict(initial['state_dict'], strict=True)
            assert tensor_digest(adapter.model) == source_digest
            del initial
            for v in vals:
                curves.append({'seed': seed, 'arm': arm, 'step': v['step'], **{k: v['metrics'][k] for k in FIELDS}})
                for scope, data in v['probes'].items():
                    for group, value in data['groups'].items():
                        probes.append({'seed': seed, 'arm': arm, 'step': v['step'], 'scope': scope,
                                       'group': group, **value})
            for mode, step in (('best', done['best_step']), ('last', 15000)):
                checkpoint = torch.load(folder/f'{mode}.pt', map_location='cpu', weights_only=False)
                saved = torch.load(folder/f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
                assert checkpoint['step'] == saved['step'] == step
                assert state_equal(checkpoint['state_dict'], saved['state_dict'])
                adapter.model.load_state_dict(checkpoint['state_dict'], strict=True)
                assert adapter.frozen_digest() == frozen and tensor_digest(adapter.model) != source_digest
                logged = next(v['metrics'] for v in vals if v['step'] == step)
                if mode == 'last' and done['best_step'] == 15000:
                    current = read(output/f's{seed}_{arm}_best.json')
                    fp_point = next(r for r in exact_rows if r['seed'] == seed and r['arm'] == arm and r['mode'] == 'best')
                    fp_point = {k: v for k, v in fp_point.items() if k not in ('seed', 'arm', 'step', 'mode')}
                else:
                    current, predictions = evaluate(adapter, loader, source_predictions)
                    fp_point = exact_fp_point(predictions, targets, baseline['fp'])
                assert numeric_metrics(current) == numeric_metrics(logged), (seed, arm, mode)
                atomic_json(output/f's{seed}_{arm}_{mode}.json', current)
                rows.append({'seed': seed, 'arm': arm, 'mode': mode, 'step': step,
                             **{k: current[k] for k in FIELDS},
                             **{'delta_S0_'+k: current[k]-baseline[k] for k in FIELDS},
                             'recovered': current['source_gt_transitions'].get('recovered', 0),
                             'lost': current['source_gt_transitions'].get('lost', 0)})
                exact_rows.append({'seed': seed, 'arm': arm, 'mode': mode, 'step': step, **fp_point})
                for cid, value in current['per_class'].items():
                    classes.append({'seed': seed, 'arm': arm, 'mode': mode, 'step': step, 'class_id': cid,
                                    **value, **{'delta_S0_'+k: value[k]-baseline['per_class'][cid][k]
                                              for k in ('AP50_95', 'tp', 'fp', 'fn')}})
                checks.append({'seed': seed, 'arm': arm, 'mode': mode, 'step': step,
                               'full_metrics_match': True, 'frozen_state_unchanged': True,
                               'matches_saved_evaluation_weights': True,
                               'fresh_evaluation': not (mode == 'last' and done['best_step'] == 15000)})
                print(json.dumps({'event': 'checkpoint_verified', 'seed': seed, 'arm': arm, 'mode': mode,
                                  'step': step, **{k: current[k] for k in ('mAP50_95', 'fp', 'fn')}}), flush=True)
                del checkpoint, saved
    stats, paired = [], []
    for mode in ('best', 'last'):
        for arm in ARMS:
            selected = [r for r in rows if r['arm'] == arm and r['mode'] == mode]
            for metric in FIELDS:
                stats.append({'mode': mode, 'arm': arm, 'metric': metric,
                              **describe([r[metric] for r in selected]),
                              'S0': baseline[metric],
                              'delta_S0_mean': statistics.mean(r[metric] for r in selected)-baseline[metric]})
        for metric in FIELDS:
            values = []
            for seed in SEEDS:
                a = next(r for r in rows if r['seed'] == seed and r['mode'] == mode and r['arm'] == 'native')
                b = next(r for r in rows if r['seed'] == seed and r['mode'] == mode and r['arm'] == 'dpo')
                values.append(b[metric]-a[metric])
            direction = -1 if metric in ('fp', 'fn') else 1
            paired.append({'mode': mode, 'metric': metric, **describe(values),
                           **{f'delta_seed{seed}': value for seed, value in zip(SEEDS, values)},
                           'DPO_better_seeds': sum(direction*v > 0 for v in values),
                           'DPO_tied_seeds': sum(v == 0 for v in values)})
    # Full rerun settles the earlier question about reusing the old seed42 native.
    old_native = root.parent/'yolov8n_native_background_long_s42_20261031_020350_005551/native'
    old_dpo = root.parent/'yolov8n_strength_long_lambda010_s42_20261101_180757_848916'
    rerun_checks = {}
    for arm, old in (('native', old_native), ('dpo', old_dpo)):
        prior_done = read(old/'completed.json')
        assert (root/f's42/{arm}/stream.jsonl').read_bytes() == (old/'stream.jsonl').read_bytes()
        for a, b in zip(histories[42, arm], validations(old)):
            assert a['step'] == b['step'] and numeric_metrics(a['metrics']) == numeric_metrics(b['metrics'])
            assert a['probes'] == b['probes']
        a = torch.load(root/f's42/{arm}/last.pt', map_location='cpu', weights_only=False)
        b = torch.load(old/'last.pt', map_location='cpu', weights_only=False)
        assert sha256(old/'last.pt') == prior_done['last_sha256']
        assert state_equal(a['state_dict'], b['state_dict']) and state_equal(a['optimizer'], b['optimizer'])
        rerun_checks[arm] = {'all_16_validation_metrics_and_probes_equal_previous': True,
                             'last15000_model_and_optimizer_bitwise_equal_previous': True,
                             'full_stream_equal_previous': True}
        del a, b
    write_csv(output/'summary.csv', rows)
    write_csv(output/'statistics.csv', stats)
    write_csv(output/'paired_deltas.csv', paired)
    write_csv(output/'classes.csv', classes)
    write_csv(output/'validation.csv', curves)
    write_csv(output/'exact_fp_budget.csv', exact_rows)
    write_csv(output/'probes.csv', probes)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (metric, title, factor) in zip(axes.flat, [
            ('mAP50_95', 'mAP50-95 (%)', 100), ('mAP75', 'AP75 (%)', 100),
            ('precision', 'Precision @ .25 (%)', 100), ('operating_recall', 'Recall @ .25 (%)', 100),
            ('fp', 'FP @ .25', 1), ('fn', 'FN @ .25', 1)]):
        ax.axhline(baseline[metric]*factor, color='black', linestyle=':', label='S0')
        for seed, color in zip(SEEDS, ('#0072b2', '#d55e00', '#009e73')):
            for arm, style in (('native', '--'), ('dpo', '-')):
                vals = histories[seed, arm]
                ax.plot([v['step'] for v in vals], [v['metrics'][metric]*factor for v in vals],
                        color=color, linestyle=style, label=f'{seed} {arm}', linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel('Updates')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle('Three matched seeds | DPO lambda .1 vs native | background .5 | 479 validation images')
    fig.tight_layout()
    fig.savefig(output/'learning_curves.png', dpi=160)
    fig.savefig(output/'learning_curves.svg')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mode in zip(axes, ('best', 'last')):
        for seed, color in zip(SEEDS, ('#0072b2', '#d55e00', '#009e73')):
            values = [next(r['mAP50_95']*100 for r in rows if r['seed'] == seed and
                           r['arm'] == arm and r['mode'] == mode) for arm in ARMS]
            ax.plot((0, 1), values, 'o-', color=color, label=f'Seed {seed}')
        ax.set_xticks((0, 1), ('Native', 'DPO lambda .1'))
        ax.set_ylabel('mAP50-95 (%)')
        ax.set_title('Selected best AP' if mode == 'best' else 'Fixed 15000 updates')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'paired_ap.png', dpi=160)
    fig.savefig(output/'paired_ap.svg')
    plt.close(fig)
    assert all(sha256(Path(p)) == digest for p, digest in immutable.items())
    result = {'run': str(root), 'seeds': list(SEEDS), 'steps_per_arm': 15000,
              'checkpoint_checks': checks, 'seed42_rerun_checks': rerun_checks,
              'all_96_evaluation_checkpoint_hashes_verified': True,
              'matched_within_seed_streams_independently_reconstructed': True,
              'different_streams_between_seeds': True, 'all_six_step0_weights_equal_S0': True,
              'provenance_code_bank_and_immutable_weights_verified': True,
              'summary': rows, 'statistics': stats, 'paired_deltas': paired,
              'exact_fp_budget': exact_rows, 'reviewer_sha256': sha256(Path(__file__)),
              'scope': 'Three stochastic fine-tuning seeds from one fixed S0 and bank; development validation; no test',
              'std_definition': 'Sample standard deviation, ddof=1, n=3; paired deltas computed before averaging'}
    atomic_json(output/'review.json', result)
    print(json.dumps({'event': 'review_completed', 'output': str(output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    review(args.root, args.output, args.device)
