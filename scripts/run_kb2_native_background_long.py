"""Matched 15k confirmation of native background 0.5, with the pilot recipe frozen."""
import argparse
import json
from pathlib import Path
import traceback

import torch
from torch.utils.data import DataLoader

from kb2_preference_optimization.artifacts import atomic_json, sha256, source_checkpoint, source_provenance
from kb2_preference_optimization.conditional.bank import Bank, sampling_probabilities
from kb2_preference_optimization.conditional_bg import LOSS_VERSION, code_fingerprints
from kb2_preference_optimization.conditional_bg.train import train


def read(path):
    return json.loads(Path(path).read_text())


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def preflight(pilot, steps):
    completed = read(pilot / 'completed.json')
    prior = completed['plan']
    if (not completed['matched_stream'] or prior['background_weight'] != .5
            or prior['steps'] != 2000 or prior['seed'] != 42 or prior['eval_interval'] != 1000
            or prior['code_sha256'] != code_fingerprints()):
        raise ValueError('Require the verified, unchanged background-0.5 pilot')
    bank = Bank(prior['preferences'])
    records = [r for r in bank.records if not r['probe']]
    weights, _ = sampling_probabilities(records)
    pairs = torch.multinomial(weights, 2 * steps, replacement=True,
                              generator=torch.Generator().manual_seed(10042)).reshape(-1, 2)
    native = DataLoader(list(range(len(bank.dataset))), batch_size=2, shuffle=True,
                        drop_last=True, generator=torch.Generator().manual_seed(20042))
    iterator = iter(native)
    old_stream = events(pilot / 'reduced_dpo/stream.jsonl')
    if len(old_stream) != 2000:
        raise ValueError('Incomplete pilot stream')
    for step, entry in enumerate(old_stream):
        try:
            uniform = next(iterator).tolist()
        except StopIteration:
            iterator = iter(native)
            uniform = next(iterator).tolist()
        selected = [records[i] for i in pairs[step].tolist()]
        if (entry['step'] != step + 1 or entry['pair_ids'] != [r['pair_id'] for r in selected]
                or entry['native_image_ids'] != uniform + [r['image_id'] for r in selected]):
            raise ValueError('Long-run first 2000 steps differ from pilot')
    immutable = {str(source_checkpoint('yolov8n')): sha256(source_checkpoint('yolov8n'))}
    for arm in prior['arms']:
        for name in ('best', 'last'):
            path = pilot / arm / f'{name}.pt'
            digest = sha256(path)
            if digest != completed['arms'][arm][f'{name}_sha256']:
                raise ValueError('Pilot checkpoint changed')
            immutable[str(path)] = digest
    if immutable[str(source_checkpoint('yolov8n'))] != prior['source_sha256']:
        raise ValueError('S0 changed')
    return bank, immutable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.steps != 15000:
        parser.error('This confirmation locks 15000 updates per arm')
    if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This confirmation requires an available CUDA GPU')
    pilot, root = args.pilot_root.resolve(), args.output_root.resolve()
    bank, immutable = preflight(pilot, args.steps)
    if args.check_only:
        print(json.dumps({'event': 'preflight_passed', 'gpu': torch.cuda.get_device_name(),
                          'pilot_prefix_steps': 2000, 'bank_pairs': len(bank.records),
                          'immutable_checkpoint_count': len(immutable)}), flush=True)
        return
    root.mkdir(parents=True, exist_ok=False)
    plan = {'experiment': 'native_background_05_long_confirmation', 'native_loss_version': LOSS_VERSION,
            'pilot_root': str(pilot), 'bank_path': str(bank.root), 'steps_per_arm': args.steps,
            'arms': ['dpo', 'native'], 'seed': 42, 'background_weight': .5, 'foreground_weight': 1.,
            'device': args.device, 'gpu': torch.cuda.get_device_name(), 'eval_interval': 1000,
            'eval_limit': 0, 'input': 640, 'native_batch': 4, 'preference_batch': 2,
            'beta': .1, 'dpo_weight': .01, 'lr': 1e-6, 'weight_decay': 5e-4, 'clip_norm': 1.,
            'initialization': 'Fresh S0 and optimizer for each arm; no resume of pilot',
            'primary_confidence': .25, 'fp_tolerance': .05, 'ap_tolerance': .001,
            'selection': 'Max macro recall; micro recall>=S0, FP<=1.05*S0, AP>=S0-0.001; S0 eligible',
            'pilot_prefix_stream_verified': True, 'immutable_checkpoints': immutable,
            'provenance': source_provenance('yolov8n'), 'code_sha256': code_fingerprints(),
            'runner_sha256': sha256(Path(__file__)),
            'scope': 'Fixed 15k budget; full validation; unchanged bank/data; no test or threshold tuning',
            'metadata_note': 'Inherited trainer pilot=True; this plan/config specifies actual 15000 budget'}
    atomic_json(root / 'plan.json', plan)
    print(json.dumps({'event': 'start', 'output_root': str(root), 'plan': plan}), flush=True)
    results = {}
    try:
        for arm in plan['arms']:
            result = train(bank.root, root / arm, arm=arm, device=args.device,
                           steps=args.steps, seed=42, eval_interval=1000, background_weight=.5)
            prior = events(pilot / f'reduced_{arm}/metrics.jsonl')
            prior = [r for r in prior if r['event'] == 'validation']
            current = events(root / arm / 'metrics.jsonl')
            current = [r for r in current if r['event'] == 'validation' and r['step'] <= 2000]
            assert len(prior) == len(current) == 3
            for old, new in zip(prior, current):
                assert old['step'] == new['step'] and old['probes'] == new['probes']
                assert {k: v for k, v in old['metrics'].items() if k != 'evaluation_wall_seconds'} == {
                    k: v for k, v in new['metrics'].items() if k != 'evaluation_wall_seconds'}
            assert (root / arm / 'stream.jsonl').read_bytes().splitlines()[:2000] == (
                pilot / f'reduced_{arm}/stream.jsonl').read_bytes().splitlines()
            result['pilot_prefix_metrics_and_stream_reproduced'] = True
            results[arm] = result
            atomic_json(root / 'progress.json', results)
            print(json.dumps({'event': 'arm_complete', 'arm': arm, 'best_step': result['best_step']}), flush=True)
        assert (root / 'dpo/stream.jsonl').read_bytes() == (root / 'native/stream.jsonl').read_bytes()
        assert len({r['stream_sha256'] for r in results.values()}) == 1
        assert code_fingerprints() == plan['code_sha256']
        assert sha256(Path(__file__)) == plan['runner_sha256']
        assert all(sha256(Path(p)) == digest for p, digest in immutable.items())
        atomic_json(root / 'completed.json', {'plan': plan, 'arms': results, 'matched_stream': True,
                                             'old_checkpoints_unchanged': True})
        print(json.dumps({'event': 'completed', 'output_root': str(root)}), flush=True)
    except BaseException:
        atomic_json(root / 'failed.json', {'traceback': traceback.format_exc(),
                                         'completed_arms': list(results)})
        raise


if __name__ == '__main__':
    main()
