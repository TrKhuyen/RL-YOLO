"""Fixed lambda sweep; new checkpoints, unchanged data/bank/native loss."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import torch

from kb2_preference_optimization.artifacts import atomic_json, sha256, source_checkpoint
from kb2_preference_optimization.conditional.bank import Bank
from kb2_preference_optimization.conditional_bg import code_fingerprints as background_fingerprints
from kb2_preference_optimization.conditional_strength import EXPERIMENT_VERSION, code_fingerprints
from kb2_preference_optimization.conditional_strength.step_audit import audit
from kb2_preference_optimization.conditional_strength.train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This experiment requires CUDA')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    root = (args.output_root or Path('kb2_preference_optimization/outputs') /
            f'yolov8n_strength_s42_{stamp}').resolve()
    pilot = Path('kb2_preference_optimization/outputs/yolov8n_native_background_s42_20261031_001256_145730').resolve()
    prior = json.loads((pilot / 'completed.json').read_text())
    assert prior['plan']['code_sha256'] == background_fingerprints()
    bank = Bank(prior['plan']['preferences'])
    immutable = {str(source_checkpoint('yolov8n')): sha256(source_checkpoint('yolov8n'))}
    for label, result in prior['arms'].items():
        for name in ('best', 'last'):
            path = pilot / label / f'{name}.pt'
            assert sha256(path) == result[f'{name}_sha256']
            immutable[str(path)] = sha256(path)
    root.mkdir(parents=True, exist_ok=False)
    plan = {'experiment': EXPERIMENT_VERSION, 'steps': 2000, 'seed': 42,
            'arms': {'native': 0., 'lambda001': .01, 'lambda003': .03, 'lambda010': .1},
            'background_weight': .5, 'foreground_weight': 1., 'beta': .1, 'lr': 1e-6,
            'native_batch': 4, 'preference_batch': 2, 'eval_interval': 1000, 'eval_limit': 0,
            'bank': str(bank.root), 'pilot': str(pilot), 'device': args.device,
            'selection': 'best AP min_delta1e-4; every eval saved; Pareto mAP/P/R diagnostic; no recall gate',
            'code_sha256': code_fingerprints(), 'background_code_sha256': background_fingerprints(),
            'runner_sha256': sha256(Path(__file__)), 'immutable_checkpoints': immutable,
            'scope': 'Single fixed-seed pilot; no test, no annotations/old checkpoints edited; no adaptive lambda changes'}
    atomic_json(root / 'plan.json', plan)
    print(json.dumps({'event': 'start', 'root': str(root), 'plan': plan}), flush=True)
    results = {}
    try:
        stream = [json.loads(s) for s in (pilot / 'reduced_dpo/stream.jsonl').read_text().splitlines()]
        audit(bank, stream, root, args.device, batches=32)
        for label, weight in plan['arms'].items():
            result = train(bank.root, root / label, arm='native' if weight == 0 else 'dpo',
                           device=args.device, steps=2000, seed=42, eval_interval=1000,
                           dpo_weight=weight or .01, background_weight=.5)
            if label in ('native', 'lambda001'):
                old_label = 'reduced_native' if label == 'native' else 'reduced_dpo'
                old_rows = [json.loads(s) for s in (pilot / old_label / 'metrics.jsonl').read_text().splitlines()]
                new_rows = [json.loads(s) for s in (root / label / 'metrics.jsonl').read_text().splitlines()]
                old_vals = [r for r in old_rows if r['event'] == 'validation']
                new_vals = [r for r in new_rows if r['event'] == 'validation']
                assert len(old_vals) == len(new_vals) == 3
                for a, b in zip(old_vals, new_vals):
                    ignored = ('evaluation_wall_seconds', 'checkpoint_selection')
                    assert {k: v for k, v in a['metrics'].items() if k not in ignored} == {
                        k: v for k, v in b['metrics'].items() if k not in ignored}
                    assert a['probes'] == b['probes']
                assert [r for r in old_rows if r['event'] == 'train'] == [r for r in new_rows if r['event'] == 'train']
                result['previous_pilot_numerically_reproduced'] = True
            results[label] = result
            atomic_json(root / 'progress.json', results)
            print(json.dumps({'event': 'arm_complete', 'label': label, 'best_step': result['best_step']}), flush=True)
        assert len({(root / label / 'stream.jsonl').read_bytes() for label in results}) == 1
        assert all(sha256(Path(path)) == digest for path, digest in immutable.items())
        assert code_fingerprints() == plan['code_sha256']
        assert background_fingerprints() == plan['background_code_sha256']
        assert sha256(Path(__file__)) == plan['runner_sha256']
        atomic_json(root / 'completed.json', {'plan': plan, 'arms': results, 'matched_stream': True,
                                             'old_checkpoints_unchanged': True})
        print(json.dumps({'event': 'completed', 'root': str(root)}), flush=True)
    except BaseException:
        atomic_json(root / 'failed.json', {'traceback': traceback.format_exc(), 'completed_arms': list(results)})
        raise


if __name__ == '__main__':
    main()
