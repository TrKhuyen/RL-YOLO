"""Compare actual matched AdamW smoke updates, at the sampled decisions."""
import argparse
import json
from pathlib import Path

import torch

from kb2_preference_optimization.artifacts import atomic_json, sha256, source_checkpoint
from kb2_preference_optimization.conditional import VERSION, code_fingerprints
from kb2_preference_optimization.conditional.bank import Bank
from kb2_preference_optimization.conditional.policy import ConditionalAdapter, configure_numerics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new audit output')
    configure_numerics()
    run = args.run.resolve()
    completed = json.loads((run / 'completed.json').read_text())
    if completed['objective_version'] != VERSION or not completed['matched_stream']:
        raise ValueError('A completed, matched V0 run is required')
    checkpoints = {arm: torch.load(run / arm / 'last.pt', map_location='cpu', weights_only=False)
                   for arm in ('native', 'dpo')}
    for arm, checkpoint in checkpoints.items():
        if checkpoint['manifest']['conditional_code_sha256'] != code_fingerprints():
            raise ValueError('Conditional code changed')
        if sha256(run / arm / 'last.pt') != completed['arms'][arm]['last_sha256']:
            raise ValueError('Checkpoint changed since completion')
    if checkpoints['native']['step'] != checkpoints['dpo']['step']:
        raise ValueError('Different update budgets')
    manifest = checkpoints['native']['manifest']
    bank = Bank(manifest['bank_path'])
    if sha256(bank.root / 'manifest.json') != manifest['bank_manifest_sha256']:
        raise ValueError('Bank changed since training')
    adapters = {arm: ConditionalAdapter(source_checkpoint('yolov8n'), args.device)
                for arm in ('S0', 'native', 'dpo')}
    for arm in ('native', 'dpo'):
        adapters[arm].model.load_state_dict(checkpoints[arm]['state_dict'])
        if adapters[arm].frozen_digest() != manifest['frozen_state_sha256']:
            raise ValueError('Frozen state differs')
    replay = bank.replay(adapters['S0'], bank.records)
    stream = [json.loads(line) for line in (run / 'dpo' / 'stream.jsonl').read_text().splitlines()]
    native_stream = (run / 'native' / 'stream.jsonl').read_bytes()
    if native_stream != (run / 'dpo' / 'stream.jsonl').read_bytes():
        raise ValueError('Native/DPO input streams differ')
    by_id = {r['pair_id']: r for r in bank.records}
    rows, max_geometry = [], 0.
    with torch.no_grad():
        for entry in stream[:8]:
            images = torch.stack([bank.dataset[i][0] for i in entry['native_image_ids']])
            raw = {arm: adapter.raw(images) for arm, adapter in adapters.items()}
            for arm in ('native', 'dpo'):
                max_geometry = max(max_geometry, float((raw[arm][1] - raw['S0'][1]).abs().max()))
            for b, pair_id in enumerate(entry['pair_ids'], 2):
                r = by_id[pair_id]
                values = {arm: float(value[0][b, r['slot'], r['class_id']]) for arm, value in raw.items()}
                scores = {arm: float(torch.tensor(z).sigmoid()) for arm, z in values.items()}
                sign = 1 if r['chosen_keep'] else -1
                rows.append({'sampled_at_step': entry['step'], 'evaluated_after_steps': checkpoints['dpo']['step'],
                             'pair_id': pair_id, 'image_id': r['image_id'], 'group': r['group'],
                             'target_gt': r['target_gt'], 'class_id': r['class_id'], 'slot': r['slot'],
                             'logits': values, 'scores': scores,
                             'signed_logit_delta_DPO_vs_native': sign * (values['dpo'] - values['native']),
                             'signed_score_delta_DPO_vs_native': sign * (scores['dpo'] - scores['native'])})
    if max_geometry > .002:
        raise ValueError('Frozen geometry drift')
    state0 = adapters['S0'].model.state_dict()
    squared = dict(native=0., dpo=0., dpo_minus_native=0.)
    for name in manifest['trainable_names']:
        initial = state0[name].cpu().double()
        native = checkpoints['native']['state_dict'][name].double()
        dpo = checkpoints['dpo']['state_dict'][name].double()
        squared['native'] += float((native - initial).square().sum())
        squared['dpo'] += float((dpo - initial).square().sum())
        squared['dpo_minus_native'] += float((dpo - native).square().sum())
    result = {'objective_version': VERSION, 'run': str(run), 'full_bank_replay': replay,
              'scope': 'Actual matched AdamW updates; fixed sampled train decisions; not generalization evidence',
              'steps': checkpoints['dpo']['step'], 'input_streams_equal': True,
              'raw_box_max_abs_delta': max_geometry, 'parameter_update_L2': {k: v**.5 for k, v in squared.items()},
              'sampled_decisions': rows,
              'mean_signed_score_delta_DPO_vs_native': sum(r['signed_score_delta_DPO_vs_native'] for r in rows)/max(len(rows), 1)}
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'sampled_decisions'}), flush=True)


if __name__ == '__main__':
    main()
