"""Measure counterfactual pilot updates on fixed GTs outside the update batch."""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import statistics

import torch

from kb2_preference_optimization.artifacts import atomic_json, sha256, source_checkpoint, tensor_digest
from kb2_preference_optimization.conditional import code_fingerprints
from kb2_preference_optimization.conditional.bank import Bank
from kb2_preference_optimization.conditional.policy import ConditionalAdapter, configure_numerics, pair_loss
from scripts.audit_kb2_conditional_gradients import grad, measure, read, write_csv


def fixed_probes(bank, per_group):
    selected = []
    for group in ('recovery', 'preservation'):
        first_by_class = {}
        for r in bank.records:
            if not r['probe'] and r['group'] == group:
                first_by_class.setdefault(r['class_id'], r)
        ordered = [first_by_class[c] for c in sorted(first_by_class)]
        if not ordered:
            raise ValueError(f'No {group} probes')
        positions = torch.linspace(0, len(ordered)-1, min(per_group, len(ordered))).round().long().tolist()
        selected.extend(ordered[i] for i in positions)
    by_image = defaultdict(list)
    for r in selected:
        by_image[r['image_id']].append(r)
    images = sorted(by_image)
    batches = []
    for offset in range(0, len(images), 4):
        ids = images[offset:offset+4]
        data = [bank.dataset[i] for i in ids]
        records = [{**r, 'batch_index': b} for b, i in enumerate(ids) for r in by_image[i]]
        batches.append((torch.stack([item[0] for item in data]), [item[1] for item in data], records))
    return selected, batches


def probe_state(adapter, batches):
    rows, boxes = {}, []
    for images, targets, records in batches:
        result, raw_boxes = measure(adapter, images, targets, records)
        rows.update({r['pair_id']: r for r in result['decisions']})
        boxes.append(raw_boxes)
    return rows, boxes


def audit(run, output, device, batches, per_group):
    configure_numerics()
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = read(run/'dpo/manifest.json')
    config = manifest['config']
    bank = Bank(manifest['bank_path'])
    if manifest['conditional_code_sha256'] != code_fingerprints() or sha256(bank.root/'manifest.json') != manifest['bank_manifest_sha256']:
        raise ValueError('Code or bank changed')
    path = run/'dpo/last.pt'
    checkpoint_hash = sha256(path)
    if checkpoint_hash != read(run/'dpo/completed.json')['last_sha256']:
        raise ValueError('Checkpoint changed')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    source = source_checkpoint('yolov8n')
    source_hash = sha256(source)
    policy, reference = (ConditionalAdapter(source, device) for _ in range(2))
    reference.model.requires_grad_(False)
    reference_hash = tensor_digest(reference.model)
    params = [p for p in policy.model.parameters() if p.requires_grad]
    records = {r['pair_id']: r for r in bank.records}
    entries = [json.loads(line) for line in (run/'dpo/stream.jsonl').read_text().splitlines()][:batches]
    selected, probe_batches = fixed_probes(bank, per_group)
    rows = []
    arms = ('zero_current_gradient', 'native_only', 'dpo_only', 'native_plus_dpo', 'dpo_fresh_optimizer')
    for stage in ('S0', 'dpo_last'):
        initial = copy.deepcopy(reference.model.state_dict() if stage == 'S0' else checkpoint['state_dict'])
        policy.model.load_state_dict(initial, strict=True)
        frozen = policy.frozen_digest()
        before, boxes = probe_state(policy, probe_batches)
        optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
        if stage == 'dpo_last':
            optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
        history = copy.deepcopy(optimizer.state_dict())
        for entry in entries:
            policy.model.load_state_dict(initial, strict=True)
            data = [bank.dataset[i] for i in entry['native_image_ids']]
            images = torch.stack([item[0] for item in data]).to(device)
            targets = [item[1] for item in data]
            pairs = [{**records[pair_id], 'batch_index': b} for b, pair_id in enumerate(entry['pair_ids'], 2)]
            native = policy.native_loss(images, targets)
            logits, _ = policy.raw(images)
            with torch.no_grad():
                ref_logits, _ = reference.raw(images)
            dpo, _, _, _ = pair_loss(logits, ref_logits, pairs, config['beta'])
            ng, dg = grad(native, params), grad(config['dpo_weight']*dpo, params)
            del native, logits, ref_logits, dpo
            measured = {}
            for arm in arms:
                policy.model.load_state_dict(initial, strict=True)
                optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
                if arm != 'dpo_fresh_optimizer':
                    optimizer.load_state_dict(copy.deepcopy(history))
                for p, n, d in zip(params, ng, dg):
                    if arm == 'zero_current_gradient':
                        p.grad = torch.zeros_like(p)
                    elif arm == 'native_only':
                        p.grad = None if n is None else n.clone()
                    elif arm in ('dpo_only', 'dpo_fresh_optimizer'):
                        p.grad = None if d is None else d.clone()
                    else:
                        p.grad = (torch.zeros_like(p) if n is None else n.clone()) + (0 if d is None else d)
                torch.nn.utils.clip_grad_norm_(params, config['clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                measured[arm], after_boxes = probe_state(policy, probe_batches)
                if any(not torch.allclose(a, b, atol=.002, rtol=0) for a, b in zip(after_boxes, boxes)):
                    raise ValueError('Fixed-probe boxes changed')
                if policy.frozen_digest() != frozen:
                    raise ValueError('Frozen state changed')
            for r in selected:
                pair_id = r['pair_id']
                exposure = ('pair_selected' if pair_id in entry['pair_ids'] else
                            'image_only' if r['image_id'] in entry['native_image_ids'] else 'image_absent')
                row = {'stage': stage, 'historical_step': entry['step'], 'exposure': exposure,
                       **{k: r[k] for k in ('pair_id', 'image_id', 'class_id', 'target_gt', 'group')},
                       'before_score': before[pair_id]['score'],
                       'before_detected': before[pair_id]['target_detected']}
                for arm in arms:
                    row[arm+'_score_delta'] = measured[arm][pair_id]['score']-before[pair_id]['score']
                    row[arm+'_target_detected'] = measured[arm][pair_id]['target_detected']
                row['incremental_DPO_vs_native'] = measured['native_plus_dpo'][pair_id]['score']-measured['native_only'][pair_id]['score']
                row['DPO_current_vs_zero_gradient'] = measured['dpo_only'][pair_id]['score']-measured['zero_current_gradient'][pair_id]['score']
                rows.append(row)
            if entry['step'] % 8 == 0:
                print(json.dumps({'event': 'fixed_probe_updates', 'stage': stage, 'batches': entry['step']}), flush=True)
        policy.model.load_state_dict(initial, strict=True)
    if sha256(path) != checkpoint_hash or sha256(source) != source_hash or tensor_digest(reference.model) != reference_hash:
        raise ValueError('Checkpoint/source/reference changed')
    summary = []
    for stage in ('S0', 'dpo_last'):
        for group in ('recovery', 'preservation'):
            for exposure in ('pair_selected', 'image_only', 'image_absent'):
                selected_rows = [r for r in rows if (r['stage'], r['group'], r['exposure']) == (stage, group, exposure)]
                if not selected_rows:
                    continue
                entry = {'stage': stage, 'group': group, 'exposure': exposure,
                         'comparisons': len(selected_rows), 'unique_targets': len({r['pair_id'] for r in selected_rows})}
                for field in [arm+'_score_delta' for arm in arms]+['incremental_DPO_vs_native', 'DPO_current_vs_zero_gradient']:
                    entry[field+'_mean'] = statistics.mean(r[field] for r in selected_rows)
                    entry[field+'_positive_count'] = sum(r[field] > 0 for r in selected_rows)
                    entry[field+'_negative_count'] = sum(r[field] < 0 for r in selected_rows)
                summary.append(entry)
    write_csv(output/'fixed_probe_updates.csv', rows)
    atomic_json(output/'audit.json', {'run': str(run), 'batches_per_stage': batches,
                'probes_per_group': per_group, 'probes': selected, 'summary': summary,
                'scope': 'Fixed train targets, stratified across available classes; repeated independent counterfactuals, not independent data or training continuation',
                'optimizer_scope': 'Native/DPO/total/zero use identical history; fresh optimizer is a separate diagnostic',
                'checkpoints_unchanged': True, 'frozen_state_unchanged': True, 'source_sha256': source_hash,
                'last_checkpoint_sha256': checkpoint_hash, 'script_sha256': sha256(Path(__file__))})
    print(json.dumps({'event': 'interference_audit_complete', 'comparisons': len(rows), 'output': str(output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batches', type=int, default=32)
    parser.add_argument('--per-group', type=int, default=8)
    args = parser.parse_args()
    if min(args.batches, args.per_group) < 1:
        parser.error('Positive batch/probe budgets required')
    audit(args.run, args.output, args.device, args.batches, args.per_group)
