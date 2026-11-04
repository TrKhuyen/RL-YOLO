"""Audit V0 at GT slots: assignment, gradients, paired AdamW and the full bank.

Only reads original checkpoints. Counterfactual steps are independent in-memory
updates from the same weights/optimizer history; they are not a training run.
"""
import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F
from torchvision.ops import box_iou

from kb2_preference_optimization.artifacts import (
    append_jsonl, atomic_json, sha256, source_checkpoint, tensor_digest,
)
from kb2_preference_optimization.conditional import VERSION, code_fingerprints
from kb2_preference_optimization.conditional.bank import Bank
from kb2_preference_optimization.conditional.pairs import matches
from kb2_preference_optimization.conditional.policy import (
    ConditionalAdapter, configure_numerics, pair_loss,
)


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dot(first, second):
    return sum(float((a.detach().double() * b.detach().double()).sum())
               for a, b in zip(first, second) if a is not None and b is not None)


def grad(loss, params):
    return [None if g is None else g.detach() for g in
            torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)]


@torch.no_grad()
def measure(adapter, images, targets, records):
    logits, boxes = adapter.raw(images)
    predictions = adapter.predict(images, .001, .6, 300)
    matched, counts, rows = [], Counter(), []
    for prediction, target in zip(predictions, targets):
        found, fp, _ = matches(prediction, target, .25)
        matched.append(sorted(found))
        counts.update(tp=len(found), fp=fp, fn=len(target['labels'])-len(found))
    for r in records:
        b, j, c, g = r['batch_index'], r['slot'], r['class_id'], r['target_gt']
        scores = logits[b, j].sigmoid()
        others = scores.clone()
        others[c] = -1
        rows.append({'pair_id': r['pair_id'], 'image_id': r['image_id'],
                     'class_id': c, 'target_gt': g, 'group': r['group'],
                     'chosen_keep': r['chosen_keep'], 'score': float(scores[c]),
                     'logit': float(logits[b, j, c]), 'argmax': int(scores.argmax()),
                     'class_gap': float(scores[c] - others.max()),
                     'target_detected': g in matched[b] if g is not None else None,
                     'slot_iou': float(box_iou(boxes[b, j:j+1].cpu(),
                                     targets[b]['boxes'][g:g+1])[0, 0]) if g is not None else None})
    return {'decisions': rows, 'counts': dict(counts), 'matched_gt_by_image': matched}, boxes.cpu()


def pool_audit(adapters, bank):
    by_image = defaultdict(list)
    for r in bank.records:
        by_image[r['image_id']].append(r)
    image_ids = sorted(by_image)
    rows, max_box_error = [], 0.
    for offset in range(0, len(image_ids), 4):
        selected = image_ids[offset:offset+4]
        data = [bank.dataset[i] for i in selected]
        images = torch.stack([item[0] for item in data])
        targets = [item[1] for item in data]
        records = [{**r, 'batch_index': b} for b, i in enumerate(selected) for r in by_image[i]]
        measured = {arm: measure(adapter, images, targets, records) for arm, adapter in adapters.items()}
        for arm in ('native_last', 'dpo_last'):
            max_box_error = max(max_box_error, float((measured[arm][1]-measured['S0'][1]).abs().max()))
        for p, r in enumerate(records):
            row = {k: r[k] for k in ('pair_id', 'image_id', 'class_id', 'target_gt', 'group', 'probe', 'chosen_keep')}
            for arm, value in measured.items():
                for key in ('score', 'logit', 'argmax', 'class_gap', 'target_detected', 'slot_iou'):
                    row[f'{arm}_{key}'] = value[0]['decisions'][p][key]
            row['score_DPO_minus_native'] = row['dpo_last_score']-row['native_last_score']
            rows.append(row)
    summary = []
    for heldout in (False, True):
        for group in sorted({r['group'] for r in rows}):
            selected = [r for r in rows if r['probe'] == heldout and r['group'] == group]
            if not selected:
                continue
            entry = {'probe': heldout, 'group': group, 'pairs': len(selected)}
            for arm in adapters:
                entry[f'{arm}_mean_score'] = statistics.mean(r[f'{arm}_score'] for r in selected)
                entry[f'{arm}_targets_detected'] = sum(r[f'{arm}_target_detected'] is True for r in selected)
            entry['mean_score_DPO_minus_native'] = statistics.mean(r['score_DPO_minus_native'] for r in selected)
            summary.append(entry)
    if max_box_error > .002:
        raise ValueError('Raw boxes changed in full-bank audit')
    return rows, summary, max_box_error


def audit(run, output, device='cuda', batches=32):
    configure_numerics()
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    completed = read(run/'completed.json')
    if completed['objective_version'] != VERSION or not completed['matched_stream']:
        raise ValueError('Completed matched V0 run required')
    manifest = read(run/'dpo/manifest.json')
    config = manifest['config']
    if manifest['conditional_code_sha256'] != code_fingerprints():
        raise ValueError('Conditional code changed')
    bank = Bank(manifest['bank_path'])
    if sha256(bank.root/'manifest.json') != manifest['bank_manifest_sha256']:
        raise ValueError('Pair bank changed since training')
    hashes = {str(run/arm/f'{stem}.pt'): sha256(run/arm/f'{stem}.pt')
              for arm in ('native', 'dpo') for stem in ('best', 'last')}
    for arm in ('native', 'dpo'):
        for stem in ('best', 'last'):
            if hashes[str(run/arm/f'{stem}.pt')] != completed['arms'][arm][f'{stem}_sha256']:
                raise ValueError('Checkpoint hash changed')
    source_path = source_checkpoint('yolov8n')
    source_hash = sha256(source_path)
    stream_bytes = (run/'dpo/stream.jsonl').read_bytes()
    if stream_bytes != (run/'native/stream.jsonl').read_bytes():
        raise ValueError('Pilot streams differ')
    stream = [json.loads(line) for line in stream_bytes.decode().splitlines()]
    digest = hashlib.sha256()
    for entry in stream:
        digest.update(json.dumps({k: v for k, v in entry.items() if k != 'step'}, sort_keys=True).encode())
    if digest.hexdigest() != completed['arms']['dpo']['stream_sha256']:
        raise ValueError('Stream digest changed')
    entries = stream[:batches]
    if len(entries) != batches:
        raise ValueError('Insufficient historical batches')
    checkpoints = {arm: torch.load(run/arm/'last.pt', map_location='cpu', weights_only=False)
                   for arm in ('native', 'dpo')}
    adapters = {arm: ConditionalAdapter(source_path, device) for arm in ('S0', 'native_last', 'dpo_last')}
    for arm in ('native', 'dpo'):
        adapters[f'{arm}_last'].model.load_state_dict(checkpoints[arm]['state_dict'], strict=True)
    reference = adapters['S0']
    reference.model.requires_grad_(False)
    reference_digest = tensor_digest(reference.model)
    pool_rows, pool_summary, pool_geometry = pool_audit(adapters, bank)
    write_csv(output/'pool_targets.csv', pool_rows)
    atomic_json(output/'pool_summary.json', pool_summary)
    print(json.dumps({'event': 'full_bank_measured', 'pairs': len(pool_rows), 'groups': pool_summary}), flush=True)
    # No optimizer is ever attached to reference or native_last.
    policy = adapters['dpo_last']
    params = [p for p in policy.model.parameters() if p.requires_grad]
    by_id = {r['pair_id']: r for r in bank.records}
    directions, outcomes, batch_results = [], [], []
    logs = [json.loads(line) for line in (run/'dpo/metrics.jsonl').read_text().splitlines()]
    first_logged = next(e for e in logs if e['event'] == 'train' and e['step'] == 1)
    for stage in ('S0', 'dpo_last'):
        initial = copy.deepcopy(reference.model.state_dict() if stage == 'S0' else checkpoints['dpo']['state_dict'])
        policy.model.load_state_dict(initial, strict=True)
        base_optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
        if stage == 'dpo_last':
            base_optimizer.load_state_dict(copy.deepcopy(checkpoints['dpo']['optimizer']))
        optimizer_state = copy.deepcopy(base_optimizer.state_dict())
        frozen = policy.frozen_digest()
        if policy.model.criterion is None:
            policy.model.criterion = policy.model.init_criterion()
        criterion = policy.model.criterion
        for historical in entries:
            policy.model.load_state_dict(initial, strict=True)
            data = [bank.dataset[i] for i in historical['native_image_ids']]
            images = torch.stack([item[0] for item in data]).to(device)
            targets = [item[1] for item in data]
            records = [{**by_id[pair_id], 'batch_index': b} for b, pair_id in enumerate(historical['pair_ids'], 2)]
            for r in records:
                if r['probe'] or r['image_id'] != historical['native_image_ids'][r['batch_index']]:
                    raise ValueError('Invalid DPO exposure')
            before, before_boxes = measure(policy, images, targets, records)
            capture = {}

            def capture_bce(module, inputs):
                capture['logits'], capture['targets'] = inputs

            def capture_assignment(module, inputs, outputs):
                capture['foreground'] = outputs[3]
                capture['gt_index'] = outputs[4]

            handles = [criterion.bce.register_forward_pre_hook(capture_bce),
                       criterion.assigner.register_forward_hook(capture_assignment)]
            try:
                native = policy.native_loss(images, targets)
            finally:
                for handle in handles:
                    handle.remove()
            logits, boxes = policy.raw(images)
            with torch.no_grad():
                ref_logits, _ = reference.raw(images)
            dpo, margin, z, z0 = pair_loss(logits, ref_logits, records, config['beta'])
            native_grads = grad(native, params)
            dpo_grads = grad(config['dpo_weight']*dpo, params)
            native_output_grad = torch.autograd.grad(native, capture['logits'], retain_graph=True)[0].detach()
            if not torch.allclose(logits.detach(), capture['logits'].detach(), atol=1e-5, rtol=0):
                raise ValueError('Native/eval logits disagree')
            dpo_output_grad = torch.autograd.grad(config['dpo_weight']*dpo, z, retain_graph=True)[0].detach()
            native_image_grads = {}
            for i, r in enumerate(records):
                b, j, c = r['batch_index'], r['slot'], r['class_id']
                jacobian = grad(z[i], params)
                native_jacobian = grad(capture['logits'][b, j, c], params)
                own_term = F.softplus(-config['beta']*(1 if r['chosen_keep'] else -1)*(z[i]-z0[i]))/len(records)
                own_dpo = grad(config['dpo_weight']*own_term, params)
                if b not in native_image_grads:
                    native_image_grads[b] = [g.detach() for g in torch.autograd.grad(
                        capture['logits'][b], params, grad_outputs=native_output_grad[b],
                        retain_graph=True, allow_unused=False)]
                coeff = float(native_output_grad[b, j, c])
                same_image_effect = -dot(jacobian, native_image_grads[b])
                native_effect = -dot(jacobian, native_grads)
                dpo_effect = -dot(jacobian, dpo_grads)
                own_native_effect = -coeff*dot(jacobian, native_jacobian)
                own_dpo_effect = -dot(jacobian, own_dpo)
                directions.append({'stage': stage, 'historical_step': historical['step'],
                    **{k: r[k] for k in ('pair_id', 'image_id', 'class_id', 'target_gt', 'group', 'chosen_keep')},
                    'score': float(z[i].detach().sigmoid()),
                    'native_soft_target': float(capture['targets'][b, j, c]),
                    'native_foreground': bool(capture['foreground'][b, j]),
                    'native_assigned_gt': int(capture['gt_index'][b, j]) if capture['foreground'][b, j] else None,
                    'native_dloss_dlogit': coeff, 'weighted_DPO_dloss_dlogit': float(dpo_output_grad[i]),
                    'negative_dot_native': native_effect, 'negative_dot_DPO': dpo_effect,
                    'negative_dot_total': native_effect+dpo_effect,
                    'negative_dot_native_own_output': own_native_effect,
                    'negative_dot_native_same_image': same_image_effect,
                    'negative_dot_native_other_images': native_effect-same_image_effect,
                    'negative_dot_native_other_outputs_same_image': same_image_effect-own_native_effect,
                    'negative_dot_DPO_own_pair': own_dpo_effect,
                    'negative_dot_DPO_other_pair': dpo_effect-own_dpo_effect})
            n_loss, d_loss = float(native.detach()), float(dpo.detach())
            del native, logits, boxes, ref_logits, dpo, margin, z, z0, own_term, capture, native_output_grad
            arm_results = {}
            for arm in ('zero_current_gradient', 'native_only', 'dpo_only', 'native_plus_dpo', 'dpo_fresh_optimizer'):
                policy.model.load_state_dict(initial, strict=True)
                optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
                if arm != 'dpo_fresh_optimizer':
                    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
                optimizer.zero_grad(set_to_none=True)
                for p, ng, dg in zip(params, native_grads, dpo_grads):
                    if arm == 'zero_current_gradient':
                        p.grad = torch.zeros_like(p)
                    elif arm == 'native_only':
                        p.grad = None if ng is None else ng.clone()
                    elif arm in ('dpo_only', 'dpo_fresh_optimizer'):
                        p.grad = None if dg is None else dg.clone()
                    else:
                        p.grad = (torch.zeros_like(p) if ng is None else ng.clone()) + (0 if dg is None else dg)
                clip_norm = float(torch.nn.utils.clip_grad_norm_(params, config['clip_norm'], error_if_nonfinite=True))
                if stage == 'S0' and historical['step'] == 1 and arm == 'native_plus_dpo':
                    if (abs(n_loss-first_logged['native_loss']) > 1e-6
                            or abs(d_loss-first_logged['dpo_loss']) > 1e-8
                            or abs(clip_norm-first_logged['clip_input_norm']) > 1e-5):
                        raise ValueError('First real pilot update does not replay')
                optimizer.step()
                after, after_boxes = measure(policy, images, targets, records)
                box_delta = float((after_boxes-before_boxes).abs().max())
                if box_delta > .002 or policy.frozen_digest() != frozen:
                    raise ValueError('Frozen geometry/state changed in counterfactual update')
                recovered = sum(len(set(a)-set(b)) for a, b in zip(after['matched_gt_by_image'], before['matched_gt_by_image']))
                lost = sum(len(set(b)-set(a)) for a, b in zip(after['matched_gt_by_image'], before['matched_gt_by_image']))
                arm_results[arm] = {'clip_input_norm': clip_norm, 'counts': after['counts'],
                                    'recovered': recovered, 'lost': lost, 'raw_box_delta': box_delta}
                for row, old in zip(after['decisions'], before['decisions']):
                    outcomes.append({'stage': stage, 'historical_step': historical['step'], 'arm': arm,
                                     **row, 'before_score': old['score'], 'score_delta': row['score']-old['score'],
                                     'logit_delta': row['logit']-old['logit'], 'before_target_detected': old['target_detected'],
                                     'box_iou_delta': row['slot_iou']-old['slot_iou'] if row['slot_iou'] is not None else None})
            batch_result = {'stage': stage, 'historical_step': historical['step'],
                            'image_ids': historical['native_image_ids'], 'pair_ids': historical['pair_ids'],
                            'native_loss': n_loss, 'DPO_loss': d_loss, 'before': before, 'arms': arm_results}
            batch_results.append(batch_result)
            append_jsonl(output/'adam_batches.jsonl', batch_result)
            if historical['step'] % 8 == 0:
                print(json.dumps({'event': 'counterfactual', 'stage': stage,
                                  'batches_complete': historical['step'], 'total': batches}), flush=True)
        policy.model.load_state_dict(initial, strict=True)
    if tensor_digest(reference.model) != reference_digest or sha256(source_path) != source_hash:
        raise ValueError('S0/reference changed')
    for name, expected in hashes.items():
        if sha256(name) != expected:
            raise ValueError('Original checkpoint changed')
    if manifest['conditional_code_sha256'] != code_fingerprints():
        raise ValueError('Conditional code changed during audit')
    write_csv(output/'directions.csv', directions)
    write_csv(output/'adam_decisions.csv', outcomes)
    summary = []
    for stage in ('S0', 'dpo_last'):
        for group in sorted({r['group'] for r in directions}):
            selected = [r for r in directions if r['stage'] == stage and r['group'] == group]
            entry = {'stage': stage, 'group': group, 'decisions': len(selected)}
            for field in ('native_soft_target', 'score', 'native_dloss_dlogit', 'weighted_DPO_dloss_dlogit',
                          'negative_dot_native', 'negative_dot_DPO', 'negative_dot_total',
                          'negative_dot_native_own_output', 'negative_dot_native_same_image',
                          'negative_dot_native_other_images', 'negative_dot_native_other_outputs_same_image',
                          'negative_dot_DPO_own_pair', 'negative_dot_DPO_other_pair'):
                entry[field+'_mean'] = statistics.mean(r[field] for r in selected)
                entry[field+'_positive_count'] = sum(r[field] > 0 for r in selected)
            entry['foreground_count'] = sum(r['native_foreground'] for r in selected)
            for arm in ('zero_current_gradient', 'native_only', 'dpo_only', 'native_plus_dpo', 'dpo_fresh_optimizer'):
                measured = [r for r in outcomes if r['stage'] == stage and r['group'] == group and r['arm'] == arm]
                entry[arm] = {'mean_score_delta': statistics.mean(r['score_delta'] for r in measured),
                              'score_increased': sum(r['score_delta'] > 0 for r in measured),
                              'score_decreased': sum(r['score_delta'] < 0 for r in measured)}
            summary.append(entry)
    result = {'objective_version': VERSION, 'run': str(run), 'batches_per_stage': batches,
              'stages': ['S0 fresh optimizer', 'DPO last weights and AdamW history at 2000'],
              'scope': 'First historical pilot batches replayed independently at each stage; train only, not future continuation or validation gain',
              'derivative_scope': 'negative_dot is unclipped infinitesimal SGD direction on raw logit; AdamW measured separately',
              'fresh_optimizer_scope': 'Diagnostic only; adaptive first step rescales gradient, not strength-matched to historical optimizer',
              'pool': pool_summary, 'pool_raw_box_max_delta': pool_geometry,
              'directions': summary, 'checks': {'first_pilot_update_reproduced': True,
                  'same_weights_and_optimizer_before_each_arm': True, 'reference_unchanged': True,
                  'checkpoints_unchanged': True, 'code_unchanged': True, 'no_weights_written': True},
              'script_sha256': sha256(Path(__file__)), 'source_sha256': source_hash,
              'checkpoint_sha256': hashes, 'stream_sha256': digest.hexdigest()}
    atomic_json(output/'audit.json', result)
    print(json.dumps({'event': 'audit_completed', 'output': str(output), 'checks': result['checks']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batches', type=int, default=32)
    args = parser.parse_args()
    if args.batches < 1:
        parser.error('batches must be positive')
    audit(args.run, args.output, args.device, args.batches)


if __name__ == '__main__':
    main()
