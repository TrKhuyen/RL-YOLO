"""Isolate DPO's incremental effect in three in-memory AdamW steps.

Replays exact independent native/DPO loaders, then compares native-only vs
native+.01*DPO from the SAME checkpoint and optimizer history. Never saves weights.
"""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from kb2_preference_optimization.adapter import DFLAdapter
from kb2_preference_optimization.artifacts import (
    assert_provenance, atomic_json, sha256, source_checkpoint, source_provenance, tensor_digest,
)
from kb2_preference_optimization.preferences import PreferenceDataset
from kb2_preference_optimization.recovery import code_fingerprints
from kb2_preference_optimization.recovery.selection import balanced_probabilities
from kb2_preference_optimization.train_dpo import pair_losses
from scripts.audit_kb2_dpo_gt_alignment import forward, norm, write_csv


def read(path):
    return json.loads(path.read_text())


def replay_stream(records, size, done, exposure, logged):
    native_loader = DataLoader(list(range(size)), batch_size=4, shuffle=True,
                              generator=torch.Generator().manual_seed(20042), num_workers=0)
    weights, _ = balanced_probabilities(records)
    preference_loader = DataLoader(list(range(len(records))), batch_size=4,
        sampler=WeightedRandomSampler(weights, len(records), replacement=True,
                                      generator=torch.Generator().manual_seed(42)),
        generator=torch.Generator().manual_seed(30042), num_workers=0)
    ni, pi = iter(native_loader), iter(preference_loader)
    native_digest, image_counts = hashlib.sha256(), Counter()
    selected = {}
    for step in range(1, 15002):
        try:
            native_ids = next(ni).tolist()
        except StopIteration:
            ni = iter(native_loader)
            native_ids = next(ni).tolist()
        try:
            pref_indices = next(pi).tolist()
        except StopIteration:
            pi = iter(preference_loader)
            pref_indices = next(pi).tolist()
        if step <= 15000:
            native_digest.update((json.dumps(native_ids)+'\n').encode())
            image_counts.update(str(records[i]['image_id']) for i in pref_indices)
            if step in logged:
                assert logged[step]['native_ids'] == native_ids
        if step in (1, 14001, 15001):
            selected[step] = {'native_ids': native_ids, 'preference_indices': pref_indices,
                              'preference_image_ids': [records[i]['image_id'] for i in pref_indices]}
    assert native_digest.hexdigest() == done['native_sequence_sha256']
    assert dict(image_counts) == exposure['image_draws']
    return selected


@torch.no_grad()
def measure(adapter, dataset, probes):
    result = []
    old_cudnn, old_matmul = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    # Paired inference uses full precision to resolve very small one-step changes.
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for image_id, targets in probes.items():
            image, target = dataset.base[image_id]
            _, cls, _, boxes, _, _ = forward(adapter, image[None].to(adapter.device), dataset.policy_config)
            from torchvision.ops import box_iou
            for entry in targets:
                g, c, slot = entry['gt_index'], entry['class_id'], entry['slot']
                result.append({**entry, 'score': float(cls[slot, c].sigmoid()),
                    'box_iou': float(box_iou(boxes[slot:slot+1], target['boxes'][g:g+1].to(adapter.device))[0, 0])})
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old_cudnn, old_matmul
    return result


@torch.no_grad()
def measure_pair_margins(policy, reference, images, config, records):
    old_cudnn, old_matmul = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        pd, _, _, _ = policy.distributions(images, config)
        rd, _, _, _ = reference.distributions(images, config)
        return [{'image_id': r['image_id'], 'relative_margin': float(
            p.log_prob(r['chosen'])-p.log_prob(r['rejected'])-
            q.log_prob(r['chosen'])+q.log_prob(r['rejected']))}
            for p, q, r in zip(pd, rd, records)]
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old_cudnn, old_matmul


def audit(root, alignment, output, device):
    output.mkdir(parents=True, exist_ok=False)
    provenance = source_provenance('yolov8n')
    folder = root/'pilot_balanced'
    done, manifest = read(folder/'completed.json'), read(folder/'manifest.json')
    assert_provenance(manifest['provenance'], provenance)
    assert manifest['recovery_code_sha256'] == code_fingerprints()
    assert sha256(folder/'manifest.json') == done['manifest_sha256']
    dataset = PreferenceDataset(root/'selected', provenance)
    records = [r for r in dataset.records if not r['probe']]
    logged = {r['step']: r for r in (json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()) if r['event'] == 'train'}
    selected = replay_stream(records, len(dataset.base), done, read(folder/'exposure.json'), logged)
    probes = {}
    raw = read(alignment/'S0_progress.json')['targets']
    for r in raw:
        entry = {k: r[k] for k in ('image_id', 'gt_index', 'class_id', 'slot', 'probe', 's0_missed')}
        probes.setdefault(r['image_id'], []).append(entry)
    reference = DFLAdapter(source_checkpoint('yolov8n'), device)
    reference.model.requires_grad_(False)
    ref_hash = tensor_digest(reference.model)
    output_rows, step_rows = [], []
    for stage, step in (('S0', 1), ('best', 14001), ('last', 15001)):
        policy = DFLAdapter(source_checkpoint('yolov8n'), device)
        params = [p for p in policy.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-6, weight_decay=5e-4)
        if stage != 'S0':
            assert sha256(folder/f'{stage}.pt') == done[f'{stage}_sha256']
            payload = torch.load(folder/f'{stage}.pt', map_location='cpu', weights_only=False)
            assert payload['manifest'] == manifest and payload['step'] == step-1
            policy.model.load_state_dict(payload['state_dict'], strict=True)
            optimizer.load_state_dict(payload['optimizer'])
        initial = copy.deepcopy(policy.model.state_dict())
        opt_initial = copy.deepcopy(optimizer.state_dict())
        initial_hash = tensor_digest(policy.model)
        batch = selected[step]
        native_data = [dataset.base[i] for i in batch['native_ids']]
        native_images = torch.stack([r[0] for r in native_data]).to(device)
        native_targets = [r[1] for r in native_data]
        pair_records = [records[i] for i in batch['preference_indices']]
        pair_images = torch.stack([dataset.base[r['image_id']][0] for r in pair_records]).to(device)
        with torch.no_grad():
            rd, _, _, _ = reference.distributions(pair_images, dataset.policy_config)
        measured = {'before': measure(policy, dataset, probes)}
        before_margins = measure_pair_margins(policy, reference, pair_images, dataset.policy_config, pair_records)
        states, losses = {}, {}
        for arm in ('native_only', 'native_plus_dpo'):
            policy.model.load_state_dict(initial, strict=True)
            optimizer.load_state_dict(copy.deepcopy(opt_initial))
            optimizer.zero_grad(set_to_none=True)
            native = policy.native_loss(native_images, native_targets)
            pd, _, _, _ = policy.distributions(pair_images, dataset.policy_config)
            preference, diagnostic = pair_losses(pd, rd, pair_records, .1)
            loss = native if arm == 'native_only' else native+.01*preference
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            if stage == 'S0' and arm == 'native_plus_dpo':
                first = logged[1]
                assert abs(float(native.detach())-first['native_loss']) < 1e-6
                assert abs(float(preference.detach())-first['dpo_loss']) < 1e-8
                assert abs(float(grad_norm)-first['grad_norm']) < 1e-5
                assert diagnostic['relative_margin_max_abs'] <= 1e-5
            optimizer.step()
            states[arm] = [p.detach().clone() for p in params]
            losses[arm] = {'native': float(native.detach()), 'dpo': float(preference.detach()),
                           'grad_norm_before_clip': float(grad_norm), **diagnostic}
            losses[arm]['post_update_pair_margins'] = measure_pair_margins(
                policy, reference, pair_images, dataset.policy_config, pair_records)
            measured[arm] = measure(policy, dataset, probes)
        parameter_diff = [a-b for a, b in zip(states['native_plus_dpo'], states['native_only'])]
        originals = [initial[name] for name, p in policy.model.named_parameters() if p.requires_grad]
        native_delta = [a-b for a, b in zip(states['native_only'], originals)]
        for a, b, base in zip(measured['native_plus_dpo'], measured['native_only'], measured['before']):
            assert (a['image_id'], a['gt_index']) == (b['image_id'], b['gt_index']) == (base['image_id'], base['gt_index'])
            output_rows.append({**{k: a[k] for k in ('image_id', 'gt_index', 'class_id', 'slot', 'probe', 's0_missed')},
                'stage': stage, 'update': step,
                'before_score': base['score'], 'before_iou': base['box_iou'],
                'native_score_change': b['score']-base['score'], 'native_iou_change': b['box_iou']-base['box_iou'],
                'full_score_change': a['score']-base['score'], 'full_iou_change': a['box_iou']-base['box_iou'],
                'incremental_dpo_score_change': a['score']-b['score'],
                'incremental_dpo_iou_change': a['box_iou']-b['box_iou']})
        policy.model.load_state_dict(initial, strict=True)
        assert tensor_digest(policy.model) == initial_hash
        step_rows.append({'stage': stage, 'update': step, **batch, 'losses': losses,
                          'before_pair_margins': before_margins,
                          'incremental_dpo_parameter_delta_L2': norm(parameter_diff),
                          'native_step_parameter_delta_L2': norm(native_delta),
                          'counterfactual_delta_native_step_norm_ratio': norm(parameter_diff)/norm(native_delta)})
        print(f'{stage}: exact-loader AdamW counterfactual {step} verified, weights restored', flush=True)
    assert tensor_digest(reference.model) == ref_hash
    for stem in ('best', 'last'):
        assert sha256(folder/f'{stem}.pt') == done[f'{stem}_sha256']
    summary = []
    for stage in ('S0', 'best', 'last'):
        for probe in (False, True):
            for missed in (False, True):
                rows = [r for r in output_rows if r['stage'] == stage and r['probe'] == probe and r['s0_missed'] == missed]
                row = {'stage': stage, 'probe': probe, 's0_missed': missed, 'n': len(rows)}
                for field in ('native_score_change', 'native_iou_change', 'full_score_change', 'full_iou_change',
                              'incremental_dpo_score_change', 'incremental_dpo_iou_change'):
                    values = [r[field] for r in rows]
                    row[field+'_mean'] = statistics.mean(values)
                    row[field+'_positive_gt1e-7'] = sum(v > 1e-7 for v in values)
                    row[field+'_negative_lt_minus1e-7'] = sum(v < -1e-7 for v in values)
                    row[field+'_within_1e-7'] = sum(abs(v) <= 1e-7 for v in values)
                summary.append(row)
    checks = {'full_15000_native_stream_sha_matches': True,
              'full_15000_DPO_image_exposure_matches': True,
              'first_full_AdamW_step_losses_and_grad_norm_match_original_log': True,
              'same_weights_optimizer_history_native_DPO_batch_between_arms': True,
              'all_in_memory_weights_restored_and_disk_checkpoints_unchanged': True,
              'no_training_run_or_checkpoint_created': True}
    atomic_json(output/'counterfactual.json', {'script_sha256': sha256(Path(__file__)), 'steps': step_rows,
        'summary': summary, 'checks': checks,
        'scope': 'step1 reconstructs historical batch; steps14001/15001 are local counterfactuals, not full training ablations; all inference TF32 off in paired measurement'})
    write_csv(output/'targets.csv', output_rows)
    write_csv(output/'summary.csv', summary)
    print(json.dumps({'steps': step_rows, 'checks': checks}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    audit(args.root.resolve(), args.alignment.resolve(), args.output.resolve(), args.device)
