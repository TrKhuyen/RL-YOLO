"""Read-only DPO/GT alignment audit: exact margin terms and local gradients.

No optimizer step is executed. Parameter directions are infinitesimal SGD
directions, not a reconstruction of the historical AdamW trajectory.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torchvision.ops import batched_nms, box_iou
from ultralytics.utils.tal import make_anchors

from kb2_preference_optimization.adapter import DFLAdapter
from kb2_preference_optimization.artifacts import (
    assert_provenance, atomic_json, sha256, source_checkpoint, source_provenance, tensor_digest,
)
from kb2_preference_optimization.policy import DetectionDistribution, dpo_loss, render_action
from kb2_preference_optimization.preferences import PreferenceDataset
from kb2_preference_optimization.quality import match_boxes
from kb2_preference_optimization.recovery import code_fingerprints
from kb2_preference_optimization.recovery.diagnostics import canonical_gt_ids
from scripts.audit_kb2_class_fn import write_csv


def read(path):
    return json.loads(path.read_text())


def forward(adapter, image, config):
    output, raw = adapter.model(image)
    anchors, strides = make_anchors(raw['feats'], adapter.head.stride, .5)
    cls = raw['scores'].permute(0, 2, 1)[0]
    box = raw['boxes'].reshape(1, 4, adapter.head.reg_max, -1).permute(0, 3, 1, 2)[0]
    dist = DetectionDistribution(cls, box, config)
    # Reuse the actual integral convolution and xywh conversion. A manual
    # weighted sum can differ on CUDA/TF32 and should not replace inference.
    integral_input = box.permute(1, 2, 0).reshape(1, 4*adapter.head.reg_max, -1)
    decoded_xywh = (adapter.head.decode_bboxes(adapter.head.dfl(integral_input), anchors.T[None])
                    * strides.T)[0].T
    expected = torch.cat((decoded_xywh[:, :2]-decoded_xywh[:, 2:]/2,
                          decoded_xywh[:, :2]+decoded_xywh[:, 2:]/2), -1)
    xywh = output[0, :4].T
    ordinary = torch.cat((xywh[:, :2]-xywh[:, 2:]/2, xywh[:, :2]+xywh[:, 2:]/2), -1)
    assert torch.allclose(expected, ordinary, atol=2e-4, rtol=1e-5)
    return dist, cls, box, expected, anchors, strides


def action_parts(dist, action):
    """Log-prob terms by slot; background normalizer cancels in pair gaps."""
    idx, labels, bins = dist.action_tensors(action)
    cat = dist.class_logps[:, 0].double().clone()
    cat[idx] = dist.class_logps[idx, labels+1].double()
    box = torch.zeros(dist.slots, dtype=torch.float64, device=cat.device)
    box[idx] = dist.box_logps[idx[:, None], torch.arange(4, device=cat.device), bins].double().sum(-1)
    assert torch.allclose(cat.sum()+box.sum(), dist.log_prob(action), atol=1e-8, rtol=1e-10)
    return torch.stack((cat, box), -1)


def render_slots(dist, action, anchors, strides, priorities, image_size):
    """Same reference renderer, retaining the source slot for each kept box."""
    idx, labels, bins = dist.action_tensors(action)
    boxes = (torch.cat((anchors[idx]-bins[:, :2], anchors[idx]+bins[:, 2:]), -1)*strides[idx]).float().clamp(0, image_size)
    scores = priorities[idx, labels]
    valid = (boxes[:, 2:] > boxes[:, :2]).all(-1)
    boxes, labels, scores, idx = boxes[valid], labels[valid], scores[valid], idx[valid]
    order = scores.argsort(descending=True, stable=True)
    boxes, labels, scores, idx = boxes[order], labels[order], scores[order], idx[order]
    ranks = torch.arange(len(boxes), 0, -1, device=boxes.device, dtype=torch.float32)
    keep = batched_nms(boxes, ranks, labels, dist.config.nms_iou)[:dist.config.max_det]
    output = {'boxes': boxes[keep].cpu(), 'labels': labels[keep].cpu(), 'scores': scores[keep].cpu()}
    original = render_action(dist, action, anchors, strides, priorities, image_size)
    assert all(torch.equal(output[k], original[k]) for k in output)
    return output, idx[keep].tolist()


def dot(a, b):
    terms = [(x.detach().double()*y.detach().double()).sum() for x, y in zip(a, b)
             if x is not None and y is not None]
    return float(torch.stack(terms).sum()) if terms else 0.


def norm(a):
    return math.sqrt(max(dot(a, a), 0.))


def grads(scalar, inputs):
    return torch.autograd.grad(scalar, inputs, retain_graph=True, allow_unused=True)


def synthetic_checks():
    torch.manual_seed(17)
    c = torch.randn(4, 3, requires_grad=True)
    b = torch.randn(4, 4, 5, requires_grad=True)
    d = DetectionDistribution(c, b)
    a = {'slots': 4, 'indices': [0, 1], 'classes': [1, 2], 'bins': [[1, 1, 2, 2], [2, 2, 1, 1]]}
    z = {'slots': 4, 'indices': [0, 2], 'classes': [1, 0], 'bins': [[2, 2, 1, 1], [1, 1, 2, 2]]}
    gap = d.log_prob(a)-d.log_prob(z)
    split = action_parts(d, a)-action_parts(d, z)
    assert torch.allclose(split.sum(), gap, atol=1e-8)
    g = torch.autograd.grad(gap, c)[0]
    expected = torch.zeros_like(c)
    expected[1, 2], expected[2, 0] = 1., -1.
    assert torch.equal(g, expected), (g, expected)
    # Shared emitted class and shared background have exactly zero class gradient.
    return {'exact_slot_decomposition': True, 'shared_class_and_background_cancel': True}


def verify_parameter_direction(policy, reference, images, target, record, primary, config, params):
    """Check analytical derivatives without TF32's finite-difference rounding."""
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    originals = [p.detach().clone() for p in params]
    results = []
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        pd, cls, _, boxes, _, _ = forward(policy, images, config)
        with torch.no_grad():
            rd, _, _, _, _, _ = forward(reference, images, config)
        loss, _ = dpo_loss(pd.log_prob(record['chosen']), pd.log_prob(record['rejected']),
                           rd.log_prob(record['chosen']), rd.log_prob(record['rejected']))
        gd = grads(loss, params)
        nd = norm(gd)
        g = record['selection']['rescued_gt'][0]
        slot, c = primary[g], int(target['labels'][g])
        score = cls[slot, c].sigmoid()
        iou = box_iou(boxes[slot:slot+1], target['boxes'][g:g+1].to(policy.device))[0, 0]
        analytical = [-dot(grads(value, params), gd)/nd for value in (score, iou)]
        for eps in (1e-3, 5e-4):
            values = {}
            for sign in (-1, 1):
                with torch.no_grad():
                    for p, original, grad in zip(params, originals, gd):
                        p.copy_(original if grad is None else original-sign*eps*grad/nd)
                    _, fc, _, fb, _, _ = forward(policy, images, config)
                    values[sign] = (float(fc[slot, c].sigmoid()), float(box_iou(
                        fb[slot:slot+1], target['boxes'][g:g+1].to(policy.device))[0, 0]))
            numerical = [(values[1][k]-values[-1][k])/(2*eps) for k in (0, 1)]
            for a, b in zip(analytical, numerical):
                assert abs(a-b) < max(.002, .03*abs(a)), (eps, a, b)
            results.append({'epsilon': eps, 'analytical': analytical, 'numerical': numerical,
                            'precision': 'TF32 disabled for both analytical and numerical check only'})
    finally:
        with torch.no_grad():
            for p, original in zip(params, originals):
                p.copy_(original)
        torch.backends.cudnn.allow_tf32 = old_cudnn
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
    return results


def summarize(rows, group_keys, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    result = []
    for key, group in groups.items():
        entry = dict(zip(group_keys, key))
        entry['n'] = len(group)
        for field in fields:
            values = [r[field] for r in group if r.get(field) is not None]
            if not values:
                continue
            entry[field+'_mean'] = statistics.mean(values)
            entry[field+'_positive'] = sum(v > 1e-10 for v in values)
            entry[field+'_negative'] = sum(v < -1e-10 for v in values)
            entry[field+'_zero'] = sum(abs(v) <= 1e-10 for v in values)
        result.append(entry)
    return result


def audit(root, output, device, stages):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    checks = synthetic_checks()
    provenance = source_provenance('yolov8n')
    folder = root/'pilot_balanced'
    done, manifest = read(folder/'completed.json'), read(folder/'manifest.json')
    assert done['status'] == 'complete' and done['steps'] == 15000
    assert_provenance(manifest['provenance'], provenance)
    assert manifest['recovery_code_sha256'] == code_fingerprints()
    assert sha256(folder/'manifest.json') == done['manifest_sha256']
    for stem in ('best', 'last'):
        assert sha256(folder/f'{stem}.pt') == done[f'{stem}_sha256']
    dataset = PreferenceDataset(root/'selected', provenance)
    records = [r for r in dataset.records if r['selection']['group'] == 'recovery']
    reference = DFLAdapter(source_checkpoint('yolov8n'), device)
    reference.model.requires_grad_(False)
    reference_digest = tensor_digest(reference.model)
    assert reference_digest == manifest['reference_state_sha256']
    baseline, pair_rows, gt_rows, finite_checks = {}, [], [], []
    for stage in stages:
        policy = DFLAdapter(source_checkpoint('yolov8n'), device)
        if stage != 'S0':
            payload = torch.load(folder/f'{stage}.pt', map_location='cpu', weights_only=False)
            assert payload['manifest'] == manifest
            policy.model.load_state_dict(payload['state_dict'], strict=True)
        before = tensor_digest(policy.model)
        params = [p for p in policy.model.parameters() if p.requires_grad]
        for record_index, record in enumerate(records):
            image, target = dataset.base[record['image_id']]
            images = image[None].to(device)
            with torch.no_grad():
                rd, reference_cls, _, _, ra, rs = forward(reference, images, dataset.policy_config)
                ref_parts = action_parts(rd, record['chosen'])-action_parts(rd, record['rejected'])
                ref_chosen, ref_rejected = rd.log_prob(record['chosen']), rd.log_prob(record['rejected'])
                pred, slots = render_slots(rd, record['chosen'], ra, rs, reference_cls.sigmoid(), 640)
                saved = record['chosen_detection']
                assert torch.equal(pred['labels'], torch.tensor(saved['labels']))
                assert torch.allclose(pred['boxes'], torch.tensor(saved['boxes']).float(), atol=1e-4)
                match = {g: (p, iou) for p, g, iou in match_boxes(pred, target, .5)}
                targets = record['selection']['rescued_gt']
                primary = {g: slots[match[g][0]] for g in targets}
                other_local = set()
                for side in ('chosen', 'rejected'):
                    action = record[side]
                    idx, labels, bins = rd.action_tensors(action)
                    sampled = (torch.cat((ra[idx]-bins[:, :2], ra[idx]+bins[:, 2:]), -1)*rs[idx]).clamp(0, 640)
                    ious = box_iou(sampled, target['boxes'][targets].to(device))
                    correct = labels[:, None] == target['labels'][targets].to(device)[None]
                    other_local.update(idx[((ious >= .5) & correct).any(-1)].tolist())
                buckets = torch.full((rd.slots,), 2, device=device, dtype=torch.long)
                if other_local:
                    buckets[list(other_local)] = 1
                buckets[list(primary.values())] = 0
                if stage == 'S0':
                    s0_prediction = reference.predict(images)[0]
                    s0_ids = canonical_gt_ids(s0_prediction, target)
                    baseline[record['image_id']] = {'matched': s0_ids, 'primary': primary}
                assert primary == baseline[record['image_id']]['primary']

            pd, cls, box, decoded, _, _ = forward(policy, images, dataset.policy_config)
            cp, rp = pd.log_prob(record['chosen']), pd.log_prob(record['rejected'])
            loss, margin = dpo_loss(cp, rp, ref_chosen, ref_rejected, .1)
            parts = action_parts(pd, record['chosen'])-action_parts(pd, record['rejected'])-ref_parts
            assert abs(float(parts.sum().detach())-float(margin)) < 1e-7
            all_grads = grads(loss, params+[cls, box])
            gd, gc, gb = all_grads[:-2], all_grads[-2], all_grads[-1]
            nd = norm(gd)
            native = policy.native_loss(images, [target])
            gn = grads(native, params)
            nn = norm(gn)
            combined = [None if a is None and b is None else
                        ((torch.zeros_like(b) if a is None else a.detach())+
                         .01*(torch.zeros_like(a) if b is None else b.detach()))
                        for a, b in zip(gn, gd)]
            nc = norm(combined)
            with torch.no_grad():
                ordinary = policy.predict(images)[0]
                matched = canonical_gt_ids(ordinary, target)
            pair = {'stage': stage, 'image_id': record['image_id'], 'probe': record['probe'],
                    'margin': float(margin), 'dpo_loss': float(loss.detach()),
                    'chosen_logp_delta': float((cp-ref_chosen).detach()),
                    'rejected_logp_delta': float((rp-ref_rejected).detach()),
                    'target_gt': len(targets), 's0_detected': sum(g in baseline[record['image_id']]['matched'] for g in targets),
                    'detected': sum(g in matched for g in targets), 'dpo_grad_norm': nd, 'native_grad_norm': nn,
                    'weighted_dpo_native_norm_ratio_same_image': .01*nd/max(nn, 1e-30)}
            for bucket_id, bucket_name in enumerate(('primary_rescue', 'other_rescue_compatible', 'remaining')):
                for kind_id, kind_name in enumerate(('class', 'box')):
                    values = parts[buckets == bucket_id, kind_id].detach()
                    pair[f'{bucket_name}_{kind_name}_delta'] = float(values.sum())
                    pair[f'{bucket_name}_{kind_name}_delta_abs_sum'] = float(values.abs().sum())
            rejected_by_slot = {i: (c, bins) for i, c, bins in zip(record['rejected']['indices'], record['rejected']['classes'], record['rejected']['bins'])}
            for g in targets:
                slot, c = primary[g], int(target['labels'][g])
                gt = target['boxes'][g:g+1].to(device)
                score = cls[slot, c].sigmoid()
                iou = box_iou(decoded[slot:slot+1], gt)[0, 0]
                gs = grads(score, params)
                gi_all = grads(iou, params+[box])
                gi, gi_box = gi_all[:-1], gi_all[-1]
                local_box = -float((gi_box[slot].detach().double()*gb[slot].detach().double()).sum())
                score_local = -float(gc[slot, c].detach()*score.detach()*(1-score.detach()))
                row = {'stage': stage, 'image_id': record['image_id'], 'probe': record['probe'],
                       'gt_index': g, 'class_id': c, 'slot': slot,
                       's0_missed': g not in baseline[record['image_id']]['matched'], 'detected': g in matched,
                       'score': float(score.detach()), 'expected_box_iou': float(iou.detach()),
                       'ordinary_argmax_correct': int(cls[slot].argmax()) == c,
                       'sampled_chosen_iou': match[g][1],
                       'rejected_same_class_at_primary': slot in rejected_by_slot and rejected_by_slot[slot][0] == c,
                       'rejected_background_at_primary': slot not in rejected_by_slot,
                       'direct_output_DPO_score_slope': score_local,
                       'direct_output_DPO_box_iou_slope': local_box,
                       'DPO_param_unit_score_slope': -dot(gs, gd)/max(nd, 1e-30),
                       'DPO_param_unit_box_iou_slope': -dot(gi, gd)/max(nd, 1e-30),
                       'native_param_unit_score_slope': -dot(gs, gn)/max(nn, 1e-30),
                       'native_param_unit_box_iou_slope': -dot(gi, gn)/max(nn, 1e-30),
                       'combined_param_unit_score_slope': -dot(gs, combined)/max(nc, 1e-30),
                       'combined_param_unit_box_iou_slope': -dot(gi, combined)/max(nc, 1e-30)}
                if row['rejected_same_class_at_primary']:
                    assert abs(score_local) < 1e-9
                gt_rows.append(row)
            pair_rows.append(pair)
            # Numerical first-order check on one pair per checkpoint. Only
            # temporary in-memory weights change; restore the exact tensors.
            if record_index == 0:
                finite_checks.extend({'stage': stage, **check} for check in verify_parameter_direction(
                    policy, reference, images, target, record, primary, dataset.policy_config, params))
                assert tensor_digest(policy.model) == before
            if (record_index+1) % 25 == 0 or record_index+1 == len(records):
                print(f'{stage}: {record_index+1}/{len(records)} recovery images audited', flush=True)
        assert tensor_digest(policy.model) == before
        atomic_json(output/f'{stage}_progress.json', {'pairs': [r for r in pair_rows if r['stage'] == stage],
                    'targets': [r for r in gt_rows if r['stage'] == stage]})
        del policy
    assert tensor_digest(reference.model) == reference_digest
    metrics = ['score', 'expected_box_iou', 'detected', 'direct_output_DPO_score_slope',
               'direct_output_DPO_box_iou_slope', 'DPO_param_unit_score_slope', 'DPO_param_unit_box_iou_slope',
               'native_param_unit_score_slope', 'native_param_unit_box_iou_slope',
               'combined_param_unit_score_slope', 'combined_param_unit_box_iou_slope']
    summary = summarize(gt_rows, ['stage', 'probe', 's0_missed'], metrics)
    class_summary = summarize(gt_rows, ['stage', 'probe', 'class_id', 's0_missed'], metrics)
    pair_summary = summarize(pair_rows, ['stage', 'probe'], [
        'margin', 'dpo_loss', 'chosen_logp_delta', 'rejected_logp_delta',
        *[f'{bucket}_{kind}_delta{suffix}' for bucket in ('primary_rescue', 'other_rescue_compatible', 'remaining')
          for kind in ('class', 'box') for suffix in ('', '_abs_sum')]])
    checks.update({'checkpoint_data_base_recovery_code_hashes_verified': True,
                   'all_primary_slots_replay_reference_render_and_GT_matching': True,
                   'all_expected_boxes_match_ordinary_DFL_decoder': True,
                   'all_margin_decompositions_exact': True,
                   'finite_difference_parameter_direction_checks': finite_checks,
                   'all_policy_reference_weights_unchanged': True,
                   'no_optimizer_step_or_checkpoint_written': True})
    for stem in ('best', 'last'):
        assert sha256(folder/f'{stem}.pt') == done[f'{stem}_sha256']
    result = {'run': str(root), 'script_sha256': sha256(Path(__file__)),
              'scope': 'all 267 selected recovery images / 328 target GT; train images only, no valid/test',
              'gradient_protocol': 'single-image native and DPO; first-order Euclidean parameter descent, no AdamW update; fixed S0 chosen slots',
              'summary': summary, 'pair_summary': pair_summary, 'class_summary': class_summary, 'checks': checks}
    atomic_json(output/'audit.json', result)
    write_csv(output/'pairs.csv', pair_rows)
    write_csv(output/'targets.csv', gt_rows)
    write_csv(output/'summary.csv', summary)
    write_csv(output/'classes.csv', class_summary)
    print(json.dumps({'summary': summary, 'pair_summary': pair_summary, 'checks': checks}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stages', nargs='+', default=['S0', 'best', 'last'])
    args = parser.parse_args()
    if args.stages[0] != 'S0' or any(s not in ('S0', 'best', 'last') for s in args.stages):
        parser.error('Stages must start with S0 and contain only S0/best/last')
    audit(args.root.resolve(), args.output.resolve(), args.device, args.stages)
