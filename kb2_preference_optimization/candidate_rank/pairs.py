"""Swap two existing S0 boxes for one GT; judge the whole image after fixed NMS."""
from dataclasses import asdict, dataclass

import torch
from torchvision.ops import box_iou

from ..conditional.pairs import PairConfig, matches, render


@dataclass(frozen=True)
class RankConfig:
    confidence: float = .25
    nms_iou: float = .6
    match_iou: float = .5
    max_det: int = 300
    max_source_iou: float = .85
    min_iou_gain: float = .1
    min_total_iou_gain: float = .05
    attempts_per_gt: int = 3
    max_pairs_per_image: int = 4

    def manifest(self):
        return asdict(self)


def render_swap(context, good, bad, choose_good, config=RankConfig()):
    if good == bad:
        raise ValueError('Swap requires two distinct candidates')
    positions = {slot: i for i, slot in enumerate(context['slots'])}
    if good not in positions or bad not in positions:
        raise ValueError('Swap candidates must both exist in the immutable context')
    active = list(context['active'])
    active[positions[good]] = bool(choose_good)
    active[positions[bad]] = not choose_good
    return render({**context, 'active': active}, config=PairConfig(
        nms_iou=config.nms_iou, max_det=config.max_det))


def matched_iou(prediction, target, assignment):
    overlaps = box_iou(prediction['boxes'], target['boxes'])
    return {g: float(overlaps[p, g]) for p, g in assignment.items()}


def judge_swap(context, target, good, bad, target_gt, config=RankConfig()):
    a = render_swap(context, good, bad, True, config)
    b = render_swap(context, good, bad, False, config)
    ag, afp, aa = matches(a, target, config.confidence, config.match_iou)
    bg, bfp, ba = matches(b, target, config.confidence, config.match_iou)
    ai, bi = matched_iou(a, target, aa), matched_iou(b, target, ba)
    actual_good = any(g == target_gt and int(a['slots'][p]) == good for p, g in aa.items())
    actual_bad = any(g == target_gt and int(b['slots'][p]) == bad for p, g in ba.items())
    gain = ai.get(target_gt, 0.)-bi.get(target_gt, 0.)
    valid = (actual_good and actual_bad and bg <= ag and afp <= bfp
             and gain >= config.min_iou_gain-1e-6
             and sum(ai.values())-sum(bi.values()) >= config.min_total_iou_gain-1e-6)
    thresholds = []
    for i in range(10):
        threshold = .5+.05*i
        at, _, _ = matches(a, target, config.confidence, threshold)
        bt, _, _ = matches(b, target, config.confidence, threshold)
        thresholds.append({'iou': round(threshold, 2), 'chosen_gt': sorted(at), 'rejected_gt': sorted(bt)})
        valid &= bt <= at
    evidence = {'chosen_gt': sorted(ag), 'rejected_gt': sorted(bg),
                'chosen_fp': afp, 'rejected_fp': bfp,
                'chosen_target_iou': ai.get(target_gt), 'rejected_target_iou': bi.get(target_gt),
                'target_iou_gain': gain, 'total_matched_iou_gain': sum(ai.values())-sum(bi.values()),
                'threshold_coverage': thresholds,
                'chosen_slots': a['slots'].tolist(), 'rejected_slots': b['slots'].tolist()}
    return bool(valid), evidence


def build_rank_pairs(boxes, logits, target, config=RankConfig()):
    boxes, logits = boxes.detach().cpu().float(), logits.detach().cpu().float()
    scores, labels = logits.sigmoid().max(-1)
    slots = (scores >= config.confidence).nonzero().flatten()
    context = {'slots': slots.tolist(), 'boxes': boxes[slots].tolist(),
               'scores': scores[slots].tolist(), 'labels': labels[slots].tolist(),
               'active': [True]*len(slots)}
    result = build_rank_context(context, target, config)
    for record in result[1]:
        record['good_logit'] = float(logits[record['good_slot'], record['class_id']])
        record['bad_logit'] = float(logits[record['bad_slot'], record['class_id']])
    return result


def build_rank_context(source_context, target, config=RankConfig()):
    """Reuse complete verified V0 pre-NMS contexts; never invent absent class logits."""
    indices = [i for i, (active, score) in enumerate(zip(source_context['active'], source_context['scores']))
               if active and score >= config.confidence]
    context = {key: [source_context[key][i] for i in indices]
               for key in ('slots', 'boxes', 'scores', 'labels')}
    context['active'] = [True]*len(indices)
    slots = torch.tensor(context['slots'], dtype=torch.long)
    boxes = torch.tensor(context['boxes'], dtype=torch.float32).reshape(-1, 4)
    scores = torch.tensor(context['scores'], dtype=torch.float32)
    labels = torch.tensor(context['labels'], dtype=torch.long)
    positions = {slot: i for i, slot in enumerate(context['slots'])}
    ordinary = render(context, config=PairConfig(nms_iou=config.nms_iou, max_det=config.max_det))
    found, fp, assignments = matches(ordinary, target, config.confidence, config.match_iou)
    overlaps = box_iou(boxes, target['boxes'])
    ordinary_slots = set(ordinary['slots'].tolist())
    diagnostics = {'s0_gt': sorted(found), 's0_fp': fp, 'gt': len(target['labels']),
                   'matched_gt': len(found), 'eligible_source_gt': 0,
                   'gt_with_better_suppressed_candidate': 0, 'attempted_pairs': 0,
                   'gate_rejected': 0, 'valid_before_cap': 0}
    pairs, gt_rows = [], []
    for g, c in enumerate(target['labels'].tolist()):
        row = {'target_gt': g, 'class_id': c, 's0_detected': g in found, 'status': 'S0_missed'}
        if g not in found:
            gt_rows.append(row)
            continue
        p = next(p for p, gt in assignments.items() if gt == g)
        bad_slot = int(ordinary['slots'][p])
        bad = positions[bad_slot]
        source_iou = float(overlaps[bad, g])
        row.update(source_iou=source_iou, source_slot=bad_slot, status='source_already_tight')
        if source_iou > config.max_source_iou:
            gt_rows.append(row)
            continue
        diagnostics['eligible_source_gt'] += 1
        inter = box_iou(boxes, boxes[bad:bad+1]).flatten()
        candidates = ((labels == c) & (scores <= scores[bad])
                      & (overlaps[:, g] >= source_iou+config.min_iou_gain-1e-6)
                      & (inter > config.nms_iou)).nonzero().flatten()
        candidates = torch.tensor([j for j in candidates.tolist() if int(slots[j]) not in ordinary_slots], dtype=torch.long)
        row['status'] = 'no_better_suppressed_candidate'
        if len(candidates):
            diagnostics['gt_with_better_suppressed_candidate'] += 1
            row['status'] = 'whole_image_gate_rejected'
        order = overlaps[candidates, g].argsort(descending=True)[:config.attempts_per_gt]
        for good in candidates[order].tolist():
            good_slot = int(slots[good])
            diagnostics['attempted_pairs'] += 1
            rejected = render_swap(context, good_slot, bad_slot, False, config)
            if rejected['slots'].tolist() != ordinary['slots'].tolist():
                diagnostics['gate_rejected'] += 1
                continue
            valid, evidence = judge_swap(context, target, good_slot, bad_slot, g, config)
            if not valid:
                diagnostics['gate_rejected'] += 1
                continue
            diagnostics['valid_before_cap'] += 1
            row['status'] = 'valid_capped' if len(pairs) >= config.max_pairs_per_image else 'accepted'
            if len(pairs) < config.max_pairs_per_image:
                pairs.append({'good_slot': good_slot, 'bad_slot': bad_slot, 'class_id': c, 'target_gt': g,
                              'good_box': boxes[good].tolist(), 'bad_box': boxes[bad].tolist(),
                              'good_score': float(scores[good]), 'bad_score': float(scores[bad]),
                              'evidence': evidence})
            break
        gt_rows.append(row)
    return context, pairs, diagnostics, gt_rows
