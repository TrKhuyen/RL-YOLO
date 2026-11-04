"""Full-image GT judging of one changed emission, with fixed S0 priorities."""
from dataclasses import asdict, dataclass

import torch
from torchvision.ops import batched_nms, box_iou


@dataclass(frozen=True)
class PairConfig:
    operating_conf: float = .25
    match_iou: float = .5
    recovery_min_score: float = .05
    recovery_min_iou: float = .6
    background_max_iou: float = .1
    nms_iou: float = .6
    max_det: int = 300
    attempts_per_gt: int = 3
    max_pairs_per_group: int = 4

    def manifest(self):
        return asdict(self)


def render(context, slot=None, keep=None, config=PairConfig()):
    """Keep bypasses confidence, but never modifies a box/class/S0 score.

    Context stores every ordinary emitted candidate plus proposed low-score
    candidates. Only one active bit changes. NMS and cap run on the whole image.
    """
    active = torch.tensor(context['active'], dtype=torch.bool)
    slots = torch.tensor(context['slots'], dtype=torch.long)
    if slot is not None:
        positions = (slots == slot).nonzero().flatten()
        if len(positions) != 1:
            raise ValueError('Decision must reference exactly one context candidate')
        active[positions[0]] = bool(keep)
    indices = active.nonzero().flatten()
    boxes = torch.tensor(context['boxes'], dtype=torch.float32).reshape(-1, 4)
    scores = torch.tensor(context['scores'], dtype=torch.float32)
    labels = torch.tensor(context['labels'], dtype=torch.long)
    # CPU float32 uses the same torchvision NMS as canonical inference.
    selected = batched_nms(boxes[indices], scores[indices], labels[indices], config.nms_iou)
    indices = indices[selected[:config.max_det]]
    return {'boxes': boxes[indices], 'scores': scores[indices],
            'labels': labels[indices], 'slots': slots[indices]}


def matches(prediction, target, conf=0., iou=.5):
    """Canonical confidence-ordered class-aware greedy matching, exposing IDs."""
    gt = target['boxes'].cpu().float()
    labels = target['labels'].cpu().long()
    overlaps = box_iou(prediction['boxes'].cpu().float(), gt)
    matched, assignments = set(), {}
    for p in prediction['scores'].argsort(descending=True).tolist():
        if float(prediction['scores'][p]) < conf:
            continue
        eligible = [g for g in range(len(gt)) if g not in matched
                    and int(labels[g]) == int(prediction['labels'][p])]
        if eligible:
            values = overlaps[p, eligible]
            g = eligible[int(values.argmax())]
            if float(overlaps[p, g]) >= iou:
                matched.add(g)
                assignments[p] = g
    fp = int((prediction['scores'] >= conf).sum()) - len(matched)
    return matched, fp, assignments


def judge(context, target, slot, chosen_keep, target_gt, config=PairConfig()):
    chosen = render(context, slot, chosen_keep, config)
    rejected = render(context, slot, not chosen_keep, config)
    cg, cfp, _ = matches(chosen, target, iou=config.match_iou)
    rg, rfp, _ = matches(rejected, target, iou=config.match_iou)
    valid = rg <= cg and cfp <= rfp
    if chosen_keep:
        valid &= target_gt is not None and target_gt in cg - rg
    else:
        valid &= cfp < rfp
    evidence = {'chosen_gt': sorted(cg), 'rejected_gt': sorted(rg),
                'chosen_fp': cfp, 'rejected_fp': rfp,
                'chosen_slots': chosen['slots'].tolist(),
                'rejected_slots': rejected['slots'].tolist()}
    return bool(valid), evidence


def build_pairs(boxes, logits, target, config=PairConfig()):
    """Proposals come only from S0 outputs. GT only selects and judges them."""
    boxes, logits = boxes.detach().cpu().float(), logits.detach().cpu().float()
    scores, labels = logits.sigmoid().max(-1)
    active = scores >= config.operating_conf
    overlaps = box_iou(boxes, target['boxes'].cpu().float())
    low_slots, attempts = set(), []
    # Start from ordinary S0 output; low-confidence choices are inactive in C.
    base_context = {'slots': active.nonzero().flatten().tolist(),
                    'boxes': boxes[active].tolist(), 'scores': scores[active].tolist(),
                    'labels': labels[active].tolist(), 'active': [True] * int(active.sum())}
    ordinary = render(base_context, config=config)
    ordinary_gt, ordinary_fp, assigned = matches(ordinary, target, iou=config.match_iou)
    for g, c in enumerate(target['labels'].tolist()):
        if g in ordinary_gt:
            # One-to-one matched source; another common detection may replace it.
            p = next(p for p, row in assigned.items() if row == g)
            attempts.append(('preservation', int(ordinary['slots'][p]), g))
            continue
        eligible = ((labels == c) & ~active & (scores >= config.recovery_min_score)
                    & (overlaps[:, g] >= config.recovery_min_iou))
        indices = eligible.nonzero().flatten()
        order = scores[indices].argsort(descending=True)[:config.attempts_per_gt]
        for j in indices[order].tolist():
            low_slots.add(j)
            attempts.append(('recovery', j, g))
    for p, j in enumerate(ordinary['slots'].tolist()):
        if p in assigned:
            continue
        same = ((target['labels'] == labels[j]) & (overlaps[j] >= config.match_iou)).nonzero().flatten()
        duplicates = [g for g in same.tolist() if g in ordinary_gt]
        if duplicates:
            attempts.append(('duplicate', j, duplicates[0]))
        else:
            # Avoid misclassifications / localization errors / boxes inside leaves.
            centers = (boxes[j, :2] + boxes[j, 2:]) / 2
            gt_boxes = target['boxes']
            inside = ((centers >= gt_boxes[:, :2]).all(-1)
                      & (centers <= gt_boxes[:, 2:]).all(-1)).any()
            if (not len(gt_boxes) or float(overlaps[j].max()) <= config.background_max_iou) and not inside:
                attempts.append(('background', j, None))
    all_slots = sorted(set(base_context['slots']) | low_slots)
    context = {'slots': all_slots, 'boxes': boxes[all_slots].tolist(),
               'scores': scores[all_slots].tolist(), 'labels': labels[all_slots].tolist(),
               'active': active[all_slots].tolist()}
    accepted, counts, seen_gt = [], {}, set()
    rejected_counts = {}
    for group, j, g in attempts:
        if counts.get(group, 0) >= config.max_pairs_per_group or (group == 'recovery' and g in seen_gt):
            continue
        keep = group in ('recovery', 'preservation')
        valid, evidence = judge(context, target, j, keep, g, config)
        if not valid:
            rejected_counts[group] = rejected_counts.get(group, 0) + 1
            continue
        accepted.append({'slot': j, 'class_id': int(labels[j]), 'group': group,
                         'chosen_keep': keep, 'target_gt': g,
                         'box': boxes[j].tolist(), 'reference_logit': float(logits[j, labels[j]]),
                         'evidence': evidence})
        counts[group] = counts.get(group, 0) + 1
        if group == 'recovery':
            seen_gt.add(g)
    return context, accepted, {'s0_gt': sorted(ordinary_gt), 's0_fp': ordinary_fp,
                               's0_fn': len(target['labels']) - len(ordinary_gt),
                               'attempts': len(attempts), 'gate_rejected': rejected_counts}
