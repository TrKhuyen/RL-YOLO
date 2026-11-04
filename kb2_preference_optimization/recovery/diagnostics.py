"""Fixed image/GT probes: preference margins and ordinary inference together."""
from collections import Counter
import math

import torch
from torchvision.ops import box_iou

from ..quality import match_boxes
from ..train_dpo import pair_losses


def choose_probes(records, probe, maximum=64):
    candidates = [r for r in records if r['probe'] == probe]
    selected, seen = [], set()
    by_class = {}
    for r in candidates:
        for c in r['selection']['rescued_classes']:
            by_class.setdefault(c, []).append(r)
    # Round robin over classes, unique images. No validation-based priority.
    position = 0
    while len(selected) < maximum:
        added = False
        for c in sorted(by_class):
            if position < len(by_class[c]):
                r = by_class[c][position]
                if r['image_id'] not in seen:
                    selected.append(r)
                    seen.add(r['image_id'])
                    added = True
                    if len(selected) == maximum:
                        break
        if not added and all(position >= len(v) for v in by_class.values()):
            break
        position += 1
    for r in candidates:
        if len(selected) >= maximum:
            break
        if r['image_id'] not in seen:
            selected.append(r)
            seen.add(r['image_id'])
    return selected


def canonical_gt_ids(prediction, target):
    keep = prediction['scores'] >= .25
    boxes, labels, scores = (prediction[k][keep] for k in ('boxes', 'labels', 'scores'))
    matched = set()
    ious = box_iou(target['boxes'].float(), boxes.float()) if len(boxes) else None
    for p in scores.argsort(descending=True).tolist():
        eligible = [g for g, label in enumerate(target['labels'].tolist())
                    if g not in matched and label == int(labels[p])]
        if eligible:
            values = ious[eligible, p]
            best = int(values.argmax())
            if float(values[best]) >= .5:
                matched.add(eligible[best])
    return matched


@torch.no_grad()
def probe_audit(policy, reference, dataset, records, beta=.1):
    counts = Counter()
    by_class = Counter()
    margins, losses, rows = [], [], []
    for record in records:
        image, target = dataset[record['image_id']]
        images = image[None].to(policy.device)
        pd, _, _, _ = policy.distributions(images)
        rd, _, _, _ = reference.distributions(images)
        loss, diagnostic = pair_losses(pd, rd, [record], beta)
        losses.append(float(loss))
        margins.append(diagnostic['relative_margin'])
        prediction = policy.predict(images, .001, .60, 300)[0]
        matched = canonical_gt_ids(prediction, target)
        op = {k: v[prediction['scores'] >= .25] for k, v in prediction.items()}
        tp = len(matched)
        counts.update(tp=tp, fp=len(op['boxes'])-tp, fn=len(target['labels'])-tp)
        rescued = record['selection']['rescued_gt']
        raw = policy.model(images)[0][0]
        xywh = raw[:4].T
        boxes = torch.cat((xywh[:, :2]-xywh[:, 2:]/2, xywh[:, :2]+xywh[:, 2:]/2), -1)
        details = []
        for g in rescued:
            c = int(target['labels'][g])
            ious = box_iou(target['boxes'][g:g+1].to(policy.device), boxes)[0]
            eligible = ious >= .5
            score = float(raw[4+c, eligible].max()) if eligible.any() else 0.
            by_class[(c, 'rescued_gt')] += 1
            by_class[(c, 'detected')] += g in matched
            by_class[(c, 'score_sum')] += score
            counts['rescued_gt'] += 1
            counts['rescued_detected'] += g in matched
            counts['rescued_score_sum'] += score
            details.append({'gt': g, 'class_id': c, 'detected': g in matched,
                            'best_true_class_score_at_iou50': score})
        rows.append({'image_id': record['image_id'], 'margin': diagnostic['relative_margin'],
                     'dpo_loss': float(loss), 'rescued_gt': details})
    number = max(len(records), 1)
    return {'scope': 'train_images_only; DPO-unseen subset excluded from DPO; both subsets used by native',
            'pairs': len(records), 'dpo_loss': sum(losses)/number,
            'relative_margin': sum(margins)/number,
            'relative_win_rate': sum(m > 0 for m in margins)/number,
            'tp': counts['tp'], 'fp': counts['fp'], 'fn': counts['fn'],
            'recall': counts['tp']/max(counts['tp']+counts['fn'], 1),
            'rescued_gt': counts['rescued_gt'], 'rescued_detected': counts['rescued_detected'],
            'rescued_recall': counts['rescued_detected']/max(counts['rescued_gt'], 1),
            'rescued_mean_true_class_score': counts['rescued_score_sum']/max(counts['rescued_gt'], 1),
            'per_class': {str(c): {'support': by_class[(c, 'rescued_gt')],
                                  'detected': by_class[(c, 'detected')],
                                  'mean_true_class_score': by_class[(c, 'score_sum')]/by_class[(c, 'rescued_gt')]}
                          for c in range(28) if by_class[(c, 'rescued_gt')]},
            'records': rows}
