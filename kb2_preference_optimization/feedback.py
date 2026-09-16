# Feedback schema and matching utilities.
from collections import Counter
import torch

SCHEMA_VERSION = '1.0'
FEEDBACK_TYPES = ('matched', 'wrong_class', 'bad_localization',
                  'false_positive', 'duplicate', 'missed')


def box_iou(a, b):
    if not len(a) or not len(b):
        return torch.zeros((len(a), len(b)))
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(2)
    a1 = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(1)
    a2 = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(1)
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-8)


def _prediction(pred, index, iou=0.0, gt_index=None):
    return {
        'prediction_index': index, 'gt_index': gt_index,
        'box': [round(float(x), 4) for x in pred['boxes'][index]],
        'class_id': int(pred['labels'][index]),
        'confidence': round(float(pred['scores'][index]), 6),
        'iou': round(float(iou), 6),
    }


def build_feedback_record(pred, target, match_iou=0.5, localization_iou=0.1):
    pb = pred['boxes'].detach().float().cpu()
    pc = pred['labels'].detach().long().cpu()
    ps = pred['scores'].detach().float().cpu()
    gb = target['boxes'].detach().float().cpu()
    gc = target['labels'].detach().long().cpu()
    clean = torch.isfinite(pb).all(1) & torch.isfinite(ps)
    pb, pc, ps = pb[clean], pc[clean], ps[clean]
    pred = {'boxes': pb, 'labels': pc, 'scores': ps}
    ious = box_iou(pb, gb)
    groups = {key: [] for key in FEEDBACK_TYPES}
    matched_gt, primary = set(), {}
    for pi in ps.argsort(descending=True).tolist():
        if not len(gb):
            groups['false_positive'].append(_prediction(pred, pi))
            continue
        gi = int(ious[pi].argmax())
        iou = float(ious[pi, gi])
        same_class = int(pc[pi]) == int(gc[gi])
        item = _prediction(pred, pi, iou, gi)
        if iou >= match_iou and same_class:
            if gi in matched_gt:
                groups['duplicate'].append(item)
            else:
                groups['matched'].append(item)
                matched_gt.add(gi)
                primary[gi] = pi
        elif iou >= match_iou:
            groups['wrong_class'].append(item)
        elif iou >= localization_iou:
            item['expected_class_id'] = int(gc[gi])
            groups['bad_localization'].append(item)
        else:
            groups['false_positive'].append(item)

    for gi in range(len(gb)):
        if gi not in matched_gt:
            groups['missed'].append({
                'gt_index': gi,
                'box': [round(float(x), 4) for x in gb[gi]],
                'class_id': int(gc[gi]),
            })

    preferences = []
    for gi, chosen in primary.items():
        candidates = [i for i in range(len(pb)) if i != chosen]
        if candidates:
            rejected = max(candidates, key=lambda i: float(ps[i]))
            preferences.append({
                'gt_index': gi,
                'chosen_prediction_index': chosen,
                'rejected_prediction_index': rejected,
                'chosen_iou': round(float(ious[chosen, gi]), 6),
                'rejected_iou': round(float(ious[rejected, gi]), 6),
            })
    return {
        'schema_version': SCHEMA_VERSION,
        'image_id': int(target['image_id'].flatten()[0]),
        'image_path': target.get('image_path', ''),
        'num_ground_truths': len(gb),
        'num_predictions': len(pb),
        'feedback': groups,
        'preferences': preferences,
    }


def summarize_records(records):
    counts, images_with = Counter(), Counter()
    for record in records:
        for kind in FEEDBACK_TYPES:
            n = len(record['feedback'][kind])
            counts[kind] += n
            images_with[kind] += int(n > 0)
    return {
        'schema_version': SCHEMA_VERSION,
        'num_images': len(records),
        'feedback_counts': dict(counts),
        'images_with_feedback': dict(images_with),
        'num_preferences': sum(len(r['preferences']) for r in records),
    }
