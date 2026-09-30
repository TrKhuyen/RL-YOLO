# Feedback schema and matching utilities.
from collections import Counter
import torch

SCHEMA_VERSION = '2.1'
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


def _maximum_cardinality_matches(choices, scores):
    """Deterministic bipartite matching, maximizing the number of covered GTs."""
    by_gt = {}

    def assign(prediction_index, seen):
        for gt_index in choices[prediction_index]:
            if gt_index in seen:
                continue
            seen.add(gt_index)
            if gt_index not in by_gt or assign(by_gt[gt_index], seen):
                by_gt[gt_index] = prediction_index
                return True
        return False

    for prediction_index in sorted(
            choices, key=lambda index: (-float(scores[index]), index)):
        assign(prediction_index, set())
    return by_gt


def build_feedback_record(pred, target, match_iou=0.5, localization_iou=0.1):
    """Assign one primary status per GT, then classify remaining predictions."""
    if not 0 <= localization_iou < match_iou <= 1:
        raise ValueError('expected 0 <= localization_iou < match_iou <= 1')
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
    used_predictions = set()
    gt_status = [None] * len(gb)
    primary = {}

    # First maximize the number of correct-class primary matches. A greedy
    # prediction-first choice can leave a nearby GT falsely missed.
    correct_choices = {
        pi: sorted(
            (gi for gi in range(len(gb))
             if int(pc[pi]) == int(gc[gi]) and float(ious[pi, gi]) >= match_iou),
            key=lambda gi: (-float(ious[pi, gi]), gi))
        for pi in range(len(pb))
    }
    primary = _maximum_cardinality_matches(correct_choices, ps)
    used_predictions.update(primary.values())
    for gi, pi in sorted(primary.items()):
        gt_status[gi] = 'matched'
        groups['matched'].append(_prediction(pred, pi, ious[pi, gi], gi))

    # Remaining GTs receive at most one error explanation each.
    for kind in ('wrong_class', 'bad_localization'):
        choices = {}
        for pi in range(len(pb)):
            if pi in used_predictions:
                continue
            candidates = []
            for gi in range(len(gb)):
                if gt_status[gi] is not None:
                    continue
                overlap = float(ious[pi, gi])
                if kind == 'wrong_class':
                    eligible = overlap >= match_iou and int(pc[pi]) != int(gc[gi])
                else:
                    eligible = localization_iou <= overlap < match_iou
                if eligible:
                    candidates.append(gi)
            choices[pi] = sorted(
                candidates, key=lambda gi: (-float(ious[pi, gi]), gi))
        selected = _maximum_cardinality_matches(choices, ps)
        for gi, pi in sorted(selected.items()):
            used_predictions.add(pi)
            gt_status[gi] = kind
            item = _prediction(pred, pi, ious[pi, gi], gi)
            if kind == 'bad_localization':
                item['expected_class_id'] = int(gc[gi])
            groups[kind].append(item)

    for gi, status in enumerate(gt_status):
        if status is None:
            gt_status[gi] = 'missed'
            groups['missed'].append({
                'gt_index': gi,
                'box': [round(float(x), 4) for x in gb[gi]],
                'class_id': int(gc[gi]),
            })

    for pi in range(len(pb)):
        if pi in used_predictions:
            continue
        duplicate_gts = [
            gi for gi in primary
            if int(pc[pi]) == int(gc[gi]) and float(ious[pi, gi]) >= match_iou
        ]
        if duplicate_gts:
            gi = max(duplicate_gts, key=lambda index: (float(ious[pi, index]), -index))
            groups['duplicate'].append(_prediction(pred, pi, ious[pi, gi], gi))
        else:
            gi = int(ious[pi].argmax()) if len(gb) else None
            overlap = float(ious[pi, gi]) if gi is not None else 0.0
            groups['false_positive'].append(_prediction(pred, pi, overlap, gi))

    preferences = []
    for gi, chosen in sorted(primary.items()):
        rejected_candidates = [
            pi for pi in range(len(pb))
            if pi != chosen and pi not in used_predictions
            and float(ious[pi, gi]) >= localization_iou
            and (
                int(pc[pi]) != int(gc[gi])
                or float(ious[pi, gi]) < float(ious[chosen, gi])
            )
        ]
        if rejected_candidates:
            rejected = max(
                rejected_candidates,
                key=lambda pi: (float(ious[pi, gi]), float(ps[pi]), -pi))
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
        'gt_status': gt_status,
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
