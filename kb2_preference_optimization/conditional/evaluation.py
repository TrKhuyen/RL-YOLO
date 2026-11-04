"""Ordinary validation, source-GT transitions and fixed-slot train diagnostics."""
from collections import Counter, defaultdict

import torch
from torchvision.ops import box_iou

from ..metrics import evaluate_adapter
from .pairs import matches
from .policy import pair_loss


def macro_recall(metrics):
    values = [v['tp'] / v['gt_support'] for v in metrics['per_class'].values() if v['gt_support']]
    return sum(values) / max(len(values), 1)


def eligible(metrics, baseline, fp_tolerance=.05, ap_tolerance=.001):
    return (metrics['operating_recall'] >= baseline['operating_recall'] - 1e-12
            and metrics['fp'] <= baseline['fp'] * (1 + fp_tolerance) + 1e-12
            and metrics['mAP50_95'] >= baseline['mAP50_95'] - ap_tolerance)


@torch.no_grad()
def evaluate(adapter, loader, source=None, image_limit=0):
    cached = []

    class Recorder:
        def predict(self, images, *args):
            predictions = adapter.predict(images, *args)
            cached.extend(predictions)
            return predictions

    metrics = evaluate_adapter(Recorder(), loader, str(adapter.device), image_limit)
    metrics['macro_recall'] = macro_recall(metrics)
    # Canonical inference protocol stays unchanged. Selection is a separate rule.
    metrics['checkpoint_selection'] = 'macro_recall_subject_to_S0_micro_recall_FP_AP_gates'
    aggregate, classes, density, sizes, overlap_bins = (defaultdict(Counter) for _ in range(5))
    common_iou = []
    thresholds = (.05, .10, .15, .20, .25, .30, .35, .40, .50, .60, .70, .80, .90)
    curve = {threshold: Counter() for threshold in thresholds}
    source_fp_budget = 0
    index = 0
    for _, targets in loader:
        for target in targets:
            if index == len(cached):
                break
            prediction = cached[index]
            baseline = prediction if source is None else source[index]
            pg, _, pa = matches(prediction, target, .25)
            sg, source_fp, sa = matches(baseline, target, .25)
            source_fp_budget += source_fp
            for threshold in thresholds:
                found, false_positives, _ = matches(prediction, target, threshold)
                curve[threshold].update(tp=len(found), fp=false_positives, gt=len(target['labels']))
            n = len(target['labels'])
            d = '1-2' if n <= 2 else '3-5' if n <= 5 else '6+'
            overlaps = box_iou(target['boxes'], target['boxes'])
            overlaps.fill_diagonal_(0)
            for g, c in enumerate(target['labels'].tolist()):
                event = ('retained' if g in pg else 'lost') if g in sg else ('recovered' if g in pg else 'persistent_miss')
                area = float((target['boxes'][g, 2:] - target['boxes'][g, :2]).prod())
                size = 'small' if area < 32**2 else 'medium' if area < 96**2 else 'large'
                overlap = 'iou>=.3' if float(overlaps[g].max()) >= .3 else 'iou<.3'
                for counter in (aggregate['all'], classes[str(c)], density[d], sizes[size], overlap_bins[overlap]):
                    counter.update({event: 1, 'gt': 1})
                if g in pg & sg:
                    p = next(p for p, row in pa.items() if row == g)
                    s = next(p for p, row in sa.items() if row == g)
                    piou = float(box_iou(prediction['boxes'][p:p+1], target['boxes'][g:g+1])[0, 0])
                    siou = float(box_iou(baseline['boxes'][s:s+1], target['boxes'][g:g+1])[0, 0])
                    common_iou.append((siou, piou))
            index += 1
        if index == len(cached):
            break
    transitions = dict(aggregate['all'])
    transitions['net_recovery'] = transitions.get('recovered', 0) - transitions.get('lost', 0)
    transitions.update(per_class=dict(classes), density=dict(density), size=dict(sizes), gt_overlap=dict(overlap_bins))
    transitions['common_detected_mean_iou_S0'] = sum(v[0] for v in common_iou) / max(len(common_iou), 1)
    transitions['common_detected_mean_iou_policy'] = sum(v[1] for v in common_iou) / max(len(common_iou), 1)
    transitions['common_detected_support'] = len(common_iou)
    metrics['source_gt_transitions'] = transitions
    curve_records = [{'confidence': threshold, 'recall': counts['tp']/max(counts['gt'], 1),
                      'fp': counts['fp']} for threshold, counts in curve.items()]
    allowed = [point for point in curve_records if point['fp'] <= source_fp_budget]
    metrics['threshold_control'] = {
        'scope': 'validation diagnostic on a fixed coarse threshold grid; not calibration or test tuning',
        'S0_FP_budget_at_conf25': source_fp_budget,
        'max_recall_at_S0_FP_budget_on_grid': max((point['recall'] for point in allowed), default=None),
        'curve': curve_records}
    return metrics, cached


def choose_probes(records, maximum=32):
    # Class/group round robin, unique decisions; deterministic across arms.
    pools = defaultdict(lambda: defaultdict(list))
    for r in records:
        pools[r['group']][r['class_id']].append(r)
    by_group = {}
    for group, classes in pools.items():
        ordered, position = [], 0
        while len(ordered) < sum(map(len, classes.values())):
            for c in sorted(classes):
                if position < len(classes[c]):
                    ordered.append(classes[c][position])
            position += 1
        by_group[group] = ordered
    result, position = [], 0
    while len(result) < min(maximum, len(records)):
        for group in sorted(by_group):
            if position < len(by_group[group]):
                result.append(by_group[group][position])
                if len(result) == maximum:
                    return result
        position += 1
    return result


@torch.no_grad()
def probe(adapter, reference, bank, records, beta):
    rows, group_totals = [], defaultdict(Counter)
    for record in records:
        image, target = bank.dataset[record['image_id']]
        logits, boxes = adapter.raw(image[None])
        ref_logits, ref_boxes = reference.raw(image[None])
        r = {**record, 'batch_index': 0}
        loss, margin, z, z0 = pair_loss(logits, ref_logits, [r], beta)
        j, c, g = r['slot'], r['class_id'], r['target_gt']
        probabilities = logits[0, j].sigmoid()
        others = probabilities.clone()
        others[c] = -1
        ordinary = adapter.predict(image[None], .001, .6, 300)[0]
        detected, _, _ = matches(ordinary, target, .25)
        row = {'pair_id': r['pair_id'], 'image_id': r['image_id'], 'group': r['group'],
               'target_gt': g, 'class_id': c, 'slot': j, 'margin': float(margin[0]),
               'loss': float(loss), 'score': float(z.sigmoid()[0]), 's0_score': float(z0.sigmoid()[0]),
               'class_gap': float(probabilities[c] - others.max()),
               'argmax_class': int(probabilities.argmax()), 'ordinary_target_detected': g in detected if g is not None else None,
               'raw_box_max_abs_delta': float((boxes - ref_boxes).abs().max()),
               'slot_iou': float(box_iou(boxes[0, j:j+1].cpu(), target['boxes'][g:g+1])[0, 0]) if g is not None else None}
        rows.append(row)
        totals = group_totals[r['group']]
        totals.update(pairs=1, margin=row['margin'], score=row['score'], s0_score=row['s0_score'],
                      detected=int(row['ordinary_target_detected'] or False))
    return {'scope': bank.manifest['probe_scope'], 'records': rows,
            'groups': {k: {'pairs': v['pairs'], 'mean_margin': v['margin']/v['pairs'],
                           'mean_score': v['score']/v['pairs'], 'mean_s0_score': v['s0_score']/v['pairs'],
                           'ordinary_detected_targets': v['detected']} for k, v in group_totals.items()},
            'raw_box_max_abs_delta': max((r['raw_box_max_abs_delta'] for r in rows), default=0.)}
