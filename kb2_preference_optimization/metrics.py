"""Deterministic S0/DPO comparison under KB1's canonical inference contract."""
from collections import defaultdict
import time

import torch
from torchmetrics.detection import MeanAveragePrecision
from kb1_reward_guided_training.canonical_eval import _operating_point_counts, _summarize_counts
from .quality import match_boxes

EVALUATION_PROTOCOL = {
    'version': 'kb2_canonical_v1_ar300', 'conf': .001, 'nms_iou': .60, 'max_det': 300,
    'operating_conf': .25, 'operating_iou': .5, 'selection': 'mAP50_95',
    'area_coordinates': 'transformed_image', 'inference': 'ordinary_deterministic_yolo',
}


@torch.no_grad()
def evaluate_adapter(adapter, loader, device='cpu', image_limit=0):
    metric = MeanAveragePrecision(class_metrics=True, max_detection_thresholds=[1, 10, 300],
                                  backend='faster_coco_eval')
    counts = defaultdict(lambda: [0, 0, 0])
    images_seen = num_gt = matched = 0
    matched_iou_sum = 0.
    area_support = {'small': 0, 'medium': 0, 'large': 0}
    started = time.perf_counter()
    for images, targets in loader:
        if image_limit:
            remaining = image_limit - images_seen
            if remaining <= 0:
                break
            images, targets = images[:remaining], targets[:remaining]
        predictions = adapter.predict(images.to(device), .001, .60, 300)
        cpu_targets = [{'boxes': t['boxes'].float().cpu(), 'labels': t['labels'].long().cpu()}
                       for t in targets]
        metric.update(predictions, cpu_targets)
        for label, values in _operating_point_counts(predictions, cpu_targets, .25, .5).items():
            counts[label] = [a + b for a, b in zip(counts[label], values)]
        for prediction, target in zip(predictions, cpu_targets):
            op = {k: v[prediction['scores'] >= .25] for k, v in prediction.items()}
            geometry_matches = match_boxes(op, target, .5)
            matched += len(geometry_matches)
            matched_iou_sum += sum(iou for _, _, iou in geometry_matches)
            num_gt += len(target['boxes'])
            areas = (target['boxes'][:, 2:] - target['boxes'][:, :2]).prod(-1)
            area_support['small'] += int((areas < 32 ** 2).sum())
            area_support['medium'] += int(((areas >= 32 ** 2) & (areas < 96 ** 2)).sum())
            area_support['large'] += int((areas >= 96 ** 2).sum())
        images_seen += len(images)
    if not images_seen:
        raise ValueError('Empty evaluation')
    result = metric.compute()
    precision, recall, f1, macro_f1 = _summarize_counts(counts)
    tp, fp, fn = (sum(v[i] for v in counts.values()) for i in range(3))
    per_class = {}
    for label, ap, ar in zip(result['classes'].reshape(-1).tolist(),
                             result['map_per_class'].reshape(-1).tolist(),
                             result['mar_300_per_class'].reshape(-1).tolist()):
        ctp, cfp, cfn = counts[label]
        per_class[str(label)] = {'AP50_95': ap, 'AR300': ar, 'tp': ctp, 'fp': cfp, 'fn': cfn,
                                 'gt_support': ctp + cfn}
    return {
        'mAP50_95': float(result['map']), 'mAP50': float(result['map_50']),
        'mAP75': float(result['map_75']), 'AR300': float(result['mar_300']),
        'APs': float(result['map_small']), 'APm': float(result['map_medium']),
        'APl': float(result['map_large']), 'precision': precision,
        'operating_recall': recall, 'f1': f1, 'macro_f1': macro_f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'fp_per_image': fp / images_seen,
        'fn_rate': fn / max(num_gt, 1),
        'matched_mean_iou': matched_iou_sum / matched if matched else None,
        'geometry_gt_coverage': matched / max(num_gt, 1),
        'gt_normalized_iou': matched_iou_sum / max(num_gt, 1),
        'geometry_matcher': 'class_aware_max_cardinality_then_iou',
        'operating_matcher': 'confidence_ordered_greedy_same_as_kb1',
        'area_gt_support': area_support, 'per_class': per_class,
        'num_images': images_seen, 'num_gt': num_gt,
        'evaluation_wall_seconds': time.perf_counter() - started,
        'protocol': EVALUATION_PROTOCOL, 'partial_evaluation': bool(image_limit),
    }
