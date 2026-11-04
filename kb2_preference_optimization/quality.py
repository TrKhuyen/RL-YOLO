"""GT-based whole-detection-set preferences (no confidence in quality labels)."""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou


@dataclass(frozen=True)
class QualityConfig:
    detection_weight: float = 0.8
    min_gap: float = 0.02
    f_beta: float = 2.0
    version: str = 'whole_set_fbeta_iou_v2'

    def __post_init__(self):
        if not 0 <= self.detection_weight <= 1:
            raise ValueError('detection_weight must be in [0,1]')
        if not math.isfinite(self.min_gap) or self.min_gap <= 0:
            raise ValueError('min_gap must be finite and positive')
        if not math.isfinite(self.f_beta) or self.f_beta <= 0:
            raise ValueError('f_beta must be finite and positive')
        if self.version != 'whole_set_fbeta_iou_v2':
            raise ValueError('Unsupported preference quality version')

    def to_dict(self):
        return asdict(self)


def match_boxes(prediction, target, threshold=0.5):
    """Maximum-cardinality class-aware matching; break ties by summed IoU.

    This is a preference/geometry matcher, not COCO's confidence-ordered matcher.
    """
    boxes = prediction['boxes'].detach().float().cpu()
    gt = target['boxes'].detach().float().cpu()
    labels = prediction['labels'].detach().long().cpu()
    gt_labels = target['labels'].detach().long().cpu()
    if not len(boxes) or not len(gt):
        return []
    ious = box_iou(boxes, gt).numpy()
    eligible = (labels[:, None] == gt_labels[None, :]).numpy() & (ious >= threshold)
    # A cardinality bonus larger than any possible total IoU difference.
    bonus = min(len(boxes), len(gt)) + 1
    weights = np.where(eligible, bonus + ious, 0.)
    rows, cols = linear_sum_assignment(-weights)
    return [(int(p), int(g), float(ious[p, g]))
            for p, g in zip(rows, cols) if eligible[p, g]]


def quality_score(prediction, target, config=QualityConfig()):
    n_pred, n_gt = len(prediction['boxes']), len(target['boxes'])
    if n_gt == 0:
        return {'score': 1.0 / (1 + n_pred), 'mean_f1': 1.0 / (1 + n_pred),
                'mean_f2': 1.0 / (1 + n_pred), 'mean_fbeta': 1.0 / (1 + n_pred),
                'gt_normalized_iou': 0., 'tp': 0, 'fp': n_pred, 'fn': 0,
                'matched_iou_sum': 0., 'num_gt': 0, 'num_predictions': n_pred}
    f1s, f2s, fbetas = [], [], []
    matches50 = None
    for i in range(10):
        matches = match_boxes(prediction, target, 0.5 + 0.05 * i)
        if i == 0:
            matches50 = matches
        # 2 TP + FP + FN == n_pred + n_gt.
        f1s.append(2 * len(matches) / (n_pred + n_gt))
        # (1 + beta^2) TP + beta^2 FN + FP == beta^2 Ngt + Npred.
        f2s.append(5 * len(matches) / (4 * n_gt + n_pred))
        beta_sq = config.f_beta ** 2
        fbetas.append((1 + beta_sq) * len(matches) / (beta_sq * n_gt + n_pred))
    tp = len(matches50)
    iou_sum = sum(v for _, _, v in matches50)
    mean_f1 = sum(f1s) / len(f1s)
    mean_f2 = sum(f2s) / len(f2s)
    mean_fbeta = sum(fbetas) / len(fbetas)
    coverage_iou = iou_sum / n_gt
    return {'score': config.detection_weight * mean_fbeta
            + (1 - config.detection_weight) * coverage_iou,
            'mean_f1': mean_f1, 'mean_f2': mean_f2, 'mean_fbeta': mean_fbeta,
            'gt_normalized_iou': coverage_iou,
            'tp': tp, 'fp': n_pred - tp, 'fn': n_gt - tp,
            'matched_iou_sum': iou_sum, 'num_gt': n_gt, 'num_predictions': n_pred}


def select_pair(outputs, qualities, min_gap):
    """Best/worst distinct rendered outputs; drop near ties and identical sets."""
    def signature(output):
        return sorted((int(label), *(round(float(v), 6) for v in box))
                      for label, box in zip(output['labels'], output['boxes']))
    best = max(range(len(qualities)), key=lambda i: qualities[i]['score'])
    worst = min(range(len(qualities)), key=lambda i: qualities[i]['score'])
    gap = qualities[best]['score'] - qualities[worst]['score']
    if gap < min_gap or signature(outputs[best]) == signature(outputs[worst]):
        return None
    return best, worst
