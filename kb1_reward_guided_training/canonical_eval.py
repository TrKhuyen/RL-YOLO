"""Canonical, framework-neutral evaluation for every KB1 detector."""

from __future__ import annotations

import time
from collections import defaultdict

import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou


def _sync(device: str) -> None:
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def _operating_point_counts(preds, targets, conf_thres=0.25, iou_thres=0.5):
    """Return class-aware one-to-one TP/FP/FN counts at a fixed threshold."""
    counts = defaultdict(lambda: [0, 0, 0])
    for pred, target in zip(preds, targets):
        keep = pred['scores'] >= conf_thres
        boxes = pred['boxes'][keep]
        labels = pred['labels'][keep]
        scores = pred['scores'][keep]
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        matched = set()
        ious = box_iou(gt_boxes.float(), boxes.float()) if len(boxes) else None
        for pred_idx in scores.argsort(descending=True).tolist():
            label = int(labels[pred_idx])
            candidates = [
                idx for idx in range(len(gt_boxes))
                if idx not in matched and int(gt_labels[idx]) == label
            ]
            if not candidates:
                counts[label][1] += 1
                continue
            values = ious[candidates, pred_idx]
            pos = int(values.argmax())
            if float(values[pos]) >= iou_thres:
                matched.add(candidates[pos])
                counts[label][0] += 1
            else:
                counts[label][1] += 1
        for gt_idx, label in enumerate(gt_labels.tolist()):
            if gt_idx not in matched:
                counts[int(label)][2] += 1
    return counts


def _summarize_counts(counts):
    tp = sum(value[0] for value in counts.values())
    fp = sum(value[1] for value in counts.values())
    fn = sum(value[2] for value in counts.values())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    class_f1 = []
    for class_tp, class_fp, class_fn in counts.values():
        class_p = class_tp / max(class_tp + class_fp, 1)
        class_r = class_tp / max(class_tp + class_fn, 1)
        class_f1.append(
            2 * class_p * class_r / max(class_p + class_r, 1e-12)
        )
    return precision, recall, f1, sum(class_f1) / max(len(class_f1), 1)


def evaluate_adapter(
    adapter,
    dataloader,
    device='cuda',
    conf_thres=0.001,
    iou_thres=0.60,
    max_det=300,
    operating_conf=0.25,
    class_metrics=False,
    measure_latency=False,
):
    """Evaluate one loaded adapter with shared preprocessing, NMS and metrics."""
    if max_det != 300:
        raise ValueError('Protocol v1 fixes max_det=300; use a new version to change it')
    adapter.eval_mode()
    metric = MeanAveragePrecision(
        class_metrics=class_metrics,
        extended_summary=False,
        max_detection_thresholds=[1, 10, max_det],
        backend='faster_coco_eval',
    )
    total_images = 0
    measured_images = 0
    inference_seconds = 0.0
    counts = defaultdict(lambda: [0, 0, 0])
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            if images.ndim != 4 or images.shape[1] != 3 or not torch.isfinite(images).all():
                raise ValueError('Expected finite BCHW RGB input')
            if float(images.min()) < 0 or float(images.max()) > 1:
                raise ValueError('Canonical input must be in [0, 1]')
            _sync(device)
            started = time.perf_counter()
            preds = adapter.forward_with_grad(
                images, conf_thres=conf_thres,
                iou_thres=iou_thres, max_det=max_det,
            )
            _sync(device)
            if measure_latency and batch_idx > 0:
                inference_seconds += time.perf_counter() - started
                measured_images += len(images)
            cpu_preds = [{
                'boxes': pred['boxes'].detach().float().cpu(),
                'scores': pred['scores'].detach().float().cpu(),
                'labels': pred['labels'].detach().long().cpu(),
            } for pred in preds]
            cpu_targets = [{
                'boxes': target['boxes'].float().cpu(),
                'labels': target['labels'].long().cpu(),
            } for target in targets]
            metric.update(cpu_preds, cpu_targets)
            batch_counts = _operating_point_counts(
                cpu_preds, cpu_targets, operating_conf, 0.5,
            )
            for label, values in batch_counts.items():
                counts[label] = [a + b for a, b in zip(counts[label], values)]
            total_images += len(images)
    if not total_images:
        raise ValueError('Cannot evaluate an empty dataset')
    result = metric.compute()
    precision, op_recall, f1, macro_f1 = _summarize_counts(counts)
    output = {
        'precision': precision,
        'operating_recall': op_recall,
        'f1': f1,
        'macro_f1': macro_f1,
        'mAP50': float(result['map_50']),
        'mAP50_95': float(result['map']),
        'APs': float(result.get('map_small', torch.tensor(-1.0))),
        'APm': float(result.get('map_medium', torch.tensor(-1.0))),
        'APl': float(result.get('map_large', torch.tensor(-1.0))),
        'recall': float(result.get(f'mar_{max_det}', torch.tensor(-1.0))),
        'AR300': float(result[f'mar_{max_det}']),
        'eval_protocol': 'kb1_canonical_v2_ar300',
        'fps': measured_images / inference_seconds if inference_seconds else 0.0,
        'n_images': total_images,
    }
    if class_metrics:
        output['classes'] = result.get('classes', torch.empty(0)).tolist()
        output['map_per_class'] = result.get(
            'map_per_class', torch.empty(0)
        ).tolist()
        output['mar_per_class'] = result.get(
            f'mar_{max_det}_per_class', torch.empty(0)
        ).tolist()
    return output
