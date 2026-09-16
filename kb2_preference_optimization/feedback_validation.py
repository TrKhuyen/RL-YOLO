"""Validation metrics and early stopping for feedback-guided training."""
import time

import torch
from torchmetrics.detection import MeanAveragePrecision


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, mode='max'):
        if patience < 1:
            raise ValueError('patience must be positive')
        if min_delta < 0:
            raise ValueError('min_delta must be non-negative')
        if mode not in ('max', 'min'):
            raise ValueError("mode must be 'max' or 'min'")
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best = None
        self.bad_evaluations = 0

    def update(self, value):
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError('validation metric must be finite')
        improved = self.best is None or (
            value > self.best + self.min_delta if self.mode == 'max'
            else value < self.best - self.min_delta)
        if improved:
            self.best, self.bad_evaluations = float(value), 0
        else:
            self.bad_evaluations += 1
        return improved, self.bad_evaluations >= self.patience

    def state_dict(self):
        return {'best': self.best, 'bad_evaluations': self.bad_evaluations}

    def load_state_dict(self, state):
        self.best = state.get('best')
        self.bad_evaluations = int(state.get('bad_evaluations', 0))


def evaluate_adapter(adapter, val_loader, device='cuda', conf=0.25, iou=0.45):
    """Compute COCO mAP on validation data without materializing all images."""
    metric = MeanAveragePrecision(
        iou_thresholds=[.50 + .05 * i for i in range(10)],
        class_metrics=False, extended_summary=False,
        backend='faster_coco_eval')
    started, num_images = time.time(), 0
    adapter.eval_mode()
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            predictions = adapter.forward_with_grad(images, conf, iou)
            metric.update([{
                'boxes': p['boxes'].detach().float().cpu(),
                'scores': p['scores'].detach().float().cpu(),
                'labels': p['labels'].detach().int().cpu(),
            } for p in predictions], [{
                'boxes': t['boxes'].float().cpu(),
                'labels': t['labels'].int().cpu(),
            } for t in targets])
            num_images += len(images)
    result = metric.compute()
    adapter.train_mode()
    elapsed = max(time.time() - started, 1e-9)
    return {
        'mAP50-95': float(result['map']),
        'mAP50': float(result['map_50']),
        'AP_small': float(result['map_small']),
        'Recall': float(result['mar_100']),
        'FPS': num_images / elapsed,
        'num_images': num_images,
    }
