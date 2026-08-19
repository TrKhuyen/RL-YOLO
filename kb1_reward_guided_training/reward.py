"""
reward.py – Reward functions cho RL fine-tuning YOLO trên bài toán sâu bệnh UAV.

Thiết kế dựa trên yanivnik/tuning_cv_models_with_rl_torch nhưng điều chỉnh:
1. recall_reward:            recall per-class với penalize duplicate (yanivnik)
2. small_object_recall_reward: bonus cho vật thể nhỏ (diện tích < small_thresh²)
3. composite_reward:         kết hợp 2 reward trên với trọng số alpha

Tại sao không dùng mAP làm reward trực tiếp?
→ yanivnik: "mAP reward currently has some problems" – mAP per-image không ổn định.
→ Tính mAP toàn tập chậm hơn recall ~10×, không phù hợp vòng lặp RL mỗi batch.
"""

import torch
import torchvision
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iou_matrix(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> torch.Tensor:
    """Tính IoU matrix (n_gt × n_pred). Trả về zeros nếu một trong hai rỗng."""
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return torch.zeros(
            len(gt_boxes), len(pred_boxes),
            device=gt_boxes.device if gt_boxes.numel() > 0 else pred_boxes.device,
        )
    return torchvision.ops.box_iou(
        gt_boxes.float().cpu(),
        pred_boxes.float().cpu(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Recall Reward
# ─────────────────────────────────────────────────────────────────────────────

def recall_reward(
    preds:             list[dict],
    targets:           list[dict],
    iou_threshold:     float = 0.5,
    duplicate_penalty: float = 0.3,
) -> torch.Tensor:
    """
    Recall-based reward cho mỗi ảnh trong batch.

    Công thức (từ yanivnik, điều chỉnh normalize):
        R_i = mean_over_classes(matched_GT - 0.3 × duplicate_preds) / n_GT_c

    - matched_GT:     số GT box được match với ít nhất 1 prediction (IoU ≥ thr)
    - duplicate_preds: số prediction dư thừa match cùng 1 GT box
    - Hệ số 0.3 < 1: ưu tiên tăng recall hơn là phạt duplicate
      → bỏ sót nguy hiểm hơn false positive trong bài toán sâu bệnh

    Args:
        preds:   list[dict] – mỗi dict: {'boxes'(xyxy), 'labels', 'scores'}
        targets: list[dict] – mỗi dict: {'boxes'(xyxy), 'labels'}

    Returns:
        Tensor shape (batch_size,), reward cho mỗi ảnh
    """
    rewards = torch.zeros(len(preds))

    for i, (pred, target) in enumerate(zip(preds, targets)):
        gt_boxes   = target['boxes'].detach().cpu()
        gt_labels  = target['labels'].detach().cpu()
        pred_boxes = pred['boxes'].detach().cpu()
        pred_labels = pred['labels'].detach().cpu()

        if len(gt_boxes) == 0:
            continue

        classes      = gt_labels.unique()
        class_reward = 0.0

        for cls in classes:
            gt_mask   = (gt_labels  == cls)
            pred_mask = (pred_labels == cls)
            gt_cls    = gt_boxes[gt_mask]
            pred_cls  = pred_boxes[pred_mask]

            if len(pred_cls) == 0:
                continue

            iou_mat  = _iou_matrix(gt_cls, pred_cls)          # (n_gt, n_pred)
            matched  = iou_mat > iou_threshold                 # bool mask

            n_matched_gt = torch.any(matched, dim=1).sum().float()
            n_duplicates = (matched.sum(dim=1) - 1).clamp(min=0).sum().float()

            # Normalize theo số GT để không bias ảnh có nhiều object
            score = (n_matched_gt - duplicate_penalty * n_duplicates) / len(gt_cls)
            class_reward += score.item()

        rewards[i] = class_reward / len(classes)

    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# 2. Small-Object Recall Reward
# ─────────────────────────────────────────────────────────────────────────────

def small_object_recall_reward(
    preds:         list[dict],
    targets:       list[dict],
    small_thresh:  int   = 32,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Bonus reward tập trung vào GT boxes nhỏ (diện tích < small_thresh² px).

    Đặc thù UAV: sâu non, trứng, đốm bệnh giai đoạn đầu chiếm diện tích rất nhỏ.
    Metric này trực tiếp đo vấn đề cốt lõi của đề tài.

    Returns:
        Tensor shape (batch_size,): tỉ lệ GT nhỏ được phát hiện [0, 1]
    """
    rewards = torch.zeros(len(preds))

    for i, (pred, target) in enumerate(zip(preds, targets)):
        gt_boxes   = target['boxes'].detach().cpu().float()
        gt_labels  = target['labels'].detach().cpu()
        pred_boxes = pred['boxes'].detach().cpu().float()
        pred_labels = pred['labels'].detach().cpu()

        if len(gt_boxes) == 0:
            continue

        # Lọc GT boxes nhỏ: diện tích = (x2-x1) × (y2-y1)
        areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]) *
                 (gt_boxes[:, 3] - gt_boxes[:, 1]))
        small_mask = areas < (small_thresh ** 2)
        small_gt = gt_boxes[small_mask]
        small_labels = gt_labels[small_mask]

        if len(small_gt) == 0:
            # Whole-leaf annotations usually have no box below 32x32.
            # Returning 1 created a constant (1-alpha)=0.4 reward floor.
            # No applicable small-object target contributes no bonus.
            rewards[i] = 0.0
            continue

        if len(pred_boxes) == 0:
            rewards[i] = 0.0
            continue

        iou_mat = _iou_matrix(small_gt, pred_boxes)
        same_class = small_labels[:, None] == pred_labels[None, :]
        n_matched = torch.any((iou_mat > iou_threshold) & same_class, dim=1).sum().float()
        rewards[i] = n_matched / (len(small_gt) + 1e-6)

    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# 3. Composite Reward
# ─────────────────────────────────────────────────────────────────────────────

def composite_reward(
    preds:         list[dict],
    targets:       list[dict],
    alpha:         float = 0.6,
    iou_threshold: float = 0.5,
    small_thresh:  int   = 32,
    duplicate_penalty: float = 0.3,
) -> torch.Tensor:
    """
    Reward tổng hợp: alpha × recall + (1-alpha) × small_recall.

    alpha = 0.6: ưu tiên recall tổng thể nhưng vẫn chú trọng vật thể nhỏ UAV.
    Điều chỉnh alpha khi muốn đẩy mạnh hơn về một phía.

    Returns:
        Tensor shape (batch_size,), giá trị trong khoảng [0, ~1]
    """
    r_recall = recall_reward(
        preds, targets, iou_threshold=iou_threshold,
        duplicate_penalty=duplicate_penalty,
    )
    r_small  = small_object_recall_reward(preds, targets,
                                          small_thresh=small_thresh,
                                          iou_threshold=iou_threshold)
    return alpha * r_recall + (1.0 - alpha) * r_small


def detection_composite_reward(
    preds: list[dict], targets: list[dict],
    iou_thresholds=(0.5, 0.6, 0.7, 0.8),
    precision_weight: float = 0.35, recall_weight: float = 0.35,
    iou_weight: float = 0.30, **_,
) -> torch.Tensor:
    '''Balanced precision, recall and localization reward over multiple IoUs.'''
    weight_sum = precision_weight + recall_weight + iou_weight
    if weight_sum <= 0:
        raise ValueError('Detection reward weights must be positive')
    wp, wr, wi = (precision_weight / weight_sum,
                  recall_weight / weight_sum, iou_weight / weight_sum)
    rewards = torch.zeros(len(preds), dtype=torch.float32)
    for image_idx, (pred, target) in enumerate(zip(preds, targets)):
        gb = target['boxes'].detach().cpu().float()
        gl = target['labels'].detach().cpu().long()
        pb = pred['boxes'].detach().cpu().float()
        pl = pred['labels'].detach().cpu().long()
        ps = pred['scores'].detach().cpu().float()
        if len(gb) == 0:
            rewards[image_idx] = float(len(pb) == 0)
            continue
        if len(pb) == 0:
            continue
        order, ious, values = ps.argsort(descending=True), _iou_matrix(gb, pb), []
        for threshold in iou_thresholds:
            used, matched_ious = set(), []
            for pred_idx in order.tolist():
                valid = gl == pl[pred_idx]
                if used:
                    valid[list(used)] = False
                candidates = torch.where(valid)[0]
                if not len(candidates):
                    continue
                candidate_ious = ious[candidates, pred_idx]
                pos = int(candidate_ious.argmax())
                gt_idx, best_iou = int(candidates[pos]), float(candidate_ious[pos])
                if best_iou >= float(threshold):
                    used.add(gt_idx)
                    matched_ious.append(best_iou)
            tp = len(used)
            precision, recall = tp / max(len(pb), 1), tp / len(gb)
            mean_iou = sum(matched_ious) / tp if tp else 0.0
            values.append(wp * precision + wr * recall + wi * mean_iou)
        rewards[image_idx] = sum(values) / len(values)
    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reward dispatcher (dùng trong train_rl.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(
    preds:       list[dict],
    targets:     list[dict],
    reward_type: str  = 'composite',
    alpha:       float = 0.6,
    **kwargs,
) -> torch.Tensor:
    """
    Dispatcher: chọn reward function theo tên.

    Args:
        reward_type: 'recall' | 'small' | 'composite'
    """
    if reward_type != 'detection_composite':
        kwargs.pop('iou_thresholds', None)
        kwargs.pop('precision_weight', None)
        kwargs.pop('recall_weight', None)
        kwargs.pop('iou_weight', None)
    if reward_type == 'recall':
        kwargs.pop('small_thresh', None)
        return recall_reward(preds, targets, **kwargs)
    elif reward_type == 'small':
        kwargs.pop('duplicate_penalty', None)
        return small_object_recall_reward(preds, targets, **kwargs)
    elif reward_type == 'composite':
        return composite_reward(preds, targets, alpha=alpha, **kwargs)
    elif reward_type == 'detection_composite':
        kwargs.pop('small_thresh', None)
        kwargs.pop('duplicate_penalty', None)
        kwargs.pop('iou_threshold', None)
        return detection_composite_reward(preds, targets, **kwargs)
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}. "
                         f"Choose from: recall | small | composite")
