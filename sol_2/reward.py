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
        pred_boxes = pred['boxes'].detach().cpu().float()

        if len(gt_boxes) == 0:
            continue

        # Lọc GT boxes nhỏ: diện tích = (x2-x1) × (y2-y1)
        areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]) *
                 (gt_boxes[:, 3] - gt_boxes[:, 1]))
        small_mask = areas < (small_thresh ** 2)
        small_gt = gt_boxes[small_mask]

        if len(small_gt) == 0:
            rewards[i] = 1.0   # không có vật thể nhỏ → không phạt
            continue

        if len(pred_boxes) == 0:
            rewards[i] = 0.0
            continue

        iou_mat   = _iou_matrix(small_gt, pred_boxes)
        n_matched = torch.any(iou_mat > iou_threshold, dim=1).sum().float()
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
) -> torch.Tensor:
    """
    Reward tổng hợp: alpha × recall + (1-alpha) × small_recall.

    alpha = 0.6: ưu tiên recall tổng thể nhưng vẫn chú trọng vật thể nhỏ UAV.
    Điều chỉnh alpha khi muốn đẩy mạnh hơn về một phía.

    Returns:
        Tensor shape (batch_size,), giá trị trong khoảng [0, ~1]
    """
    r_recall = recall_reward(preds, targets,
                             iou_threshold=iou_threshold)
    r_small  = small_object_recall_reward(preds, targets,
                                          small_thresh=small_thresh,
                                          iou_threshold=iou_threshold)
    return alpha * r_recall + (1.0 - alpha) * r_small


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
    if reward_type == 'recall':
        return recall_reward(preds, targets, **kwargs)
    elif reward_type == 'small':
        return small_object_recall_reward(preds, targets, **kwargs)
    elif reward_type == 'composite':
        return composite_reward(preds, targets, alpha=alpha, **kwargs)
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}. "
                         f"Choose from: recall | small | composite")
