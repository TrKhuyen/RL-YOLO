"""
models/dp_yolo/loss.py  –  W3F_MPDIoU composite IoU loss cho DP-YOLO.

W3F_MPDIoU  =  r  ×  R_WIoU  ×  L_F_MPDIoU

Trong đó:
  L_MPDIoU    = 1 − IoU + (d1² + d2²) / c²
                d1 = khoảng cách góc trên-trái (top-left corner distance)
                d2 = khoảng cách góc dưới-phải (bottom-right corner distance)
                c² = đường chéo hộp bao (enclosing box diagonal)²

  L_F_MPDIoU  = L_MPDIoU + IoU − IoU^γ          (Focaler-IoU wrapper, γ=0.5)

  R_WIoU      = exp(center_dist² / gt_diag²)      (outlier degree)
  r           = β / (δ·α^β − δ)                   (hằng số WIoU v3)

Hàm `patch_loss()` monkey-patch `bbox_iou` trong utils.loss của YOLOv5
để tự động dùng W3F_MPDIoU thay cho CIoU loss.

Tham khảo:
  MPDIoU:      https://arxiv.org/abs/2307.07662
  Focaler-IoU: https://arxiv.org/abs/2311.10861
  WIoU v3:     https://arxiv.org/abs/2301.10051
"""

import math
import os
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Core loss function
# ─────────────────────────────────────────────────────────────────────────────

def bbox_iou_w3f(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool  = True,
    eps:  float = 1e-7,
    # Focaler-IoU
    focaler_gamma: float = 0.5,
    # WIoU v3 hyperparameters (từ paper gốc)
    wiou_alpha: float = 1.9,
    wiou_beta:  float = 0.6,
    wiou_delta: float = 0.5,
) -> torch.Tensor:
    """
    Tính W3F_MPDIoU loss và trả về (1 − loss) để tương thích với
    pattern ``lbox += (1.0 − iou).mean()`` của YOLOv5.

    Args:
        box1:  Tensor (N, 4) – predicted boxes
        box2:  Tensor (N, 4) – target boxes
        xywh:  True = format [cx, cy, w, h]; False = [x1, y1, x2, y2]

    Returns:
        Tensor (N,): `(1 − W3F_loss)`. Khi YOLOv5 tính `1 − return_value`,
        kết quả chính là W3F_MPDIoU loss mà ta muốn minimize.
    """
    # ── Convert to xyxy ─────────────────────────────────────────────────
    if xywh:
        b1_x1 = box1[..., 0] - box1[..., 2] * 0.5
        b1_y1 = box1[..., 1] - box1[..., 3] * 0.5
        b1_x2 = box1[..., 0] + box1[..., 2] * 0.5
        b1_y2 = box1[..., 1] + box1[..., 3] * 0.5
        b2_x1 = box2[..., 0] - box2[..., 2] * 0.5
        b2_y1 = box2[..., 1] - box2[..., 3] * 0.5
        b2_x2 = box2[..., 0] + box2[..., 2] * 0.5
        b2_y2 = box2[..., 1] + box2[..., 3] * 0.5
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = (
            box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        )
        b2_x1, b2_y1, b2_x2, b2_y2 = (
            box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
        )

    # ── IoU ─────────────────────────────────────────────────────────────
    inter = (
        (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) *
        (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    )
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union   = w1 * h1 + w2 * h2 - inter + eps
    iou     = (inter / union).clamp(0.0, 1.0)

    # ── Enclosing box diagonal² (normalization cho MPDIoU) ───────────────
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps

    # ── MPDIoU: top-left + bottom-right corner distances ────────────────
    d1 = (b1_x1 - b2_x1) ** 2 + (b1_y1 - b2_y1) ** 2   # góc trên-trái
    d2 = (b1_x2 - b2_x2) ** 2 + (b1_y2 - b2_y2) ** 2   # góc dưới-phải
    L_mpdiou = 1.0 - iou + (d1 + d2) / c2

    # ── Focaler-IoU: L_F = L_MPDIoU + IoU − IoU^γ ──────────────────────
    L_f_mpdiou = L_mpdiou + iou - iou.clamp_min(eps).pow(focaler_gamma)

    # ── WIoU v3: focal weight theo outlier degree ────────────────────────
    cx_p = (b1_x1 + b1_x2) * 0.5
    cy_p = (b1_y1 + b1_y2) * 0.5
    cx_g = (b2_x1 + b2_x2) * 0.5
    cy_g = (b2_y1 + b2_y2) * 0.5

    center_dist  = (cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2
    gt_diag2     = w2 ** 2 + h2 ** 2 + eps
    outlier_degree = (center_dist / gt_diag2).clamp(0.0, math.log(10.0))
    R_wiou = torch.exp(outlier_degree).detach()

    # Hằng số r = β / (δ·α^β − δ)
    denom   = wiou_delta * (wiou_alpha ** wiou_beta) - wiou_delta
    r_const = float(wiou_beta / (denom + 1e-9))
    r_const = float(min(max(r_const, 0.0), 10.0))   # safety clamp

    # W3F_MPDIoU final loss
    W3F_loss = r_const * R_wiou * L_f_mpdiou

    # Trả về (1 − W3F_loss) để YOLOv5 tính `lbox += (1 − iou).mean()`
    # → `1 − (1 − W3F_loss) = W3F_loss` ✓
    return 1.0 - W3F_loss


# ─────────────────────────────────────────────────────────────────────────────
# Patch function
# ─────────────────────────────────────────────────────────────────────────────

def patch_loss() -> bool:
    """
    Monkey-patch ``bbox_iou`` trong ``utils.loss`` của YOLOv5 để dùng
    W3F_MPDIoU thay cho CIoU khi ``CIoU=True``.

    Gọi SAU KHI yolov5 đã được thêm vào sys.path (trong patch_yolov5.patch()).
    Chỉ ảnh hưởng khi ``CIoU=True``; các mode GIoU, DIoU không bị thay đổi.

    Returns:
        True nếu patch thành công, False nếu import thất bại.
    """
    try:
        import utils.loss    as _loss_mod

        _original_bbox_iou = _loss_mod.bbox_iou  # luu ham goc tu utils.loss

        def _bbox_iou_w3f_compat(
            box1, box2, xywh=True,
            GIoU=False, DIoU=False, CIoU=False,
            eps=1e-7, **kwargs
        ):
            """
            Use W3F for the box-regression gradient while preserving CIoU as
            the forward value. ComputeLoss also reuses this value (detached)
            as its objectness target, which must remain a valid IoU-quality
            score rather than the unbounded W3F surrogate.
            """
            if CIoU:
                w3f_score = bbox_iou_w3f(box1, box2, xywh=xywh, eps=eps)
                ciou_score = _original_bbox_iou(
                    box1, box2, xywh=xywh, CIoU=True, eps=eps,
                ).squeeze(-1)
                # Forward: CIoU (correct objectness quality target).
                # Backward: W3F surrogate (custom box-regression gradient).
                return ciou_score.detach() + w3f_score - w3f_score.detach()
            return _original_bbox_iou(
                box1, box2, xywh=xywh,
                GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, eps=eps,
            )

        # Patch module-level reference trong utils.loss
        _loss_mod.bbox_iou = _bbox_iou_w3f_compat

        print("  -> W3F_MPDIoU loss patched  ->  utils.loss.bbox_iou (CIoU path)")
        return True

    except (ImportError, AttributeError) as e:
        print(f"  [WARN] patch_loss failed: {e}")
        print("  -> Continuing with default CIoU loss.")
        return False
