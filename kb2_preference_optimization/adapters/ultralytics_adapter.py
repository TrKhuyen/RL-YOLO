"""
adapters/ultralytics_adapter.py – Adapter cho YOLOv8 và YOLOv11 (Ultralytics).

Tương tự YOLOv5Adapter nhưng dành cho framework Ultralytics mới hơn.
YOLOv8/v11 là anchor-free: output format khác YOLOv5.

Output raw của Ultralytics DetectionModel:
    Tensor (B, 4+nc, num_anchors) với num_anchors = 8400 (default 640px input)
    - [:4, :]  = raw box regression (cxcywh, chưa decode)
    - [4:, :]  = class logits (chưa sigmoid)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class UltralyticsAdapter:
    """
    Adapter cho YOLOv8n/s và YOLOv11n/s từ Ultralytics.

    Cần: pip install ultralytics>=8.0.0
    """

    def __init__(self, checkpoint: str, device: str = 'cuda'):
        self.device = torch.device(device)
        self.model  = self._load(checkpoint)

    def _load(self, checkpoint: str):
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import DetectionModel

        yolo = YOLO(checkpoint)
        # Lấy nn.Module bên trong để có full control
        inner: DetectionModel = yolo.model.to(self.device)
        if isinstance(inner.args, dict):
            inner.args = get_cfg(overrides=inner.args)
        inner.requires_grad_(True)
        inner.train()
        return inner

    # ── Interface ──────────────────────────────────────────────────────────

    def train_mode(self):
        self.model.eval()

    def eval_mode(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def state_dict(self):
        return self.model.state_dict()

    def raw_predictions(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        raw = self.model(images.to(self.device))
        return raw[0] if isinstance(raw, (list, tuple)) else raw

    def supervised_loss(self, images: torch.Tensor, targets: list[dict]):
        images = images.to(self.device)
        h, w = images.shape[-2:]
        batch_idx, classes, boxes = [], [], []
        for i, target in enumerate(targets):
            xyxy = target['boxes'].to(self.device)
            if not len(xyxy):
                continue
            xywh = torch.stack((
                (xyxy[:, 0] + xyxy[:, 2]) / (2 * w),
                (xyxy[:, 1] + xyxy[:, 3]) / (2 * h),
                (xyxy[:, 2] - xyxy[:, 0]) / w,
                (xyxy[:, 3] - xyxy[:, 1]) / h,
            ), dim=1)
            batch_idx.append(torch.full((len(xyxy),), i, device=self.device))
            classes.append(target['labels'].to(self.device).float().view(-1, 1))
            boxes.append(xywh)
        batch = {
            'img': images,
            'batch_idx': torch.cat(batch_idx) if batch_idx else torch.zeros(0, device=self.device),
            'cls': torch.cat(classes) if classes else torch.zeros((0, 1), device=self.device),
            'bboxes': torch.cat(boxes) if boxes else torch.zeros((0, 4), device=self.device),
        }
        self.model.train()
        # Small feedback batches corrupt BatchNorm running statistics quickly.
        # Keep affine parameters trainable but use the pretrained running stats.
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        loss, items = self.model.loss(batch)
        self.model.eval()
        return loss.sum() / max(len(images), 1), items

    # ── Forward ────────────────────────────────────────────────────────────

    def forward_with_grad(
        self,
        images:     torch.Tensor,
        conf_thres: float = 0.20,
        iou_thres:  float = 0.45,
    ) -> list[dict]:
        """
        Forward pass giữ gradient qua confidence scores.

        Ultralytics v8/v11 raw output:
            Tuple[Tensor, list], trong đó Tensor[0] shape: (B, 4+nc, 8400)
            - [b, :4, :]  = cxcywh box predictions
            - [b, 4:, :]  = class logits

        Chiến lược:
        - Tính scores (sigmoid of max class logit) có gradient.
        - Chạy NMS detach để biết predictions nào được giữ.
        - Map confidence → predictions được chọn (top-k approximation).
        """
        from ultralytics.utils.nms import non_max_suppression as nms_v8

        images = images.to(self.device)

        # Raw forward – giữ grad
        raw = self.model(images)

        # Ultralytics trả về (pred_tensor, ...) hoặc chỉ pred_tensor
        feat = raw[0] if isinstance(raw, (list, tuple)) else raw
        # feat: (B, 4+nc, 8400)

        nc = feat.shape[1] - 4
        B  = feat.shape[0]

        # Class scores với gradient: sigmoid(max class logit)
        cls_probs = feat[:, 4:, :]
        scores_all = cls_probs.max(dim=1).values

        # NMS once for the complete batch. Calling it per image multiplies the
        # Python/CUDA synchronization overhead in DPO (G policy views + ref).
        with torch.no_grad():
            det_list, kept_indices = nms_v8(
                feat.detach(),
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=100,
                max_nms=5000,
                max_time_img=0.5,
                nc=nc,
                return_idxs=True,
            )

        preds = []
        for b, det in enumerate(det_list):
            keep = kept_indices[b].flatten().long()

            if det is not None and len(det) > 0:
                preds.append({
                    'boxes':  det[:, :4].detach(),
                    'labels': det[:, 5].long().detach(),
                    'scores': scores_all[b, keep],
                })
            else:
                preds.append({
                    'boxes':  torch.zeros((0, 4), device=self.device),
                    'labels': torch.zeros(0, dtype=torch.long,
                                          device=self.device),
                    'scores': torch.zeros(0, device=self.device,
                                          requires_grad=True),
                })

        return preds
