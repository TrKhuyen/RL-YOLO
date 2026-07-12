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
        from ultralytics.nn.tasks import DetectionModel

        yolo = YOLO(checkpoint)
        # Lấy nn.Module bên trong để có full control
        inner: DetectionModel = yolo.model.to(self.device)
        inner.train()
        return inner

    # ── Interface ──────────────────────────────────────────────────────────

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def state_dict(self):
        return self.model.state_dict()

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
        from ultralytics.utils.ops import non_max_suppression as nms_v8
        from ultralytics.utils.ops import xywh2xyxy

        images = images.to(self.device)

        # Raw forward – giữ grad
        raw = self.model(images)

        # Ultralytics trả về (pred_tensor, ...) hoặc chỉ pred_tensor
        feat = raw[0] if isinstance(raw, (list, tuple)) else raw
        # feat: (B, 4+nc, 8400)

        nc = feat.shape[1] - 4
        B  = feat.shape[0]

        # Class scores với gradient: sigmoid(max class logit)
        cls_logits = feat[:, 4:, :]           # (B, nc, 8400)
        scores_all = cls_logits.sigmoid().max(dim=1).values  # (B, 8400) – GIỮ GRAD

        preds = []
        for b in range(B):
            # ── Decode boxes (detach, chỉ cần để tính reward) ────────────
            with torch.no_grad():
                # Reshape cho NMS: (1, 8400, 4+nc)
                feat_b = feat[b:b+1].permute(0, 2, 1).detach()  # (1, 8400, 4+nc)

                # Thêm objectness score = max class prob cho anchor-free format
                boxes_cxcywh = feat_b[..., :4]
                cls_prob_det = feat_b[..., 4:].sigmoid()
                conf_det     = cls_prob_det.max(dim=-1, keepdim=True).values
                # Format NMS yêu cầu: (B, num, 5+nc) = xyxy conf cls...
                boxes_xyxy = xywh2xyxy(boxes_cxcywh)
                nms_input  = torch.cat([boxes_xyxy, conf_det,
                                        cls_prob_det], dim=-1)

                det_list = nms_v8(
                    nms_input,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                    max_det=300,
                )
                det = det_list[0]  # (M, 6): x1 y1 x2 y2 conf cls

            if det is not None and len(det) > 0:
                M = len(det)
                # Lấy top-M scores từ scores_all[b] (có gradient)
                topk_scores, _ = scores_all[b].topk(
                    min(M, scores_all.shape[1])
                )
                preds.append({
                    'boxes':  det[:, :4].detach(),
                    'labels': det[:, 5].long().detach(),
                    'scores': topk_scores[:M],   # GIỮ GRADIENT
                })
            else:
                preds.append({
                    'boxes':  torch.zeros((0, 4), device=self.device),
                    'labels': torch.zeros(0, dtype=torch.long,
                                          device=self.device),
                    'scores': torch.zeros(1, device=self.device,
                                          requires_grad=True),
                })

        return preds
