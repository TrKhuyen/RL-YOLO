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
        inner.eval()
        inner.requires_grad_(True)
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

    def native_detection_loss(self, images, targets):
        # Restore default loss gains omitted from exported checkpoint args.
        from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
        if isinstance(self.model.args, dict):
            self.model.args = get_cfg(DEFAULT_CFG_DICT,
                                      overrides=self.model.args)
            self.model.criterion = None

        batch_idx, classes, boxes = [], [], []
        height, width = images.shape[-2:]
        for image_idx, target in enumerate(targets):
            xyxy = target['boxes']
            xywh = torch.empty_like(xyxy)
            xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / (2 * width)
            xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / (2 * height)
            xywh[:, 2] = (xyxy[:, 2] - xyxy[:, 0]) / width
            xywh[:, 3] = (xyxy[:, 3] - xyxy[:, 1]) / height
            batch_idx.append(torch.full((len(xyxy),), image_idx))
            classes.append(target['labels'].float().view(-1, 1))
            boxes.append(xywh)

        native_batch = {
            'img': images,
            'batch_idx': torch.cat(batch_idx).to(self.device),
            'cls': torch.cat(classes).to(self.device),
            'bboxes': torch.cat(boxes).to(self.device),
        }
        loss_components, loss_items = self.model.loss(native_batch)
        return loss_components.sum() / images.shape[0], loss_items

    def freeze_except_detection_head(self) -> tuple[int, int]:
        '''Freeze the feature extractor and train only the final Detect module.'''
        for param in self.model.parameters():
            param.requires_grad = False
        detect = self.model.model[-1]
        for param in detect.parameters():
            param.requires_grad = True
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return frozen, trainable

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
        from torchvision.ops import batched_nms
        from ultralytics.utils.ops import xywh2xyxy

        images = images.to(self.device)

        # Raw forward – giữ grad
        raw = self.model(images)

        # Ultralytics trả về (pred_tensor, ...) hoặc chỉ pred_tensor
        feat = raw[0] if isinstance(raw, (list, tuple)) else raw
        # feat: (B, 4+nc, 8400)

        B  = feat.shape[0]

        # Class scores với gradient: sigmoid(max class logit)
        scores_all, labels_all = feat[:, 4:, :].max(dim=1)
        boxes_all = xywh2xyxy(feat[:, :4, :].permute(0, 2, 1))

        preds = []
        for b in range(B):
            with torch.no_grad():
                candidates = torch.where(scores_all[b].detach() >= conf_thres)[0]
                keep = batched_nms(
                    boxes_all[b, candidates].detach(), scores_all[b, candidates].detach(),
                    labels_all[b, candidates], iou_thres,
                )[:300]
                selected = candidates[keep]

            if selected.numel() > 0:
                selected_scores = scores_all[b, selected]
                preds.append({
                    'boxes':  boxes_all[b, selected].detach(),
                    'labels': labels_all[b, selected].long().detach(),
                    'scores': selected_scores,
                    'policy_log_prob': torch.log(selected_scores.mean().clamp_min(1e-20)),
                    'max_score_all': scores_all[b].max(),
                })
            else:
                max_score = scores_all[b].max().clamp(1e-7, 1.0 - 1e-7)
                preds.append({
                    'boxes':  torch.zeros((0, 4), device=self.device),
                    'labels': torch.zeros(0, dtype=torch.long,
                                          device=self.device),
                    'scores': scores_all[b].new_empty((0,)),
                    'policy_log_prob': torch.log1p(-max_score),
                    'max_score_all': max_score,
                })

        return preds
