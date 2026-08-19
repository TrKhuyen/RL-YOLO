"""
adapters/yolov5_adapter.py – Adapter cho YOLOv5s và DP-YOLO.

Nhiệm vụ:
- Wrap YOLOv5 model (DetectMultiBackend) về interface chung.
- forward_with_grad(): giữ gradient qua confidence scores để REINFORCE hoạt động.
- Boxes và labels được tách ra (detach) vì chỉ cần grad qua scores (log_prob).
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class YOLOv5Adapter:
    """
    Adapter cho YOLOv5s và bất kỳ model nào dùng kiến trúc YOLOv5
    (bao gồm DP-YOLO custom được định nghĩa qua --cfg).

    Cần yolov5/ được clone và có trong sys.path.
    """

    def __init__(self, checkpoint: str, device: str = 'cuda',
                 yolov5_path: str | None = None):
        self.device = torch.device(device)
        if yolov5_path is None:
            yolov5_path = str(Path(__file__).resolve().parents[1] / 'yolov5')
        self._setup_path(yolov5_path)
        self.model = self._load(checkpoint)

    def _setup_path(self, yolov5_path: str):
        p = str(Path(yolov5_path).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)

    def _load(self, checkpoint: str):
        from models.common import DetectMultiBackend
        model = DetectMultiBackend(checkpoint, device=self.device)
        # Detection heads return decoded predictions in eval mode. Eval mode
        # does not disable autograd, so RL gradients still flow through scores.
        model.model.eval()
        model.model.requires_grad_(True)
        return model

    # ── Interface ──────────────────────────────────────────────────────────

    def train_mode(self):
        self.model.model.eval()

    def eval_mode(self):
        self.model.model.eval()

    def parameters(self):
        return self.model.model.parameters()

    def named_parameters(self):
        return self.model.model.named_parameters()

    def state_dict(self):
        return self.model.model.state_dict()

    def freeze_except_detection_head(self) -> tuple[int, int]:
        '''Freeze the feature extractor and train only the final Detect module.'''
        inner = self.model.model
        for param in inner.parameters():
            param.requires_grad = False
        detect = inner.model[-1]
        for param in detect.parameters():
            param.requires_grad = True
        frozen = sum(p.numel() for p in inner.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in inner.parameters() if p.requires_grad)
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

        YOLOv5 raw output shape: list of Tensors, mỗi scale là (B, A, H, W, 5+nc)
        trong đó A=3 anchors, 5 = [cx, cy, w, h, obj_conf], nc = class scores.

        Chiến lược gradient:
        - Chạy NMS trên output đã detach để lấy indices predictions hợp lệ.
        - Ghép lại confidence từ raw output (chưa detach) → giữ grad.

        Args:
            images:     Tensor (B, 3, H, W), đã normalize
            conf_thres: threshold confidence khi lấy predictions (thấp hơn lúc eval)
            iou_thres:  NMS IoU threshold

        Returns:
            list[dict]: mỗi dict có 'boxes'(xyxy), 'labels'(long), 'scores'(grad)
        """
        from torchvision.ops import batched_nms
        from utils.general import xywh2xyxy

        images = images.to(self.device)

        # YOLOv5 forward trả về tuple: (inference_out, train_out)
        # inference_out: Tensor (B, num_all_anchors, 5+nc) – đã sigmoid
        out = self.model.model(images)
        if isinstance(out, tuple):
            inference_out = out[0]  # (B, num_anchors_all, 5+nc) với grad
        else:
            inference_out = out

        preds = []
        B = images.shape[0]

        for b in range(B):
            single = inference_out[b]  # (N, 5+nc) – có gradient

            # Confidence = obj_conf × max_cls_conf
            obj_conf  = single[:, 4]                        # (N,) – grad intact
            cls_conf  = single[:, 5:].max(dim=-1).values   # (N,) – grad intact
            scores_all = obj_conf * cls_conf                # (N,) – grad intact

            labels_all = single[:, 5:].argmax(dim=-1)
            boxes_all = xywh2xyxy(single[:, :4])

            with torch.no_grad():
                candidates = torch.where(scores_all.detach() >= conf_thres)[0]
                keep = batched_nms(
                    boxes_all[candidates].detach(), scores_all[candidates].detach(),
                    labels_all[candidates], iou_thres,
                )[:300]
                selected = candidates[keep]

            if selected.numel() > 0:
                selected_scores = scores_all[selected]
                preds.append({
                    'boxes':  boxes_all[selected].detach(),
                    'labels': labels_all[selected].long().detach(),
                    'scores': selected_scores,
                    'policy_log_prob': torch.log(selected_scores.mean().clamp_min(1e-20)),
                    'max_score_all': scores_all.max(),
                })
            else:
                max_score = scores_all.max().clamp(1e-7, 1.0 - 1e-7)
                preds.append({
                    'boxes':  torch.zeros((0, 4), device=self.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=self.device),
                    'scores': scores_all.new_empty((0,)),
                    'policy_log_prob': torch.log1p(-max_score),
                    'max_score_all': max_score,
                })

        return preds
