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
                 yolov5_path: str = 'yolov5'):
        self.device = torch.device(device)
        self._setup_path(yolov5_path)
        self.model = self._load(checkpoint)

    def _setup_path(self, yolov5_path: str):
        p = str(Path(yolov5_path).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)

    def _load(self, checkpoint: str):
        from models.common import DetectMultiBackend
        model = DetectMultiBackend(checkpoint, device=self.device)
        model.model.train()
        return model

    # ── Interface ──────────────────────────────────────────────────────────

    def train_mode(self):
        self.model.model.train()

    def eval_mode(self):
        self.model.model.eval()

    def parameters(self):
        return self.model.model.parameters()

    def named_parameters(self):
        return self.model.model.named_parameters()

    def state_dict(self):
        return self.model.model.state_dict()

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
        from utils.general import non_max_suppression

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

            # NMS trên version đã detach để lấy indices hợp lệ
            with torch.no_grad():
                det_list = non_max_suppression(
                    inference_out[b:b+1].detach(),
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                )
            det = det_list[0]  # (M, 6): x1 y1 x2 y2 conf cls

            if det is not None and len(det) > 0:
                # Lấy confidence của M predictions từ scores_all có grad
                # Xấp xỉ: dùng top-M scores theo giá trị (không có index map chính xác)
                # Đây là xấp xỉ được dùng trong yanivnik – đủ cho gradient đúng hướng
                M = len(det)
                topk_scores, _ = scores_all.topk(min(M, len(scores_all)))
                preds.append({
                    'boxes':  det[:, :4].detach(),
                    'labels': det[:, 5].long().detach(),
                    'scores': topk_scores[:M],          # GIỮ GRADIENT
                })
            else:
                preds.append({
                    'boxes':  torch.zeros((0, 4), device=self.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=self.device),
                    'scores': torch.zeros(1, device=self.device,
                                          requires_grad=True),
                })

        return preds
