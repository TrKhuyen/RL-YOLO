"""DFL YOLO adapter with raw logits, native loss and canonical inference."""
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.utils.tal import make_anchors
from torchvision.ops import batched_nms

from .policy import DetectionDistribution, PolicyConfig

SUPPORTED_MODELS = ('yolov8n', 'yolov8s', 'yolov11n', 'yolov11s')


class DFLAdapter:
    def __init__(self, checkpoint, device='cpu', head_only=True):
        self.device = torch.device(device)
        self.model = YOLO(str(checkpoint)).model.to(self.device).float().eval()
        self.head = self.model.model[-1]
        if getattr(self.head, 'end2end', False) or self.head.reg_max <= 1:
            raise ValueError('Structured DPO v1 requires an unfused DFL YOLOv8/YOLO11 head; YOLO26 is unsupported')
        if not hasattr(self.head, 'cv2') or self.head.cv2 is None:
            raise ValueError('Unfused regression head required')
        if isinstance(self.model.args, dict):
            self.model.args = get_cfg(DEFAULT_CFG_DICT, overrides=self.model.args)
        self.model.requires_grad_(not head_only)
        if head_only:
            self.head.requires_grad_(True)
        self.head.dfl.requires_grad_(False)
        self.model.criterion = None

    def distributions(self, images, config=PolicyConfig()):
        self.model.eval()  # BN buffers remain immutable; autograd stays enabled.
        output = self.model(images.to(self.device))
        if not isinstance(output, tuple) or not isinstance(output[1], dict):
            raise RuntimeError('Unsupported Ultralytics raw-output API; expected 8.4.163 boxes/scores/feats')
        raw = output[1]
        anchors, strides = make_anchors(raw['feats'], self.head.stride, 0.5)
        classes = raw['scores'].permute(0, 2, 1)
        boxes = raw['boxes'].reshape(len(images), 4, self.head.reg_max, -1).permute(0, 3, 1, 2)
        return ([DetectionDistribution(c, b, config) for c, b in zip(classes, boxes)],
                anchors, strides, classes.sigmoid())

    def native_loss(self, images, targets):
        images = images.to(self.device)
        height, width = images.shape[-2:]
        box_list, class_list, index_list = [], [], []
        for index, target in enumerate(targets):
            boxes = target['boxes'].to(self.device)
            xywh = torch.cat(((boxes[:, :2] + boxes[:, 2:]) / 2,
                              boxes[:, 2:] - boxes[:, :2]), -1)
            xywh = xywh / xywh.new_tensor((width, height, width, height))
            box_list.append(xywh)
            class_list.append(target['labels'].to(self.device).float().view(-1, 1))
            index_list.append(torch.full((len(boxes),), index, device=self.device))
        batch = {'img': images, 'batch_idx': torch.cat(index_list),
                 'cls': torch.cat(class_list), 'bboxes': torch.cat(box_list)}
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        try:
            loss, _ = self.model.loss(batch)
        finally:
            self.model.eval()
        return loss.sum() / len(images)

    @torch.no_grad()
    def predict(self, images, conf=0.001, nms_iou=0.60, max_det=300):
        """Ordinary deterministic YOLO inference, used identically for S0/DPO."""
        self.model.eval()
        raw = self.model(images.to(self.device))[0]
        xywh = raw[:, :4].permute(0, 2, 1)
        boxes = torch.cat((xywh[..., :2] - xywh[..., 2:] / 2,
                           xywh[..., :2] + xywh[..., 2:] / 2), -1)
        scores, labels = raw[:, 4:].max(1)
        results = []
        for b, s, c in zip(boxes, scores, labels):
            selected = (s >= conf).nonzero().flatten()
            keep = batched_nms(b[selected], s[selected], c[selected], nms_iou)[:max_det]
            selected = selected[keep]
            results.append({'boxes': b[selected].cpu(), 'scores': s[selected].cpu(),
                            'labels': c[selected].cpu()})
        return results
