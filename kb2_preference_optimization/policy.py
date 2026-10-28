"""A normalized policy over a complete indexed detection action.

The action records background/class at EVERY grid location and four DFL bins
at emitted locations. Its likelihood is exact under this factorized policy.
It is NOT the marginal likelihood of an unordered, post-NMS box set.
"""
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms


@dataclass(frozen=True)
class PolicyConfig:
    class_temperature: float = 1.0
    box_temperature: float = 1.0
    background_logit: float = 0.0
    nms_iou: float = 0.60
    max_det: int = 300

    def __post_init__(self):
        for name in ('class_temperature', 'box_temperature'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.background_logit):
            raise ValueError('background_logit must be finite')
        if not 0 < self.nms_iou <= 1 or self.max_det < 1:
            raise ValueError('invalid rendering NMS/cap')

    def to_dict(self):
        return asdict(self)


class DetectionDistribution:
    """One image, [M,C] raw class logits and [M,4,R] raw DFL logits."""

    def __init__(self, class_logits, box_logits, config=PolicyConfig()):
        if (class_logits.ndim != 2 or box_logits.ndim != 3
                or box_logits.shape[:2] != (len(class_logits), 4)
                or len(class_logits) < 1 or class_logits.shape[1] < 1
                or box_logits.shape[2] <= 1):
            raise ValueError('expected [M,C] and [M,4,R>1] logits')
        if not torch.isfinite(class_logits).all() or not torch.isfinite(box_logits).all():
            raise ValueError('nonfinite policy logits')
        self.config = config
        self.slots, self.classes = class_logits.shape
        self.reg_max = box_logits.shape[-1]
        background = torch.full_like(class_logits[:, :1], config.background_logit)
        self.class_logps = F.log_softmax(
            torch.cat((background, class_logits.float()), -1) / config.class_temperature, -1)
        self.box_logps = F.log_softmax(box_logits.float() / config.box_temperature, -1)

    @torch.no_grad()
    def sample(self, generator=None):
        categories = torch.multinomial(self.class_logps.exp(), 1,
                                       generator=generator).squeeze(-1)
        indices = (categories > 0).nonzero().flatten()
        bins = torch.multinomial(
            self.box_logps[indices].exp().reshape(-1, self.reg_max), 1,
            generator=generator).reshape(-1, 4) if len(indices) else indices.new_empty((0, 4))
        return {'slots': self.slots, 'indices': indices.cpu().tolist(),
                'classes': (categories[indices] - 1).cpu().tolist(),
                'bins': bins.cpu().tolist()}

    def action_tensors(self, action):
        if action.get('slots') != self.slots:
            raise ValueError('action grid differs from policy')
        device = self.class_logps.device
        # Reject fractional values, duplicate slots and out-of-support actions.
        for key in ('indices', 'classes', 'bins'):
            values = action[key]
            flat = [v for row in values for v in row] if key == 'bins' else values
            if any(type(v) is not int for v in flat):
                raise ValueError(f'{key} must contain integers')
        indices = torch.tensor(action['indices'], dtype=torch.long, device=device)
        classes = torch.tensor(action['classes'], dtype=torch.long, device=device)
        bins = torch.tensor(action['bins'], dtype=torch.long, device=device).reshape(-1, 4)
        if len(indices) != len(classes) or len(indices) != len(bins):
            raise ValueError('action lengths differ')
        if (len(set(action['indices'])) != len(indices) or (indices < 0).any()
                or (indices >= self.slots).any() or (classes < 0).any()
                or (classes >= self.classes).any() or (bins < 0).any()
                or (bins >= self.reg_max).any()):
            raise ValueError('action outside policy support')
        return indices, classes, bins

    def log_prob(self, action):
        indices, classes, bins = self.action_tensors(action)
        # Float64 reduction avoids loss of precision on large complete grids.
        value = self.class_logps[:, 0].double().sum()
        if len(indices):
            value = value + (self.class_logps[indices, classes + 1].double()
                             - self.class_logps[indices, 0].double()).sum()
            sides = torch.arange(4, device=indices.device)
            value = value + self.box_logps[indices[:, None], sides, bins].double().sum()
        return value


@torch.no_grad()
def render_action(distribution, action, anchors, strides, reference_scores, image_size):
    """Decode a stored action using FIXED reference priorities for NMS.

    No current-policy geometry/scores or GT can change a saved output. Scores
    only order NMS; categorical sampling already decided whether to emit a box.
    Degenerate boxes disappear in rendering but remain in action likelihood.
    """
    indices, labels, bins = distribution.action_tensors(action)
    if anchors.shape != (distribution.slots, 2) or strides.shape != (distribution.slots, 1):
        raise ValueError('anchor/stride shape mismatch')
    if reference_scores.shape != (distribution.slots, distribution.classes):
        raise ValueError('reference score shape mismatch')
    device = anchors.device
    indices, labels, bins = indices.to(device), labels.to(device), bins.to(device)
    boxes = torch.cat((anchors[indices] - bins[:, :2],
                       anchors[indices] + bins[:, 2:]), -1) * strides[indices]
    boxes = boxes.float().clamp(0, image_size)
    scores = reference_scores[indices, labels].float()
    valid = (boxes[:, 2:] > boxes[:, :2]).all(-1)
    boxes, labels, scores = boxes[valid], labels[valid], scores[valid]
    # Strictly ordered priorities make tied reference scores deterministic.
    order = scores.argsort(descending=True, stable=True)
    boxes, labels, scores = boxes[order], labels[order], scores[order]
    priorities = torch.arange(len(boxes), 0, -1, dtype=torch.float32, device=device)
    keep = batched_nms(boxes, priorities, labels, distribution.config.nms_iou)
    keep = keep[:distribution.config.max_det]
    return {'boxes': boxes[keep].cpu(), 'labels': labels[keep].cpu(),
            'scores': scores[keep].cpu()}


def dpo_loss(chosen_logp, rejected_logp, reference_chosen_logp,
             reference_rejected_logp, beta=0.1):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('beta must be finite and positive')
    reference_gap = (torch.as_tensor(reference_chosen_logp, device=chosen_logp.device, dtype=chosen_logp.dtype)
                     - torch.as_tensor(reference_rejected_logp, device=chosen_logp.device, dtype=chosen_logp.dtype)).detach()
    margin = (chosen_logp - rejected_logp) - reference_gap
    return F.softplus(-beta * margin).mean(), margin.detach()
