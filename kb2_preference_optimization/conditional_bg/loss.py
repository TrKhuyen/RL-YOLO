"""Weight native background slots; preserve assignment, positives and denominator."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.utils.loss import v8DetectionLoss

from ..conditional.policy import ConditionalAdapter


class BackgroundBCE(nn.Module):
    """Only unassigned slots are downweighted, across all their classes.

    Zero targets at foreground slots retain weight one. No target-score sum or
    class weighting is changed in Ultralytics' surrounding loss implementation.
    """
    def __init__(self, background_weight, cls_gain=1., class_weights=None):
        super().__init__()
        if not math.isfinite(background_weight) or not 0 <= background_weight <= 1:
            raise ValueError('background_weight must be finite and in [0, 1]')
        self.background_weight = float(background_weight)
        self.cls_gain = float(cls_gain)
        self.class_weights = class_weights
        self.foreground = None
        self.last_parts = {}

    def set_foreground(self, foreground):
        self.foreground = foreground.detach().bool()

    def forward(self, logits, target):
        mask, self.foreground = self.foreground, None
        if mask is None or mask.shape != logits.shape[:-1]:
            raise ValueError('Fresh native foreground assignment required before each BCE call')
        base = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        weighted = base if self.background_weight == 1 else torch.where(
            mask.unsqueeze(-1), base, self.background_weight * base)
        # Diagnostic values reflect optional class weighting performed later by
        # the original loss. They are detached, never used to normalize training.
        with torch.no_grad():
            measured = base.detach()
            if self.class_weights is not None:
                measured = measured * self.class_weights
            denominator = target.sum().clamp_min(1)
            fg = measured[mask].sum() * self.cls_gain / denominator
            bg = measured[~mask].sum() * self.cls_gain / denominator
            self.last_parts = {'native_cls_foreground': fg,
                               'native_cls_background_unweighted': bg,
                               'native_cls_background_weighted': bg*self.background_weight,
                               'native_cls_weighted_total': fg+bg*self.background_weight,
                               'native_target_score_sum': denominator,
                               'native_foreground_slots': mask.sum(),
                               'native_background_slots': (~mask).sum()}
        return weighted


class BackgroundDetectionLoss(v8DetectionLoss):
    """Reuse native TAL assignment, box/DFL loss, gains and reduction unchanged."""
    def __init__(self, model, background_weight):
        super().__init__(model)
        self.bce = BackgroundBCE(background_weight, self.hyp.cls, self.class_weights)
        self.assignment_hook = self.assigner.register_forward_hook(self._assignment)

    def _assignment(self, module, inputs, output):
        self.bce.set_foreground(output[3])


class BackgroundAdapter(ConditionalAdapter):
    def __init__(self, checkpoint, device='cuda', background_weight=.5):
        super().__init__(checkpoint, device)
        self.model.criterion = BackgroundDetectionLoss(self.model, background_weight)

    def native_parts(self):
        return {k: float(v) for k, v in self.model.criterion.bce.last_parts.items()}
