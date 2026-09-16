"""Shared post-processing contract for all detector adapters."""

import torch
from torchvision.ops import batched_nms


def canonical_postprocess(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    conf_thres: float,
    iou_thres: float,
    max_det: int = 300,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply identical confidence filtering and class-aware NMS."""
    with torch.no_grad():
        candidates = torch.where(scores.detach() >= conf_thres)[0]
        if candidates.numel() == 0:
            return candidates, boxes.new_zeros((0, 4)), labels.new_zeros((0,))
        keep = batched_nms(
            boxes[candidates].detach(),
            scores[candidates].detach(),
            labels[candidates].detach(),
            iou_thres,
        )[:max_det]
        selected = candidates[keep]
    return selected, boxes[selected].detach(), labels[selected].long().detach()
