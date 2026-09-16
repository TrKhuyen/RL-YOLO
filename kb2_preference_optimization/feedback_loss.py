"""Stable feedback-guided objective built on the native YOLO detection loss."""
import torch


def _xywh_to_xyxy(boxes):
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def dynamic_object_feedback_codes_from_predictions(
        predictions, targets, match_iou=0.5, localization_iou=0.1,
        confidence=0.25):
    boxes = _xywh_to_xyxy(predictions[:, :4].permute(0, 2, 1)).detach()
    probabilities = predictions[:, 4:].permute(0, 2, 1).detach()
    output = []
    for batch_index, target in enumerate(targets):
        gt_boxes = target['boxes'].to(predictions.device)
        gt_labels = target['labels'].to(predictions.device).long()
        codes = []
        for gt, label in zip(gt_boxes, gt_labels):
            candidates = boxes[batch_index]
            lt = torch.maximum(candidates[:, :2], gt[:2])
            rb = torch.minimum(candidates[:, 2:], gt[2:])
            inter = (rb - lt).clamp_min(0).prod(1)
            areas = (candidates[:, 2:] - candidates[:, :2]).clamp_min(0).prod(1)
            gt_area = (gt[2:] - gt[:2]).clamp_min(0).prod()
            ious = inter / (areas + gt_area - inter + 1e-8)
            selected = int(ious.argmax())
            best_iou = float(ious[selected])
            score, predicted_label = probabilities[batch_index, selected].max(0)
            if best_iou < localization_iou or float(score) < confidence:
                code = 3
            elif best_iou < match_iou:
                code = 2
            elif int(predicted_label) != int(label):
                code = 1
            else:
                code = 0
            codes.append(code)
        output.append(torch.tensor(codes, dtype=torch.long,
                                   device=predictions.device))
    return output


def object_feedback_loss_from_predictions(predictions, targets,
                                          cls_weight=1.0, box_weight=1.0):
    pred_boxes = _xywh_to_xyxy(predictions[:, :4].permute(0, 2, 1))
    probabilities = predictions[:, 4:].permute(0, 2, 1).clamp(1e-6, 1 - 1e-6)
    losses = []
    for batch_index, target in enumerate(targets):
        boxes = target['boxes'].to(predictions.device)
        labels = target['labels'].to(predictions.device).long()
        codes = target.get('object_feedback_codes')
        if codes is None:
            continue
        codes = codes.to(predictions.device).long()
        for object_index in torch.where(codes > 0)[0].tolist():
            gt = boxes[object_index]
            candidates = pred_boxes[batch_index]
            detached = candidates.detach()
            lt = torch.maximum(detached[:, :2], gt[:2])
            rb = torch.minimum(detached[:, 2:], gt[2:])
            inter = (rb - lt).clamp_min(0).prod(1)
            candidate_area = (detached[:, 2:] - detached[:, :2]).clamp_min(0).prod(1)
            gt_area = (gt[2:] - gt[:2]).clamp_min(0).prod()
            selected = int((inter / (candidate_area + gt_area - inter + 1e-8)).argmax())
            code = int(codes[object_index])
            item_loss = predictions.sum() * 0
            if code in (1, 3):
                item_loss = item_loss - cls_weight * torch.log(
                    probabilities[batch_index, selected, labels[object_index]])
            if code in (2, 3):
                box = candidates[selected]
                overlap = (torch.minimum(box[2:], gt[2:]) -
                           torch.maximum(box[:2], gt[:2])).clamp_min(0).prod()
                union = ((box[2:] - box[:2]).clamp_min(0).prod() +
                         gt_area - overlap + 1e-8)
                item_loss = item_loss + box_weight * (1 - overlap / union)
            losses.append(item_loss)
    return torch.stack(losses).mean() if losses else predictions.sum() * 0


def compute_object_feedback_loss(adapter, images, targets,
                                 cls_weight=1.0, box_weight=1.0,
                                 dynamic=False):
    predictions = adapter.raw_predictions(images)
    if dynamic:
        codes = dynamic_object_feedback_codes_from_predictions(
            predictions, targets)
        targets = [dict(target, object_feedback_codes=code)
                   for target, code in zip(targets, codes)]
    return object_feedback_loss_from_predictions(
        predictions, targets, cls_weight, box_weight)


def combine_projected_gradients(native_grads, feedback_grads, alpha=0.01,
                                eps=1e-12):
    if alpha < 0:
        raise ValueError('gradient feedback alpha must be non-negative')
    if len(native_grads) != len(feedback_grads):
        raise ValueError('gradient lists must have equal length')
    dot = sum((native * feedback).sum()
              for native, feedback in zip(native_grads, feedback_grads))
    native_norm_sq = sum(native.square().sum() for native in native_grads)
    feedback_norm_sq = sum(feedback.square().sum() for feedback in feedback_grads)
    coefficient = (dot / native_norm_sq.clamp_min(eps)).clamp_max(0)
    projected = [feedback - coefficient * native
                 for native, feedback in zip(native_grads, feedback_grads)]
    combined = [native + alpha * feedback
                for native, feedback in zip(native_grads, projected)]
    cosine = dot / (native_norm_sq.sqrt() * feedback_norm_sq.sqrt()).clamp_min(eps)
    return combined, {
        'gradient_dot': dot.detach(),
        'gradient_cosine': cosine.detach(),
        'gradient_projected': bool(float(dot.detach()) < 0),
    }


def apply_gradient_conflict_control(adapter, images, targets, alpha=0.01,
                                    cls_weight=1.0, box_weight=1.0,
                                    dynamic=True):
    parameters = [parameter for parameter in adapter.parameters()
                  if parameter.requires_grad]
    native_loss, loss_items = adapter.supervised_loss(images, targets)
    native_raw = torch.autograd.grad(
        native_loss, parameters, allow_unused=True)
    feedback_loss = compute_object_feedback_loss(
        adapter, images, targets, cls_weight, box_weight, dynamic)
    feedback_raw = torch.autograd.grad(
        feedback_loss, parameters, allow_unused=True)
    native_grads = [torch.zeros_like(parameter) if grad is None else grad
                    for parameter, grad in zip(parameters, native_raw)]
    feedback_grads = [torch.zeros_like(parameter) if grad is None else grad
                      for parameter, grad in zip(parameters, feedback_raw)]
    combined, diagnostics = combine_projected_gradients(
        native_grads, feedback_grads, alpha)
    for parameter, gradient in zip(parameters, combined):
        parameter.grad = gradient
    diagnostics.update({
        'native_loss': native_loss.detach(),
        'feedback_loss': native_loss.detach(),
        'object_feedback_loss': feedback_loss.detach(),
        'loss_items': loss_items,
    })
    return native_loss.detach() + alpha * feedback_loss.detach(), diagnostics


def normalized_difficulty_weights(targets, max_weight=3.0, eps=1e-6):
    """Return mean-one, bounded weights; neutral when a batch has no errors."""
    if max_weight < 1.0:
        raise ValueError('max_weight must be at least 1')
    if not targets:
        return torch.zeros(0)
    device = targets[0]['feedback_difficulty'].device
    difficulty = torch.stack([
        torch.as_tensor(t['feedback_difficulty'], device=device).float()
        for t in targets
    ])
    if not torch.isfinite(difficulty).all() or (difficulty < 0).any():
        raise ValueError('feedback_difficulty must be finite and non-negative')
    if float(difficulty.sum()) <= eps:
        return torch.ones_like(difficulty)
    # Adding one keeps easy examples active; normalization preserves loss scale.
    weights = (1.0 + difficulty).clamp(max=max_weight)
    return weights / weights.mean().clamp_min(eps)


def blend_native_losses(batch_loss, per_image_losses, weights, alpha=0.25):
    """Blend ordinary native loss and its feedback-weighted per-image form."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('feedback_alpha must be in [0, 1]')
    if per_image_losses.ndim != 1 or weights.ndim != 1:
        raise ValueError('per_image_losses and weights must be vectors')
    if len(per_image_losses) != len(weights) or not len(weights):
        raise ValueError('losses and weights must have the same non-zero length')
    weighted = (per_image_losses * weights.to(per_image_losses.device)).mean()
    return (1.0 - alpha) * batch_loss + alpha * weighted, weighted


def compute_hybrid_loss(adapter, images, targets, feedback_alpha=0.25,
                        max_feedback_weight=3.0, object_feedback_alpha=0.0,
                        object_cls_weight=1.0, object_box_weight=1.0,
                        dynamic_object_feedback=False):
    """Compute native batch loss plus a small error-aware native-loss term.

    The auxiliary term uses GT-based native YOLO loss, not frozen prediction
    boxes, so geometric augmentation remains valid.
    """
    batch_loss, loss_items = adapter.supervised_loss(images, targets)
    if feedback_alpha == 0:
        total = batch_loss
        object_loss = batch_loss.detach() * 0
        if object_feedback_alpha:
            object_loss = compute_object_feedback_loss(
                adapter, images, targets, object_cls_weight, object_box_weight,
                dynamic_object_feedback)
            total = total + object_feedback_alpha * object_loss
        return total, {
            'native_loss': batch_loss.detach(),
            'feedback_loss': batch_loss.detach(),
            'object_feedback_loss': object_loss.detach(),
            'weights': torch.ones(len(targets), device=batch_loss.device),
            'loss_items': loss_items,
        }
    weights = normalized_difficulty_weights(targets, max_feedback_weight)
    per_image = []
    for index, target in enumerate(targets):
        sample_loss, _ = adapter.supervised_loss(images[index:index + 1], [target])
        per_image.append(sample_loss)
    total, feedback_loss = blend_native_losses(
        batch_loss, torch.stack(per_image), weights, feedback_alpha)
    object_loss = batch_loss.detach() * 0
    if object_feedback_alpha:
        object_loss = compute_object_feedback_loss(
            adapter, images, targets, object_cls_weight, object_box_weight,
            dynamic_object_feedback)
        total = total + object_feedback_alpha * object_loss
    return total, {
        'native_loss': batch_loss.detach(),
        'feedback_loss': feedback_loss.detach(),
        'object_feedback_loss': object_loss.detach(),
        'weights': weights.detach(),
        'loss_items': loss_items,
    }
