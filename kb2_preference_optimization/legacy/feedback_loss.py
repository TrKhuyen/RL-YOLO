"""Feedback, pairwise ranking and candidate-selection DPO objectives."""
import math
import torch
import torch.nn.functional as F

from feedback import box_iou

DPO_PAIR_SOURCE = 'online_policy_gt_iou_v2'
DPO_OBJECTIVE_VERSION = 'candidate_selection_online_dpo_v2'


def _xywh_to_xyxy(boxes):
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def select_preference_pairs(predictions, targets, chosen_iou=0.5, rejected_iou=0.1,
                            return_quality=False, rank_all_lower_iou=False):
    """Return [image, GT class, chosen anchor, rejected anchor] indices.

    Candidate selection uses detached boxes; class scores retain gradients.
    Each candidate belongs to its highest-IoU GT, so a positive for one GT
    cannot be used as a rejected candidate for another.
    With return_quality=True, also return detached [IoU(A), IoU(B)] per pair.
    rank_all_lower_iou includes inferior candidates above chosen_iou as well;
    the default retains the original binary-cutoff pairwise ablation.
    """
    if not 0 <= rejected_iou < chosen_iou <= 1:
        raise ValueError('invalid preference IoU thresholds')
    if predictions.ndim != 3 or predictions.shape[1] < 5:
        raise ValueError('expected decoded predictions [B, 4 + classes, anchors]')
    if len(predictions) != len(targets):
        raise ValueError('prediction and target batch sizes differ')
    boxes = _xywh_to_xyxy(predictions[:, :4].permute(0, 2, 1)).detach()
    probabilities = predictions[:, 4:, :].permute(0, 2, 1)
    pairs = []
    qualities = []
    for batch_index, target in enumerate(targets):
        gt_boxes = target['boxes'].to(predictions.device)
        gt_labels = target['labels'].to(predictions.device).long()
        if not len(gt_boxes):
            continue
        if gt_labels.min() < 0 or gt_labels.max() >= probabilities.shape[-1]:
            raise ValueError('GT label outside prediction class range')
        ious = box_iou(boxes[batch_index], gt_boxes)
        owner = ious.argmax(dim=1)
        for gt_index, label in enumerate(gt_labels):
            same_gt = owner == gt_index
            quality = ious[:, gt_index]
            chosen_mask = same_gt & (quality >= chosen_iou)
            if not chosen_mask.any():
                continue
            chosen = quality.masked_fill(~chosen_mask, -1).argmax()
            cutoff = quality[chosen] - 1e-6 if rank_all_lower_iou else chosen_iou
            rejected_mask = same_gt & (quality >= rejected_iou) & (quality < cutoff)
            if not rejected_mask.any():
                continue
            scores = probabilities[batch_index, :, label].clamp(1e-6, 1 - 1e-6)
            # Highest-IoU A; highest GT-class score among inferior candidates B.
            rejected = scores.detach().masked_fill(~rejected_mask, -1).argmax()
            pairs.append((batch_index, int(label), int(chosen), int(rejected)))
            if return_quality:
                qualities.append(torch.stack((quality[chosen], quality[rejected])))
    pairs = torch.tensor(pairs, device=predictions.device, dtype=torch.long).reshape(-1, 4)
    if return_quality:
        return pairs, (torch.stack(qualities) if qualities else boxes.new_empty((0, 2)))
    return pairs


def pairwise_preference_loss_from_predictions(
        predictions, targets, margin=0.1, chosen_iou=0.5, rejected_iou=0.1):
    """Rank candidates selected from the current model, preserving the old loss."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('invalid pairwise margin')
    pairs = select_preference_pairs(predictions, targets, chosen_iou, rejected_iou)
    if not len(pairs):
        return predictions.sum() * 0, 0
    image, label, chosen, rejected = pairs.unbind(1)
    logits = torch.logit(predictions[:, 4:, :].clamp(1e-6, 1 - 1e-6))
    gaps = logits[image, label, chosen] - logits[image, label, rejected]
    return F.softplus(margin - gaps).mean(), len(pairs)


def candidate_selection_log_probs(predictions):
    """Categorical policy over anchor indices, conditional on image and GT class.

    YOLO's independent sigmoid class scores are converted to logits, then
    softmax-normalized over ALL anchors for that class. This defines a policy
    for selecting an anchor; it is not a likelihood of a complete detection set.
    """
    if predictions.ndim != 3 or predictions.shape[1] < 5 or predictions.shape[2] < 1:
        raise ValueError('expected decoded predictions [B, 4 + classes, anchors]')
    # Keep the policy calculation stable if inference later uses fp16/bfloat16.
    scores = predictions[:, 4:, :].float()
    if not torch.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError('candidate class probabilities must be finite and in [0, 1]')
    return F.log_softmax(torch.logit(scores.clamp(1e-6, 1 - 1e-6)), dim=-1)


def dpo_preference_loss_from_predictions(predictions, reference_predictions,
                                         targets, beta=0.1):
    """Online DPO: current policy box quality against GT supplies preferences.

    Detached CURRENT predictions select A/B; reference geometry never labels
    preferences. Both policies score the same selected anchor indices. This is
    a candidate-index policy, not a likelihood of complete detection sets.
    The discrete choice has no gradient; native loss still learns box regression.
    Online labels can change next step as the current geometry changes, unlike
    the original offline DPO data distribution. Reference scores are detached.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('dpo beta must be finite and positive')
    if predictions.shape != reference_predictions.shape:
        raise ValueError('policy and reference prediction shapes must match')
    reference_predictions = reference_predictions.detach()
    log_policy = candidate_selection_log_probs(predictions)
    log_reference = candidate_selection_log_probs(reference_predictions)
    pairs, qualities = select_preference_pairs(
        predictions.detach(), targets, return_quality=True, rank_all_lower_iou=True)
    zero = predictions.sum() * 0
    if not len(pairs):
        return zero, {'pair_count': 0, 'relative_margin': zero.detach(),
                      'policy_win_rate': zero.detach(), 'reference_win_rate': zero.detach(),
                      'relative_win_rate': zero.detach(), 'policy_margin': zero.detach(),
                      'chosen_iou': zero.detach(), 'rejected_iou': zero.detach(),
                      'quality_gap': zero.detach()}
    image, label, chosen, rejected = pairs.unbind(1)
    policy_gap = log_policy[image, label, chosen] - log_policy[image, label, rejected]
    reference_gap = (log_reference[image, label, chosen]
                     - log_reference[image, label, rejected])
    relative_margin = policy_gap - reference_gap
    loss = -F.logsigmoid(beta * relative_margin).mean()
    return loss, {'pair_count': len(pairs),
                  'relative_margin': relative_margin.detach().mean(),
                  'policy_win_rate': (policy_gap.detach() > 0).float().mean(),
                  'reference_win_rate': (reference_gap > 0).float().mean(),
                  'relative_win_rate': (relative_margin.detach() > 0).float().mean(),
                  'policy_margin': policy_gap.detach().mean(),
                  'chosen_iou': qualities[:, 0].mean(),
                  'rejected_iou': qualities[:, 1].mean(),
                  'quality_gap': (qualities[:, 0] - qualities[:, 1]).mean()}


def compute_dpo_preference_loss(adapter, reference, images, targets, beta=0.1):
    with torch.no_grad():
        reference_predictions = reference.raw_predictions(images)
    return dpo_preference_loss_from_predictions(
        adapter.raw_predictions(images), reference_predictions, targets, beta)


def compute_pairwise_preference_loss(adapter, images, targets, margin=0.1):
    return pairwise_preference_loss_from_predictions(
        adapter.raw_predictions(images), targets, margin=margin)


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
    if dynamic:
        # Error status is defined on actual detections, not on dense candidates.
        from feedback import build_feedback_record
        with torch.no_grad():
            detections = adapter.forward_with_grad(images, 0.25, 0.45)
            records = [build_feedback_record(detection, target)
                       for detection, target in zip(detections, targets)]
        code_map = {'matched': 0, 'wrong_class': 1,
                    'bad_localization': 2, 'missed': 3}
        targets = [
            dict(target, object_feedback_codes=torch.tensor(
                [code_map[status] for status in record['gt_status']],
                dtype=torch.long, device=images.device))
            for target, record in zip(targets, records)
        ]
    predictions = adapter.raw_predictions(images)
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
        'feedback_loss': feedback_loss.detach(),
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
