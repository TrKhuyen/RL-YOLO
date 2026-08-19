"""
train_rl.py - Giai doan 2: RL Fine-tuning YOLO bang REINFORCE.

Thiet ke theo:
- bwconrad/cv-rl:  pattern CE-pretrain -> RL fine-tune (tranh train tu dau)
- yanivnik:        recall reward + log(avg_confidence) lam log-prob xap xi
- Cai tien:        EMA baseline (on dinh hon ca 2 paper nguon)

Chay:
    python train_rl.py                       # fine-tune tat ca model
    python train_rl.py --model dp_yolo       # chi fine-tune DP-YOLO
    python train_rl.py --model yolov8n --steps 30000 --lr 5e-7
    python train_rl.py --model dp_yolo --freeze  # freeze backbone

Fix list:
    [v1.1] Checkpoint luu theo rolling average (200 buoc) thay vi spike tuc thoi
    [v1.1] Them eval_interval: danh gia tren val set dinh ky
    [v1.1] Format checkpoint: dict co is_rl_checkpoint=True (tuong thich evaluate.py)
    [v1.1] Them quick_eval() dung torchmetrics
"""

import argparse
import importlib.util
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from reward import compute_reward
from dataloader import get_pest_dataloader

# ── Hardware constants (i9-14900HX, 16GB RAM, RTX 4060 8GB VRAM) ────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT.parent / 'pre-data' / 'data' / 'v2i_cleanned'
NUM_WORKERS = 4           # giam tu 8 -> 4 cho 16GB RAM


# =============================================================================
# 1. EMA Baseline - giam variance REINFORCE
# =============================================================================

class EMABaseline:
    """
    Exponential Moving Average baseline de giam variance cua REINFORCE.

    advantage_i = R_i - b   (unbiased khi E[b] = E[R])

    Ly do dung EMA thay vi:
    - bwconrad: dung sample thu 2 (Monte Carlo) -> ton 2x forward
    - yanivnik: chua implement baseline -> variance cao
    """
    def __init__(self, alpha: float = 0.99):
        self.alpha = alpha
        self.value: float | None = None
        self.sq_value: float | None = None

    def update(self, reward: float) -> float:
        if self.value is None:
            self.value = reward
        else:
            self.value = self.alpha * self.value + (1.0 - self.alpha) * reward
        return self.value

    def advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        """Tra ve advantage = rewards - baseline (Tensor, no grad)."""
        b = self.update(rewards.mean().item())
        return rewards - b

    def normalized_advantage(self, rewards, clip=3.0):
        mean = self.update(rewards.mean().item())
        sq_mean = rewards.square().mean().item()
        if self.sq_value is None:
            self.sq_value = sq_mean
        else:
            self.sq_value = (self.alpha * self.sq_value
                             + (1.0 - self.alpha) * sq_mean)
        std = max(self.sq_value - mean * mean, 1e-4) ** 0.5
        return ((rewards - mean) / std).clamp(-clip, clip)


# =============================================================================
# 2a. Log-probability xap xi
# =============================================================================

def compute_log_prob(preds: list[dict]) -> torch.Tensor:
    """
    Xap xi log pi_theta(a|s) = log(average confidence) tren batch.

    Ly do dung xap xi (tu yanivnik):
    YOLO khong phai autoregressive model nen khong co log-prob day du.
    avg_confidence la proxy hop ly: confidence cao -> policy "chac chan"
    -> gradient di dung huong toi uu reward.

    Clamp ve [1e-20, 1.0] de tranh log(0) = -inf.
    """
    log_probs = []
    for pred in preds:
        if 'policy_log_prob' in pred:
            log_probs.append(pred['policy_log_prob'])
            continue
        scores = pred['scores']
        if scores.numel() == 0:
            # Khong co prediction -> log prob thap nhat
            log_probs.append(
                torch.tensor(-20.0, requires_grad=True,
                             device=scores.device if scores.numel() > 0 else 'cpu')
            )
        else:
            avg_conf = scores.mean().clamp(1e-20, 1.0)
            log_probs.append(torch.log(avg_conf))
    return torch.stack(log_probs)


def match_aware_objective(preds, targets, iou_threshold=0.5,
                          tp_weight=1.0, fp_weight=0.5, fn_weight=1.5):
    # TP/FP/FN-aware action likelihood and confidence surrogate.
    from torchvision.ops import box_iou
    log_probs = []
    for pred, target in zip(preds, targets):
        scores = pred['scores'].clamp(1e-6, 1.0 - 1e-6)
        boxes, labels = pred['boxes'].detach(), pred['labels'].detach()
        gt_boxes = target['boxes'].to(boxes.device).detach()
        gt_labels = target['labels'].to(labels.device).detach()
        tp = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)
        used = set()
        if len(scores) and len(gt_boxes):
            ious = box_iou(gt_boxes.float(), boxes.float())
            for pred_idx in scores.detach().argsort(descending=True).tolist():
                valid = gt_labels == labels[pred_idx]
                if used:
                    valid[list(used)] = False
                candidates = torch.where(valid)[0]
                if not len(candidates):
                    continue
                candidate_ious = ious[candidates, pred_idx]
                pos = int(candidate_ious.argmax())
                gt_idx = int(candidates[pos])
                if float(candidate_ious[pos]) >= iou_threshold:
                    used.add(gt_idx)
                    tp[pred_idx] = True

        parts = []
        if tp.any():
            parts.append(tp_weight * torch.log(scores[tp]).mean())
        if (~tp).any():
            parts.append(fp_weight * torch.log1p(-scores[~tp]).mean())
        missed = max(len(gt_boxes) - len(used), 0)
        candidate_score = pred.get('max_score_all')
        if missed and candidate_score is not None:
            candidate_score = candidate_score.clamp(1e-6, 1.0 - 1e-6)
            parts.append(fn_weight * missed / max(len(gt_boxes), 1)
                         * torch.log(candidate_score))
        log_prob = (torch.stack(parts).sum() if parts
                    else pred['policy_log_prob'] * 0.0)
        log_probs.append(log_prob)
    values = torch.stack(log_probs)
    return values, -values.mean()


# =============================================================================
# 2b. Quick Evaluation (dung trong eval_interval)
# =============================================================================

def quick_eval(
    adapter,
    val_loader,
    device:     str   = 'cuda',
    conf_thres: float = 0.001,
    iou_thres:  float = 0.60,
) -> dict:
    """
    Danh gia nhanh tren val set sau moi eval_interval buoc RL.

    Dung torchmetrics.detection.MeanAveragePrecision.
    Tu dong switch sang eval mode roi tra lai train mode.

    Returns:
        dict voi keys: 'mAP50', 'recall' (max recall @ 100 det)
    """
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError:
        print("    [WARN] torchmetrics not installed. Skip eval_interval.")
        return {'mAP': 0.0, 'mAP50': 0.0, 'recall': 0.0}

    adapter.eval_mode()
    metric = MeanAveragePrecision(
        class_metrics=False,
        backend='faster_coco_eval',
    )

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            if batch_idx == 0:
                validate_yolo_input(images)
            preds  = adapter.forward_with_grad(
                images, conf_thres=conf_thres, iou_thres=iou_thres
            )
            tm_preds, tm_targets = [], []
            for pred, tgt in zip(preds, targets):
                tm_preds.append({
                    'boxes':  pred['boxes'].cpu(),
                    'scores': pred['scores'].detach().float().cpu(),
                    'labels': pred['labels'].cpu(),
                })
                tm_targets.append({
                    'boxes':  tgt['boxes'].cpu(),
                    'labels': tgt['labels'].cpu(),
                })
            metric.update(tm_preds, tm_targets)

    res = metric.compute()
    adapter.train_mode()

    return {
        'mAP':    float(res.get('map',     torch.tensor(0.0)).item()),
        'mAP50':  float(res.get('map_50',  torch.tensor(0.0)).item()),
        'recall': float(res.get('mar_100', torch.tensor(0.0)).item()),
    }


def validation_score(metrics: dict) -> float:
    '''Checkpoint score balancing easy IoU, strict IoU and recall.'''
    return (0.4 * metrics['mAP50']
            + 0.4 * metrics['mAP']
            + 0.2 * metrics['recall'])


def validate_yolo_input(images: torch.Tensor, tolerance: float = 1e-6) -> None:
    '''Fail fast when a dataloader violates the native YOLO [0, 1] contract.'''
    if not torch.isfinite(images).all():
        raise ValueError('RL input contains NaN or Inf values')
    low, high = float(images.min()), float(images.max())
    if low < -tolerance or high > 1.0 + tolerance:
        raise ValueError(
            f'YOLO RL input must be RGB float in [0, 1], got [{low:.4f}, {high:.4f}]'
        )


def l2sp_loss(
    trainable_named: list[tuple[str, torch.nn.Parameter]],
    reference: dict[str, torch.Tensor],
) -> torch.Tensor:
    '''Mean parameter drift from the supervised initialization.'''
    if not trainable_named:
        raise ValueError('No trainable parameters available for L2-SP')
    terms = [(param - reference[name]).pow(2).mean()
             for name, param in trainable_named]
    return torch.stack(terms).mean()


# =============================================================================
# 3. Adapter factory
# =============================================================================

def load_adapter(model_name: str, checkpoint: str, device: str):
    """
    Tao adapter phu hop cho tung model family.
    Tra ve object co .forward_with_grad(), .parameters(), .named_parameters().
    """
    from adapters import YOLOv5Adapter, UltralyticsAdapter

    if model_name == 'dp_yolo':
        patch_path = PROJECT_ROOT / 'models' / 'dp_yolo' / 'patch_yolov5.py'
        spec = importlib.util.spec_from_file_location('dp_yolo_patch_runtime', patch_path)
        patch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch_module)
        if not patch_module.patch():
            raise RuntimeError(f'Failed to apply DP-YOLO runtime patch: {patch_path}')
        return YOLOv5Adapter(checkpoint, device=device)
    if model_name == 'yolov5s':
        return YOLOv5Adapter(checkpoint, device=device)
    elif model_name in ('yolov8n', 'yolov8s', 'yolov11n', 'yolov11s'):
        return UltralyticsAdapter(checkpoint, device=device)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            "Them vao adapter factory neu can."
        )


# =============================================================================
# 4. Freeze backbone helper
# =============================================================================

def freeze_backbone(adapter, model_name: str) -> int:
    """
    Dong bang tham so backbone de tranh catastrophic forgetting.
    Tra ve so tham so bi freeze.
    """
    frozen = 0
    for name, param in adapter.named_parameters():
        should_freeze = False
        if model_name in ('yolov5s', 'dp_yolo'):
            # YOLOv5: backbone la model.0 den model.9
            if any(f'model.{i}.' in name for i in range(10)):
                should_freeze = True
        else:
            # Ultralytics YOLOv8/v11: backbone nam trong model.model[:-2]
            if 'model.model.' in name:
                try:
                    layer_idx = int(name.split('model.model.')[1].split('.')[0])
                    if layer_idx < 10:
                        should_freeze = True
                except (IndexError, ValueError):
                    pass

        if should_freeze:
            param.requires_grad = False
            frozen += 1

    return frozen


# =============================================================================
# 5. Main RL training loop
# =============================================================================

def rl_finetune(
    model_name: str,
    checkpoint: str,
    cfg:        dict,
    device:     str = 'cuda',
):
    """
    Vong lap REINFORCE fine-tuning cho 1 YOLO model.

    Args:
        model_name: ten model ('yolov5s', 'yolov8n', 'yolov11n', 'dp_yolo')
        checkpoint: path den best.pt tu supervised training
        cfg:        dict hyperparameters tu hyp.rl.yaml
        device:     'cuda' hoac 'cpu'
    """
    seed = int(cfg.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = PROJECT_ROOT / 'rl_checkpoints'
    output_dir.mkdir(exist_ok=True)

    run_id = f"{model_name}_{int(time.time())}"
    writer = SummaryWriter(f"results/tensorboard/rl_{run_id}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"  Loading {model_name} from {checkpoint}...")
    adapter = load_adapter(model_name, checkpoint, device)
    adapter.train_mode()
    mode = cfg.get('mode', 'kb1b')
    has_native_loss = hasattr(adapter, 'native_detection_loss')
    if mode == 'native-only' and not has_native_loss:
        raise ValueError(
            'native-only currently supports YOLOv8/YOLOv11 only')
    if mode == 'native-only':
        method_name = 'native_only_control'
        checkpoint_tag = 'native_only'
    else:
        method_name = ('kb1b_native_reward_v31' if has_native_loss
                       else 'kb1b_reward_guided_v2')
        checkpoint_tag = 'rl'

    # ── Freeze backbone (tuy chon) ──────────────────────────────────────────
    if cfg.get('train_head_only', True):
        frozen, trainable_count = adapter.freeze_except_detection_head()
        print('  Detect-head only:', trainable_count, 'trainable /',
              frozen, 'frozen parameters.')
    elif cfg.get('freeze_backbone', False):
        n = freeze_backbone(adapter, model_name)
        print(f"  Frozen {n} backbone parameters.")

    # ── Optimizer ───────────────────────────────────────────────────────────
    trainable_named = [(name, param) for name, param in adapter.named_parameters()
                       if param.requires_grad]
    trainable = [param for _, param in trainable_named]
    if not trainable:
        raise RuntimeError('KB1-B has no trainable parameters')
    supervised_reference = {
        name: param.detach().clone() for name, param in trainable_named
    }
    optimizer = torch.optim.Adam(trainable, lr=cfg['lr'])

    # ── DataLoaders (train + val) ─────────────────────────────────────
    loader = get_pest_dataloader(
        DATA_ROOT, split='train',
        batch_size=cfg.get('batch_size', 16),
        img_size=640,
        num_workers=NUM_WORKERS,
    )
    val_loader = get_pest_dataloader(
        DATA_ROOT, split='val',
        batch_size=cfg.get('batch_size', 16),
        img_size=640,
        num_workers=NUM_WORKERS,
    )

    # ── RL state ────────────────────────────────────────────────────────────
    baseline        = EMABaseline(alpha=cfg.get('ema_alpha', 0.99))
    reward_hist     = deque(maxlen=200)
    best_avg_reward = -float('inf')
    best_val_score  = -float('inf')
    no_improve_evals = 0
    best_ckpt = output_dir / (
        f'{model_name}_seed{seed}_{checkpoint_tag}_best.pt')
    reward_ckpt = output_dir / (
        f'{model_name}_seed{seed}_{checkpoint_tag}_reward_best.pt')
    data_iter       = iter(loader)
    steps           = cfg.get('steps', 50_000)
    log_interval    = cfg.get('log_interval',  100)
    save_interval   = cfg.get('save_interval', 5_000)
    eval_interval   = cfg.get('eval_interval', 5_000)   # [fix] implement
    early_patience  = cfg.get('early_stopping_patience', 3)
    min_delta       = cfg.get('early_stopping_min_delta', 1e-4)
    reward_loss_weight = float(cfg.get('reward_loss_weight', 1.0))
    native_loss_weight = float(
        cfg.get('native_supervised_loss_weight', 1.0))
    supervised_loss_weight = float(cfg.get('supervised_loss_weight', 1.0))
    stability_loss_weight = float(cfg.get('stability_loss_weight', 0.01))
    accumulation_steps = max(int(cfg.get('gradient_accumulation_steps', 1)), 1)
    if mode == 'native-only':
        reward_loss_weight = 0.0
        supervised_loss_weight = 0.0
    print('  Mode:', mode, '| method:', method_name)

    print(f"\n{'='*60}")
    print(f"  RL Fine-tuning: {model_name}")
    print(f"  Steps: {steps}  LR: {cfg['lr']}  "
          f"Freeze: {cfg.get('freeze_backbone', False)}")
    print(f"  Reward: {cfg.get('reward_type','composite')}  "
          f"alpha={cfg.get('reward_alpha', 0.6)}")
    print(f"{'='*60}\n")

    print('  [Eval step 0] Measuring supervised baseline...')
    baseline_metrics = quick_eval(adapter, val_loader, device=device)
    best_val_score = validation_score(baseline_metrics)
    torch.save({
        'is_rl_checkpoint': True, 'method': method_name,
        'model_name': model_name, 'mode': mode,
        'seed': seed, 'step': 0,
        'val_score': best_val_score,
        'val_mAP': baseline_metrics['mAP'],
        'val_mAP50': baseline_metrics['mAP50'],
        'val_recall': baseline_metrics['recall'],
        'state_dict': adapter.state_dict(),
    }, str(best_ckpt))
    print('    -> Baseline mAP50={mAP50:.4f} mAP50-95={mAP:.4f} '
          'recall={recall:.4f}'.format(**baseline_metrics))

    optimizer.zero_grad()
    for step in range(1, steps + 1):
        if no_improve_evals >= early_patience:
            print('    -> Early stopping: validation did not improve.')
            break

        warmup_steps = int(cfg.get('warmup_steps', 0))
        max_lr = float(cfg['lr'])
        min_lr = float(cfg.get('min_lr', max_lr))
        if warmup_steps and step <= warmup_steps:
            current_lr = min_lr + (max_lr - min_lr) * step / warmup_steps
        elif cfg.get('scheduler') == 'cosine':
            progress = ((step - warmup_steps)
                        / max(steps - warmup_steps, 1))
            current_lr = min_lr + 0.5 * (max_lr - min_lr) * (
                1.0 + math.cos(math.pi * progress))
        else:
            current_lr = max_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # ── Lay batch ────────────────────────────────────────────────────────
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, targets = next(data_iter)

        images = images.to(device)
        if step == 1:
            validate_yolo_input(images)

        # ── Forward (giu grad qua confidence scores) ─────────────────────────
        preds = adapter.forward_with_grad(
            images,
            conf_thres=cfg.get('conf_thres', 0.20),
            iou_thres=cfg.get('iou_thres',  0.45),
        )

        if hasattr(adapter, 'native_detection_loss'):
            native_loss, native_items = adapter.native_detection_loss(
                images, targets)
            proxy_weight = supervised_loss_weight
        else:
            native_loss = images.new_zeros(())
            native_items = {}
            proxy_weight = float(cfg.get('fallback_proxy_loss_weight', 1.0))

        # ── Tinh reward (khong can grad) ─────────────────────────────────────
        with torch.no_grad():
            rewards = compute_reward(
                preds, targets,
                reward_type=cfg.get('reward_type', 'composite'),
                alpha=cfg.get('reward_alpha', 0.6),
                iou_threshold=cfg.get('iou_threshold', 0.5),
                small_thresh=cfg.get('small_thresh', 32),
                duplicate_penalty=cfg.get('duplicate_penalty', 0.3),
                iou_thresholds=cfg.get('reward_iou_thresholds',
                                       [0.5, 0.6, 0.7, 0.8]),
                precision_weight=cfg.get('precision_weight', 0.35),
                recall_weight=cfg.get('recall_weight', 0.35),
                iou_weight=cfg.get('iou_weight', 0.30),
            ).to(device)

        # ── EMA baseline -> advantage ─────────────────────────────────────────
        if cfg.get('normalize_advantage', True):
            advantage = baseline.normalized_advantage(
                rewards, float(cfg.get('advantage_clip', 3.0)))
        else:
            advantage = baseline.advantage(rewards)

        # ── Log-probability xap xi ────────────────────────────────────────────
        log_probs, supervised_loss = match_aware_objective(
            preds, targets,
            iou_threshold=float(cfg.get('iou_threshold', 0.5)),
            tp_weight=float(cfg.get('tp_weight', 1.0)),
            fp_weight=float(cfg.get('fp_weight', 0.5)),
            fn_weight=float(cfg.get('fn_weight', 1.5)),
        )

        # ── REINFORCE loss ────────────────────────────────────────────────────
        # L = -E[log pi(a|s) * advantage]  (dau tru: minimize -> maximize reward)
        reward_loss = -torch.mean(log_probs * advantage.detach())
        stability_loss = l2sp_loss(trainable_named, supervised_reference)
        loss = (reward_loss_weight * reward_loss
                + native_loss_weight * native_loss
                + proxy_weight * supervised_loss
                + stability_loss_weight * stability_loss)

        # ── Backprop ──────────────────────────────────────────────────────────
        (loss / accumulation_steps).backward()
        if step % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                trainable, max_norm=cfg.get('grad_clip', 1.0))
            optimizer.step()
            optimizer.zero_grad()

        # ── Logging ──────────────────────────────────────────────────────────
        r_val = rewards.mean().item()
        reward_hist.append(r_val)

        if step % log_interval == 0:
            avg_r = float(np.mean(reward_hist))
            writer.add_scalar(f'{model_name}/reward',     r_val,          step)
            writer.add_scalar(f'{model_name}/reward_avg', avg_r,          step)
            writer.add_scalar(f'{model_name}/loss',       loss.item(),    step)
            writer.add_scalar(f'{model_name}/reward_loss', reward_loss.item(), step)
            writer.add_scalar(f'{model_name}/supervised_proxy_loss',
                              supervised_loss.item(), step)
            writer.add_scalar(f'{model_name}/native_detection_loss',
                              native_loss.item(), step)
            for loss_name, loss_value in native_items.items():
                writer.add_scalar(f'{model_name}/native_{loss_name}',
                                  loss_value.item(), step)
            writer.add_scalar(f'{model_name}/stability_loss',
                              stability_loss.item(), step)
            writer.add_scalar(f'{model_name}/baseline',   baseline.value, step)
            writer.add_scalar(f'{model_name}/learning_rate', current_lr, step)
            print(f"  step {step:6d} | R={r_val:.4f} (avg200={avg_r:.4f}) "
                  f"| loss={loss.item():.6f} | b={baseline.value:.4f}")

            # [fix] Save best checkpoint dung rolling average (tranh noise spike)
            reward_improved = avg_r > best_avg_reward
            if reward_improved:
                best_avg_reward = avg_r
            if mode == 'kb1b' and reward_improved:
                torch.save({
                    'is_rl_checkpoint': True,
                    'method':           method_name,
                    'model_name':       model_name,
                    'mode':             mode,
                    'seed':             seed,
                    'step':             step,
                    'avg_reward':       avg_r,
                    'state_dict':       adapter.state_dict(),
                }, str(reward_ckpt))
                print(f"    -> Best ckpt updated (avg_r={avg_r:.4f}): {best_ckpt}")

        # ── Save periodic checkpoint ──────────────────────────────────────────
        if step % save_interval == 0:
            ckpt = output_dir / (
                f'{model_name}_seed{seed}_{checkpoint_tag}_step{step}.pt')
            torch.save({
                'is_rl_checkpoint': True,
                'method':           method_name,
                'model_name':       model_name,
                'mode':             mode,
                'seed':             seed,
                'step':             step,
                'state_dict':       adapter.state_dict(),
            }, str(ckpt))
            print(f"    -> Saved: {ckpt}")

        # [fix] Periodic val evaluation (eval_interval)
        if step % eval_interval == 0:
            print(f"    [Eval step {step}] Running val set evaluation...")
            val_m = quick_eval(adapter, val_loader, device=device)
            val_score = validation_score(val_m)
            writer.add_scalar(f'{model_name}/val_mAP', val_m['mAP'], step)
            writer.add_scalar(f'{model_name}/val_mAP50',  val_m['mAP50'],  step)
            writer.add_scalar(f'{model_name}/val_recall', val_m['recall'], step)
            writer.add_scalar(f'{model_name}/val_score', val_score, step)
            if val_score > best_val_score + min_delta:
                best_val_score = val_score
                no_improve_evals = 0
                torch.save({
                    'is_rl_checkpoint': True,
                    'method': method_name,
                    'model_name': model_name,
                    'mode': mode,
                    'seed': seed,
                    'step': step,
                    'avg_reward': float(np.mean(reward_hist)),
                    'val_score': best_val_score,
                    'val_mAP': val_m['mAP'],
                    'val_mAP50': val_m['mAP50'],
                    'val_recall': val_m['recall'],
                    'state_dict': adapter.state_dict(),
                }, str(best_ckpt))
                print(f'    -> Best val checkpoint updated: {best_ckpt}')
            else:
                no_improve_evals += 1
            print(
                f"    -> Val  mAP50={val_m['mAP50']:.4f}  "
                f"recall={val_m['recall']:.4f}"
            )

    writer.close()
    print(f"\n  Done: {model_name}  best_avg_reward={best_avg_reward:.4f}")
    print(f"  Best checkpoint: {best_ckpt}")


# =============================================================================
# 6. Entry point
# =============================================================================

CHECKPOINTS = {
    'yolov5s':  PROJECT_ROOT / 'checkpoints/yolov5s/weights/best.pt',
    'yolov8n':  PROJECT_ROOT / 'checkpoints/yolov8n/weights/best.pt',
    'yolov8s':  PROJECT_ROOT / 'checkpoints/yolov8s/weights/best.pt',
    'yolov11n': PROJECT_ROOT / 'checkpoints/yolov11n/weights/best.pt',
    'yolov11s': PROJECT_ROOT / 'checkpoints/yolov11s/weights/best.pt',
    'dp_yolo':  PROJECT_ROOT / 'checkpoints/dp_yolo/weights/best.pt',
}

for _name, _checkpoint in list(CHECKPOINTS.items()):
    _legacy = PROJECT_ROOT.parent / 'runs' / 'detect' / 'checkpoints' / _name / 'weights' / 'best.pt'
    if not _checkpoint.exists() and _legacy.exists():
        CHECKPOINTS[_name] = _legacy


def main():
    parser = argparse.ArgumentParser(
        description='RL Fine-tuning - Giai doan 2')
    parser.add_argument('--model',  default='all',
                        choices=['all'] + list(CHECKPOINTS.keys()))
    parser.add_argument('--steps',  type=int,   default=None,
                        help='Override so buoc RL (mac dinh: tu hyp.rl.yaml)')
    parser.add_argument('--lr',     type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--freeze', action='store_true',
                        help='Freeze backbone khi fine-tune')
    parser.add_argument('--cfg',    default='configs/hyp.rl.yaml',
                        help='Path den file hyperparameter RL')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed and isolate checkpoints')
    parser.add_argument('--mode', choices=['kb1b', 'native-only'],
                        default='kb1b',
                        help='Training mode; default keeps KB1-B v3.1')
    args = parser.parse_args()

    # Load hyperparameters
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Override tu CLI
    if args.steps is not None:
        cfg['steps'] = args.steps
    if args.lr is not None:
        cfg['lr'] = args.lr
    if args.seed is not None:
        cfg['seed'] = args.seed
    cfg['mode'] = args.mode
    if args.freeze:
        cfg['freeze_backbone'] = True

    # Chon model can train
    targets = (CHECKPOINTS if args.model == 'all'
               else {args.model: CHECKPOINTS[args.model]})

    for name, ckpt in targets.items():
        if not Path(ckpt).exists():
            print(f"  SKIP {name}: checkpoint not found at {ckpt}")
            print(f"  -> Chay train_supervised.py truoc.")
            continue
        rl_finetune(name, ckpt, cfg, device=args.device)


if __name__ == '__main__':
    main()
