"""
train_rl.py – RL Fine-tuning YOLO với 3 cấp độ từ DPO/GRPO/DAPO.

Cấp độ:
  Level 1 – REINFORCE + EMA Baseline  (từ kb1_reward_guided_training, cải tiến)
  Level 2 – GRPO-style Group Aug      (từ GRPO DeepSeek: group relative advantage)
  Level 3 – GRPO + DAPO improvements  (clip-higher + dynamic batch filtering)

Chạy:
    cd kb2_preference_optimization
    python train_rl.py --model dp_yolo --level 2 --steps 50000
    python train_rl.py --model dp_yolo --level 3 --G 4 --eps-high 0.3
    python train_rl.py --model all --level 1  # baseline nhanh

Khác kb1_reward_guided_training/train_rl.py:
    - Thêm Level 2 (GRPO-style): group augmentation, group relative advantage
    - Thêm Level 3 (DAPO): asymmetric clip-higher, dynamic batch filtering
    - augment.py: module riêng cho group augmentation
    - hyp.rl.grpo.yaml: config riêng cho Level 2/3
"""

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from reward import compute_reward
from dataloader import get_pest_dataloader
from augment import create_group_views


# =============================================================================
# 1. Shared utilities
# =============================================================================

class EMABaseline:
    """
    EMA baseline (Level 1).
    Level 2/3 dùng group mean thay thế – không cần EMA.
    """
    def __init__(self, alpha: float = 0.99):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, r: float) -> float:
        self.value = r if self.value is None else (
            self.alpha * self.value + (1 - self.alpha) * r
        )
        return self.value

    def advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        b = self.update(rewards.mean().item())
        return rewards - b


def compute_log_prob(preds: list[dict]) -> torch.Tensor:
    """
    Xấp xỉ log π_θ(action|state) = log(avg_confidence).
    Gradient chảy qua confidence scores → cập nhật model weights.
    """
    log_probs = []
    for pred in preds:
        scores = pred['scores']
        if scores.numel() == 0:
            dev = scores.device if scores.numel() > 0 else torch.device('cpu')
            log_probs.append(torch.tensor(-20.0, requires_grad=True, device=dev))
        else:
            avg_conf = scores.mean().clamp(1e-20, 1.0)
            log_probs.append(torch.log(avg_conf))
    return torch.stack(log_probs)  # (B,), requires_grad=True


def load_adapter(model_name: str, checkpoint: str, device: str):
    """Factory: trả về adapter phù hợp với model family."""
    from adapters import YOLOv5Adapter, UltralyticsAdapter

    if model_name in ('yolov5s', 'dp_yolo'):
        return YOLOv5Adapter(checkpoint, device=device)
    elif model_name in ('yolov8n', 'yolov8s', 'yolov11n', 'yolov11s'):
        return UltralyticsAdapter(checkpoint, device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def freeze_backbone(adapter, model_name: str) -> int:
    """Đóng băng backbone. Dùng khi mAP drop > 5% sau RL warmup."""
    frozen = 0
    for name, param in adapter.named_parameters():
        should_freeze = False
        if model_name in ('yolov5s', 'dp_yolo'):
            if any(f'model.{i}.' in name for i in range(10)):
                should_freeze = True
        else:
            if 'model.model.' in name:
                try:
                    idx = int(name.split('model.model.')[1].split('.')[0])
                    if idx < 10:
                        should_freeze = True
                except (IndexError, ValueError):
                    pass
        if should_freeze:
            param.requires_grad = False
            frozen += 1
    return frozen


def quick_eval(adapter, val_loader, device='cuda', conf_thres=0.25, iou_thres=0.45) -> dict:
    """Eval nhanh trên val set. Dùng trong eval_interval."""
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError:
        print('    [WARN] torchmetrics not installed. Skip quick_eval.')
        return {'mAP50': 0.0, 'recall': 0.0}

    adapter.eval_mode()
    metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=False)

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            preds  = adapter.forward_with_grad(images, conf_thres, iou_thres)
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
        'mAP50':  float(res.get('map_50',  torch.tensor(0.0)).item()),
        'recall': float(res.get('mar_100', torch.tensor(0.0)).item()),
    }


def save_checkpoint(adapter, output_dir: Path, model_name: str,
                    step: int, tag: str, extra: dict = None):
    """Lưu checkpoint với metadata đầy đủ."""
    ckpt = output_dir / f'{model_name}_rl_{tag}.pt'
    data = {
        'is_rl_checkpoint': True,
        'model_name':       model_name,
        'step':             step,
        'state_dict':       adapter.state_dict(),
    }
    if extra:
        data.update(extra)
    torch.save(data, str(ckpt))
    return ckpt


# =============================================================================
# 2. Level 1 – REINFORCE + EMA Baseline
#    Nguồn: kb1_reward_guided_training/train_rl.py (cải tiến: cùng interface với Level 2/3)
# =============================================================================

def train_level1(
    adapter,
    train_loader,
    val_loader,
    cfg:        dict,
    writer:     SummaryWriter,
    model_name: str,
    output_dir: Path,
    device:     str = 'cuda',
):
    """
    REINFORCE với EMA Baseline.
    Đơn giản nhất, tốn ít nhất tài nguyên (1× forward/step).
    Dùng để baseline hoặc khi GPU hạn chế.
    """
    baseline     = EMABaseline(alpha=cfg.get('ema_alpha', 0.99))
    trainable    = [p for p in adapter.parameters() if p.requires_grad]
    optimizer    = torch.optim.Adam(trainable, lr=cfg['lr'])
    reward_hist  = deque(maxlen=200)
    best_avg_r   = -float('inf')
    data_iter    = iter(train_loader)
    steps        = cfg.get('steps', 50_000)

    print(f'\n[Level 1 – REINFORCE+EMA] {model_name} | lr={cfg["lr"]} | steps={steps}')

    for step in range(1, steps + 1):
        # ── Batch ──────────────────────────────────────────────────────────
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)
        images = images.to(device)

        # ── Forward (có gradient qua scores) ───────────────────────────────
        preds = adapter.forward_with_grad(
            images,
            conf_thres=cfg.get('conf_thres', 0.20),
            iou_thres=cfg.get('iou_thres', 0.45),
        )

        # ── Reward (rule-based, no grad) ────────────────────────────────────
        with torch.no_grad():
            rewards = compute_reward(
                preds, targets,
                reward_type=cfg.get('reward_type', 'composite'),
                alpha=cfg.get('reward_alpha', 0.6),
                iou_threshold=cfg.get('iou_threshold', 0.5),
                small_thresh=cfg.get('small_thresh', 32),
            ).to(device)

        # ── EMA Advantage ───────────────────────────────────────────────────
        advantage = baseline.advantage(rewards)

        # ── Log-prob + REINFORCE loss ───────────────────────────────────────
        log_probs = compute_log_prob(preds)
        loss      = -torch.mean(log_probs * advantage.detach())

        # ── Backprop ────────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.get('grad_clip', 1.0))
        optimizer.step()

        # ── Logging + Checkpoint ────────────────────────────────────────────
        r_val = rewards.mean().item()
        reward_hist.append(r_val)

        if step % cfg.get('log_interval', 100) == 0:
            avg_r = float(np.mean(reward_hist))
            writer.add_scalar(f'{model_name}/L1/reward',     r_val,       step)
            writer.add_scalar(f'{model_name}/L1/reward_avg', avg_r,       step)
            writer.add_scalar(f'{model_name}/L1/loss',       loss.item(), step)
            writer.add_scalar(f'{model_name}/L1/baseline',   baseline.value or 0, step)
            print(f'  [L1] {step:6d} | R={r_val:.4f} avg200={avg_r:.4f} '
                  f'loss={loss.item():.6f} base={baseline.value:.4f}')

            if avg_r > best_avg_r:
                best_avg_r = avg_r
                ckpt = save_checkpoint(
                    adapter, output_dir, model_name, step, 'l1_best',
                    extra={'avg_reward': avg_r, 'level': 1},
                )
                print(f'    -> Best L1 ckpt (avg_r={avg_r:.4f}): {ckpt}')

        if step % cfg.get('save_interval', 5_000) == 0:
            save_checkpoint(adapter, output_dir, model_name, step, f'l1_step{step}')

        if step % cfg.get('eval_interval', 5_000) == 0:
            val_m = quick_eval(adapter, val_loader, device=device)
            writer.add_scalar(f'{model_name}/L1/val_mAP50',  val_m['mAP50'],  step)
            writer.add_scalar(f'{model_name}/L1/val_recall', val_m['recall'], step)
            print(f'    [Eval L1 step {step}] mAP50={val_m["mAP50"]:.4f} '
                  f'recall={val_m["recall"]:.4f}')

    return best_avg_r


# =============================================================================
# 3. Level 2 – GRPO-style (Group Augmentation)
#    Nguồn: GRPO (DeepSeek) → augmentation thay text sampling
# =============================================================================

def train_level2(
    adapter,
    train_loader,
    val_loader,
    cfg:        dict,
    writer:     SummaryWriter,
    model_name: str,
    output_dir: Path,
    device:     str = 'cuda',
    G:          int = 4,
):
    """
    GRPO-style: sinh G augmented views, tính group relative advantage.

    Ý tưởng cốt lõi từ GRPO (DeepSeek):
      - Không cần value model → baseline = mean(reward trong group)
      - Group advantage tự normalize → không cần EMA warmup
      - Rule-based reward (recall) = "verified reward" như DeepSeek math

    Args:
        G: số augmented views (4 là tốt nhất theo thực nghiệm)
    """
    trainable    = [p for p in adapter.parameters() if p.requires_grad]
    optimizer    = torch.optim.Adam(trainable, lr=cfg['lr'])
    reward_hist  = deque(maxlen=200)
    best_avg_r   = -float('inf')
    data_iter    = iter(train_loader)
    steps        = cfg.get('steps', 50_000)

    print(f'\n[Level 2 – GRPO Group Aug] {model_name} | G={G} | lr={cfg["lr"]} | steps={steps}')

    for step in range(1, steps + 1):
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)
        images = images.to(device)

        # ── Sinh G augmented views ─────────────────────────────────────────
        views = create_group_views(images, G=G, device=device)

        all_log_probs: list[torch.Tensor] = []
        all_rewards:   list[torch.Tensor] = []

        for g, aug_images in enumerate(views):
            preds = adapter.forward_with_grad(
                aug_images,
                conf_thres=cfg.get('conf_thres', 0.20),
                iou_thres=cfg.get('iou_thres', 0.45),
            )
            with torch.no_grad():
                r = compute_reward(
                    preds, targets,
                    reward_type=cfg.get('reward_type', 'composite'),
                    alpha=cfg.get('reward_alpha', 0.6),
                    iou_threshold=cfg.get('iou_threshold', 0.5),
                    small_thresh=cfg.get('small_thresh', 32),
                ).to(device)
                all_rewards.append(r)
            all_log_probs.append(compute_log_prob(preds))

        # ── Stack: (G, B) ───────────────────────────────────────────────────
        log_probs_mat = torch.stack(all_log_probs)   # (G, B), has grad
        rewards_mat   = torch.stack(all_rewards)     # (G, B), no grad

        # ── Group relative advantage (GRPO core) ───────────────────────────
        # Baseline = mean(reward trong group) thay EMA
        # Không cần warmup, tự normalize theo std của group
        mean_r = rewards_mat.mean(dim=0, keepdim=True)   # (1, B)
        std_r  = rewards_mat.std(dim=0,  keepdim=True) + 1e-8
        advantage_mat = (rewards_mat - mean_r) / std_r    # (G, B), normalized

        # ── GRPO loss ────────────────────────────────────────────────────────
        loss = -torch.mean(log_probs_mat * advantage_mat.detach())

        # ── Backprop ─────────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.get('grad_clip', 1.0))
        optimizer.step()

        # ── Logging ──────────────────────────────────────────────────────────
        avg_r_step  = rewards_mat.mean().item()
        group_std   = rewards_mat.std().item()
        reward_hist.append(avg_r_step)

        if step % cfg.get('log_interval', 100) == 0:
            avg_r = float(np.mean(reward_hist))
            writer.add_scalar(f'{model_name}/L2/reward',     avg_r_step,  step)
            writer.add_scalar(f'{model_name}/L2/reward_avg', avg_r,       step)
            writer.add_scalar(f'{model_name}/L2/group_std',  group_std,   step)
            writer.add_scalar(f'{model_name}/L2/loss',       loss.item(), step)
            print(f'  [L2] {step:6d} | R={avg_r_step:.4f} avg200={avg_r:.4f} '
                  f'group_std={group_std:.4f} loss={loss.item():.6f}')

            if avg_r > best_avg_r:
                best_avg_r = avg_r
                ckpt = save_checkpoint(
                    adapter, output_dir, model_name, step, 'l2_best',
                    extra={'avg_reward': avg_r, 'level': 2, 'G': G},
                )
                print(f'    -> Best L2 ckpt (avg_r={avg_r:.4f}): {ckpt}')

        if step % cfg.get('save_interval', 5_000) == 0:
            save_checkpoint(adapter, output_dir, model_name, step, f'l2_step{step}')

        if step % cfg.get('eval_interval', 5_000) == 0:
            val_m = quick_eval(adapter, val_loader, device=device)
            writer.add_scalar(f'{model_name}/L2/val_mAP50',  val_m['mAP50'],  step)
            writer.add_scalar(f'{model_name}/L2/val_recall', val_m['recall'], step)
            print(f'    [Eval L2 step {step}] mAP50={val_m["mAP50"]:.4f} '
                  f'recall={val_m["recall"]:.4f}')

    return best_avg_r


# =============================================================================
# 4. Level 3 – GRPO + DAPO Improvements
#    Nguồn: DAPO paper → clip-higher + dynamic batch filtering
# =============================================================================

def _dapo_loss(
    log_probs:     torch.Tensor,   # (G, B), has grad
    log_probs_ref: torch.Tensor,   # (G, B), no grad (reference = view 0)
    advantage:     torch.Tensor,   # (G, B), no grad
    eps_low:       float = 0.1,
    eps_high:      float = 0.3,
) -> torch.Tensor:
    """
    DAPO asymmetric clipping.

    Thay vì PPO clip đối xứng [1-ε, 1+ε]:
      DAPO khi advantage > 0: clip [1-eps_low, 1+eps_high]   (cho phép cải thiện nhiều)
      DAPO khi advantage < 0: clip [1-eps_high, 1+eps_low]   (hạn chế suy giảm)

    Lợi ích: không bỏ qua signal tốt khi model đột nhiên detect chính xác hơn.
    """
    ratio = torch.exp(log_probs - log_probs_ref.detach())
    adv   = advantage.detach()

    clipped = torch.where(
        adv >= 0,
        ratio.clamp(1 - eps_low,  1 + eps_high),
        ratio.clamp(1 - eps_high, 1 + eps_low),
    )
    # Lấy min như PPO: tránh over-optimization
    loss_unclipped = ratio   * adv
    loss_clipped   = clipped * adv
    return -torch.mean(torch.min(loss_unclipped, loss_clipped))


def _filter_learnable(
    rewards_mat: torch.Tensor,   # (G, B)
    min_std:     float = 0.01,
    min_mean:    float = 0.01,
    max_mean:    float = 0.99,
) -> torch.Tensor:
    """
    DAPO dynamic batch filtering.
    Loại bỏ samples trong batch không có signal học hữu ích:
      - Std quá nhỏ: tất cả views cho reward giống nhau → advantage ≈ 0
      - Mean = 1.0: model đã detect hoàn hảo → không cần train thêm
      - Mean ≈ 0.0: model hoàn toàn fail → không có gradient signal

    Returns:
        bool Tensor (B,), True = giữ lại
    """
    r_std  = rewards_mat.std(dim=0)    # (B,)
    r_mean = rewards_mat.mean(dim=0)   # (B,)
    return (r_std > min_std) & (r_mean > min_mean) & (r_mean < max_mean)


def train_level3(
    adapter,
    train_loader,
    val_loader,
    cfg:        dict,
    writer:     SummaryWriter,
    model_name: str,
    output_dir: Path,
    device:     str = 'cuda',
    G:          int = 4,
):
    """
    GRPO + DAPO improvements:
      1. Group relative advantage (GRPO)
      2. Asymmetric clip-higher (DAPO)
      3. Dynamic batch filtering (DAPO)

    Phức tạp hơn Level 2 nhưng tận dụng signal tốt hơn.
    """
    trainable  = [p for p in adapter.parameters() if p.requires_grad]
    optimizer  = torch.optim.Adam(trainable, lr=cfg['lr'])
    reward_hist = deque(maxlen=200)
    best_avg_r  = -float('inf')
    data_iter   = iter(train_loader)
    steps       = cfg.get('steps', 50_000)
    skipped     = 0

    eps_low  = cfg.get('eps_low',  0.1)
    eps_high = cfg.get('eps_high', 0.3)
    min_std  = cfg.get('min_std',  0.01)

    print(f'\n[Level 3 – GRPO+DAPO] {model_name} | G={G} | '
          f'eps=({eps_low},{eps_high}) | lr={cfg["lr"]} | steps={steps}')

    for step in range(1, steps + 1):
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)
        images = images.to(device)

        # ── Sinh G views ──────────────────────────────────────────────────
        views = create_group_views(images, G=G, device=device)

        all_log_probs: list[torch.Tensor] = []
        all_rewards:   list[torch.Tensor] = []

        for aug_images in views:
            preds = adapter.forward_with_grad(
                aug_images,
                conf_thres=cfg.get('conf_thres', 0.20),
                iou_thres=cfg.get('iou_thres', 0.45),
            )
            with torch.no_grad():
                r = compute_reward(
                    preds, targets,
                    reward_type=cfg.get('reward_type', 'composite'),
                    alpha=cfg.get('reward_alpha', 0.6),
                    iou_threshold=cfg.get('iou_threshold', 0.5),
                    small_thresh=cfg.get('small_thresh', 32),
                ).to(device)
                all_rewards.append(r)
            all_log_probs.append(compute_log_prob(preds))

        log_probs_mat = torch.stack(all_log_probs)   # (G, B)
        rewards_mat   = torch.stack(all_rewards)     # (G, B)

        # DAPO: Dynamic batch filtering
        valid = _filter_learnable(rewards_mat, min_std=min_std)
        if valid.sum() == 0:
            skipped += 1
            # Log skip ở log_interval và mội 500 bước dù không phải log step
            if (step % cfg.get('log_interval', 100) == 0
                    or skipped % 500 == 0):
                skip_rate = skipped / step
                writer.add_scalar(f'{model_name}/L3/skip_rate', skip_rate, step)
                print(f'  [L3] {step:6d} | SKIP ({skipped} total, '
                      f'skip_rate={skip_rate:.1%})')
            continue

        lp_v  = log_probs_mat[:, valid]   # (G, B_valid)
        r_v   = rewards_mat[:, valid]     # (G, B_valid)

        # ── Group relative advantage ──────────────────────────────────────
        mean_r = r_v.mean(dim=0, keepdim=True)
        std_r  = r_v.std(dim=0,  keepdim=True) + 1e-8
        adv    = (r_v - mean_r) / std_r           # (G, B_valid)

        # ── DAPO: Clip-higher ─────────────────────────────────────────────
        # View 0 (gốc, không augment) làm reference policy
        lp_ref = lp_v[0:1].detach().expand_as(lp_v)   # (G, B_valid)
        loss   = _dapo_loss(lp_v, lp_ref, adv, eps_low, eps_high)

        # ── Backprop ──────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.get('grad_clip', 1.0))
        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────
        avg_r_step = r_v.mean().item()
        group_std  = r_v.std().item()
        reward_hist.append(avg_r_step)

        if step % cfg.get('log_interval', 100) == 0:
            avg_r     = float(np.mean(reward_hist))
            skip_rate = skipped / step
            writer.add_scalar(f'{model_name}/L3/reward',     avg_r_step,  step)
            writer.add_scalar(f'{model_name}/L3/reward_avg', avg_r,       step)
            writer.add_scalar(f'{model_name}/L3/group_std',  group_std,   step)
            writer.add_scalar(f'{model_name}/L3/loss',       loss.item(), step)
            writer.add_scalar(f'{model_name}/L3/skip_rate',  skip_rate,   step)
            print(f'  [L3] {step:6d} | R={avg_r_step:.4f} avg200={avg_r:.4f} '
                  f'skip={skip_rate:.1%} group_std={group_std:.4f} '
                  f'loss={loss.item():.6f}')

            if avg_r > best_avg_r:
                best_avg_r = avg_r
                ckpt = save_checkpoint(
                    adapter, output_dir, model_name, step, 'l3_best',
                    extra={'avg_reward': avg_r, 'level': 3, 'G': G},
                )
                print(f'    -> Best L3 ckpt (avg_r={avg_r:.4f}): {ckpt}')

        if step % cfg.get('save_interval', 5_000) == 0:
            save_checkpoint(adapter, output_dir, model_name, step, f'l3_step{step}')

        if step % cfg.get('eval_interval', 5_000) == 0:
            val_m = quick_eval(adapter, val_loader, device=device)
            writer.add_scalar(f'{model_name}/L3/val_mAP50',  val_m['mAP50'],  step)
            writer.add_scalar(f'{model_name}/L3/val_recall', val_m['recall'], step)
            print(f'    [Eval L3 step {step}] mAP50={val_m["mAP50"]:.4f} '
                  f'recall={val_m["recall"]:.4f}')

    return best_avg_r


# =============================================================================
# 5. Main dispatcher
# =============================================================================

CHECKPOINTS = {
    'yolov5s':  'checkpoints/yolov5s/weights/best.pt',
    'yolov8n':  'checkpoints/yolov8n/weights/best.pt',
    'yolov8s':  'checkpoints/yolov8s/weights/best.pt',
    'yolov11n': 'checkpoints/yolov11n/weights/best.pt',
    'yolov11s': 'checkpoints/yolov11s/weights/best.pt',
    'dp_yolo':  'checkpoints/dp_yolo/weights/best.pt',
}

TRAIN_FUNCS = {
    1: train_level1,
    2: train_level2,
    3: train_level3,
}


def run_rl(model_name: str, checkpoint: str, cfg: dict, args):
    """Chạy RL fine-tuning cho 1 model với level được chỉ định."""
    device     = args.device
    output_dir = Path('rl_checkpoints')
    output_dir.mkdir(exist_ok=True)

    run_id = f'{model_name}_l{args.level}_{int(time.time())}'
    writer = SummaryWriter(f'results/tensorboard/{run_id}')

    print(f'\n{"="*60}')
    print(f'  RL Fine-tuning: {model_name}  [Level {args.level}]')
    print(f'  Checkpoint: {checkpoint}')
    print(f'{"="*60}')

    # ── Load model ───────────────────────────────────────────────────────
    adapter = load_adapter(model_name, checkpoint, device)
    adapter.train_mode()

    if cfg.get('freeze_backbone', False):
        n = freeze_backbone(adapter, model_name)
        print(f'  Frozen {n} backbone params.')

    # ── Resume from RL checkpoint (optional) ─────────────────────────────
    if getattr(args, 'resume', None) and Path(args.resume).exists():
        rl_data = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(rl_data, dict) and rl_data.get('is_rl_checkpoint', False):
            state = rl_data.get('state_dict', {})
            if hasattr(adapter, 'model'):
                adapter.model.load_state_dict(state, strict=False)
            else:
                adapter.load_state_dict(state, strict=False)
            resume_step = rl_data.get('step', 0)
            cfg = dict(cfg)  # copy để không sửa cfg gốc
            cfg['steps'] = max(1, cfg.get('steps', 50_000) - resume_step)
            print(f'  Resumed from: {args.resume}')
            print(f'  -> step={resume_step}, remaining steps={cfg["steps"]}')
        else:
            print(f'  [WARN] --resume: {args.resume} không phải RL checkpoint hợp lệ, bỏ qua.')

    # ── DataLoaders ──────────────────────────────────────────────────────
    data_root = cfg.get('data_root', '../pre-data/data/v2i')
    bs        = cfg.get('batch_size', 16)
    train_loader = get_pest_dataloader(data_root, split='train', batch_size=bs, img_size=640)
    val_loader   = get_pest_dataloader(data_root, split='val',   batch_size=bs, img_size=640)

    # ── Train ────────────────────────────────────────────────────────────
    train_fn = TRAIN_FUNCS[args.level]
    kwargs = dict(
        adapter=adapter, train_loader=train_loader, val_loader=val_loader,
        cfg=cfg, writer=writer, model_name=model_name,
        output_dir=output_dir, device=device,
    )
    if args.level in (2, 3):
        kwargs['G'] = args.G

    best_r = train_fn(**kwargs)
    writer.close()
    print(f'\n  Done: {model_name} | best_avg_reward={best_r:.4f}')


def main():
    parser = argparse.ArgumentParser(
        description='RL Fine-tuning YOLO (Level 1/2/3: REINFORCE / GRPO / DAPO)')
    parser.add_argument('--model',  default='all',
                        choices=['all'] + list(CHECKPOINTS.keys()),
                        help='Model cần fine-tune')
    parser.add_argument('--level',  type=int, default=2, choices=[1, 2, 3],
                        help='RL level: 1=REINFORCE+EMA, 2=GRPO, 3=GRPO+DAPO')
    parser.add_argument('--G',      type=int, default=4,
                        help='Số augmented views (Level 2/3, default=4)')
    parser.add_argument('--steps',  type=int,   default=None)
    parser.add_argument('--lr',     type=float, default=None)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--eps-high', type=float, default=None,
                        help='DAPO clip upper bound (Level 3)')
    parser.add_argument('--resume', default=None, metavar='CKPT',
                        help='Path đến RL checkpoint để resume training bị dừ dượng giữa chừng. '
                             'Ví dụ: rl_checkpoints/dp_yolo_rl_l2_step25000.pt')
    parser.add_argument('--cfg',    default='configs/hyp.rl.yaml')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.steps    is not None: cfg['steps']          = args.steps
    if args.lr       is not None: cfg['lr']             = args.lr
    if args.freeze:               cfg['freeze_backbone'] = True
    if args.eps_high is not None: cfg['eps_high']        = args.eps_high

    targets = (CHECKPOINTS if args.model == 'all'
               else {args.model: CHECKPOINTS[args.model]})

    for name, ckpt in targets.items():
        if not Path(ckpt).exists():
            print(f'  SKIP {name}: not found at {ckpt}')
            print(f'  -> Run train_supervised.py first.')
            continue
        run_rl(name, ckpt, cfg, args)


if __name__ == '__main__':
    main()
