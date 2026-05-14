"""
train_rl.py – Giai đoạn 2: RL Fine-tuning YOLO bằng REINFORCE.

Thiết kế theo:
- bwconrad/cv-rl:  pattern CE-pretrain → RL fine-tune (tránh train từ đầu)
- yanivnik:        recall reward + log(avg_confidence) làm log-prob xấp xỉ
- Cải tiến:        EMA baseline (ổn định hơn cả 2 paper nguồn)

Chạy:
    python train_rl.py                       # fine-tune tất cả model
    python train_rl.py --model dp_yolo       # chỉ fine-tune DP-YOLO
    python train_rl.py --model yolov8n --steps 30000 --lr 5e-7
    python train_rl.py --model dp_yolo --freeze  # freeze backbone
"""

import argparse
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


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMA Baseline – giảm variance REINFORCE
# ─────────────────────────────────────────────────────────────────────────────

class EMABaseline:
    """
    Exponential Moving Average baseline để giảm variance của REINFORCE.

    advantage_i = R_i - b   (unbiased khi E[b] = E[R])

    Lý do dùng EMA thay vì:
    - bwconrad: dùng sample thứ 2 (Monte Carlo) → tốn 2× forward
    - yanivnik: TODO, chưa implement baseline → variance cao
    """
    def __init__(self, alpha: float = 0.99):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, reward: float) -> float:
        if self.value is None:
            self.value = reward
        else:
            self.value = self.alpha * self.value + (1.0 - self.alpha) * reward
        return self.value

    def advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        """Trả về advantage = rewards - baseline (Tensor, no grad)."""
        b = self.update(rewards.mean().item())
        return rewards - b


# ─────────────────────────────────────────────────────────────────────────────
# 2. Log-probability xấp xỉ
# ─────────────────────────────────────────────────────────────────────────────

def compute_log_prob(preds: list[dict]) -> torch.Tensor:
    """
    Xấp xỉ log π_θ(a|s) = log(average confidence) trên batch.

    Lý do dùng xấp xỉ (từ yanivnik):
    YOLO không phải autoregressive model nên không có log-prob đầy đủ.
    avg_confidence là proxy hợp lý: confidence cao → policy "chắc chắn"
    → gradient đi đúng hướng tối ưu reward.

    Clamp về [1e-20, 1.0] để tránh log(0) = -inf.
    """
    log_probs = []
    for pred in preds:
        scores = pred['scores']
        if scores.numel() == 0:
            # Không có prediction → log prob thấp nhất
            log_probs.append(torch.tensor(-20.0, requires_grad=True,
                                           device=scores.device
                                           if scores.numel() > 0 else 'cpu'))
        else:
            avg_conf = scores.mean().clamp(1e-20, 1.0)
            log_probs.append(torch.log(avg_conf))
    return torch.stack(log_probs)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Adapter factory
# ─────────────────────────────────────────────────────────────────────────────

def load_adapter(model_name: str, checkpoint: str, device: str):
    """
    Tạo adapter phù hợp cho từng model family.
    Trả về object có .forward_with_grad(), .parameters(), .named_parameters().
    """
    from adapters import YOLOv5Adapter, UltralyticsAdapter

    if model_name in ('yolov5s', 'dp_yolo'):
        return YOLOv5Adapter(checkpoint, device=device)
    elif model_name in ('yolov8n', 'yolov8s', 'yolov11n', 'yolov11s'):
        return UltralyticsAdapter(checkpoint, device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         "Thêm vào adapter factory nếu cần.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Freeze backbone helper
# ─────────────────────────────────────────────────────────────────────────────

def freeze_backbone(adapter, model_name: str) -> int:
    """
    Đóng băng tham số backbone để tránh catastrophic forgetting.
    Trả về số tham số bị freeze.
    """
    frozen = 0
    for name, param in adapter.named_parameters():
        should_freeze = False
        if model_name in ('yolov5s', 'dp_yolo'):
            # YOLOv5: backbone là model.0 đến model.9
            if any(f'model.{i}.' in name for i in range(10)):
                should_freeze = True
        else:
            # Ultralytics YOLOv8/v11: backbone nằm trong model.model[:-2]
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


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main RL training loop
# ─────────────────────────────────────────────────────────────────────────────

def rl_finetune(
    model_name:      str,
    checkpoint:      str,
    cfg:             dict,
    device:          str = 'cuda',
):
    """
    Vòng lặp REINFORCE fine-tuning cho 1 YOLO model.

    Args:
        model_name: tên model ('yolov5s', 'yolov8n', 'yolov11n', 'dp_yolo')
        checkpoint: path đến best.pt từ supervised training
        cfg:        dict hyperparameters từ hyp.rl.yaml
        device:     'cuda' hoặc 'cpu'
    """
    output_dir = Path('rl_checkpoints')
    output_dir.mkdir(exist_ok=True)

    run_id = f"{model_name}_{int(time.time())}"
    writer = SummaryWriter(f"results/tensorboard/rl_{run_id}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"  Loading {model_name} from {checkpoint}...")
    adapter = load_adapter(model_name, checkpoint, device)
    adapter.train_mode()

    # ── Freeze backbone (tùy chọn) ──────────────────────────────────────────
    if cfg.get('freeze_backbone', False):
        n = freeze_backbone(adapter, model_name)
        print(f"  Frozen {n} backbone parameters.")

    # ── Optimizer ───────────────────────────────────────────────────────────
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg['lr'])

    # ── DataLoader ─────────────────────────────────────────────────────────
    loader = get_pest_dataloader(
        'data/pest', split='train',
        batch_size=cfg.get('batch_size', 16),
        img_size=640,
    )

    # ── RL state ────────────────────────────────────────────────────────────
    baseline     = EMABaseline(alpha=cfg.get('ema_alpha', 0.99))
    reward_hist  = deque(maxlen=200)
    best_reward  = -float('inf')
    data_iter    = iter(loader)
    steps        = cfg.get('steps', 50_000)
    log_interval  = cfg.get('log_interval', 100)
    save_interval = cfg.get('save_interval', 5_000)

    print(f"\n{'='*60}")
    print(f"  RL Fine-tuning: {model_name}")
    print(f"  Steps: {steps}  LR: {cfg['lr']}  "
          f"Freeze: {cfg.get('freeze_backbone', False)}")
    print(f"  Reward: {cfg.get('reward_type','composite')}  "
          f"alpha={cfg.get('reward_alpha', 0.6)}")
    print(f"{'='*60}\n")

    for step in range(1, steps + 1):

        # ── Lấy batch ────────────────────────────────────────────────────
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, targets = next(data_iter)

        images = images.to(device)

        # ── Forward (giữ grad qua confidence scores) ──────────────────────
        preds = adapter.forward_with_grad(
            images,
            conf_thres=cfg.get('conf_thres', 0.20),
            iou_thres=cfg.get('iou_thres',  0.45),
        )

        # ── Tính reward (không cần grad) ──────────────────────────────────
        with torch.no_grad():
            rewards = compute_reward(
                preds, targets,
                reward_type=cfg.get('reward_type', 'composite'),
                alpha=cfg.get('reward_alpha', 0.6),
                iou_threshold=cfg.get('iou_threshold', 0.5),
                small_thresh=cfg.get('small_thresh', 32),
            ).to(device)

        # ── EMA baseline → advantage ──────────────────────────────────────
        advantage = baseline.advantage(rewards)    # Tensor, no grad

        # ── Log-probability xấp xỉ ────────────────────────────────────────
        log_probs = compute_log_prob(preds)        # Tensor (B,), có grad

        # ── REINFORCE loss ────────────────────────────────────────────────
        # L = -E[log π(a|s) * advantage]  (dấu trừ: minimize → maximize reward)
        loss = -torch.mean(log_probs * advantage.detach())

        # ── Backprop ──────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=cfg.get('grad_clip', 1.0))
        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────
        r_val = rewards.mean().item()
        reward_hist.append(r_val)

        if step % log_interval == 0:
            avg_r = float(np.mean(reward_hist))
            writer.add_scalar(f'{model_name}/reward',     r_val,             step)
            writer.add_scalar(f'{model_name}/reward_avg', avg_r,             step)
            writer.add_scalar(f'{model_name}/loss',       loss.item(),       step)
            writer.add_scalar(f'{model_name}/baseline',   baseline.value,    step)
            print(f"  step {step:6d} | R={r_val:.4f} (avg200={avg_r:.4f}) "
                  f"| loss={loss.item():.6f} | b={baseline.value:.4f}")

        # ── Save checkpoint ───────────────────────────────────────────────
        if step % save_interval == 0:
            ckpt = output_dir / f'{model_name}_rl_step{step}.pt'
            torch.save(adapter.state_dict(), str(ckpt))
            print(f"    → Saved: {ckpt}")

        # ── Save best ────────────────────────────────────────────────────
        if r_val > best_reward:
            best_reward = r_val
            best_ckpt = output_dir / f'{model_name}_rl_best.pt'
            torch.save(adapter.state_dict(), str(best_ckpt))

    writer.close()
    print(f"\n  Done: {model_name}  best_reward={best_reward:.4f}")
    print(f"  Best checkpoint: {best_ckpt}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINTS = {
    'yolov5s':  'checkpoints/yolov5s/weights/best.pt',
    'yolov8n':  'checkpoints/yolov8n/weights/best.pt',
    'yolov8s':  'checkpoints/yolov8s/weights/best.pt',
    'yolov11n': 'checkpoints/yolov11n/weights/best.pt',
    'yolov11s': 'checkpoints/yolov11s/weights/best.pt',
    'dp_yolo':  'checkpoints/dp_yolo/weights/best.pt',
}


def main():
    parser = argparse.ArgumentParser(
        description='RL Fine-tuning – Giai đoạn 2')
    parser.add_argument('--model',  default='all',
                        choices=['all'] + list(CHECKPOINTS.keys()))
    parser.add_argument('--steps',  type=int,   default=None,
                        help='Override số bước RL (mặc định: từ hyp.rl.yaml)')
    parser.add_argument('--lr',     type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--freeze', action='store_true',
                        help='Freeze backbone khi fine-tune')
    parser.add_argument('--cfg',    default='configs/hyp.rl.yaml',
                        help='Path đến file hyperparameter RL')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Load hyperparameters
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    # Override từ CLI
    if args.steps is not None:
        cfg['steps'] = args.steps
    if args.lr is not None:
        cfg['lr'] = args.lr
    if args.freeze:
        cfg['freeze_backbone'] = True

    # Chọn model cần train
    targets = (CHECKPOINTS if args.model == 'all'
               else {args.model: CHECKPOINTS[args.model]})

    for name, ckpt in targets.items():
        if not Path(ckpt).exists():
            print(f"  SKIP {name}: checkpoint not found at {ckpt}")
            print(f"  → Chạy train_supervised.py trước.")
            continue
        rl_finetune(name, ckpt, cfg, device=args.device)


if __name__ == '__main__':
    main()
