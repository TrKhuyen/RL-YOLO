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


# =============================================================================
# 2b. Quick Evaluation (dung trong eval_interval)
# =============================================================================

def quick_eval(
    adapter,
    val_loader,
    device:     str   = 'cuda',
    conf_thres: float = 0.25,
    iou_thres:  float = 0.45,
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
        return {'mAP50': 0.0, 'recall': 0.0}

    adapter.eval_mode()
    metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=False)

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
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
        'mAP50':  float(res.get('map_50',  torch.tensor(0.0)).item()),
        'recall': float(res.get('mar_100', torch.tensor(0.0)).item()),
    }


# =============================================================================
# 3. Adapter factory
# =============================================================================

def load_adapter(model_name: str, checkpoint: str, device: str):
    """
    Tao adapter phu hop cho tung model family.
    Tra ve object co .forward_with_grad(), .parameters(), .named_parameters().
    """
    from adapters import YOLOv5Adapter, UltralyticsAdapter

    if model_name in ('yolov5s', 'dp_yolo'):
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
    output_dir = Path('rl_checkpoints')
    output_dir.mkdir(exist_ok=True)

    run_id = f"{model_name}_{int(time.time())}"
    writer = SummaryWriter(f"results/tensorboard/rl_{run_id}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"  Loading {model_name} from {checkpoint}...")
    adapter = load_adapter(model_name, checkpoint, device)
    adapter.train_mode()

    # ── Freeze backbone (tuy chon) ──────────────────────────────────────────
    if cfg.get('freeze_backbone', False):
        n = freeze_backbone(adapter, model_name)
        print(f"  Frozen {n} backbone parameters.")

    # ── Optimizer ───────────────────────────────────────────────────────────
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg['lr'])

    # ── DataLoaders (train + val) ───────────────────────────────────────────
    loader = get_pest_dataloader(
        'data/pest', split='train',
        batch_size=cfg.get('batch_size', 16),
        img_size=640,
    )
    val_loader = get_pest_dataloader(
        'data/pest', split='val',
        batch_size=cfg.get('batch_size', 16),
        img_size=640,
    )

    # ── RL state ────────────────────────────────────────────────────────────
    baseline        = EMABaseline(alpha=cfg.get('ema_alpha', 0.99))
    reward_hist     = deque(maxlen=200)
    best_avg_reward = -float('inf')   # [fix] dung rolling avg, tranh noise spike
    best_ckpt       = output_dir / f'{model_name}_rl_best.pt'
    data_iter       = iter(loader)
    steps           = cfg.get('steps', 50_000)
    log_interval    = cfg.get('log_interval',  100)
    save_interval   = cfg.get('save_interval', 5_000)
    eval_interval   = cfg.get('eval_interval', 5_000)   # [fix] implement

    print(f"\n{'='*60}")
    print(f"  RL Fine-tuning: {model_name}")
    print(f"  Steps: {steps}  LR: {cfg['lr']}  "
          f"Freeze: {cfg.get('freeze_backbone', False)}")
    print(f"  Reward: {cfg.get('reward_type','composite')}  "
          f"alpha={cfg.get('reward_alpha', 0.6)}")
    print(f"{'='*60}\n")

    for step in range(1, steps + 1):

        # ── Lay batch ────────────────────────────────────────────────────────
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, targets = next(data_iter)

        images = images.to(device)

        # ── Forward (giu grad qua confidence scores) ─────────────────────────
        preds = adapter.forward_with_grad(
            images,
            conf_thres=cfg.get('conf_thres', 0.20),
            iou_thres=cfg.get('iou_thres',  0.45),
        )

        # ── Tinh reward (khong can grad) ─────────────────────────────────────
        with torch.no_grad():
            rewards = compute_reward(
                preds, targets,
                reward_type=cfg.get('reward_type', 'composite'),
                alpha=cfg.get('reward_alpha', 0.6),
                iou_threshold=cfg.get('iou_threshold', 0.5),
                small_thresh=cfg.get('small_thresh', 32),
            ).to(device)

        # ── EMA baseline -> advantage ─────────────────────────────────────────
        advantage = baseline.advantage(rewards)    # Tensor, no grad

        # ── Log-probability xap xi ────────────────────────────────────────────
        log_probs = compute_log_prob(preds)        # Tensor (B,), co grad

        # ── REINFORCE loss ────────────────────────────────────────────────────
        # L = -E[log pi(a|s) * advantage]  (dau tru: minimize -> maximize reward)
        loss = -torch.mean(log_probs * advantage.detach())

        # ── Backprop ──────────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=cfg.get('grad_clip', 1.0))
        optimizer.step()

        # ── Logging ──────────────────────────────────────────────────────────
        r_val = rewards.mean().item()
        reward_hist.append(r_val)

        if step % log_interval == 0:
            avg_r = float(np.mean(reward_hist))
            writer.add_scalar(f'{model_name}/reward',     r_val,          step)
            writer.add_scalar(f'{model_name}/reward_avg', avg_r,          step)
            writer.add_scalar(f'{model_name}/loss',       loss.item(),    step)
            writer.add_scalar(f'{model_name}/baseline',   baseline.value, step)
            print(f"  step {step:6d} | R={r_val:.4f} (avg200={avg_r:.4f}) "
                  f"| loss={loss.item():.6f} | b={baseline.value:.4f}")

            # [fix] Save best checkpoint dung rolling average (tranh noise spike)
            if avg_r > best_avg_reward:
                best_avg_reward = avg_r
                torch.save({
                    'is_rl_checkpoint': True,
                    'model_name':       model_name,
                    'step':             step,
                    'avg_reward':       avg_r,
                    'state_dict':       adapter.state_dict(),
                }, str(best_ckpt))
                print(f"    -> Best ckpt updated (avg_r={avg_r:.4f}): {best_ckpt}")

        # ── Save periodic checkpoint ──────────────────────────────────────────
        if step % save_interval == 0:
            ckpt = output_dir / f'{model_name}_rl_step{step}.pt'
            torch.save({
                'is_rl_checkpoint': True,
                'model_name':       model_name,
                'step':             step,
                'state_dict':       adapter.state_dict(),
            }, str(ckpt))
            print(f"    -> Saved: {ckpt}")

        # [fix] Periodic val evaluation (eval_interval)
        if step % eval_interval == 0:
            print(f"    [Eval step {step}] Running val set evaluation...")
            val_m = quick_eval(adapter, val_loader, device=device)
            writer.add_scalar(f'{model_name}/val_mAP50',  val_m['mAP50'],  step)
            writer.add_scalar(f'{model_name}/val_recall', val_m['recall'], step)
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
    'yolov5s':  'checkpoints/yolov5s/weights/best.pt',
    'yolov8n':  'checkpoints/yolov8n/weights/best.pt',
    'yolov8s':  'checkpoints/yolov8s/weights/best.pt',
    'yolov11n': 'checkpoints/yolov11n/weights/best.pt',
    'yolov11s': 'checkpoints/yolov11s/weights/best.pt',
    'dp_yolo':  'checkpoints/dp_yolo/weights/best.pt',
}


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
    args = parser.parse_args()

    # Load hyperparameters
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    # Override tu CLI
    if args.steps is not None:
        cfg['steps'] = args.steps
    if args.lr is not None:
        cfg['lr'] = args.lr
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
