# Hướng Dẫn Thực Hành: RL Fine-tuning cho YOLO

> **Tổng hợp từ:**
> - So sánh DPO/GRPO/RLHF ↔ Detection (`guide-rl-finetune-detection-from-llm.md`)
> - Pipeline đầy đủ cho bài toán sâu bệnh (`guide-rl-finetune-yolo-sau-benh.md`)
> - DeepSeek R1-Zero (rule-based reward, GRPO-style)
> - yanivnik/Google DeepMind (recall reward, confidence proxy)

---

## 1. Tại Sao RL, Không Chỉ Supervised?

YOLO được huấn luyện để tối ưu **CIoU + BCE** – nhưng được đánh giá bằng **mAP@50, Recall, AP_small**.
Khoảng cách giữa loss training và metric đánh giá tạo ra dư địa mà RL có thể khai thác.

```
Supervised training:   tối ưu CIoU loss          → model biết detect
RL fine-tuning:        tối ưu recall/mAP trực tiếp → model detect tốt hơn theo metric thực
```

Thực nghiệm (bwconrad/cv-rl): REINFORCE từ đầu → F1 = 0.41 (tệ).
Fine-tune từ supervised checkpoint → F1 cao hơn đáng kể.

**Kết luận:** Luôn warm-start RL từ supervised checkpoint, không train từ đầu.

---

## 2. Khái Niệm Cốt Lõi

| Khái niệm RL | Trong YOLO |
|---|---|
| Policy π_θ | YOLO model (tham số θ) |
| State s | Ảnh đầu vào |
| Action a | Tập bounding box predictions |
| Reward R | Recall + Small-object bonus (rule-based) |
| log π(a\|s) | log(avg_confidence) [xấp xỉ, không phải exact] |
| Baseline b | EMA của reward qua các bước |
| Advantage | R - baseline |

> **Tại sao log(avg_confidence) thay vì log-prob đầy đủ?**
> YOLO không phải autoregressive model – không có chuỗi token để tính exact log-prob.
> avg_confidence là proxy tốt nhất được kiểm chứng thực nghiệm (yanivnik 2023).

---

## 3. Thiết Kế Reward Function

Reward được tính **hoàn toàn tự động**, không cần human label – giống DeepSeek dùng
verified math answers. Đây là lý do RL cho YOLO rẻ và không cần reward model riêng.

### 3.1. Recall Reward (thành phần chính, α = 0.6)

```python
# reward.py
import torch
import torchvision

def recall_reward(preds, targets, iou_threshold=0.5, duplicate_penalty=0.3):
    """
    Cho mỗi ảnh:
      + 1.0  cho mỗi GT box được match với ít nhất 1 prediction (IoU >= threshold)
      - 0.3  cho mỗi prediction thừa match cùng 1 GT box
    Normalize theo số lớp có trong ảnh.

    Penalty 0.3 < 1.0: ưu tiên tăng recall hơn phạt duplicate.
    """
    rewards = torch.zeros(len(preds))
    for i, (pred, tgt) in enumerate(zip(preds, targets)):
        gt_boxes, gt_labels = tgt["boxes"], tgt["labels"]
        if len(gt_boxes) == 0:
            continue

        classes = gt_labels.unique()
        class_score = 0.0
        for cls in classes:
            gt_cls   = gt_boxes[gt_labels == cls]
            pred_cls = pred["boxes"][pred["labels"] == cls]
            if len(pred_cls) == 0:
                continue

            iou_mat = torchvision.ops.box_iou(gt_cls, pred_cls)
            matched = iou_mat > iou_threshold

            n_matched_gt = torch.any(matched, dim=1).sum().float()
            n_duplicates = (matched.sum(dim=1) - 1).clamp(0).sum().float()
            class_score += (n_matched_gt - duplicate_penalty * n_duplicates).item()

        rewards[i] = class_score / len(classes)
    return rewards
```

### 3.2. Small-Object Bonus (thành phần UAV, α = 0.4)

```python
def small_object_recall(preds, targets, small_thresh=32, iou_threshold=0.5):
    """
    Bonus reward cho GT boxes nhỏ (area < 32² pixels).
    Đặc thù UAV: sâu non, trứng, đốm bệnh đầu mùa.
    """
    rewards = torch.zeros(len(preds))
    for i, (pred, tgt) in enumerate(zip(preds, targets)):
        gt_boxes = tgt["boxes"]
        if len(gt_boxes) == 0 or len(pred["boxes"]) == 0:
            continue

        areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        small_gt = gt_boxes[areas < small_thresh**2]
        if len(small_gt) == 0:
            continue

        iou_mat = torchvision.ops.box_iou(small_gt, pred["boxes"])
        n_matched = torch.any(iou_mat > iou_threshold, dim=1).sum().float()
        rewards[i] = n_matched / (len(small_gt) + 1e-6)
    return rewards
```

### 3.3. Composite Reward

```python
def composite_reward(preds, targets, alpha=0.6):
    r_recall = recall_reward(preds, targets)
    r_small  = small_object_recall(preds, targets)
    # Nếu không có small object: fallback về recall thuần túy
    has_small = (r_small > 0).float()
    r_small   = has_small * r_small + (1 - has_small) * r_recall
    return alpha * r_recall + (1 - alpha) * r_small
```

---

## 4. Log-Probability Proxy

```python
def compute_log_prob(preds):
    """
    Xấp xỉ log π_θ(action | state) = log(avg_confidence).
    Gradient chảy qua confidence scores của YOLO → cập nhật weights.
    """
    log_probs = []
    for pred in preds:
        if len(pred["scores"]) == 0:
            # Không có prediction → phạt mạnh, nhưng giữ gradient
            lp = torch.tensor(-20.0, requires_grad=True)
        else:
            avg_conf = pred["scores"].mean().clamp(1e-20, 1.0)
            lp = torch.log(avg_conf)
        log_probs.append(lp)
    return torch.stack(log_probs)   # (batch_size,), requires_grad=True
```

---

## 5. EMA Baseline

```python
class EMABaseline:
    """
    Giảm variance của REINFORCE bằng cách trừ baseline.
    
    Tương tự GRPO: baseline = mean(reward trong group)
    Ở đây:         baseline = EMA của reward qua thời gian
    
    Rẻ hơn GRPO: không cần G forward passes.
    Ổn định hơn Monte-Carlo (sample 2 lần): EMA mượt hơn.
    """
    def __init__(self, alpha=0.99):
        self.alpha = alpha
        self.value = None

    def advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        b = self.update(rewards.mean().item())
        return rewards - b   # E[advantage] ≈ 0 khi hội tụ

    def update(self, r: float) -> float:
        self.value = r if self.value is None else (
            self.alpha * self.value + (1 - self.alpha) * r
        )
        return self.value
```

---

## 6. Vòng Lặp RL Training

```python
# train_rl.py (phần cốt lõi)
from collections import deque
import numpy as np
import torch

def rl_finetune(model_wrapper, train_loader, cfg, device="cuda"):
    """
    REINFORCE fine-tuning loop.
    model_wrapper: YOLOWrapper với phương thức forward_with_grad()
    """
    baseline  = EMABaseline(alpha=cfg["ema_alpha"])   # 0.99
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_wrapper.parameters()),
        lr=cfg["lr"]    # 1e-6 – QUAN TRỌNG: phải rất nhỏ
    )

    reward_hist = deque(maxlen=200)   # rolling avg để track xu hướng
    best_avg_reward = -float("inf")
    data_iter = iter(train_loader)

    for step in range(1, cfg["steps"] + 1):
        # ── Lấy batch (infinite loop) ─────────────────────────────────
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)
        images = images.to(device)

        # ── 1. Forward – GIỮ GRADIENT qua confidence ──────────────────
        preds = model_wrapper.forward_with_grad(
            images,
            conf_thres=cfg["conf_thres"],   # 0.20 (thấp hơn eval=0.25)
            iou_thres=cfg["iou_thres"]      # 0.45
        )

        # ── 2. Reward (NO GRAD – rule-based) ──────────────────────────
        with torch.no_grad():
            rewards = composite_reward(preds, targets,
                                       alpha=cfg["reward_alpha"]).to(device)

        # ── 3. Advantage ───────────────────────────────────────────────
        advantage = baseline.advantage(rewards)

        # ── 4. Log-prob proxy ──────────────────────────────────────────
        log_probs = compute_log_prob(preds)   # (B,), requires_grad

        # ── 5. REINFORCE loss  L = -E[log π(a|s) × Adv] ───────────────
        loss = -torch.mean(log_probs * advantage.detach())

        # ── 6. Backprop ────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
        optimizer.step()

        # ── 7. Logging & Checkpoint ────────────────────────────────────
        r_val = rewards.mean().item()
        reward_hist.append(r_val)

        if step % cfg["log_interval"] == 0:
            avg_r = np.mean(reward_hist)
            print(f"Step {step:6d} | R={r_val:.4f} avg200={avg_r:.4f} "
                  f"loss={loss.item():.6f} baseline={baseline.value:.4f}")

            # Lưu theo rolling average, không phải instant reward
            if avg_r > best_avg_reward and step > 500:   # warmup 500 steps
                best_avg_reward = avg_r
                save_rl_checkpoint(model_wrapper, step, avg_r, cfg["output_dir"])
```

---

## 7. Adapter: Giữ Gradient Qua Confidence

Đây là phần kỹ thuật quan trọng nhất. YOLO cần được gọi theo cách đặc biệt để
gradient chảy qua confidence scores (không dùng `model.predict()` thông thường).

### 7.1. YOLOv5 / DP-YOLO

```python
# adapters/yolov5_adapter.py
import torch, sys
sys.path.insert(0, "yolov5")
from models.common import DetectMultiBackend
from utils.general import non_max_suppression

class YOLOv5Adapter:
    def __init__(self, checkpoint, device="cuda"):
        self.device = device
        self.model = DetectMultiBackend(checkpoint, device=device)
        self.model.train()   # train mode để gradient hoạt động

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def forward_with_grad(self, images, conf_thres=0.20, iou_thres=0.45):
        """
        raw_out: list của Tensor (B, num_anchors, 5+nc) – có gradient.
        Boxes và labels: detach (không cần grad, chỉ dùng trong reward).
        Scores: giữ gradient để loss.backward() hoạt động.
        """
        raw_out = self.model(images)   # (B, anchors, 5+nc)

        preds = []
        for b in range(images.shape[0]):
            # Scores có gradient (conf * max_cls)
            conf     = raw_out[0][b, :, 4]
            cls_max  = raw_out[0][b, :, 5:].max(dim=-1).values
            scores_grad = conf * cls_max   # GRADIENT ở đây

            # Boxes & labels: detach, dùng cho reward
            with torch.no_grad():
                det = non_max_suppression(
                    raw_out[0][b:b+1].detach(), conf_thres, iou_thres
                )[0]

            if det is not None and len(det):
                preds.append({
                    "boxes":  det[:, :4],
                    "labels": det[:, 5].long(),
                    "scores": scores_grad[:len(det)],   # gradient preserved
                })
            else:
                preds.append({
                    "boxes":  torch.zeros((0,4), device=self.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=self.device),
                    "scores": torch.zeros(0, device=self.device, requires_grad=True),
                })
        return preds
```

### 7.2. YOLOv8 / YOLOv11 (Ultralytics)

```python
# adapters/ultralytics_adapter.py
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression as nms_v8

class UltralyticsAdapter:
    def __init__(self, checkpoint, device="cuda"):
        self.device = device
        self.model = YOLO(checkpoint).model.to(device)
        self.model.train()

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def forward_with_grad(self, images, conf_thres=0.20, iou_thres=0.45):
        # Ultralytics v8: raw output (B, 4+nc, 8400)
        raw = self.model(images)
        feat = raw[0] if isinstance(raw, (list, tuple)) else raw

        boxes_raw  = feat[:, :4, :]                          # (B, 4, 8400)
        scores_raw = feat[:, 4:, :].max(dim=1).values        # (B, 8400) – grad

        preds = []
        for b in range(images.shape[0]):
            with torch.no_grad():
                det = nms_v8(
                    feat[b:b+1].detach().permute(0, 2, 1),
                    conf_thres=conf_thres, iou_thres=iou_thres
                )[0]

            if det is not None and len(det):
                preds.append({
                    "boxes":  det[:, :4].detach(),
                    "labels": det[:, 5].long().detach(),
                    "scores": scores_raw[b, :len(det)],   # gradient preserved
                })
            else:
                preds.append({
                    "boxes":  torch.zeros((0,4), device=self.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=self.device),
                    "scores": torch.zeros(0, device=self.device, requires_grad=True),
                })
        return preds
```

---

## 8. Freeze Backbone (Khi Cần)

```python
def freeze_backbone(adapter, n_layers=10):
    """
    Đóng băng n_layers đầu (backbone) – chỉ fine-tune neck + head.
    Dùng khi: GPU hạn chế, hoặc mAP drop > 5% sau RL warmup.
    """
    frozen = 0
    for name, param in adapter.named_parameters():
        # YOLOv5/v8: layers được đánh số model.0, model.1, ...
        layer_idx = None
        for part in name.split("."):
            if part.isdigit():
                layer_idx = int(part)
                break
        if layer_idx is not None and layer_idx < n_layers:
            param.requires_grad = False
            frozen += 1
    print(f"Frozen {frozen} params (layers 0-{n_layers-1})")
```

---

## 9. Hyperparameters

```yaml
# configs/hyp.rl.yaml

# ── Optimizer ──────────────────────────────────────────────────────────
lr: 1.0e-6          # RẤT NHỎ – implicit KL constraint, tránh catastrophic forgetting
optimizer: adam     # Adam ổn định hơn SGD cho RL fine-tuning
grad_clip: 1.0      # Clip gradient norm

# ── Training schedule ──────────────────────────────────────────────────
steps: 50000        # Số bước RL (1 step = 1 batch)
warmup_steps: 500   # Không lưu checkpoint trong 500 bước đầu (EMA chưa ổn định)
log_interval: 100
save_interval: 5000
eval_interval: 5000

# ── Reward ─────────────────────────────────────────────────────────────
reward_alpha: 0.6       # Trọng số recall vs small-object (0.6 = recall quan trọng hơn)
iou_threshold: 0.5      # IoU để tính "match" GT box
duplicate_penalty: 0.3  # Phạt pred thừa (nhẹ, ưu tiên recall)
small_thresh: 32        # Threshold vật thể nhỏ (pixels, area < 32² = 1024 px²)

# ── EMA Baseline ───────────────────────────────────────────────────────
ema_alpha: 0.99         # Decay EMA baseline

# ── Forward trong training loop ────────────────────────────────────────
conf_thres: 0.20        # Thấp hơn eval (0.25) → nhiều predictions → signal mạnh hơn
iou_thres: 0.45
batch_size: 16

# ── Architecture ───────────────────────────────────────────────────────
freeze_backbone: false  # Bật nếu gặp catastrophic forgetting
freeze_n_layers: 10     # Số layer backbone đóng băng

# ── Checkpoint ─────────────────────────────────────────────────────────
checkpoint_by: avg_reward_200    # Lưu theo rolling average 200 steps, không instant
output_dir: rl_checkpoints
```

---

## 10. Evaluation: Supervised vs RL

```python
# evaluate.py
import torch, pandas as pd, time
from pathlib import Path
from torchmetrics.detection import MeanAveragePrecision

def evaluate_checkpoint(checkpoint_path, val_loader, device="cuda"):
    """Chạy eval đầy đủ cho 1 checkpoint, trả về dict metrics."""
    from ultralytics import YOLO
    model = YOLO(checkpoint_path).to(device)
    model.eval()

    metric = MeanAveragePrecision(iou_thresholds=[0.5], extended_summary=True)
    preds_all, targets_all = [], []
    t0, n_imgs = time.time(), 0

    with torch.no_grad():
        for images, targets in val_loader:
            results = model(images, verbose=False)
            for r, t in zip(results, targets):
                preds_all.append({
                    "boxes":  r.boxes.xyxy.cpu(),
                    "scores": r.boxes.conf.cpu(),
                    "labels": r.boxes.cls.int().cpu(),
                })
                targets_all.append({
                    "boxes":  t["boxes"].cpu(),
                    "labels": t["labels"].int().cpu(),
                })
            n_imgs += len(images)

    metric.update(preds_all, targets_all)
    res = metric.compute()
    fps = n_imgs / (time.time() - t0)

    return {
        "mAP50":    res["map_50"].item(),
        "mAP50-95": res["map"].item(),
        "AP_small": res.get("map_small", torch.tensor(0.0)).item(),
        "Recall":   res.get("mar_100", torch.tensor(0.0)).item(),
        "FPS":      fps,
    }


def run_comparison():
    """So sánh supervised vs RL cho tất cả model."""
    EXPERIMENTS = {
        "YOLOv5s":  {"supervised": "checkpoints/yolov5s/weights/best.pt",
                     "rl":         "rl_checkpoints/yolov5s_rl_best.pt"},
        "YOLOv8n":  {"supervised": "checkpoints/yolov8n/weights/best.pt",
                     "rl":         "rl_checkpoints/yolov8n_rl_best.pt"},
        "YOLOv11n": {"supervised": "checkpoints/yolov11n/weights/best.pt",
                     "rl":         "rl_checkpoints/yolov11n_rl_best.pt"},
        "DP-YOLO":  {"supervised": "checkpoints/dp_yolo/weights/best.pt",
                     "rl":         "rl_checkpoints/dp_yolo_rl_best.pt"},
    }
    from dataloader import get_pest_dataloader
    val_loader = get_pest_dataloader("data/V2", split="val", batch_size=16)

    rows = []
    for model_name, paths in EXPERIMENTS.items():
        for stage, ckpt in paths.items():
            if not Path(ckpt).exists():
                print(f"  Skip: {ckpt}")
                continue
            print(f"Evaluating {model_name} [{stage}]...")
            metrics = evaluate_checkpoint(ckpt, val_loader)
            rows.append({"Model": model_name, "Stage": stage, **metrics})

    df = pd.DataFrame(rows)

    # Tính delta RL - Supervised
    for m in ["mAP50", "mAP50-95", "AP_small", "Recall"]:
        for model_name in df["Model"].unique():
            sup = df[(df["Model"]==model_name) & (df["Stage"]=="supervised")][m]
            rl  = df[(df["Model"]==model_name) & (df["Stage"]=="rl")][m]
            if len(sup) and len(rl):
                delta = rl.values[0] - sup.values[0]
                print(f"  {model_name} | {m} delta: {delta:+.4f}")

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/tables/comparison.csv", index=False)
    print(df.to_markdown(index=False, floatfmt=".4f"))
    return df
```

**Template bảng kết quả:**

| Model | Stage | mAP@50 | mAP@50-95 | AP_small | Recall | FPS |
|---|---|---|---|---|---|---|
| YOLOv5s | Supervised | — | — | — | — | — |
| YOLOv5s | **+RL** | — | — | — | — | — |
| YOLOv8n | Supervised | — | — | — | — | — |
| YOLOv8n | **+RL** | — | — | — | — | — |
| YOLOv11n | Supervised | — | — | — | — | — |
| YOLOv11n | **+RL** | — | — | — | — | — |
| **DP-YOLO** | Supervised | — | — | — | — | — |
| **DP-YOLO** | **+RL** | — | — | — | — | — |

> **Cột quan trọng nhất:** `AP_small` và `Recall` – đây là 2 chỉ số thể hiện
> hiệu quả trực tiếp của RL trên bài toán vật thể nhỏ UAV.

---

## 11. Ma Trận Thí Nghiệm

| Exp | Model | RL | Freeze | Steps | LR | Mục tiêu |
|---|---|---|---|---|---|---|
| E01 | YOLOv5s | ✗ | — | — | — | Baseline cổ điển |
| E02 | YOLOv5s | ✓ | ✗ | 50k | 1e-6 | RL boost YOLOv5s |
| E03 | YOLOv8n | ✗ | — | — | — | Anchor-free baseline |
| E04 | YOLOv8n | ✓ | ✗ | 50k | 1e-6 | RL boost YOLOv8n |
| E05 | YOLOv11n | ✗ | — | — | — | SOTA baseline |
| E06 | YOLOv11n | ✓ | ✗ | 50k | 1e-6 | RL boost YOLOv11n |
| E07 | DP-YOLO | ✗ | — | — | — | Custom architecture |
| E08 | **DP-YOLO** | ✓ | ✗ | 50k | 1e-6 | **Main experiment** |
| E09 | DP-YOLO | ✓ | ✓ | 50k | 1e-5 | Head-only fine-tune |
| E10 | DP-YOLO | ✓ | ✗ | 100k | 1e-6 | Longer training |

---

## 12. Xử Lý Sự Cố

| Vấn đề | Triệu chứng | Giải pháp |
|---|---|---|
| Catastrophic forgetting | mAP50 giảm > 5% sau RL | Giảm lr → 5e-7, hoặc freeze backbone |
| Reward không tăng | Reward plateau ngay từ đầu | Hạ `conf_thres` xuống 0.15, tăng batch_size |
| High variance | Reward dao động mạnh, loss spike | Tăng ema_alpha → 0.999, tăng rolling window 200→500 |
| Gradient = 0 | loss = 0 mọi bước | Kiểm tra `requires_grad` của scores, hạ conf_thres |
| Recall tăng, Precision giảm | Mô hình predict nhiều hơn | Tăng duplicate_penalty 0.3 → 0.5 |
| Mode collapse | Mọi prediction là 1 class | Thêm per-class reward thay vì average |

---

## 13. Checklist Thực Hành

```
CHUẨN BỊ
□ Train supervised ≥ 100 epochs (tốt nhất 300 epochs)
□ Lưu best.pt supervised → đây là điểm xuất phát RL
□ Verify dataset: labels đúng format (xyxy absolute cho reward.py)

RL FINE-TUNING
□ Bắt đầu: lr=1e-6, freeze_backbone=false, steps=50000
□ Theo dõi: rolling avg reward 200 steps (không nhìn instant reward)
□ Kiểm tra sau 1000 bước đầu: reward có xu hướng tăng không?
□ Kiểm tra mAP mỗi 5000 bước: có drop > 5% không?
  → Có: giảm lr hoặc freeze backbone
  → Không: tiếp tục

ĐÁNH GIÁ
□ So sánh supervised vs RL trên test set (cùng conf_thres=0.25)
□ Chú trọng AP_small và Recall (metric chính của RL)
□ Kiểm tra FPS không thay đổi (kiến trúc không đổi)
□ Phân tích per-class AP: lớp nào được cải thiện nhiều nhất?
```

---

## 14. Lệnh Chạy

```bash
# ── Bước 1: Supervised Training ────────────────────────────────────────
# YOLOv5s
cd kb1_reward_guided_training && python yolov5/train.py \
  --weights yolov5s.pt --data configs/pest.yaml \
  --epochs 300 --batch-size 32 --project checkpoints --name yolov5s

# YOLOv8n / YOLOv11n
yolo train model=yolov8n.pt data=configs/pest.yaml \
  epochs=300 batch=32 project=checkpoints name=yolov8n

# ── Bước 2: RL Fine-tuning ─────────────────────────────────────────────
cd kb1_reward_guided_training && python train_rl.py \
  --model yolov5s \
  --checkpoint checkpoints/yolov5s/weights/best.pt \
  --steps 50000 --lr 1e-6 --device cuda

# Tất cả model
python train_rl.py --model all --steps 50000 --lr 1e-6

# Với freeze backbone
python train_rl.py --model dp_yolo --freeze --lr 1e-5

# ── Bước 3: Evaluation ─────────────────────────────────────────────────
cd kb1_reward_guided_training && python evaluate.py
```

---

## 15. Tài Liệu Tham Khảo

1. **Tuning CV Models with Task Rewards** — Zhai et al., Google DeepMind, arXiv:2302.08242, 2023
2. **REINFORCE** — Williams (1992), cơ sở lý thuyết policy gradient
3. **DeepSeek-R1** — rule-based verified reward, GRPO group advantage
4. **cv-rl** (bwconrad) — CE-pretrain → RL fine-tune pattern
5. **tuning_cv_models_with_rl_torch** (yanivnik) — recall reward, confidence proxy
6. **DP-YOLO** — Wang et al., Applied Sciences MDPI, 2023

---
*Xem thêm: `guide-rl-finetune-detection-from-llm.md` để hiểu lý thuyết so sánh DPO/GRPO → YOLO.*
