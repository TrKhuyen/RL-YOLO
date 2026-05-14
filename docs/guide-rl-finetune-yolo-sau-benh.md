# HƯỚNG DẪN FINE-TUNE CÁC MODEL YOLO BẰNG REINFORCEMENT LEARNING CHO BÀI TOÁN PHÁT HIỆN SÂU BỆNH TRÊN UAV

> **Tổng hợp từ:**
> - DP-YOLO guide (guide-dp-yolo-sau-benh.md)
> - cv-rl: *Computer Vision Training using Policy Optimization* (bwconrad, dựa trên paper arXiv:2302.08242)
> - tuning_cv_models_with_rl_torch: *Tuning CV Models with Task Rewards* (yanivnik, Google DeepMind 2023)

---

## 1. BỐI CẢNH VÀ MỤC TIÊU

### 1.1. Vấn đề cốt lõi: Loss không bằng Metric

Khi huấn luyện YOLO với supervised loss (CIoU, BCE classification), mô hình tối ưu **proxy metric** – không phải metric thực tế được dùng để đánh giá:

| Giai đoạn | Tối ưu | Thực chất đo |
|---|---|---|
| Training | CIoU Loss + BCE | Sai số hồi quy box + cross-entropy |
| Evaluation | mAP@50, Recall | IoU threshold + precision-recall curve |

Trên UAV, bài toán sâu bệnh có thêm đặc thù khiến khoảng cách này lớn hơn:
- Sâu nhỏ, mờ do motion blur, hay bị che khuất → mô hình có loss thấp nhưng **recall lớp nhỏ vẫn tệ**.
- Mất cân bằng lớp nặng → supervised loss bị chi phối bởi lớp dễ (lá khỏe).
- YOLOv5/v8/v11 sau supervised training đã hội tụ nhưng vẫn còn dư địa cải thiện recall vật thể nhỏ.

### 1.2. Giải pháp: RL fine-tune sau supervised pre-train

Bài học từ 2 paper nguồn:

> *"Train REINFORCE từ đầu: F1 chỉ đạt 0.41 (rất tệ do variance cao). Fine-tune từ CE checkpoint cho kết quả cân bằng tốt nhất."* — bwconrad/cv-rl

Pattern đã được kiểm chứng:
1. **Giai đoạn 1** – Supervised training đầy đủ (YOLO chuẩn).
2. **Giai đoạn 2** – RL fine-tune: dùng **task metric làm reward** (Recall, mAP@50, hoặc composite) thay vì loss khả vi.

Cách này tương tự RLHF trong NLP (InstructGPT): warm-start với MLE, sau đó fine-tune với policy gradient.

### 1.3. Mục tiêu của guide này

Xây dựng pipeline RL fine-tune áp dụng lần lượt cho:

| Model | Vai trò |
|---|---|
| YOLOv5s | Baseline cổ điển, nền cho DP-YOLO |
| YOLOv8n/s | Baseline anchor-free hiện đại |
| YOLOv11n/s | Baseline mới nhất |
| **DP-YOLO** | Model chính (kiến trúc theo guide-dp-yolo-sau-benh.md) |

So sánh **trước và sau RL fine-tune** cho từng model → đo mức đóng góp của RL độc lập với kiến trúc.

---

## 2. LÝ THUYẾT NỀN TẢNG

### 2.1. REINFORCE (Policy Gradient)

Thuật toán cốt lõi được dùng trong cả 2 paper nguồn (Williams, 1992):

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot R\right]$$

Trong đó $J(\theta)$ là expected reward, $\pi_\theta$ là policy (mô hình YOLO), $a$ là action (tập bounding box predictions), $s$ là state (ảnh đầu vào), và $R$ là reward (task metric).

Để giảm variance, dùng **advantage** thay cho raw reward:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot (R - b)\right]$$

Trong đó $b$ là baseline (ví dụ: reward trung bình trượt exponential).

**REINFORCE loss** (dùng với PyTorch optimizer – minimize):

$$\mathcal{L}_{RL} = -\frac{1}{N}\sum_{i=1}^N \log \pi_\theta(a_i \mid s_i) \cdot (R_i - b)$$

### 2.2. Mapping sang bài toán YOLO

| Khái niệm RL | Trong pipeline này |
|---|---|
| Policy $\pi_\theta$ | YOLO model (tham số $\theta$) |
| State $s$ | Ảnh đầu vào từ UAV |
| Action $a$ | Tập bounding box predictions (boxes, labels, confidences) |
| Reward $R$ | Recall@small / mAP@50 / composite score |
| Log-probability $\log \pi_\theta(a \mid s)$ | $\log(\text{avg\_confidence})$ của các predictions |
| Baseline $b$ | EMA của reward hoặc reward của sample thứ 2 |

> **Lưu ý kỹ thuật (từ yanivnik):** Dùng `log(avg_confidence)` là **xấp xỉ** log-likelihood. YOLO không phải sequence model như DETR, nên xấp xỉ này hợp lý hơn và đủ để gradient đi đúng hướng.

### 2.3. Thiết kế Reward Function cho Sâu Bệnh

Đây là điểm thiết kế quan trọng nhất. Dựa trên reward logic từ yanivnik nhưng điều chỉnh cho bài toán sâu bệnh UAV:

#### Reward thành phần

**a) Recall reward cho vật thể nhỏ** (ưu tiên cao nhất – vấn đề cốt lõi UAV):

$$R_{recall} = \frac{1}{|\mathcal{C}|}\sum_{c \in \mathcal{C}} \frac{\text{GT}_c^{matched} - 0.3 \times \text{Pred}_c^{duplicate}}{|\text{GT}_c|}$$

Trong đó $\text{GT}_c^{matched}$ là số ground truth box của lớp $c$ được match với ít nhất 1 prediction (IoU ≥ 0.5), $\text{Pred}_c^{duplicate}$ là số prediction thừa match vào cùng 1 GT box, và $|\text{GT}_c|$ là tổng số GT box của lớp $c$.

Hệ số phạt `0.3` (nhỏ hơn 1) để ưu tiên tăng recall hơn là trừng phạt duplicate – bám sát thiết kế từ yanivnik và hợp lý cho bài toán sâu bệnh (bỏ sót nguy hiểm hơn false positive).

**b) Small-object bonus** (đặc thù UAV):

$$R_{small} = \frac{\sum_{i} \mathbf{1}[\text{area}_{gt_i} < 32^2] \cdot \mathbf{1}[\text{matched}_i]}{\sum_{i} \mathbf{1}[\text{area}_{gt_i} < 32^2] + \epsilon}$$

**c) Composite reward** (cân bằng cả 2 mục tiêu):

$$R_{total} = \alpha \cdot R_{recall} + (1 - \alpha) \cdot R_{small}, \quad \alpha = 0.6$$

#### Lý do không dùng mAP trực tiếp làm reward

- mAP yêu cầu tính toán trên toàn bộ precision-recall curve → **quá chậm** trong vòng lặp RL (mỗi batch đều phải tính).
- yanivnik xác nhận: "mAP reward currently has some problems" vì mAP per-image không ổn định.
- Recall per-image nhanh hơn ~10× và tương quan tốt với mAP trong thực nghiệm.

---

## 3. KIẾN TRÚC PIPELINE TỔNG THỂ

```
┌─────────────────────────────────────────────────────┐
│              GIAI ĐOẠN 1: SUPERVISED TRAINING        │
│                                                      │
│  YOLOv5s ──┐                                        │
│  YOLOv8n ──┤── train 300 epochs, CIoU + BCE loss    │
│  YOLOv11n ─┤── lưu best.pt cho mỗi model            │
│  DP-YOLO ──┘                                        │
└───────────────────────┬─────────────────────────────┘
                        │  (checkpoint)
┌───────────────────────▼─────────────────────────────┐
│              GIAI ĐOẠN 2: RL FINE-TUNING             │
│                                                      │
│  Load checkpoint ──► Freeze backbone (tùy chọn)     │
│        │                                            │
│  ┌─────▼──────────────────────────────────┐         │
│  │  Vòng lặp REINFORCE (N steps):         │         │
│  │  1. Forward → predictions              │         │
│  │  2. Tính reward (Recall + Small-bonus) │         │
│  │  3. RL loss = -log(conf) * advantage   │         │
│  │  4. Backprop với lr rất nhỏ (1e-6)     │         │
│  └─────────────────────────────────────┘           │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│              GIAI ĐOẠN 3: ĐÁNH GIÁ SO SÁNH          │
│                                                      │
│  Mỗi model: supervised-only vs. RL-finetuned        │
│  Metrics: mAP50, mAP50-95, APs, Recall, FPS         │
└─────────────────────────────────────────────────────┘
```

---

## 4. CÀI ĐẶT MÔI TRƯỜNG

### 4.1. Yêu cầu

```
Python:  3.10+
PyTorch: 2.0+
CUDA:    11.8+ (hoặc 12.x)
GPU:     RTX 3080/3090 trở lên (tối thiểu 8GB VRAM)
```

### 4.2. Cài đặt

```bash
conda create -n yolo-rl python=3.10 -y
conda activate yolo-rl

# PyTorch
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# YOLO frameworks
pip install ultralytics          # YOLOv8, YOLOv11
pip install yolov5               # YOLOv5

# RL + metrics
pip install torchmetrics
pip install tensorboard

# Utils
pip install albumentations opencv-python tqdm matplotlib numpy
```

### 4.3. Cấu trúc thư mục project

```
yolo-rl-pest/
├── data/
│   └── pest/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── configs/
│   ├── pest.yaml              ← dataset config
│   ├── hyp.pest.yaml          ← hyperparameters supervised
│   └── hyp.rl.yaml            ← hyperparameters RL
├── models/
│   └── dp_yolo/               ← custom model files
├── checkpoints/               ← best.pt sau supervised
├── rl_checkpoints/            ← best.pt sau RL
├── results/
│   ├── tensorboard/
│   └── tables/
├── train_supervised.py        ← Giai đoạn 1
├── train_rl.py                ← Giai đoạn 2
├── evaluate.py                ← Giai đoạn 3
└── reward.py                  ← Reward functions
```

---

## 5. GIAI ĐOẠN 1: SUPERVISED TRAINING

### 5.1. Dataset config `configs/pest.yaml`

```yaml
path: ../data/pest
train: images/train
val: images/val
test: images/test

nc: 10
names:
  0: sau_duc_than
  1: sau_cuon_la
  2: benh_dom_nau
  3: benh_dao_on
  4: benh_kho_van
  5: ran_xanh
  6: bo_tri
  7: ran_nau
  8: dom_la
  9: healthy
```

### 5.2. Script `train_supervised.py`

```python
"""
Giai đoạn 1: Train supervised cho tất cả model.
Kết quả: checkpoints/yolov5s_best.pt, yolov8n_best.pt, ...
"""
import subprocess
import shutil
from pathlib import Path

MODELS = {
    "yolov5s":  {"framework": "v5",  "weights": "yolov5s.pt"},
    "yolov8n":  {"framework": "v8",  "weights": "yolov8n.pt"},
    "yolov8s":  {"framework": "v8",  "weights": "yolov8s.pt"},
    "yolov11n": {"framework": "v11", "weights": "yolo11n.pt"},
    "dp_yolo":  {"framework": "v5",  "weights": "yolov5s.pt"},  # custom cfg
}

COMMON = {
    "data":    "configs/pest.yaml",
    "imgsz":   640,
    "epochs":  300,
    "batch":   32,
    "workers": 8,
    "device":  0,
    "patience": 50,
}

def train_yolov5(name, cfg):
    cmd = [
        "python", "yolov5/train.py",
        f"--weights={cfg['weights']}",
        f"--data={COMMON['data']}",
        f"--imgsz={COMMON['imgsz']}",
        f"--epochs={COMMON['epochs']}",
        f"--batch-size={COMMON['batch']}",
        f"--device={COMMON['device']}",
        f"--project=checkpoints",
        f"--name={name}",
        "--optimizer=SGD",
        "--hyp=configs/hyp.pest.yaml",
    ]
    if name == "dp_yolo":
        cmd.append("--cfg=models/dp_yolo/dp_yolo.yaml")
    subprocess.run(cmd, check=True)

def train_ultralytics(name, cfg):
    from ultralytics import YOLO
    model = YOLO(cfg["weights"])
    model.train(
        data=COMMON["data"],
        imgsz=COMMON["imgsz"],
        epochs=COMMON["epochs"],
        batch=COMMON["batch"],
        device=COMMON["device"],
        patience=COMMON["patience"],
        project="checkpoints",
        name=name,
        optimizer="SGD",
        lr0=0.01,
        weight_decay=0.0005,
    )

if __name__ == "__main__":
    for name, cfg in MODELS.items():
        print(f"\n{'='*50}\nTraining {name}...\n{'='*50}")
        if cfg["framework"] == "v5":
            train_yolov5(name, cfg)
        else:
            train_ultralytics(name, cfg)
        print(f"Done: {name}")
```

### 5.3. Lệnh chạy nhanh từng model

```bash
# YOLOv5s
python yolov5/train.py --weights yolov5s.pt --data configs/pest.yaml \
  --epochs 300 --batch-size 32 --imgsz 640 --project checkpoints --name yolov5s

# YOLOv8n (Ultralytics)
yolo train model=yolov8n.pt data=configs/pest.yaml epochs=300 imgsz=640 \
  batch=32 project=checkpoints name=yolov8n

# YOLOv11n
yolo train model=yolo11n.pt data=configs/pest.yaml epochs=300 imgsz=640 \
  batch=32 project=checkpoints name=yolov11n

# DP-YOLO (custom cfg trên nền YOLOv5s)
python yolov5/train.py --weights yolov5s.pt --cfg models/dp_yolo/dp_yolo.yaml \
  --data configs/pest.yaml --epochs 300 --batch-size 32 --project checkpoints --name dp_yolo
```

---

## 6. GIAI ĐOẠN 2: RL FINE-TUNING

### 6.1. Reward Function `reward.py`

```python
"""
Reward functions cho RL fine-tuning YOLO trên bài toán sâu bệnh.
Tham khảo: yanivnik/tuning_cv_models_with_rl_torch
"""
import torch
import torchvision


def compute_iou_matrix(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> torch.Tensor:
    """Tính IoU matrix giữa GT boxes và predicted boxes."""
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return torch.zeros(len(gt_boxes), len(pred_boxes))
    return torchvision.ops.box_iou(gt_boxes, pred_boxes)


def recall_reward(preds: list[dict], targets: list[dict],
                  iou_threshold: float = 0.5,
                  duplicate_penalty: float = 0.3) -> torch.Tensor:
    """
    Tính recall-based reward cho mỗi ảnh trong batch.

    Logic (từ yanivnik):
    - Cộng điểm cho mỗi GT box được match với ít nhất 1 prediction.
    - Trừ nhẹ (0.3x) khi có prediction thừa match cùng 1 GT box.
    - Normalize theo số lớp có trong ảnh.

    Args:
        preds:   list[dict], mỗi dict có keys 'boxes' (xyxy), 'labels', 'scores'
        targets: list[dict], mỗi dict có keys 'boxes' (xyxy), 'labels'
        iou_threshold: ngưỡng IoU để tính là "match"
        duplicate_penalty: hệ số phạt prediction thừa

    Returns:
        rewards: Tensor shape (batch_size,)
    """
    rewards = torch.zeros(len(preds))

    for i, (pred, target) in enumerate(zip(preds, targets)):
        gt_boxes    = target["boxes"]
        gt_labels   = target["labels"]
        pred_boxes  = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]

        if len(gt_boxes) == 0:
            continue

        classes = gt_labels.unique()
        class_reward = 0.0

        for cls in classes:
            gt_mask   = gt_labels == cls
            pred_mask = pred_labels == cls

            gt_cls   = gt_boxes[gt_mask]
            pred_cls = pred_boxes[pred_mask]

            if len(pred_cls) == 0:
                continue

            iou_mat = compute_iou_matrix(gt_cls, pred_cls)
            matched = iou_mat > iou_threshold                      # (n_gt, n_pred) bool

            n_matched_gt  = torch.any(matched, dim=1).sum().float()
            n_duplicates  = (matched.sum(dim=1) - 1).clamp(0).sum().float()

            class_reward += (n_matched_gt - duplicate_penalty * n_duplicates).item()

        rewards[i] = class_reward / len(classes)

    return rewards


def small_object_recall_reward(preds: list[dict], targets: list[dict],
                                small_thresh: int = 32,
                                iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Bonus reward tập trung vào GT boxes nhỏ (diện tích < small_thresh^2 px).
    Đặc thù UAV: sâu non, trứng, đốm bệnh đầu mùa.
    """
    rewards = torch.zeros(len(preds))

    for i, (pred, target) in enumerate(zip(preds, targets)):
        gt_boxes   = target["boxes"]
        pred_boxes = pred["boxes"]

        if len(gt_boxes) == 0:
            continue

        # Lọc GT boxes nhỏ: diện tích = (x2-x1)*(y2-y1)
        areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        small_mask = areas < (small_thresh ** 2)
        small_gt = gt_boxes[small_mask]

        if len(small_gt) == 0:
            continue

        if len(pred_boxes) == 0:
            continue

        iou_mat = compute_iou_matrix(small_gt, pred_boxes)
        n_matched = torch.any(iou_mat > iou_threshold, dim=1).sum().float()
        rewards[i] = n_matched / (len(small_gt) + 1e-6)

    return rewards


def composite_reward(preds: list[dict], targets: list[dict],
                     alpha: float = 0.6) -> torch.Tensor:
    """
    Reward tổng hợp: alpha * recall + (1-alpha) * small_recall.
    alpha=0.6: ưu tiên recall tổng nhưng vẫn chú trọng vật thể nhỏ.
    """
    r_recall = recall_reward(preds, targets)
    r_small  = small_object_recall_reward(preds, targets)
    return alpha * r_recall + (1 - alpha) * r_small
```

### 6.2. Script RL Fine-tuning `train_rl.py`

```python
"""
Giai đoạn 2: RL fine-tune YOLO models bằng REINFORCE.

Thiết kế dựa trên:
- bwconrad/cv-rl: pattern CE-pretrain → RL fine-tune
- yanivnik/tuning_cv_models_with_rl_torch: recall reward, confidence log-prob

QUAN TRỌNG:
- Learning rate rất nhỏ (1e-6 đến 1e-5) để tránh catastrophic forgetting.
- Freeze backbone nếu GPU hạn chế.
- Dùng EMA baseline để giảm variance của REINFORCE.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import yaml
import argparse
from collections import deque
import numpy as np

from reward import composite_reward


# ─────────────────────────────────────────────────────────────────────────────
# 1. Wrapper: đưa bất kỳ YOLO nào về interface chung
# ─────────────────────────────────────────────────────────────────────────────

class YOLOWrapper:
    """
    Wrapper chuẩn hóa interface cho YOLOv5/v8/v11/DP-YOLO.
    Output chuẩn: list[dict] với keys 'boxes'(xyxy), 'labels', 'scores'.
    """

    def __init__(self, model_name: str, checkpoint: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = self._load(checkpoint)

    def _load(self, checkpoint: str):
        if "v5" in self.model_name or "dp_yolo" in self.model_name:
            import sys
            sys.path.insert(0, "yolov5")
            model = torch.hub.load("ultralytics/yolov5", "custom",
                                   path=checkpoint, force_reload=False)
        else:
            from ultralytics import YOLO
            model = YOLO(checkpoint)
        return model.to(self.device)

    def parameters(self):
        if hasattr(self.model, "model"):
            return self.model.model.parameters()
        return self.model.parameters()

    def named_parameters(self):
        if hasattr(self.model, "model"):
            return self.model.model.named_parameters()
        return self.model.named_parameters()

    def train_mode(self):
        if hasattr(self.model, "model"):
            self.model.model.train()
        else:
            self.model.train()

    def eval_mode(self):
        if hasattr(self.model, "model"):
            self.model.model.eval()
        else:
            self.model.eval()

    def forward(self, images: torch.Tensor) -> list[dict]:
        """
        Trả về list[dict] predictions chuẩn hóa.
        Mỗi dict: {'boxes': Tensor(N,4 xyxy), 'labels': Tensor(N,), 'scores': Tensor(N,)}
        """
        raise NotImplementedError("Implement cho từng framework cụ thể")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Compute log-probability (xấp xỉ)
# ─────────────────────────────────────────────────────────────────────────────

def compute_log_prob(preds: list[dict]) -> torch.Tensor:
    """
    Xấp xỉ log π_θ(a|s) = log(average confidence) trên toàn batch.

    Lý do dùng xấp xỉ này (từ yanivnik):
    YOLO không phải autoregressive model, không có log-prob đầy đủ.
    avg_confidence là proxy hợp lý: khi confidence cao → policy "chắc chắn"
    về action → gradient đi đúng hướng.

    Clamp về [1e-20, 1.0] để tránh log(0).
    """
    log_probs = []
    for pred in preds:
        if len(pred["scores"]) == 0:
            # Không có prediction → confidence ≈ 0 → log ≈ -inf → clip
            log_probs.append(torch.tensor(-20.0, requires_grad=True))
        else:
            avg_conf = pred["scores"].mean().clamp(1e-20, 1.0)
            log_probs.append(torch.log(avg_conf))
    return torch.stack(log_probs)


# ─────────────────────────────────────────────────────────────────────────────
# 3. EMA Baseline – giảm variance REINFORCE
# ─────────────────────────────────────────────────────────────────────────────

class EMABaseline:
    """
    Exponential Moving Average baseline để giảm variance của REINFORCE.
    advantage = R - baseline (unbiased vì E[baseline] = E[R] khi hội tụ).

    Từ bwconrad/cv-rl: dùng sample thứ 2 làm baseline (Monte Carlo).
    Từ yanivnik: TODO baseline chưa implement.
    → Dùng EMA ổn định hơn cả 2 cách trên.
    """
    def __init__(self, alpha: float = 0.99):
        self.alpha = alpha
        self.value = None

    def update(self, reward: float) -> float:
        if self.value is None:
            self.value = reward
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * reward
        return self.value

    def advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        b = self.update(rewards.mean().item())
        return rewards - b


# ─────────────────────────────────────────────────────────────────────────────
# 4. RL Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def rl_finetune(
    model_name:     str,
    checkpoint:     str,
    dataloader:     DataLoader,
    steps:          int   = 50_000,
    lr:             float = 1e-6,
    freeze_backbone:bool  = False,
    reward_alpha:   float = 0.6,
    log_interval:   int   = 100,
    save_interval:  int   = 5_000,
    output_dir:     str   = "rl_checkpoints",
    device:         str   = "cuda",
):
    """
    Vòng lặp REINFORCE fine-tuning cho 1 YOLO model.

    Args:
        model_name:      Tên model (dùng cho logging và lưu file)
        checkpoint:      Path đến best.pt từ supervised training
        dataloader:      DataLoader của tập train
        steps:           Số bước RL (khuyến nghị 20k–100k)
        lr:              Learning rate (rất nhỏ! 1e-6 đến 1e-5)
        freeze_backbone: Đóng băng backbone, chỉ fine-tune head/neck
        reward_alpha:    Trọng số recall vs small-object reward
        log_interval:    Log tensorboard mỗi N bước
        save_interval:   Lưu checkpoint mỗi N bước
        output_dir:      Thư mục lưu checkpoint RL
        device:          'cuda' hoặc 'cpu'
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(f"results/tensorboard/rl_{model_name}")
    baseline = EMABaseline(alpha=0.99)

    # Load model
    wrapper = YOLOWrapper(model_name, checkpoint, device)
    wrapper.train_mode()

    # Freeze backbone nếu cần (tiết kiệm VRAM, tránh catastrophic forgetting)
    if freeze_backbone:
        frozen_count = 0
        for name, param in wrapper.named_parameters():
            # Freeze tầng backbone (layer0 đến layer9 trong YOLOv5)
            if any(f"model.{i}" in name for i in range(10)):
                param.requires_grad = False
                frozen_count += 1
        print(f"Frozen {frozen_count} backbone parameters.")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, wrapper.parameters()),
        lr=lr,
    )

    data_iter   = iter(dataloader)
    reward_hist = deque(maxlen=200)

    print(f"\n{'='*60}")
    print(f"RL Fine-tuning: {model_name}")
    print(f"Steps: {steps} | LR: {lr} | Freeze backbone: {freeze_backbone}")
    print(f"{'='*60}\n")

    for step in range(1, steps + 1):
        # Lấy batch tiếp theo (vòng lặp vô hạn qua dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images, targets = batch
        images = images.to(device)

        # ── 1. Forward pass (có gradient qua confidence scores) ──────────
        # NOTE: YOLO forward trong train mode trả về cả loss và preds.
        # Ở đây ta chỉ cần preds để tính reward và log_prob.
        # Implementation cụ thể phụ thuộc vào framework (xem Section 7).
        preds = wrapper.forward(images)  # list[dict]

        # ── 2. Tính reward ────────────────────────────────────────────────
        with torch.no_grad():
            rewards = composite_reward(preds, targets, alpha=reward_alpha)
            rewards = rewards.to(device)

        # ── 3. EMA baseline → advantage ───────────────────────────────────
        advantage = baseline.advantage(rewards)

        # ── 4. Log-probability xấp xỉ ─────────────────────────────────────
        log_probs = compute_log_prob(preds)  # (batch_size,), requires_grad

        # ── 5. REINFORCE loss ──────────────────────────────────────────────
        # L = -E[log π(a|s) * advantage]
        # Dấu trừ: PyTorch minimize, ta muốn maximize reward
        loss = -torch.mean(log_probs * advantage.detach())

        # ── 6. Backprop ────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping để ổn định
        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=1.0)
        optimizer.step()

        # ── 7. Logging ─────────────────────────────────────────────────────
        reward_val = rewards.mean().item()
        reward_hist.append(reward_val)

        if step % log_interval == 0:
            avg_reward = np.mean(reward_hist)
            writer.add_scalar(f"{model_name}/reward",      reward_val, step)
            writer.add_scalar(f"{model_name}/reward_avg",  avg_reward, step)
            writer.add_scalar(f"{model_name}/loss",        loss.item(), step)
            writer.add_scalar(f"{model_name}/baseline",    baseline.value, step)
            print(f"Step {step:6d} | Reward: {reward_val:.4f} "
                  f"(avg200: {avg_reward:.4f}) | Loss: {loss.item():.6f}")

        # ── 8. Save checkpoint ─────────────────────────────────────────────
        if step % save_interval == 0:
            save_path = f"{output_dir}/{model_name}_rl_step{step}.pt"
            if hasattr(wrapper.model, "model"):
                torch.save(wrapper.model.model.state_dict(), save_path)
            else:
                torch.save(wrapper.model.state_dict(), save_path)
            print(f"  → Saved: {save_path}")

    writer.close()
    print(f"\nRL fine-tuning xong: {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main: chạy RL cho tất cả model
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="all",
                        choices=["all","yolov5s","yolov8n","yolov11n","dp_yolo"])
    parser.add_argument("--steps",   type=int,   default=50_000)
    parser.add_argument("--lr",      type=float, default=1e-6)
    parser.add_argument("--freeze",  action="store_true")
    parser.add_argument("--device",  default="cuda")
    args = parser.parse_args()

    CHECKPOINTS = {
        "yolov5s":  "checkpoints/yolov5s/weights/best.pt",
        "yolov8n":  "checkpoints/yolov8n/weights/best.pt",
        "yolov11n": "checkpoints/yolov11n/weights/best.pt",
        "dp_yolo":  "checkpoints/dp_yolo/weights/best.pt",
    }

    # Import dataloader (cần implement theo format dataset của bạn)
    from dataloader import get_pest_dataloader
    dataloader = get_pest_dataloader("data/pest", split="train", batch_size=16)

    targets = CHECKPOINTS if args.model == "all" else {args.model: CHECKPOINTS[args.model]}

    for model_name, ckpt in targets.items():
        rl_finetune(
            model_name=model_name,
            checkpoint=ckpt,
            dataloader=dataloader,
            steps=args.steps,
            lr=args.lr,
            freeze_backbone=args.freeze,
            device=args.device,
        )
```

---

## 7. ADAPTER CHO TỪNG FRAMEWORK

Phần này giải quyết sự khác biệt giữa các YOLO framework về cách forward và lấy predictions.

### 7.1. YOLOv5s và DP-YOLO

```python
# adapters/yolov5_adapter.py
import torch
import sys
sys.path.insert(0, "yolov5")
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes


class YOLOv5Adapter:
    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.device = device
        self.model = DetectMultiBackend(checkpoint, device=device)
        self.model.train()

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def forward_with_grad(self, images: torch.Tensor) -> list[dict]:
        """
        Forward pass giữ gradient qua confidence scores.
        images: Tensor (B, 3, H, W), đã chuẩn hóa [0,1].
        """
        # YOLOv5 raw output: (B, num_anchors, 85) = (cx,cy,w,h,conf,cls*80)
        raw_out = self.model(images)  # list of (B, anchors, 5+nc)

        preds = []
        for b in range(images.shape[0]):
            # Lấy từng anchor's confidence (có gradient)
            # raw_out[0] là scale lớn nhất → tổng hợp sau NMS
            # Đây là phần cần custom để giữ gradient qua conf
            conf = raw_out[0][b, :, 4]   # confidence của mỗi anchor
            cls_score = raw_out[0][b, :, 5:].max(dim=-1).values
            scores = conf * cls_score    # final score = obj_conf * cls_conf

            # Detach boxes và labels để dùng trong reward (không cần grad)
            with torch.no_grad():
                det = non_max_suppression(
                    raw_out[0][b:b+1].detach(), conf_thres=0.25, iou_thres=0.45
                )[0]

            if det is not None and len(det):
                preds.append({
                    "boxes":  det[:, :4],
                    "labels": det[:, 5].long(),
                    "scores": scores[:len(det)],  # Giữ gradient ở đây
                })
            else:
                preds.append({
                    "boxes":  torch.zeros((0, 4), device=self.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=self.device),
                    "scores": torch.zeros(0, device=self.device, requires_grad=True),
                })
        return preds
```

### 7.2. YOLOv8 và YOLOv11 (Ultralytics)

```python
# adapters/ultralytics_adapter.py
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


class UltralyticsAdapter:
    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.device = device
        yolo = YOLO(checkpoint)
        self.model: DetectionModel = yolo.model.to(device)
        self.model.train()

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def forward_with_grad(self, images: torch.Tensor) -> list[dict]:
        """
        Ultralytics YOLOv8/v11: model trả về list of Detection objects.
        Ta cần giữ gradient qua confidence.
        """
        # Chạy raw forward để lấy tensor output có gradient
        raw = self.model(images)  # (B, 84, 8400) với nc=80, hoặc (B, 14+nc, anc)

        preds = []
        batch_size = images.shape[0]

        # Ultralytics v8 output format: (B, 4+nc, num_anchors)
        # raw[0]: Tensor(B, 4+nc, 8400)
        feat = raw[0] if isinstance(raw, (list, tuple)) else raw
        boxes_raw = feat[:, :4, :]       # (B, 4, 8400)
        scores_raw = feat[:, 4:, :].max(dim=1).values  # (B, 8400) – max cls score

        for b in range(batch_size):
            # Chạy NMS tách biệt (detach) để lấy indices predictions hợp lệ
            with torch.no_grad():
                from ultralytics.utils.ops import non_max_suppression as nms_v8
                det = nms_v8(feat[b:b+1].detach().permute(0, 2, 1),
                             conf_thres=0.25, iou_thres=0.45)[0]

            if det is not None and len(det):
                preds.append({
                    "boxes":  det[:, :4].detach(),
                    "labels": det[:, 5].long().detach(),
                    "scores": scores_raw[b, :len(det)],  # Giữ gradient
                })
            else:
                preds.append({
                    "boxes":  torch.zeros((0, 4), device=self.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=self.device),
                    "scores": torch.zeros(0, device=self.device, requires_grad=True),
                })
        return preds
```

---

## 8. GIAI ĐOẠN 3: ĐÁNH GIÁ SO SÁNH

### 8.1. Script `evaluate.py`

```python
"""
Đánh giá toàn bộ model: supervised-only vs. RL-finetuned.
Output: bảng so sánh CSV + markdown.
"""
import torch
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision


def evaluate_model(checkpoint: str, dataloader, device: str = "cuda") -> dict:
    """
    Chạy evaluation đầy đủ cho 1 checkpoint.
    Trả về dict với mAP50, mAP50-95, APs, recall, FPS.
    """
    model = YOLO(checkpoint).to(device)
    model.eval()

    metric = MeanAveragePrecision(iou_thresholds=[0.5],
                                  extended_summary=True)

    all_preds, all_targets = [], []
    import time

    t0 = time.time()
    n_images = 0

    with torch.no_grad():
        for images, targets in dataloader:
            results = model(images, verbose=False)
            for r, t in zip(results, targets):
                pred_dict = {
                    "boxes":  r.boxes.xyxy.cpu(),
                    "scores": r.boxes.conf.cpu(),
                    "labels": r.boxes.cls.int().cpu(),
                }
                target_dict = {
                    "boxes":  t["boxes"].cpu(),
                    "labels": t["labels"].int().cpu(),
                }
                all_preds.append(pred_dict)
                all_targets.append(target_dict)
            n_images += len(images)

    elapsed = time.time() - t0
    fps = n_images / elapsed

    metric.update(all_preds, all_targets)
    result = metric.compute()

    return {
        "mAP50":     result["map_50"].item(),
        "mAP50_95":  result["map"].item(),
        "APs":       result.get("map_small", torch.tensor(0.0)).item(),
        "recall":    result.get("mar_100", torch.tensor(0.0)).item(),
        "fps":       fps,
    }


def run_comparison(device: str = "cuda"):
    EXPERIMENTS = {
        # model_name: {supervised, rl}
        "YOLOv5s":  {
            "supervised": "checkpoints/yolov5s/weights/best.pt",
            "rl":         "rl_checkpoints/yolov5s_rl_best.pt",
        },
        "YOLOv8n":  {
            "supervised": "checkpoints/yolov8n/weights/best.pt",
            "rl":         "rl_checkpoints/yolov8n_rl_best.pt",
        },
        "YOLOv11n": {
            "supervised": "checkpoints/yolov11n/weights/best.pt",
            "rl":         "rl_checkpoints/yolov11n_rl_best.pt",
        },
        "DP-YOLO":  {
            "supervised": "checkpoints/dp_yolo/weights/best.pt",
            "rl":         "rl_checkpoints/dp_yolo_rl_best.pt",
        },
    }

    from dataloader import get_pest_dataloader
    val_loader = get_pest_dataloader("data/pest", split="val", batch_size=16)

    rows = []
    for model_name, paths in EXPERIMENTS.items():
        for stage, ckpt in paths.items():
            if not Path(ckpt).exists():
                print(f"Skip (not found): {ckpt}")
                continue
            print(f"Evaluating {model_name} [{stage}]...")
            metrics = evaluate_model(ckpt, val_loader, device)
            rows.append({"Model": model_name, "Stage": stage, **metrics})

    df = pd.DataFrame(rows)

    # Tính delta RL vs supervised
    df_pivot = df.pivot(index="Model", columns="Stage", values=["mAP50","mAP50_95","APs","recall"])
    df_pivot.columns = ["_".join(c) for c in df_pivot.columns]
    for metric in ["mAP50", "mAP50_95", "APs", "recall"]:
        sup_col = f"{metric}_supervised"
        rl_col  = f"{metric}_rl"
        if sup_col in df_pivot.columns and rl_col in df_pivot.columns:
            df_pivot[f"{metric}_delta"] = df_pivot[rl_col] - df_pivot[sup_col]

    # Lưu kết quả
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/tables/results_full.csv", index=False)
    df_pivot.to_csv("results/tables/results_comparison.csv")

    # In bảng markdown
    print("\n" + "="*70)
    print("KẾT QUẢ SO SÁNH")
    print("="*70)
    print(df.to_markdown(index=False, floatfmt=".4f"))

    return df


if __name__ == "__main__":
    run_comparison()
```

### 8.2. Template bảng kết quả

| Model | Stage | mAP@50 | mAP@50-95 | AP$_s$ | Recall | FPS |
|---|---|---|---|---|---|---|
| YOLOv5s | Supervised | — | — | — | — | — |
| YOLOv5s | **+RL** | — | — | — | — | — |
| YOLOv8n | Supervised | — | — | — | — | — |
| YOLOv8n | **+RL** | — | — | — | — | — |
| YOLOv11n | Supervised | — | — | — | — | — |
| YOLOv11n | **+RL** | — | — | — | — | — |
| **DP-YOLO** | Supervised | — | — | — | — | — |
| **DP-YOLO** | **+RL** | — | — | — | — | — |

> **Cột then chốt:** $AP_s$ và Recall – đây là 2 chỉ số phản ánh trực tiếp vấn đề vật thể nhỏ trên UAV.

---

## 9. THIẾT KẾ THÍ NGHIỆM

### 9.1. Ma trận thí nghiệm đầy đủ

| Exp | Model | RL | Freeze | Steps | LR RL | Mục tiêu kiểm chứng |
|---|---|---|---|---|---|---|
| E01 | YOLOv5s | ✗ | — | — | — | Baseline cổ điển |
| E02 | YOLOv5s | ✓ | ✗ | 50k | 1e-6 | RL boost YOLOv5s |
| E03 | YOLOv8n | ✗ | — | — | — | Anchor-free baseline |
| E04 | YOLOv8n | ✓ | ✗ | 50k | 1e-6 | RL boost YOLOv8n |
| E05 | YOLOv11n | ✗ | — | — | — | SOTA baseline |
| E06 | YOLOv11n | ✓ | ✗ | 50k | 1e-6 | RL boost YOLOv11n |
| E07 | DP-YOLO | ✗ | — | — | — | Kiến trúc tùy chỉnh |
| E08 | **DP-YOLO** | ✓ | ✗ | 50k | 1e-6 | **DP-YOLO + RL (main)** |
| E09 | DP-YOLO | ✓ | ✓ | 50k | 1e-5 | RL chỉ fine-tune head/neck |
| E10 | DP-YOLO | ✓ | ✗ | 100k | 1e-6 | Nhiều steps hơn |

### 9.2. Điều kiện đồng nhất (bắt buộc để so sánh công bằng)

- Cùng bộ dữ liệu (split ngẫu nhiên cố định, seed=42).
- Cùng augmentation trong supervised training.
- Cùng số epoch supervised (300 epochs).
- Cùng tập val và test để đánh giá.
- Cùng phần cứng khi đo FPS.
- Cùng conf_threshold=0.25 và iou_threshold=0.45 khi inference.

### 9.3. Điểm kiểm soát (sanity checks)

Trước khi báo cáo kết quả, kiểm tra:
1. **Reward curve tăng dần** theo bước RL (nếu không tăng → lr quá lớn hoặc bug reward).
2. **Supervised metrics không giảm** so với pre-RL checkpoint sau 5k bước đầu (nếu giảm → catastrophic forgetting, cần giảm lr hoặc freeze backbone).
3. **FPS không thay đổi** sau RL (chỉ là fine-tune tham số, không đổi kiến trúc).

---

## 10. PHÂN TÍCH KẾT QUẢ DỰ KIẾN

### 10.1. Kỳ vọng theo từng model

| Model | Dự kiến Supervised | Dự kiến RL boost | Lý do |
|---|---|---|---|
| YOLOv5s | mAP50 ~85%, AP$_s$ thấp | AP$_s$ tăng 1-2%, Recall tăng ~3% | Anchor-based, RL bổ sung recall lớp nhỏ |
| YOLOv8n | mAP50 ~87%, AP$_s$ khá hơn | AP$_s$ tăng 1-1.5% | Anchor-free đã tốt hơn, RL boost ít hơn |
| YOLOv11n | mAP50 ~88%, AP$_s$ tốt nhất trong baseline | Boost nhỏ (~0.5-1%) | Pipeline đã hiện đại, dư địa RL nhỏ hơn |
| **DP-YOLO** | mAP50 ~89%, AP$_s$ cao nhất | **Boost rõ nhất (~2-3%)** | Kiến trúc tối ưu cho vật thể nhỏ + RL tập trung recall |

> Kỳ vọng này dựa trên bài học từ cv-rl: *"CE-pretrain + RL fine-tune"* cho kết quả tốt nhất khi model đã có nền tảng vững.

### 10.2. Phân tích từng lớp khó

Sau khi có kết quả, phân tích per-class AP để xác định:
- Lớp nào được cải thiện nhiều nhất bởi RL (thường là lớp hiếm, nhỏ).
- Lớp nào bị ảnh hưởng tiêu cực (nếu có → điều chỉnh reward).

---

## 11. CÁC RỦI RO VÀ CÁCH XỬ LÝ

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Catastrophic forgetting | mAP50 giảm mạnh sau RL | Giảm LR (1e-7), freeze backbone, giảm steps |
| High variance (bwconrad) | Reward dao động mạnh, không hội tụ | Tăng alpha EMA (0.99→0.999), tăng batch size |
| Reward collapse | Reward tăng nhanh rồi plateau ở mức thấp | Điều chỉnh reward scale, thêm entropy reg |
| Mode collapse | Mô hình predict toàn bộ diện tích là 1 class | Thêm diversity penalty vào reward |
| mAP reward không ổn định (yanivnik) | Recall tăng nhưng precision giảm | Dùng composite reward thay mAP thuần túy |
| Gradient = 0 (không có predictions) | loss = 0 mọi bước | Hạ conf_threshold khi lấy predictions cho RL |

---

## 12. TÀI LIỆU THAM KHẢO

1. **Paper gốc RL cho CV:** Zhai et al., *"Tuning Computer Vision Models with Task Rewards"*, Google DeepMind, arXiv:2302.08242, 2023.
2. **REINFORCE:** Williams, R.J., *"Simple statistical gradient-following algorithms for connectionist reinforcement learning"*, Machine Learning, 1992.
3. **Self-Critical Sequence Training (baseline kỹ thuật):** Rennie et al., *"Self-Critical Sequence Training for Image Captioning"*, CVPR 2017.
4. **cv-rl implementation:** bwconrad/cv-rl – F1 score reward cho binary segmentation.
5. **tuning_cv_models_with_rl_torch:** yanivnik – Recall reward cho DETR object detection.
6. **DP-YOLO gốc:** Wang et al., *"DP-YOLO: Effective Improvement Based on YOLO Detector"*, Applied Sciences (MDPI), 2023.
7. **DP-YOLO Rail Fastener:** Chen et al., *"DP-YOLO: A Lightweight Real-Time Detection Algorithm for Rail Fastener Defects"*, Sensors (MDPI), 2025.
