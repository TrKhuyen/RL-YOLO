# RL Fine-tuning cho YOLO: Guide Tổng Hợp
### Từ DPO / GRPO / DAPO / KTO → Detection

> **Phạm vi:** Guide này phân tích từng hướng RL fine-tuning từ LLM, đánh giá khả năng
> áp dụng vào YOLO, rồi tổng hợp thành 3 cấp độ implementation thực tế.
>
> **Nguồn:** DPO/GRPO/DAPO guide (sol_2/docs) + REINFORCE pipeline (sol_1) +
> DeepSeek-R1/Math papers + yanivnik/Google DeepMind 2023.

---

## Phần 1 – Bối Cảnh: Tại Sao Cần RL?

### 1.1. Khoảng cách Loss ↔ Metric

YOLO được train với proxy loss, nhưng đánh giá bằng task metric khác nhau:

| Giai đoạn | Tối ưu | Metric thực tế |
|---|---|---|
| Supervised training | CIoU + BCE loss | — |
| Evaluation | — | mAP@50, Recall, AP_small |

Với bài toán sâu bệnh UAV, khoảng cách này đặc biệt lớn vì vật thể nhỏ
(sâu non, trứng) tạo gradient yếu từ CIoU, mất cân bằng lớp khiến loss
bị chi phối bởi lớp dễ.

### 1.2. Pipeline 3 Giai Đoạn (giống LLM)

```
  LLM                              YOLO
  ---                              ----
① Pre-training (internet text)  ←→  ① Supervised Training (CIoU+BCE)
② SFT (instruction following)   ←→  [không cần - supervised = SFT]
③ RL alignment (RLHF/DPO/GRPO)  ←→  ② RL Fine-tuning (Recall reward)
```

Bài học từ cv-rl: REINFORCE từ random init → F1 = 0.41 (tệ).
Fine-tune từ supervised checkpoint → tốt hơn đáng kể.
**Luôn warm-start RL từ supervised checkpoint.**

---

## Phần 2 – Phân Tích Từng Hướng LLM

### 2.1. RLHF + PPO → Không dùng

Cần reward model (human preference) + value model (tốn 2× VRAM).
Chỉ giữ lại ý tưởng PPO clipping cho Level 3.

### 2.2. DPO → Khả thi có điều kiện (DetPO)

DPO dùng (prompt, chosen, rejected) pairs. Với YOLO:

```python
def detpo_loss(model, ref_model, images, targets, beta=0.1):
    """
    chosen  = predictions của pass augment nhẹ (recall cao hơn)
    rejected = predictions của pass augment mạnh (recall thấp hơn)
    Không cần reward model. Cần lưu supervised checkpoint làm ref_model.
    """
    preds_light = model.forward_with_grad(augment_light(images))
    preds_heavy = model.forward_with_grad(augment_strong(images))

    with torch.no_grad():
        r_light = compute_recall(preds_light, targets)
        r_heavy = compute_recall(preds_heavy, targets)

    chosen   = [preds_light[i] if r_light[i]>=r_heavy[i] else preds_heavy[i]
                for i in range(len(images))]
    rejected = [preds_heavy[i] if r_light[i]>=r_heavy[i] else preds_light[i]
                for i in range(len(images))]

    log_chosen   = compute_log_prob(chosen)
    log_rejected = compute_log_prob(rejected)

    with torch.no_grad():
        log_ref_c = compute_log_prob(ref_model.forward(augment_light(images)))
        log_ref_r = compute_log_prob(ref_model.forward(augment_strong(images)))

    ratio = beta * (log_chosen - log_rejected) - (log_ref_c - log_ref_r)
    return -torch.mean(torch.log(torch.sigmoid(ratio)))
```

Nhược: 3× chậm hơn REINFORCE (2 forward + ref), chosen/rejected artificial.
Dùng như variant thử nghiệm, không phải default.

### 2.3. KTO → Phù hợp tự nhiên (không cần pairs)

KTO chỉ cần binary label good/bad, không cần paired comparison:

```python
def kto_loss(model, images, targets, beta=0.1, recall_threshold=0.6):
    """
    Không cần pairs. Không cần teacher model.
    Loss aversion: undesirable weight > desirable (Prospect Theory).
    Phù hợp: bỏ sót sâu bệnh (undesirable) bị phạt nặng hơn.
    """
    preds    = model.forward_with_grad(images)
    recalls  = compute_recall(preds, targets)
    log_probs = compute_log_prob(preds)

    good = recalls >= recall_threshold
    bad  = recalls <  recall_threshold

    loss_good = -0.5 * torch.mean(log_probs[good]  * torch.sigmoid(recalls[good]  * beta))
    loss_bad  =  1.0 * torch.mean(log_probs[bad]   * torch.sigmoid(-recalls[bad] * beta))

    if good.any() and bad.any():
        return loss_good + loss_bad
    return loss_good if good.any() else loss_bad
```

Chưa có thực nghiệm cho detection. Thử nghiệm sau khi GRPO ổn.

### 2.4. GRPO → Phù hợp nhất (Group Augmentation)

Thay "sinh G text responses" bằng "sinh G augmented views":

```
GRPO LLM:                   GRPO-YOLO:
1 prompt                    1 ảnh gốc
sinh G responses    →       sinh G augmented views
score từng response →       recall từng view
Adv = (r - mean)/std        Adv = (r - mean)/std
loss = -log_pi × Adv        loss = -log_conf × Adv
```

Điểm mấu chốt từ DeepSeek:
- Rule-based reward (recall = verified match, không thể fake)
- Group relative baseline tự normalize, không cần EMA warmup
- Loại bỏ value model hoàn toàn

### 2.5. DAPO → 3 cải tiến cho Level 3

a) **Anchor-level loss** (DAPO "token-level"):
   Thay log(avg_conf_image) bằng mean_anchor(log(conf_anchor))
   → Gradient dày đặc hơn, mỗi anchor nhận signal

b) **Clip-higher** (asymmetric clipping):
   eps_low=0.1, eps_high=0.3 → cho phép policy cải thiện nhiều hơn
   khi reward tốt bất ngờ

c) **Dynamic batch filtering**:
   Bỏ qua batch khi tất cả views có reward bằng nhau (std < 0.01)
   → Không waste compute trên batch không học được

---

## Phần 3 – Tổng Hợp Thiết Kế

```
╔═══════════════╦═════════╦══════════════╦══════════════════════════════╗
║ Hướng LLM     ║ Dùng?   ║ Adapt thành  ║ Lý do                        ║
╠═══════════════╬═════════╬══════════════╬══════════════════════════════╣
║ RLHF/PPO      ║ Không   ║ —            ║ Cần reward+value model       ║
║ DPO           ║ Optional ║ DetPO        ║ Cần teacher, 3× chậm hơn    ║
║ KTO           ║ Exp     ║ Binary KTO   ║ Chưa kiểm chứng detection    ║
║ GRPO          ║ YES     ║ Group-Aug    ║ Augmentation = group sampling ║
║ DAPO clip     ║ YES     ║ Clip-higher  ║ Không bỏ qua signal tốt     ║
║ DAPO token    ║ YES     ║ Anchor-level ║ Gradient dày đặc hơn         ║
║ DAPO dynamic  ║ YES     ║ Batch filter ║ Bỏ qua batch vô ích          ║
╚═══════════════╩═════════╩══════════════╩══════════════════════════════╝

Level 1: REINFORCE + EMA Baseline    (1× forward, đơn giản nhất)
Level 2: GRPO-style Group Aug        (G× forward, stable baseline)
Level 3: GRPO + DAPO improvements   (G× forward + anchor + clip-higher)
```

---

## Phần 4 – Reward Function (Rule-Based)

Reward hoàn toàn tự động, giống DeepSeek dùng verified math answers.

```python
# reward.py
import torch
import torchvision.ops as ops

def compute_recall_reward(preds, targets, iou_threshold=0.5, dup_penalty=0.3):
    """
    Tương tự "accuracy reward" của GRPO/DeepSeek.
    Công thức: mean_class[(matched_GT - 0.3*dup) / total_GT]
    dup_penalty=0.3: ưu tiên recall hơn phạt false positive.
    """
    rewards = torch.zeros(len(preds))
    for i, (pred, tgt) in enumerate(zip(preds, targets)):
        gt_boxes, gt_labels = tgt["boxes"], tgt["labels"]
        if len(gt_boxes) == 0:
            continue
        classes = gt_labels.unique()
        score = 0.0
        for cls in classes:
            gt_cls   = gt_boxes[gt_labels == cls]
            pred_cls = pred["boxes"][pred["labels"] == cls]
            if len(pred_cls) == 0:
                continue
            iou_mat = ops.box_iou(gt_cls.float(), pred_cls.float())
            matched = iou_mat > iou_threshold
            n_matched = torch.any(matched, dim=1).sum().float()
            n_dup     = (matched.sum(dim=1) - 1).clamp(0).sum().float()
            score    += (n_matched - dup_penalty * n_dup).item()
        rewards[i] = score / len(classes)
    return rewards.clamp(0, 1)

def compute_small_reward(preds, targets, small_thresh=32, iou_threshold=0.5):
    """
    Bonus reward cho vật thể nhỏ (area < 32^2 px). Đặc thù UAV.
    Tương tự "format reward" của GRPO.
    """
    rewards = torch.zeros(len(preds))
    for i, (pred, tgt) in enumerate(zip(preds, targets)):
        gt_boxes = tgt["boxes"]
        if len(gt_boxes) == 0 or len(pred["boxes"]) == 0:
            continue
        areas    = (gt_boxes[:,2]-gt_boxes[:,0]) * (gt_boxes[:,3]-gt_boxes[:,1])
        small_gt = gt_boxes[areas < small_thresh**2]
        if len(small_gt) == 0:
            continue
        iou_mat   = ops.box_iou(small_gt.float(), pred["boxes"].float())
        n_matched = torch.any(iou_mat > iou_threshold, dim=1).sum().float()
        rewards[i] = n_matched / (len(small_gt) + 1e-6)
    return rewards.clamp(0, 1)

def composite_reward(preds, targets, alpha=0.6):
    """alpha=0.6: recall chiếm 60%, small-object 40%."""
    r_recall = compute_recall_reward(preds, targets)
    r_small  = compute_small_reward(preds, targets)
    has_small = (r_small > 0).float()
    r_small   = has_small * r_small + (1 - has_small) * r_recall
    return alpha * r_recall + (1 - alpha) * r_small

def compute_log_prob(preds):
    """
    Xấp xỉ log pi(action|state) = log(avg_confidence).
    LLM: exact sum over tokens. YOLO: proxy qua confidence.
    """
    log_probs = []
    for pred in preds:
        if len(pred["scores"]) == 0:
            lp = torch.tensor(-20.0, requires_grad=True)
        else:
            avg_conf = pred["scores"].mean().clamp(1e-20, 1.0)
            lp = torch.log(avg_conf)
        log_probs.append(lp)
    return torch.stack(log_probs)
```

---

## Phần 5 – Level 1: REINFORCE + EMA Baseline

**Nguồn:** RLHF baseline concept + DeepSeek rule-based reward
**Resource:** 1× forward pass, đơn giản nhất

```python
class EMABaseline:
    """EMA baseline thay thế group mean của GRPO. Rẻ hơn nhưng cần warmup."""
    def __init__(self, alpha=0.99):
        self.alpha = alpha
        self.value = None

    def advantage(self, rewards):
        b = self.update(rewards.mean().item())
        return rewards - b

    def update(self, r):
        self.value = r if self.value is None else (
            self.alpha * self.value + (1 - self.alpha) * r)
        return self.value

def train_level1(model_wrapper, train_loader, cfg, device="cuda"):
    baseline  = EMABaseline(alpha=cfg["ema_alpha"])
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_wrapper.parameters()), lr=cfg["lr"])
    reward_hist = []
    data_iter   = iter(train_loader)

    for step in range(1, cfg["steps"] + 1):
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)
        images = images.to(device)

        preds = model_wrapper.forward_with_grad(
            images, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])

        with torch.no_grad():
            rewards = composite_reward(preds, targets, alpha=cfg["reward_alpha"]).to(device)

        advantage = baseline.advantage(rewards)
        log_probs = compute_log_prob(preds)
        loss      = -torch.mean(log_probs * advantage.detach())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
        optimizer.step()

        reward_hist.append(rewards.mean().item())
        if step % cfg["log_interval"] == 0:
            avg200 = sum(reward_hist[-200:]) / min(len(reward_hist), 200)
            print(f"[L1] {step:6d} | R={rewards.mean():.4f} avg200={avg200:.4f} "
                  f"loss={loss.item():.6f} base={baseline.value:.4f}")
```

---

## Phần 6 – Level 2: GRPO-style (Group Augmentation)

**Nguồn:** GRPO (DeepSeek) – group relative advantage, loại bỏ value model
**Resource:** G× forward pass (G=4 recommended)

```python
def augment_for_group(images, g_idx):
    """
    Augmentation khác nhau cho mỗi view trong group.
    Augment mạnh hơn khi g_idx cao → tạo diversity trong reward.
    """
    import torchvision.transforms.functional as TF
    import random
    aug = images.clone()
    if g_idx > 0:
        if random.random() > 0.5:
            aug = TF.hflip(aug)
        if g_idx > 1 and random.random() > 0.5:
            aug = TF.vflip(aug)
        if g_idx > 2:
            noise = torch.randn_like(aug) * 0.05
            aug   = (aug + noise).clamp(0, 1)
    return aug

def train_level2(model_wrapper, train_loader, cfg, device="cuda", G=4):
    """
    GRPO-style: group advantage tự normalize, không cần EMA warmup.
    Stable hơn Level 1 khi reward distribution thay đổi nhanh.
    """
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_wrapper.parameters()), lr=cfg["lr"])
    data_iter = iter(train_loader)

    for step in range(1, cfg["steps"] + 1):
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)
        images = images.to(device)

        all_log_probs, all_rewards = [], []
        for g in range(G):
            aug_images = augment_for_group(images, g).to(device)
            preds = model_wrapper.forward_with_grad(
                aug_images, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
            with torch.no_grad():
                all_rewards.append(
                    composite_reward(preds, targets, alpha=cfg["reward_alpha"]).to(device))
            all_log_probs.append(compute_log_prob(preds))

        log_probs_mat = torch.stack(all_log_probs)   # (G, B)
        rewards_mat   = torch.stack(all_rewards)     # (G, B)

        # Group relative advantage (GRPO core idea)
        mean_r       = rewards_mat.mean(dim=0, keepdim=True)
        std_r        = rewards_mat.std(dim=0,  keepdim=True) + 1e-8
        advantage_mat = (rewards_mat - mean_r) / std_r

        loss = -torch.mean(log_probs_mat * advantage_mat.detach())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
        optimizer.step()

        if step % cfg["log_interval"] == 0:
            print(f"[L2] {step:6d} | R={rewards_mat.mean():.4f} "
                  f"group_std={rewards_mat.std():.4f} loss={loss.item():.6f}")
```

---

## Phần 7 – Level 3: DAPO Improvements

**Nguồn:** DAPO paper – anchor-level loss, clip-higher, dynamic filtering

### 7.1. Clip-Higher (Asymmetric Clipping)

```python
def dapo_policy_loss(log_probs, log_probs_old, advantage,
                     eps_low=0.1, eps_high=0.3):
    """
    DAPO asymmetric clip:
    - Khi advantage > 0: cho phép policy tăng nhiều hơn (eps_high=0.3)
    - Khi advantage < 0: giới hạn policy giảm (eps_low=0.1)
    Tránh bỏ qua signal tốt khi model cải thiện đột biến.
    """
    ratio = torch.exp(log_probs - log_probs_old.detach())
    adv   = advantage.detach()

    clipped = torch.where(
        adv >= 0,
        ratio.clamp(1 - eps_low, 1 + eps_high),   # positive: allow more upside
        ratio.clamp(1 - eps_high, 1 + eps_low),   # negative: allow more downside
    )
    return -torch.mean(torch.min(ratio * adv, clipped * adv))
```

### 7.2. Dynamic Batch Filtering

```python
def filter_learnable(rewards_mat, min_std=0.01, min_mean=0.01, max_mean=0.99):
    """
    DAPO dynamic sampling: bỏ qua image nếu group reward không diverse.
    rewards_mat: (G, B)
    """
    r_std  = rewards_mat.std(dim=0)    # (B,)
    r_mean = rewards_mat.mean(dim=0)   # (B,)
    valid  = (r_std > min_std) & (r_mean > min_mean) & (r_mean < max_mean)
    return valid  # (B,) bool
```

### 7.3. Training Loop Level 3

```python
def train_level3(model_wrapper, train_loader, cfg, device="cuda", G=4):
    optimizer   = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_wrapper.parameters()), lr=cfg["lr"])
    data_iter   = iter(train_loader)
    skipped     = 0

    for step in range(1, cfg["steps"] + 1):
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)
        images = images.to(device)

        all_log_probs, all_rewards = [], []
        for g in range(G):
            aug_images = augment_for_group(images, g).to(device)
            preds = model_wrapper.forward_with_grad(
                aug_images, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
            with torch.no_grad():
                all_rewards.append(
                    composite_reward(preds, targets, alpha=cfg["reward_alpha"]).to(device))
            all_log_probs.append(compute_log_prob(preds))

        log_probs_mat = torch.stack(all_log_probs)
        rewards_mat   = torch.stack(all_rewards)

        # DAPO: Dynamic filtering
        valid = filter_learnable(
            rewards_mat,
            min_std=cfg["min_std"],
            max_mean=0.99
        )
        if valid.sum() == 0:
            skipped += 1
            continue

        log_probs_v   = log_probs_mat[:, valid]
        rewards_v     = rewards_mat[:, valid]

        mean_r        = rewards_v.mean(dim=0, keepdim=True)
        std_r         = rewards_v.std(dim=0,  keepdim=True) + 1e-8
        advantage_mat = (rewards_v - mean_r) / std_r

        # DAPO: Clip-higher (dùng G=0 view làm reference)
        log_probs_old = log_probs_v[0:1].detach().expand_as(log_probs_v)
        loss = dapo_policy_loss(
            log_probs_v, log_probs_old, advantage_mat,
            eps_low=cfg["eps_low"], eps_high=cfg["eps_high"]
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), 1.0)
        optimizer.step()

        if step % cfg["log_interval"] == 0:
            skip_rate = skipped / step
            print(f"[L3] {step:6d} | R={rewards_v.mean():.4f} "
                  f"skip={skip_rate:.1%} loss={loss.item():.6f}")
```

---

## Phần 8 – Adapters: Giữ Gradient Qua Confidence

### 8.1. YOLOv5 / DP-YOLO

```python
# adapters/yolov5_adapter.py
import torch, sys
sys.path.insert(0, "yolov5")
from models.common import DetectMultiBackend
from utils.general import non_max_suppression

class YOLOv5Adapter:
    def __init__(self, checkpoint, device="cuda"):
        self.device = device
        self.model  = DetectMultiBackend(checkpoint, device=device)
        self.model.train()

    def parameters(self):    return self.model.parameters()
    def named_parameters(self): return self.model.named_parameters()

    def forward_with_grad(self, images, conf_thres=0.20, iou_thres=0.45):
        raw_out = self.model(images)   # (B, anchors, 5+nc), HAS GRAD
        preds = []
        for b in range(images.shape[0]):
            obj_conf  = raw_out[0][b, :, 4]
            cls_score = raw_out[0][b, :, 5:].max(dim=-1).values
            scores_g  = obj_conf * cls_score     # GRADIENT HERE

            with torch.no_grad():
                det = non_max_suppression(
                    raw_out[0][b:b+1].detach(), conf_thres, iou_thres)[0]

            if det is not None and len(det):
                preds.append({"boxes":  det[:, :4],
                              "labels": det[:, 5].long(),
                              "scores": scores_g[:len(det)]})
            else:
                preds.append({"boxes":  torch.zeros((0,4), device=self.device),
                              "labels": torch.zeros(0, dtype=torch.long, device=self.device),
                              "scores": scores_g[:1]})
        return preds
```

### 8.2. YOLOv8 / YOLOv11 (Ultralytics)

```python
# adapters/ultralytics_adapter.py
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression as nms_v8

class UltralyticsAdapter:
    def __init__(self, checkpoint, device="cuda"):
        self.device = device
        self.model  = YOLO(checkpoint).model.to(device)
        self.model.train()

    def parameters(self):    return self.model.parameters()
    def named_parameters(self): return self.model.named_parameters()

    def forward_with_grad(self, images, conf_thres=0.20, iou_thres=0.45):
        raw = self.model(images)
        feat = raw[0] if isinstance(raw, (list, tuple)) else raw  # (B,4+nc,8400)
        scores_raw = feat[:, 4:, :].max(dim=1).values             # (B, 8400) HAS GRAD

        preds = []
        for b in range(images.shape[0]):
            with torch.no_grad():
                det = nms_v8(feat[b:b+1].detach().permute(0,2,1),
                             conf_thres=conf_thres, iou_thres=iou_thres)[0]
            if det is not None and len(det):
                preds.append({"boxes":  det[:,:4].detach(),
                              "labels": det[:,5].long().detach(),
                              "scores": scores_raw[b, :len(det)]})
            else:
                preds.append({"boxes":  torch.zeros((0,4), device=self.device),
                              "labels": torch.zeros(0, dtype=torch.long, device=self.device),
                              "scores": scores_raw[b, :1]})
        return preds
```

### 8.3. Freeze Backbone (khi mAP drop > 5%)

```python
def freeze_backbone(adapter, n=10):
    """Đóng băng n layer đầu. Tương đương LoRA trong LLM."""
    frozen = 0
    for name, param in adapter.named_parameters():
        for part in name.split("."):
            if part.isdigit() and int(part) < n:
                param.requires_grad = False
                frozen += 1
                break
    print(f"Frozen {frozen} params (layers 0-{n-1})")
```

---

## Phần 9 – Hyperparameters

```yaml
# configs/hyp.rl.yaml

level: 2              # 1=REINFORCE+EMA, 2=GRPO-style, 3=DAPO
grpo_G: 4             # Số augmented views (level 2, 3)

lr: 1.0e-6            # Implicit KL constraint. lr>1e-5: catastrophic forgetting
optimizer: adam
grad_clip: 1.0

steps: 50000
warmup_steps: 500
log_interval: 100
eval_interval: 5000

reward_alpha: 0.6     # recall weight vs small-object
iou_threshold: 0.5
dup_penalty: 0.3
small_thresh: 32

ema_alpha: 0.99       # Level 1 only

eps_low:  0.1         # DAPO clip lower bound
eps_high: 0.3         # DAPO clip upper bound (cao hơn = clip-higher)
min_std:  0.01        # DAPO dynamic filter

conf_thres: 0.20      # Thấp hơn eval (0.25): nhiều preds, signal mạnh hơn
iou_thres:  0.45
batch_size: 16

freeze_backbone: false
freeze_n_layers: 10
```

---

## Phần 10 – Evaluation

```python
# evaluate.py
import torch, time, pandas as pd
from pathlib import Path
from torchmetrics.detection import MeanAveragePrecision

def evaluate_checkpoint(ckpt_path, val_loader, device="cuda"):
    from ultralytics import YOLO
    model  = YOLO(ckpt_path).to(device)
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5], extended_summary=True)
    preds_all, targets_all = [], []
    t0, n = time.time(), 0

    with torch.no_grad():
        for images, targets in val_loader:
            results = model(images, verbose=False)
            for r, t in zip(results, targets):
                preds_all.append({
                    "boxes": r.boxes.xyxy.cpu(), "scores": r.boxes.conf.cpu(),
                    "labels": r.boxes.cls.int().cpu()})
                targets_all.append({
                    "boxes": t["boxes"].cpu(), "labels": t["labels"].int().cpu()})
            n += len(images)

    metric.update(preds_all, targets_all)
    res = metric.compute()
    return {"mAP50": res["map_50"].item(), "mAP50-95": res["map"].item(),
            "AP_small": res.get("map_small", torch.tensor(0.0)).item(),
            "Recall": res.get("mar_100", torch.tensor(0.0)).item(),
            "FPS": n / (time.time() - t0)}

def run_comparison():
    from dataloader import get_pest_dataloader
    val_loader = get_pest_dataloader("../data/V2", split="val", batch_size=16)
    EXPERIMENTS = {
        "DP-YOLO Supervised": "checkpoints/dp_yolo/weights/best.pt",
        "DP-YOLO RL L1":      "rl_checkpoints/dp_yolo_l1_best.pt",
        "DP-YOLO RL L2":      "rl_checkpoints/dp_yolo_l2_best.pt",
        "DP-YOLO RL L3":      "rl_checkpoints/dp_yolo_l3_best.pt",
    }
    rows = []
    for name, ckpt in EXPERIMENTS.items():
        if Path(ckpt).exists():
            rows.append({"Model": name, **evaluate_checkpoint(ckpt, val_loader)})
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/rl_comparison.csv", index=False)
    print(df.to_markdown(index=False, floatfmt=".4f"))
```

**Template kết quả:**

| Model | mAP@50 | mAP@50-95 | AP_small | Recall | FPS |
|---|---|---|---|---|---|
| Supervised | — | — | — | — | — |
| + RL L1 (REINFORCE+EMA) | — | — | **↑?** | **↑?** | = |
| + RL L2 (GRPO Group Aug) | — | — | **↑?** | **↑?** | = |
| + RL L3 (GRPO+DAPO) | — | — | **↑?** | **↑?** | = |

---

## Phần 11 – Xử Lý Sự Cố

| Vấn đề | Triệu chứng | Giải pháp |
|---|---|---|
| Catastrophic forgetting | mAP50 giảm > 5% | Giảm lr → 5e-7, freeze backbone |
| Reward plateau | Không tăng sau 1000 steps | Hạ conf_thres → 0.15 |
| L1 high variance | Reward dao động mạnh | Tăng ema_alpha → 0.999 |
| L2 group_std ≈ 0 | loss ≈ 0 mọi bước | Tăng augmentation diversity |
| L3 skip_rate > 50% | Quá nhiều batch bị skip | Giảm min_std → 0.005 |
| Gradient = 0 | weights không thay đổi | Kiểm tra scores.requires_grad=True trong adapter |
| Precision giảm | FP tăng nhiều | Tăng dup_penalty → 0.5 |
| Mode collapse | Chỉ predict 1 class | Thêm per-class normalization trong reward |

---

## Phần 12 – Checklist & Lệnh Chạy

### Checklist

```
CHUẨN BỊ
□ Supervised training xong (>= 100 epochs)
□ Lưu best.pt supervised
□ Verify: targets dict có "boxes" (xyxy abs) và "labels"
□ Verify: adapter.forward_with_grad() → scores.requires_grad=True

RL TRAINING
□ Level 1 (REINFORCE): reward tăng sau 1000 steps?
□ Level 2 (GRPO): group_std > 0.05 (đủ diversity)?
□ Level 3 (DAPO): skip_rate < 30%?
□ mAP không drop > 5% sau 5000 steps?
  → Có: giảm lr hoặc freeze backbone

EVALUATION
□ conf_thres=0.25 khi eval (giống supervised)
□ So sánh AP_small và Recall (metric chính RL)
□ FPS phải bằng nhau (kiến trúc không đổi)
```

### Lệnh Chạy

```bash
# Supervised training
cd sol_1 && python yolov5/train.py \
  --weights yolov5s.pt --data configs/pest.yaml \
  --epochs 300 --batch-size 32 --project checkpoints --name yolov5s

# RL Level 1
python train_rl.py --model dp_yolo --level 1 --steps 50000 --lr 1e-6

# RL Level 2 (GRPO-style, recommended)
python train_rl.py --model dp_yolo --level 2 --G 4 --steps 50000 --lr 1e-6

# RL Level 3 (DAPO improvements)
python train_rl.py --model dp_yolo --level 3 --G 4 --steps 50000 \
  --eps-low 0.1 --eps-high 0.3

# Evaluation
python evaluate.py
```

---

## Ma Trận Thí Nghiệm

| Exp | Model | Level | G | Steps | Mục tiêu |
|---|---|---|---|---|---|
| E01 | YOLOv5s | — | — | — | Baseline |
| E02 | YOLOv5s | L1 | — | 50k | REINFORCE cơ bản |
| E03 | YOLOv8n | — | — | — | Anchor-free baseline |
| E04 | YOLOv8n | L2 | 4 | 50k | GRPO-style |
| E05 | YOLOv11n | — | — | — | SOTA baseline |
| E06 | YOLOv11n | L2 | 4 | 50k | SOTA + GRPO |
| E07 | DP-YOLO | — | — | — | Custom baseline |
| **E08** | **DP-YOLO** | **L2** | **4** | **50k** | **Main experiment** |
| E09 | DP-YOLO | L3 | 4 | 50k | DAPO improvements |
| E10 | DP-YOLO | L2 | 4 | 100k | Longer training |

---

*Tài liệu tổng hợp từ: DPO/GRPO/DAPO guide (sol_2/docs), DeepSeek-R1/Math papers,
DAPO paper, yanivnik/Google DeepMind 2023, cv-rl (bwconrad), sol_1 REINFORCE pipeline.*
