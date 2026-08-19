# Hướng Dẫn RL Fine-tuning cho Object Detection (YOLO)
## Đối chiếu từ LLM (DPO / GRPO / RLHF) sang Detection

> **Mục đích:** Đọc xong guide này, bạn sẽ hiểu TẠI SAO và LÀM THẾ NÀO để áp dụng
> Reinforcement Learning fine-tune mô hình YOLO, bằng cách liên hệ sang RLHF/DPO/GRPO
> mà bạn đã quen trong LLM.

---

## Phần 1 – Bức Tranh Toàn Cảnh: Tại sao cần RL?

### 1.1. Vấn đề chung cho cả LLM và Detection

Cả LLM lẫn YOLO đều gặp **cùng một vấn đề gốc rễ**:

| Khía cạnh | LLM | YOLO Detection |
|---|---|---|
| **Loss khi train** | Cross-entropy (next-token prediction) | CIoU + BCE (box regression + classification) |
| **Metric thực tế** | Win-rate, helpfulness, safety | mAP@50, Recall, AP_small |
| **Khoảng cách** | Model tốt ở perplexity nhưng chưa chắc helpful | Model loss thấp nhưng recall vật thể nhỏ vẫn tệ |

> **Nguyên tắc vàng:** RL giúp tối ưu trực tiếp **metric thực tế** thay vì proxy loss.
> Đây là lý do InstructGPT vượt trội GPT-3, và lý do RL fine-tune YOLO cải thiện recall UAV.

### 1.2. Pipeline 3 giai đoạn – Cùng mẫu hình ở cả 2 domain

```
LLM Pipeline                          YOLO Pipeline (bài toán này)
═══════════════                        ══════════════════════════════

① Pre-training                        ① Supervised Training
   Internet text (raw)                   YOLOv5/v8/v11 + CIoU + BCE
   → base model (biết ngôn ngữ)          → best.pt (biết detect)

② SFT / Instruction Tuning            (không cần bước riêng)
   Labeled (prompt, response) pairs      supervised training = SFT

③ RLHF / DPO / GRPO                   ② RL Fine-tuning (REINFORCE)
   Human preference optimization         reward = Recall + Small-bonus
   → aligned, safe, helpful model        → cải thiện recall UAV

Evaluation                             ③ Evaluation & Comparison
   Benchmark, human rating               mAP50 | mAP50-95 | APs | FPS
                                          → supervised-only vs RL
```

**Điểm khác biệt quan trọng:** YOLO không cần bước SFT riêng vì supervised training
(bước ①) đã tương đương SFT. RL được áp dụng thẳng lên checkpoint supervised,
giống DeepSeek R1-Zero (GRPO ngay từ base model, không qua SFT).

---

## Phần 2 – Đối Chiếu Khái Niệm 1-1

### 2.1. Policy, State, Action, Reward

```
╔════════════════╦══════════════════════════════╦══════════════════════════════════╗
║ Khái niệm RL   ║ LLM (RLHF/GRPO)              ║ YOLO Detection                   ║
╠════════════════╬══════════════════════════════╬══════════════════════════════════╣
║ Policy π_θ     ║ Language model với param θ   ║ YOLO model với param θ           ║
║ State s        ║ Prompt / conversation so far ║ Ảnh đầu vào từ UAV               ║
║ Action a       ║ Token được chọn (discrete)   ║ Tập bbox predictions (continuous) ║
║ Reward R       ║ Human preference score        ║ Recall / mAP / composite score   ║
║ log π(a|s)     ║ log P(token | context)       ║ log(avg_confidence) [xấp xỉ]     ║
║ Baseline b     ║ Value model / group mean     ║ EMA của reward (rolling mean)     ║
╚════════════════╩══════════════════════════════╩══════════════════════════════════╝
```

### 2.2. Tại sao log π(a|s) của YOLO khác LLM?

**LLM:** Mô hình autoregressive → log-probability đầy đủ tính được chính xác:
```python
# LLM: cộng log-prob của từng token
log_prob = sum(log_softmax(logits)[token_ids])
```

**YOLO:** Không phải autoregressive, không có chain of tokens →
phải **xấp xỉ** bằng confidence score:
```python
# YOLO: dùng avg_confidence làm proxy log π(a|s)
avg_conf = predictions['scores'].mean().clamp(1e-20, 1.0)
log_prob = torch.log(avg_conf)
```

> **Tại sao xấp xỉ này hợp lý?**
> Confidence cao → model "chắc chắn" về predictions → tương đương policy "committed".
> Gradient của log(conf) đẩy confidence đúng hướng với reward.
> Đây là heuristic được kiểm chứng thực nghiệm bởi yanivnik (Google DeepMind 2023).

---

## Phần 3 – So Sánh Các Thuật Toán

### 3.1. RLHF (PPO) → REINFORCE cho YOLO

**RLHF/PPO trong LLM:**
- Cần reward model riêng (train từ human preference data)
- Cần value model (critic) song song với policy
- PPO clip giữ update ổn định
- KL divergence với reference model

**REINFORCE trong YOLO:**
- Không có reward model (reward = task metric tính được trực tiếp ✓)
- Không có value model (thay bằng EMA baseline ✓)
- Không cần clip (lr rất nhỏ 1e-6 giữ ổn định ✓)
- Không cần KL constraint tường minh (lr nhỏ + gradient clipping đủ ✓)

```
RLHF/PPO (LLM)          REINFORCE (YOLO)
══════════════           ════════════════
Policy model        →    YOLO model
Reward model        →    compute_reward() [rule-based, không cần train]
Value model         →    EMA baseline [đơn giản hơn nhiều]
PPO clip            →    grad_clip(max_norm=1.0)
KL divergence       →    lr=1e-6 (implicit constraint)
```

> **Giống DeepSeek R1-Zero:** YOLO cũng dùng rule-based reward thay vì reward model.
> Recall@IoU=0.5 là "verified reward" – đúng/sai rõ ràng như đáp án toán.

### 3.2. GRPO → Adaptation cho YOLO

**GRPO trong LLM (DeepSeek):**
```
Với mỗi prompt:
  1. Sinh G câu trả lời {a₁, a₂, ..., aG}
  2. Tính reward {r₁, r₂, ..., rG}
  3. Advantage_i = (r_i - mean({r})) / std({r})
  4. Update: -mean(log_π × advantage)
```
→ Loại bỏ value model, dùng group mean làm baseline.

**GRPO-style cho YOLO (augmentation = "sampling"):**
```python
for images, targets in train_loader:
    all_log_probs, all_rewards = [], []
    for g in range(G):  # G=4 augmented views
        aug_images = augment(images).to(device)
        preds = model.forward_with_grad(aug_images, ...)
        with torch.no_grad():
            rewards = compute_reward(preds, targets)
        all_log_probs.append(compute_log_prob(preds))
        all_rewards.append(rewards)

    log_probs_mat = torch.stack(all_log_probs)  # (G, B)
    rewards_mat   = torch.stack(all_rewards)    # (G, B)

    # Group relative advantage – không cần warmup EMA
    mean_r = rewards_mat.mean(dim=0, keepdim=True)
    std_r  = rewards_mat.std(dim=0, keepdim=True) + 1e-8
    advantage_mat = (rewards_mat - mean_r) / std_r

    loss = -torch.mean(log_probs_mat * advantage_mat.detach())
```

> **Lợi ích:** Advantage tự normalize theo nhóm → training ổn định ngay từ bước đầu.
> Nhược điểm: tốn G× forward passes (G=4 → 4× chậm hơn REINFORCE đơn giản).

### 3.3. DPO → Adaptation cho Detection

**DPO trong LLM:**
```
Dữ liệu: (prompt, chosen_response, rejected_response)
Loss: -log σ(β × [log π(chosen) - log π(rejected)] - [log πref(chosen) - log πref(rejected)])
```
→ Tăng xác suất chosen, giảm rejected, không cần reward model.

**DPO-style cho YOLO – tại sao KHÓ áp dụng:**
1. Không có "chosen/rejected pair" tự nhiên như NLP
2. YOLO output là continuous (boxes, scores) – không phải discrete token distribution
3. Reference model πref cần là checkpoint supervised (phải lưu và load song song)
4. Không có cách tự nhiên tính log π(detection | image) cho continuous output

**Kết luận:** Với YOLO, **REINFORCE (hoặc GRPO-style) thực tế hơn DPO**.
DPO chỉ hữu ích nếu bạn có teacher/ensemble model để tạo "chosen" predictions
và student model để tạo "rejected" predictions.

---

## Phần 4 – Thiết Kế Reward Function

Đây là điểm **khác biệt lớn nhất** so với LLM: reward của YOLO hoàn toàn **rule-based**
(không cần human preference), tương tự DeepSeek dùng verified math answers.

### 4.1. Reward Components

```
R_total = α × R_recall + (1-α) × R_small    [α = 0.6]

R_recall (tương tự "accuracy reward" trong GRPO):
  = mean_per_class [ (matched_GT - 0.3 × duplicate_preds) / total_GT ]
  - matched_GT:      số GT box được match với ít nhất 1 pred (IoU ≥ 0.5)
  - duplicate_preds: số pred thừa match cùng 1 GT box
  - penalty 0.3 < 1: ưu tiên tăng recall hơn trừng phạt duplicate

R_small (bonus đặc thù UAV – tương tự "format reward" trong GRPO):
  = matched_small_objects / (total_small_objects + ε)
  với small = area < 32² pixels
```

### 4.2. So sánh với GRPO Reward Design

| | GRPO (LLM – DeepSeek) | REINFORCE (YOLO – bài này) |
|---|---|---|
| **Accuracy reward** | Kiểm tra đáp án toán đúng/sai | Recall@IoU≥0.5 (match GT box) |
| **Format reward** | Kiểm tra `<think>...</think>` tags | Small-object recall bonus |
| **Reward hacking** | Không thể fake verified answers | Không thể fake IoU match |
| **Human needed?** | ❌ Không | ❌ Không |
| **Tính được?** | ✅ Deterministic | ✅ Deterministic |

### 4.3. Code: Tính Reward

```python
def compute_reward(preds, targets, iou_threshold=0.5, small_thresh=32,
                   alpha=0.6, duplicate_penalty=0.3):
    """
    Rule-based reward cho object detection.
    Không cần reward model – tương tự DeepSeek verified reward.
    """
    batch_rewards = []
    for pred, tgt in zip(preds, targets):
        gt_boxes   = tgt['boxes']    # (N, 4) xyxy absolute
        pred_boxes = pred['boxes']   # (M, 4) xyxy absolute

        if gt_boxes.numel() == 0:
            batch_rewards.append(0.0)
            continue

        # ── Recall reward ──────────────────────────────────────────────
        iou_mat = box_iou(gt_boxes.float(), pred_boxes.float())  # (N, M)
        matched_gt    = (iou_mat.max(dim=1).values >= iou_threshold).sum()
        duplicate_pred = max(0, iou_mat.shape[1] - iou_mat.shape[0])
        r_recall = ((matched_gt - duplicate_penalty * duplicate_pred) / len(gt_boxes)).clamp(0, 1).item()

        # ── Small-object bonus ──────────────────────────────────────────
        areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        small_mask = areas < (small_thresh ** 2)
        n_small = small_mask.sum().item()
        if n_small > 0:
            small_matched = (iou_mat[small_mask].max(dim=1).values >= iou_threshold).sum()
            r_small = small_matched.item() / n_small
        else:
            r_small = r_recall  # fallback khi không có vật thể nhỏ

        r_total = alpha * r_recall + (1 - alpha) * r_small
        batch_rewards.append(r_total)

    return torch.tensor(batch_rewards)
```

---

## Phần 5 – Vòng Lặp RL Training

### 5.1. REINFORCE với EMA Baseline (recommended)

```python
class EMABaseline:
    """
    Trong GRPO: baseline = mean(reward trong group)
    Ở đây: baseline = EMA theo thời gian – rẻ hơn, không cần G forward passes
    """
    def __init__(self, alpha=0.99):
        self.alpha = alpha
        self.value = None

    def advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        b = self.update(rewards.mean().item())
        return rewards - b

    def update(self, r):
        self.value = r if self.value is None else self.alpha * self.value + (1-self.alpha) * r
        return self.value


def rl_finetune(model, train_loader, val_loader, cfg, device='cuda'):
    baseline  = EMABaseline(alpha=cfg['ema_alpha'])   # 0.99
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])  # 1e-6

    for step, (images, targets) in enumerate(train_loader):
        images = images.to(device)

        # 1. Forward – GIỮ GRADIENT qua confidence scores
        preds = model.forward_with_grad(images,
                    conf_thres=cfg['conf_thres'],   # 0.20 thấp hơn eval
                    iou_thres=cfg['iou_thres'])     # 0.45

        # 2. Reward (NO GRAD – rule-based, không khả vi)
        with torch.no_grad():
            rewards = compute_reward(preds, targets).to(device)

        # 3. Advantage = R - baseline (tương tự GRPO group advantage)
        advantage = baseline.advantage(rewards)

        # 4. Log-prob proxy (xấp xỉ log π(a|s))
        log_probs = compute_log_prob(preds)  # log(avg_confidence)

        # 5. REINFORCE loss: L = -E[log π(a|s) × advantage]
        loss = -torch.mean(log_probs * advantage.detach())

        # 6. Update với gradient clipping (thay PPO clip)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 7. Logging & Checkpoint (lưu theo rolling average, không instant)
        if step % cfg['log_interval'] == 0:
            avg_r = rolling_mean(rewards, window=200)
            if avg_r > best_avg_reward:
                save_checkpoint(model, avg_r, step)
```

### 5.2. GRPO-style (nâng cao – ổn định hơn từ bước đầu)

```python
def rl_finetune_grpo_style(model, train_loader, cfg, G=4, device='cuda'):
    for images, targets in train_loader:
        all_log_probs, all_rewards = [], []

        for g in range(G):  # G augmented views
            aug_images = random_augment(images).to(device)
            preds = model.forward_with_grad(aug_images, ...)
            with torch.no_grad():
                all_rewards.append(compute_reward(preds, targets))
            all_log_probs.append(compute_log_prob(preds))

        rewards_mat   = torch.stack(all_rewards)    # (G, B)
        log_probs_mat = torch.stack(all_log_probs)  # (G, B)

        # Group relative advantage (GRPO core idea)
        mean_r = rewards_mat.mean(dim=0, keepdim=True)
        std_r  = rewards_mat.std(dim=0, keepdim=True) + 1e-8
        advantage_mat = (rewards_mat - mean_r) / std_r

        loss = -torch.mean(log_probs_mat * advantage_mat.detach())
        # ... backward, clip, step
```

---

## Phần 6 – Các Quyết Định Thiết Kế Quan Trọng

### 6.1. Freeze backbone hay không?

| | LLM | YOLO |
|---|---|---|
| **Tương đương** | LoRA (chỉ update adapter layers) | Freeze layers 0-9 (backbone) |
| **Lý do freeze** | Tránh catastrophic forgetting | Backbone đã học feature tốt |
| **Khi nào dùng** | Model nhỏ, fine-tune specific task | Khi mAP drop > 5% sau RL warmup |
| **Config** | `lora_r=8, target_modules=[...]` | `freeze_backbone: true` |

```python
def freeze_backbone(model, model_name):
    for name, param in model.named_parameters():
        if model_name in ('yolov5s', 'dp_yolo'):
            if any(f'model.{i}.' in name for i in range(10)):  # layers 0-9
                param.requires_grad = False
```

### 6.2. Learning rate: tại sao 1e-6?

**LLM (DPO):** Tham số β kiểm soát mức thay đổi từ reference model.
**YOLO (REINFORCE):** lr = 1e-6 đóng vai trò **implicit KL constraint**:

```
lr nhỏ → Δweight nhỏ mỗi step → π_new ≈ π_old
       → tương đương KL(π_new || π_ref) nhỏ
       
lr > 1e-5 → catastrophic forgetting → mAP drop đột ngột
```

Từ thực nghiệm: lr ∈ [5e-7, 2e-6] là vùng ổn định nhất.

### 6.3. Checkpoint: rolling average, không phải instant reward

```
Vấn đề: Instant reward dao động mạnh do stochastic batch sampling
         (giống loss spike trong LLM training)

Giải pháp: Rolling average 200 steps (tương tự EMA loss monitoring)

reward_history = deque(maxlen=200)
reward_history.append(current_step_reward)
avg_reward = mean(reward_history)

if avg_reward > best_avg_reward:  # Save khi xu hướng dài hạn tốt
    best_avg_reward = avg_reward
    save_checkpoint(model, avg_reward, step)
```

### 6.4. Confidence threshold trong training loop

```
Eval thông thường: conf_thres = 0.25 (chỉ lấy predictions chắc chắn)
RL training loop:  conf_thres = 0.20 (thấp hơn để có nhiều preds hơn)

Lý do: RL cần gradient chảy qua log(conf) của NHIỀU predictions
        để signal reward reach được nhiều hơn.
        Nếu conf_thres quá cao → ít predictions → reward signal yếu → học chậm.
```

---

## Phần 7 – Hyperparameters Reference

```yaml
# hyp.rl.yaml
# ── Optimizer ─────────────────────────────────────────────
lr: 1.0e-6          # Rất nhỏ – implicit KL constraint (so với DPO beta)
optimizer: adam
grad_clip: 1.0      # Tương đương PPO clip (nhưng đơn giản hơn)

# ── Schedule ──────────────────────────────────────────────
steps: 50000        # Tổng số bước RL
log_interval: 100
save_interval: 5000
eval_interval: 5000

# ── Reward (rule-based – không cần reward model) ──────────
reward_type: composite    # recall | small | composite
reward_alpha: 0.6         # α: trọng số recall vs small-object
iou_threshold: 0.5        # IoU để tính match GT-pred
duplicate_penalty: 0.3    # Phạt prediction thừa (nhẹ, ưu tiên recall)
small_thresh: 32          # Threshold vật thể nhỏ (pixels)

# ── Baseline (tương tự GRPO group mean, rẻ hơn) ──────────
ema_alpha: 0.99     # EMA decay

# ── Architecture ──────────────────────────────────────────
freeze_backbone: false    # true nếu gặp catastrophic forgetting

# ── Inference trong training loop ─────────────────────────
conf_thres: 0.20    # Thấp hơn eval để nhiều predictions hơn
iou_thres: 0.45
batch_size: 16
```

---

## Phần 8 – Tóm Tắt So Sánh Cuối

```
═══════════════════════════════════════════════════════════════════════════
                    RLHF/PPO    DPO        GRPO       REINFORCE(YOLO)
═══════════════════════════════════════════════════════════════════════════
Reward model        ✅ Cần      ❌          ❌          ❌ (rule-based)
Value model         ✅ Cần      ❌          ❌          ❌ (EMA baseline)
Human labels        ✅ Cần      ✅ Cần      ❌          ❌ (IoU computed)
Reference model     ✅          ✅           ✅           ❌ (lr thay thế)
Group sampling      ❌          ❌           ✅ G views   Optional
KL constraint       ✅ Explicit  ✅ beta     ✅ Penalty   ❌ Implicit (lr)
Applicable to YOLO  ⚠️ Phức tạp ⚠️ Khó       ✅ Optional  ✅ Recommended
Training cost       💰💰💰      💰           💰💰         💰 (rẻ nhất)
═══════════════════════════════════════════════════════════════════════════
```

### Kết luận

1. **REINFORCE với EMA baseline** là lựa chọn tối ưu cho YOLO RL fine-tuning:
   - Đơn giản như DPO (không cần reward/value model)
   - Ổn định như GRPO (EMA baseline ≈ group mean)
   - Không cần human labels (rule-based reward như DeepSeek verified)

2. **GRPO-style** (G augmented views) cải thiện ổn định nhưng tốn G× tài nguyên.
   Dùng khi training không ổn định hoặc reward variance cao.

3. **DPO cho YOLO** về lý thuyết khả thi nhưng không tự nhiên: YOLO output là
   continuous, không có "chosen/rejected pair" tự nhiên.

4. Pattern chung: **Supervised pretrain → RL fine-tune** luôn tốt hơn train RL từ đầu,
   cả LLM lẫn YOLO. (REINFORCE từ đầu → F1 = 0.41, từ CE checkpoint → kết quả tốt: bwconrad/cv-rl)

---

## Phần 9 – Checklist Thực Hành

```
□ 1. Train supervised đầy đủ (≥ 100 epochs, tốt nhất 300 epochs)
□ 2. Lưu best.pt supervised – đây là "SFT model" (điểm xuất phát RL)
□ 3. Implement reward function rule-based: Recall + Small-object bonus
□ 4. Bắt đầu với lr = 1e-6, freeze_backbone = false
□ 5. Monitor rolling average reward 200 steps (không nhìn instant)
□ 6. Eval mAP50 định kỳ mỗi 5000 steps (torchmetrics.MeanAveragePrecision)
□ 7. Nếu mAP drop > 5% so với supervised: thử freeze backbone
□ 8. Nếu training không ổn định: giảm lr xuống 5e-7 hoặc thử GRPO-style (G=4)
□ 9. So sánh supervised-only vs RL-finetuned trên test set
```

---

*Tài liệu tổng hợp từ: DPO/GRPO/RLHF guide (kb2_preference_optimization/docs/DPO, GRPO, DAPO guide.md),
cv-rl (bwconrad, arXiv:2302.08242), tuning_cv_models_with_rl_torch (yanivnik/Google DeepMind),
DeepSeek-R1 paper, DeepSeek-Math paper.*
