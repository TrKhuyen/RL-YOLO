# Báo Cáo Kỹ Thuật: RL-YOLO Pest Detection

> **Cập nhật lần cuối:** 2026-06-24  
> **Trạng thái tổng quan:** ✅ Code hoàn thiện — chờ data để train  
> **Test suite:** ✅ 19/19 PASS (`uv run python tests/test_fixes.py`)

---

## 1. Tổng Quan Dự Án

### 1.1. Mục Tiêu

Xây dựng hệ thống phát hiện sâu bệnh lúa từ ảnh UAV (10 class) bằng cách kết hợp:

- **DP-YOLO** — kiến trúc cải tiến YOLOv5 dùng Deformable Convolution (DCNv2/v3), PTCSP, C3Ghost, multi-scale head P2+P3+P4 cho vật thể nhỏ
- **REINFORCE** — RL fine-tuning tối ưu trực tiếp Recall/mAP (metric không khả vi), thay cho loss cross-entropy thông thường

### 1.2. Motivation Học Thuật

| Hạn chế của approach cũ | Giải pháp |
|---|---|
| YOLOv5 vanilla kém với vật thể nhỏ (sâu non, trứng sâu) | DP-YOLO thêm P2 head (160×160), DCNv2 học biến dạng hình thể |
| CIoU loss không tối ưu corner alignment | W3F_MPDIoU = MPDIoU + Focaler-IoU + WIoU v3 |
| Label assignment chuẩn bỏ sót positive sample | PSA (Petal-like Sample Amplification) tăng ~5% positive |
| Loss ≠ metric (cross-entropy ≠ Recall/mAP) | REINFORCE optimize trực tiếp composite reward |

### 1.3. Pipeline 3 Giai Đoạn

```
Giai đoạn 1 – Supervised Training
  └─ DP-YOLO + YOLOv5s + YOLOv8n + YOLOv11n
     Loss: W3F_MPDIoU, Label assign: PSA
     → checkpoints/*/weights/best.pt

Giai đoạn 2 – RL Fine-tuning (REINFORCE)
  └─ EMA Baseline + composite_reward + grad_clip
     → rl_checkpoints/*_rl_best.pt  (dict format)

Giai đoạn 3 – Evaluation & Comparison
  └─ mAP50 | mAP50-95 | AP_small | Recall | FPS
     Supervised vs. RL  →  results/tables/
```

### 1.4. Tập Dữ Liệu — 10 Class

| ID | Tên | Loại |
|---|---|---|
| 0 | `sau_duc_than` | Sâu đục thân |
| 1 | `sau_cuon_la` | Sâu cuốn lá |
| 2 | `benh_dom_nau` | Bệnh đốm nâu |
| 3 | `benh_dao_on` | Bệnh đạo ôn |
| 4 | `benh_kho_van` | Bệnh khô vằn |
| 5 | `ran_xanh` | Rầy xanh |
| 6 | `bo_tri` | Bọ trĩ |
| 7 | `ran_nau` | Rầy nâu |
| 8 | `dom_la` | Đốm lá |
| 9 | `healthy` | Lúa khỏe |

---

## 2. Kiến Trúc DP-YOLO

### 2.1. Backbone — 4 Stage Deformable

```
Input (640×640)
  │
  ├─ Stage 1-3: D2C3 (DCNv2 deformable conv)
  │    - Học hình dạng biến dạng của sâu bệnh
  │    - offset_mask: 3*k² channels (2*k² offset + k² mask)
  │
  └─ Stage 4: D3C3 chiến lược "3+1"
       - 3 bottleneck dùng Conv thường (tránh overfitting ở feature nhỏ)
       - 1 bottleneck cuối dùng DCNv3 (shared offset, groups=4)
```

**File:** [modules.py](file:///d:/graduate/RL-YOLO/yolo-rl-pest/models/dp_yolo/modules.py)

| Module | Dòng | Mô tả |
|---|---|---|
| `DCNv2` | [L66–L110](file:///d:/graduate/RL-YOLO/yolo-rl-pest/models/dp_yolo/modules.py#L66-L110) | DeformConv2d + offset/mask predictor, fallback Conv2d |
| `D2Bottleneck` | [L117–L128](file:///d:/graduate/RL-YOLO/yolo-rl-pest/models/dp_yolo/modules.py#L117-L128) | 1×1 → DCNv2 3×3 → 1×1 với shortcut |
| `D2C3` | [L131–L148](file:///d:/graduate/RL-YOLO/yolo-rl-pest/models/dp_yolo/modules.py#L131-L148) | C3 block stage 1-3 backbone |
| `DCNv3` | [L155–L206](file:///d:/graduate/RL-YOLO/yolo-rl-pest/models/dp_yolo/modules.py#L155-L206) | Group deformable, offset shared (3*k² channels) |
| `D3C3` | [L223–L247](file:///d:/graduate/RL-YOLO/yolo-rl-pest/models/dp_yolo/modules.py#L223-L247) | C3 block stage 4, chiến lược "3+1" |

> **Lưu ý DCNv3:** `torchvision.ops.DeformConv2d` dùng **1 shared offset** cho tất cả groups (không phải per-group như DCNv3 gốc). Offset shape đúng là `(B, 2*k², H, W)`, không phải `(B, groups*2*k², H, W)`. Đây là trade-off hợp lý khi không dùng mmcv — cần ghi chú trong thesis.

### 2.2. Neck — PTCSP + C3Ghost

```
P2 (160×160) ─→ PTCSP (Parallel CNN + Transformer)
P3 (80×80)   ─→ C3Ghost
P4 (40×40)   ─→ C3Ghost
```

| Module | Dòng | Mô tả |
|---|---|---|
| `TransformerLayer` | [L254–L277](file:///d:/graduate/RL-YOLO/yolo-rl-pest/models/dp_yolo/modules.py#L254-L277) | MHSA → Add&Norm → FFN → Add&Norm |
| `PTCSP` | [L280–L314](file:///d:/graduate/RL-YOLO/yolo-rl-pest/models/dp_yolo/modules.py#L280-L314) | CNN branch ‖ Transformer branch → concat |
| `GhostConv` | [L321–L332](file:///d:/graduate/RL-YOLO/yolo-rl-pest/models/dp_yolo/modules.py#L321-L332) | 1×1 cheap op + depthwise 5×5 |
| `GhostBottleneck` | [L335–L353](file:///d:/graduate/RL-YOLO/yolo-rl-pest/models/dp_yolo/modules.py#L335-L353) | MobileNetV3-style |
| `C3Ghost` | [L356–L370](file:///d:/graduate/RL-YOLO/yolo-rl-pest/models/dp_yolo/modules.py#L356-L370) | C3 block với GhostBottleneck |

### 2.3. Config

- **[dp_yolo.yaml](file:///d:/graduate/RL-YOLO/yolo-rl-pest/models/dp_yolo/dp_yolo.yaml)** — Cấu trúc P2+P3+P4 (bỏ P5), ghi chú Loss W3F_MPDIoU
- **[pest.yaml](file:///d:/graduate/RL-YOLO/yolo-rl-pest/configs/pest.yaml)** — 10 class, đường dẫn data
- **[hyp.rl.yaml](file:///d:/graduate/RL-YOLO/yolo-rl-pest/configs/hyp.rl.yaml)** — Hyperparameters cho RL (lr, steps, ema_alpha, eval_interval, ...)

---

## 3. Custom Loss: W3F_MPDIoU

**File:** [models/dp_yolo/loss.py](file:///d:/graduate/RL-YOLO/yolo-rl-pest/models/dp_yolo/loss.py)

### 3.1. Công Thức

```
W3F_MPDIoU = r × R_WIoU × L_F_MPDIoU

Trong đó:
  IoU         = intersection / union

  L_MPDIoU    = 1 − IoU + (d₁² + d₂²) / c²
                d₁ = khoảng cách góc top-left
                d₂ = khoảng cách góc bottom-right
                c² = đường chéo enclosing box²

  L_F_MPDIoU  = L_MPDIoU + IoU − IoU^γ     (Focaler-IoU, γ=0.5)

  R_WIoU      = exp(center_dist² / gt_diag²)   (WIoU v3 outlier degree)

  r           = β / (δ·α^β − δ)                (WIoU v3 constant)
                α=1.9, β=0.6, δ=0.5
```

### 3.2. Tại Sao Tốt Hơn CIoU?

| Loss | Ưu điểm | Nhược điểm |
|---|---|---|
| CIoU | Phổ biến, ổn định | Không penalize lệch corner |
| MPDIoU | Penalize cả 2 góc (top-left + bottom-right) | Không focal weight |
| Focaler-IoU | Down-weight box gần perfect (tập trung hard cases) | Cần kết hợp metric khác |
| WIoU v3 | Focal weight theo outlier degree (penalize object xa tâm ảnh) | Hằng số phức tạp |
| **W3F_MPDIoU** | **Kết hợp cả 3 → corner + focal + outlier** | Nhiều hyperparameter hơn |

### 3.3. Tích Hợp vào YOLOv5

```python
# patch_yolov5.py gọi:
from models.dp_yolo.loss import patch_loss
patch_loss()  # monkey-patch utils.loss.bbox_iou khi CIoU=True
```

Hàm `patch_loss()` tại [loss.py L127–L171](file:///d:/graduate/RL-YOLO/yolo-rl-pest/models/dp_yolo/loss.py#L127-L171):
- Lưu hàm `bbox_iou` gốc từ `utils.metrics`
- Thay bằng wrapper: khi `CIoU=True` → dùng `bbox_iou_w3f`, ngược lại dùng hàm gốc
- Patch cả `utils.metrics.bbox_iou` và `utils.loss.bbox_iou`

---

## 4. PSA Label Assignment

**File:** [models/dp_yolo/psa.py](file:///d:/graduate/RL-YOLO/yolo-rl-pest/models/dp_yolo/psa.py)

### 4.1. Nguyên Tắc

```
Standard YOLOv5:
  Anchor là positive nếu tâm GT nằm trong ±0.5 grid unit từ biên cell
  → Tối đa 5 positive cells (center + 4 neighbors)

PSA (Petal-like Sample Amplification):
  Anchor là positive nếu tâm cell nằm trong VÒNG TRÒN bán kính r=1
  xung quanh tâm GT (tính theo grid unit)
  → Tối đa 9 candidate cells (3×3 grid), lọc qua điều kiện dist² < r²
  → Hai vòng tròn từ 2 cells kề nhau chồng lên tạo hình "cánh hoa"
  → Tăng ~5% positive samples (theo paper DP-YOLO gốc)
```

### 4.2. Logic Lọc Cell Hợp Lệ

```python
# Với mỗi offset (dx, dy) trong [-1,0,1]²:
gi_cand = floor(GT.x) + dx
gj_cand = floor(GT.y) + dy
cx_cell = gi_cand + 0.5   # tâm cell
dist²   = (GT.x - cx_cell)² + (GT.y - cy_cell)²
valid   = (dist² < 1.0²) AND (0 ≤ gi_cand < W) AND (0 ≤ gj_cand < H)
```

### 4.3. Tích Hợp vào YOLOv5

```python
from models.dp_yolo.psa import patch_psa
patch_psa()  # monkey-patch ComputeLoss.build_targets
```

---

## 5. RL Fine-tuning Pipeline

**File:** [train_rl.py](file:///d:/graduate/RL-YOLO/yolo-rl-pest/train_rl.py)

### 5.1. Thuật Toán REINFORCE

```
Mỗi bước:
  1. Lấy batch (images, targets) từ train set
  2. forward_with_grad(images) → preds (giữ gradient qua scores)
  3. compute_reward(preds, targets) → rewards (no grad)
  4. advantage = rewards - EMA_baseline
  5. log_prob = log(avg_confidence)  [xấp xỉ log π_θ(a|s)]
  6. loss = -mean(log_prob × advantage.detach())
  7. loss.backward() → clip_grad_norm(1.0) → optimizer.step()
  8. Mỗi log_interval bước: log TensorBoard, update best ckpt
  9. Mỗi eval_interval bước: quick_eval() trên val set
```

### 5.2. Các Thành Phần

| Component | File | Dòng | Mô Tả |
|---|---|---|---|
| `EMABaseline` | [train_rl.py](file:///d:/graduate/RL-YOLO/yolo-rl-pest/train_rl.py) | [L41–L65](file:///d:/graduate/RL-YOLO/yolo-rl-pest/train_rl.py#L41-L65) | EMA(α=0.99) giảm variance, không cần 2× forward |
| `compute_log_prob` | [train_rl.py](file:///d:/graduate/RL-YOLO/yolo-rl-pest/train_rl.py) | [L72–L95](file:///d:/graduate/RL-YOLO/yolo-rl-pest/train_rl.py#L72-L95) | `log(avg_conf.clamp(1e-20))`, trả Tensor có grad |
| `quick_eval` | [train_rl.py](file:///d:/graduate/RL-YOLO/yolo-rl-pest/train_rl.py) | [L102–L152](file:///d:/graduate/RL-YOLO/yolo-rl-pest/train_rl.py#L102-L152) | torchmetrics MeanAveragePrecision trên val set |
| `rl_finetune` | [train_rl.py](file:///d:/graduate/RL-YOLO/yolo-rl-pest/train_rl.py) | [L214–L373](file:///d:/graduate/RL-YOLO/yolo-rl-pest/train_rl.py#L214-L373) | Vòng lặp chính |
| `freeze_backbone` | [train_rl.py](file:///d:/graduate/RL-YOLO/yolo-rl-pest/train_rl.py) | [L181–L207](file:///d:/graduate/RL-YOLO/yolo-rl-pest/train_rl.py#L181-L207) | Freeze layers 0-9 (YOLOv5) hoặc layers <10 (Ultralytics) |

### 5.3. Checkpoint Format (v1.1)

```python
# Checkpoint được lưu khi avg_r (rolling 200 bước) > best_avg_reward
torch.save({
    'is_rl_checkpoint': True,      # flag phân biệt với supervised ckpt
    'model_name': model_name,
    'step': step,
    'avg_reward': avg_r,
    'state_dict': adapter.state_dict(),   # model.model.state_dict()
}, best_ckpt)
```

> **Tại sao rolling average?** Dùng instant reward để save checkpoint dẫn đến lưu model tại các "spike" ngẫu nhiên (batch đặc biệt dễ). Rolling average 200 bước ổn định hơn, phản ánh xu hướng dài hạn.

### 5.4. Reward Functions

**File:** [reward.py](file:///d:/graduate/RL-YOLO/yolo-rl-pest/reward.py)

```python
# 1. Recall reward (alpha=0.6)
recall_reward(preds, targets, iou_threshold=0.5)
  → Normalize per-class, penalty duplicate detection

# 2. Small-object recall reward (alpha=0.4)
small_object_recall_reward(preds, targets, small_thresh=32)
  → Bonus cho vật thể có area < 32×32 px

# 3. Composite reward (default)
composite_reward = 0.6 × recall_reward + 0.4 × small_object_recall_reward
```

### 5.5. Model Adapters

| Adapter | File | Framework |
|---|---|---|
| `YOLOv5Adapter` | [adapters/yolov5_adapter.py](file:///d:/graduate/RL-YOLO/yolo-rl-pest/adapters/yolov5_adapter.py) | YOLOv5s, DP-YOLO |
| `UltralyticsAdapter` | [adapters/ultralytics_adapter.py](file:///d:/graduate/RL-YOLO/yolo-rl-pest/adapters/ultralytics_adapter.py) | YOLOv8n/s, YOLOv11n/s |

Cả hai đều implement `forward_with_grad()` — chạy inference nhưng giữ gradient qua confidence scores để backprop được.

---

## 6. Evaluation

**File:** [evaluate.py](file:///d:/graduate/RL-YOLO/yolo-rl-pest/evaluate.py)

### 6.1. Metrics

| Metric | Mô tả |
|---|---|
| `mAP50` | mAP tại IoU=0.5 |
| `mAP50-95` | mAP trung bình IoU∈[0.5:0.95:0.05] (COCO standard) |
| `APs` | AP cho small objects (area < 32²) |
| `APm` | AP cho medium objects |
| `Recall` | max recall @ 100 detections per image |
| `FPS` | Tốc độ inference (images/second) |

### 6.2. Load RL Checkpoint (Fix v1.1)

```python
# evaluate.py _load_model_for_eval():
ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
is_rl = isinstance(ckpt, dict) and ckpt.get('is_rl_checkpoint', False)

if is_rl:
    # Load supervised checkpoint để lấy model structure
    # Apply RL state_dict lên trên
    inner_model.load_state_dict(ckpt['state_dict'], strict=False)
else:
    # Standard format: torch.hub.load hoặc YOLO()
```

### 6.3. Output

```
results/tables/results_full.csv   — raw metrics mọi model × stage
results/tables/results_delta.csv  — delta RL - supervised (dương = RL tốt hơn)

Bảng markdown in ra terminal:
  Model     | Stage      | mAP50 | mAP50_95 | APs  | recall | fps
  DP-YOLO   | supervised | ...   | ...      | ...  | ...    | ...
  DP-YOLO   | rl         | ...   | ...      | ...  | ...    | ...
  ...
```

---

## 7. Cấu Trúc Project

```
yolo-rl-pest/
├── models/dp_yolo/
│   ├── modules.py          # D2C3, D3C3, PTCSP, C3Ghost, DCNv2, DCNv3
│   ├── loss.py             # W3F_MPDIoU (MPDIoU + Focaler + WIoU v3)
│   ├── psa.py              # PSA label assignment
│   ├── patch_yolov5.py     # Entry point: đăng ký modules + patch loss/PSA
│   ├── dp_yolo.yaml        # Model config P2+P3+P4
│   └── __init__.py
├── adapters/
│   ├── yolov5_adapter.py   # forward_with_grad cho YOLOv5/DP-YOLO
│   └── ultralytics_adapter.py  # forward_with_grad cho YOLOv8/v11
├── configs/
│   ├── hyp.rl.yaml         # RL hyperparameters
│   └── pest.yaml           # Dataset config 10 class
├── tests/
│   └── test_fixes.py       # 19/19 PASS
├── train_supervised.py     # Giai đoạn 1
├── train_rl.py             # Giai đoạn 2
├── evaluate.py             # Giai đoạn 3
├── dataloader.py           # PestDataset + albumentations
├── reward.py               # recall + small-object + composite reward
├── dp_yolo_train.py        # Convenience wrapper cho DP-YOLO
├── pyproject.toml          # uv environment
└── requirements.txt
```

---

## 8. Môi Trường & Chạy

### 8.1. Setup (lần đầu)

```bash
# Tạo venv + cài deps với uv
uv sync --extra dev

# Verify
uv run python tests/test_fixes.py
# → PASSED: 19/19
```

### 8.2. Workflow Khi Có Data

```bash
# Bước 0: Chuẩn bị data
#   ├─ ≥2000 ảnh UAV đa điều kiện (sáng/chiều, mùa khô/mưa)
#   ├─ Annotation YOLO format (10 class, tham khảo pest.yaml)
#   └─ Split 7:2:1 (seed=42) → data/pest/{train,val,test}/

# Bước 1: Supervised Training (Giai đoạn 1)
uv run python train_supervised.py --model dp_yolo --epochs 300
uv run python train_supervised.py --model yolov5s  --epochs 300  # baseline
uv run python train_supervised.py --model yolov8n  --epochs 300  # baseline
uv run python train_supervised.py --model yolov11n --epochs 300  # baseline

# Bước 2: RL Fine-tuning (Giai đoạn 2)
uv run python train_rl.py --model dp_yolo --steps 50000
uv run python train_rl.py --model yolov5s --steps 50000  # optional

# Monitor training
tensorboard --logdir results/tensorboard/

# Bước 3: Evaluation (Giai đoạn 3)
uv run python evaluate.py --split test
# → results/tables/results_full.csv
# → results/tables/results_delta.csv
```

### 8.3. Commands Tham Khảo

```bash
# Patch DP-YOLO vào YOLOv5 (verify modules được đăng ký)
uv run python models/dp_yolo/patch_yolov5.py

# Fine-tune với freeze backbone (tránh catastrophic forgetting)
uv run python train_rl.py --model dp_yolo --freeze --steps 30000

# Evaluate chỉ một model
uv run python evaluate.py --model DP-YOLO --split val

# Chạy test suite
uv run python tests/test_fixes.py
```

---

## 9. Các Quyết Định Kỹ Thuật Quan Trọng

### 9.1. Tại sao `log(avg_confidence)` là log π_θ(a|s)?

YOLO không phải autoregressive model nên không có log-prob đầy đủ. `avg_confidence` là proxy hợp lý: confidence cao ↔ policy "chắc chắn" về prediction → gradient đúng hướng tối ưu reward. Đây là heuristic được chấp nhận (theo yanivnik/cv-rl). **Cần nêu rõ limitation này trong thesis.**

### 9.2. Tại sao EMA Baseline thay vì Monte Carlo?

| Approach | Chi phí | Variance |
|---|---|---|
| bwconrad (2nd sample) | 2× forward pass / step | Thấp |
| yanivnik (no baseline) | 1× forward | Cao |
| **EMA (α=0.99)** | **1× forward, bộ nhớ O(1)** | **Trung bình-thấp** |

EMA là trade-off tốt nhất cho bài toán này.

### 9.3. Tại sao DCNv3 dùng shared offset thay vì per-group?

`torchvision.ops.DeformConv2d` không hỗ trợ per-group offset (khác DCNv3 gốc của InternImage). Thay vì thêm dependency nặng (mmcv), ta dùng shared offset — vẫn capture spatial deformation, chỉ không độc lập theo group. **Ghi chú trong thesis: "simplified DCNv3 with shared offset".**

### 9.4. Composite Reward: α=0.6 recall + 0.4 small

Chọn α=0.6 vì recall quan trọng hơn small-object recall (bài toán phát hiện sâu bệnh: false negative rất tốn kém). Giá trị α có thể tune qua `hyp.rl.yaml`.

---

## 10. Trạng Thái & Bước Tiếp Theo

### 10.1. Tóm Tắt Trạng Thái

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Kiến trúc DP-YOLO (D2C3, D3C3, PTCSP, C3Ghost) | ✅ Hoàn thiện | DCNv3 dùng shared offset |
| Config YAML (dp_yolo.yaml, pest.yaml, hyp files) | ✅ Đầy đủ | |
| Custom Loss W3F_MPDIoU | ✅ Implement + test pass | loss.py |
| PSA Label Assignment | ✅ Implement + test pass | psa.py |
| Patch system (loss + PSA + modules) | ✅ patch_yolov5.py | |
| RL Pipeline (REINFORCE + EMA + eval_interval) | ✅ Hoàn thiện | train_rl.py v1.1 |
| Reward functions (recall + small + composite) | ✅ Hoàn thiện | reward.py |
| Model Adapters (YOLOv5 + Ultralytics) | ✅ Hoàn thiện | |
| Dataloader (YOLO format + albumentations) | ✅ Hoàn thiện | |
| Evaluate script (RL checkpoint compatible) | ✅ Fixed | evaluate.py |
| Test suite | ✅ **19/19 PASS** | `uv run python tests/test_fixes.py` |
| uv environment | ✅ Sẵn sàng | pyproject.toml + .venv |
| **Data** | ❌ **Chưa có** | **Bước tiếp theo duy nhất** |

### 10.2. Bước Tiếp Theo

1. **Thu thập ≥2000 ảnh UAV** — đa điều kiện (sáng/chiều, mùa khô/mưa, độ cao UAV khác nhau)
2. **Annotation** — YOLO format, 10 class theo `pest.yaml`
3. **Split dataset** — 70% train / 20% val / 10% test (stratified, seed=42)
4. **Chạy pipeline** theo mục 8.2

### 10.3. Hạn Chế Còn Lại (Không Blocking)

| Hạn chế | Ảnh hưởng | Xử lý trong thesis |
|---|---|---|
| DCNv3 dùng shared offset (không phải per-group) | Thấp: vẫn deformable | Nêu rõ là "simplified DCNv3" |
| `log(avg_conf)` là proxy của log π_θ | Trung bình: YOLO không có log-prob đầy đủ | Cite yanivnik, nêu limitation |
| Type hints `list[dict]` (Python 3.10+) | Thấp: uv dùng Python 3.10 sẵn | Không cần fix |
