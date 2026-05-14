# Review Chi Tiết: cv-rl — Computer Vision Training using Policy Optimization

> Repo gốc: [bwconrad/cv-rl](https://github.com/bwconrad/cv-rl)  
> Được lấy cảm hứng từ paper: **"Tuning computer vision models with task rewards"** ([arXiv:2302.08242](https://arxiv.org/abs/2302.08242))

---

## 1. Project Này Làm Gì?

### Mục tiêu cốt lõi

Project thực hiện thí nghiệm **dùng Policy Optimization (Reinforcement Learning) để huấn luyện các mô hình Computer Vision tối ưu trực tiếp các metric đánh giá không khả vi (non-differentiable)**. Cụ thể là metric **F1 Score** trong bài toán **Binary Segmentation**.

> **Dẫn chứng** — `README.md` dòng 1–3:
> ```
> # Computer Vision Training using Policy Optimization
> This repo contains experiments using policy optimization to train
> non-differentiable objectives for computer vision tasks.
> ```

### Vấn đề cần giải quyết

Trong Deep Learning truyền thống, hàm mất mát phải **khả vi** (differentiable) để lan truyền ngược (backprop). Tuy nhiên, các metric thực tế như **F1 Score, IoU, BLEU, ROUGE** lại **không khả vi** — chúng phụ thuộc vào các bước lấy ngưỡng (thresholding) hay argmax, vốn tạo ra gradient bằng 0 hoặc không xác định.

| Metric | Khả vi? | Vấn đề |
|--------|---------|--------|
| Cross-Entropy | ✅ | Không trực tiếp tối ưu F1 |
| F1 Score | ❌ | Có argmax/thresholding |
| Dice | ❌ | Tương tự F1 |
| IoU | ❌ | Tương tự |

Giải pháp của project này: **dùng REINFORCE (policy gradient) để lấy gradient gián tiếp thông qua sampling**.

---

## 2. Cấu Trúc Project

```
cv-rl/
├── train_segmentation.py          # Baseline: train với Cross-Entropy
├── train_segmentation_reinforce.py # RL version: train với REINFORCE + F1 reward
├── requirements.txt
├── README.md
└── src/
    ├── base_model.py              # Base LightningModule (optimizer, scheduler)
    ├── callbacks.py               # Visualization callback
    └── datamodules/
        ├── __init__.py
        └── segmentation.py        # Dataset + DataModule cho TNBC
```

---

## 3. Setup

### Yêu cầu hệ thống

- Python 3.10+
- GPU với CUDA (dùng `--trainer.accelerator gpu`)

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

**Các thư viện chính** (`requirements.txt`):

| Thư viện | Phiên bản | Vai trò |
|----------|-----------|---------|
| `torch` | 2.0.1 | Deep learning framework |
| `pytorch_lightning` | 2.0.2 | Training loop quản lý |
| `segmentation_models_pytorch` | 0.3.3 | U-Net, encoders pre-trained |
| `torchmetrics` | 1.0.1 | Tính F1, Dice, Precision, Recall |
| `torchvision` | 0.15.2 | Augmentation pipeline |
| `transformers` | 4.30.0 | Cosine LR scheduler |
| `jsonargparse` | 4.21.1 | CLI từ class args |
| `opencv_python` | 4.7.0.72 | Xử lý ảnh |

### Chuẩn bị dữ liệu

Dataset: **TNBC** (Triple Negative Breast Cancer) — dataset segmentation tế bào ung thư vú.  
Link tải: https://zenodo.org/record/1175282

Cấu trúc thư mục dữ liệu cần có:
```
data/tnbc/
├── train/
│   ├── images/   ← ảnh RGB gốc
│   └── masks/    ← mask nhị phân (0/1)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

> **Dẫn chứng** — `src/datamodules/segmentation.py` dòng 119–130:
> ```python
> self.img_paths = sorted([
>     f for f in glob(f"{root}/images/**/*", recursive=True)
>     if os.path.isfile(f)
> ])
> self.mask_paths = sorted([
>     f for f in glob(f"{root}/masks/**/*", recursive=True)
>     if os.path.isfile(f)
> ])
> ```

---

## 4. Kiến Trúc Mô Hình

### Mạng nơ-ron: U-Net + ResNet-18 Encoder

> **Dẫn chứng** — `train_segmentation.py` dòng 64–69 (và tương tự trong `train_segmentation_reinforce.py`):
> ```python
> self.net = create_model(
>     self.arch,           # default = "unet"
>     encoder_name=self.encoder,  # default = "resnet18"
>     encoder_weights="imagenet",
>     in_channels=3,
>     classes=1,
> )
> ```

- **Backbone (Encoder)**: ResNet-18 pre-trained trên ImageNet
- **Decoder**: U-Net style (skip connections)
- **Output**: 1 channel (logit cho binary mask)
- **Thư viện**: `segmentation_models_pytorch`

### DataModule & Augmentation

> **Dẫn chứng** — `src/datamodules/segmentation.py` dòng 51–62:
> ```python
> self.transforms_train = Compose([
>     RandomResizedCrop(size, (min_scale, max_scale), antialias=True),
>     RandomHorizontalFlip(p=0.5),
>     RandomVerticalFlip(p=0.5),
>     Normalize(mean, std),
> ])
> self.transforms_test = Compose([
>     Resize((size, size), antialias=True),
>     Normalize(mean, std)
> ])
> ```

- Train: RandomResizedCrop + Flip ngang + Flip dọc + Normalize
- Val/Test: chỉ Resize + Normalize
- Dataset training được set `length = 1000 * batch_size` → lặp lại ảnh để tạo epoch lớn hơn

### Optimizer & Scheduler

> **Dẫn chứng** — `src/base_model.py` dòng 14–55:

Hỗ trợ 3 optimizer: `adam`, `adamw`, `sgd`  
Hỗ trợ 2 scheduler: `cosine` (dùng HuggingFace warmup cosine), `none`

---

## 5. Training

### 5.1 Baseline — Cross-Entropy (train_segmentation.py)

```bash
python train_segmentation.py fit \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.precision 16-mixed \
  --data.root data/tnbc \
  --data.batch_size 8 \
  --trainer.max_steps 1000 \
  --trainer.val_check_interval 100 \
  --model.lr 0.0005 \
  --model.schedule cosine
```

**Loss function**: Binary Cross-Entropy with Logits  
> **Dẫn chứng** — `train_segmentation.py` dòng 96–100:
> ```python
> pred = self.net(x).squeeze(1)
> loss = F.binary_cross_entropy_with_logits(pred, y)
> ```

### 5.2 REINFORCE — F1 Score (train_segmentation_reinforce.py)

```bash
python train_segmentation_reinforce.py fit \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.precision 16-mixed \
  --data.root data/tnbc \
  --data.batch_size 8 \
  --trainer.max_steps 1000 \
  --trainer.val_check_interval 100 \
  --model.lr 0.0005 \
  --model.schedule cosine
```

### 5.3 Fine-tune from CE checkpoint → F1

```bash
python train_segmentation_reinforce.py fit \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.precision 16-mixed \
  --data.root data/tnbc \
  --data.batch_size 8 \
  --trainer.max_steps 1000 \
  --trainer.val_check_interval 100 \
  --model.lr 0.00005 \
  --model.schedule cosine \
  --model.weights output/weights-ce.ckpt
```

> Lưu ý: lr giảm 10x (0.0005 → 0.00005) khi fine-tune

---

## 6. Cách RL Được Dùng Trong Mô Hình CV — Phân Tích Chi Tiết

### 6.1 Thuật toán: REINFORCE (Williams, 1992)

REINFORCE là thuật toán policy gradient cơ bản nhất trong RL. Công thức gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot R \right]$$

Trong đó:
- $\pi_\theta$: policy (mô hình segmentation với tham số $\theta$)
- $a$: action (binary mask được sample)
- $s$: state (ảnh đầu vào)
- $R$: reward (F1 score)

### 6.2 Mapping sang bài toán Segmentation

| Khái niệm RL | Tương ứng trong Project |
|---|---|
| Policy $\pi_\theta$ | U-Net (đầu ra là xác suất từng pixel) |
| State $s$ | Ảnh đầu vào |
| Action $a$ | Binary mask được sample từ Bernoulli |
| Reward $R$ | F1 Score so với ground truth |
| Baseline | F1 của sample thứ 2 (Monte Carlo baseline) |

### 6.3 Code Chi Tiết — Từng Bước

> **Dẫn chứng đầy đủ** — `train_segmentation_reinforce.py` dòng 95–117:

```python
def shared_step(self, batch, mode="train"):
    x, y = batch

    # 1. Forward pass: U-Net trả về logit (chưa sigmoid)
    pred = self.net(x).squeeze(1)

    # 2. Tạo phân phối Bernoulli từ logit
    #    Mỗi pixel ~ Bernoulli(sigmoid(logit))
    dist = Bernoulli(logits=pred)

    # 3. Sample 2 mask từ phân phối (sample + baseline)
    sample = dist.sample()    # action a ~ π_θ(·|s)
    baseline = dist.sample()  # baseline b ~ π_θ(·|s)

    # 4. Tính reward = F1(sample) - F1(baseline)
    #    Baseline giảm variance của gradient
    reward = binary_f1_score(sample, y) - binary_f1_score(baseline, y)

    # 5. Tính log prob trung bình trên toàn batch theo không gian (H×W)
    log_prob = torch.mean(dist.log_prob(sample), dim=[1, 2])

    # 6. REINFORCE loss = -E[log_prob * reward]
    #    Dấu trừ vì ta muốn maximize reward, còn optimizer minimize loss
    loss = torch.mean(-log_prob * reward)

    return loss
```

### 6.4 Tại Sao Phải Dùng RL Thay Vì Optimize F1 Trực Tiếp?

**Nguyên nhân kỹ thuật**: F1 Score được tính dựa trên **binary predictions** (0 hoặc 1), không phải xác suất liên tục:

$$F1 = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

Để có binary prediction, ta phải thresholding: $\hat{y} = \mathbf{1}[\sigma(\text{logit}) > 0.5]$

Hàm step function `1[·]` có **gradient = 0 ở mọi nơi** (trừ điểm không liên tục), nên không thể backprop qua được.

**Giải pháp RL**: Thay vì tính gradient trực tiếp qua F1, ta:
1. **Sample** binary mask từ phân phối Bernoulli → vẫn discrete nhưng có log prob
2. **Dùng log-derivative trick**: $\nabla_\theta \mathbb{E}[R] = \mathbb{E}[R \cdot \nabla_\theta \log p_\theta(a)]$
3. Gradient $\nabla_\theta \log p_\theta(a)$ **khả vi hoàn toàn** vì tính qua Bernoulli distribution

### 6.5 Kỹ Thuật Baseline — Giảm Variance

Thay vì dùng reward trực tiếp $R$, project dùng **advantage** = $R - b$ với $b$ là baseline:

```python
reward = binary_f1_score(sample, y) - binary_f1_score(baseline, y)
```

> **Tại sao cần baseline?** REINFORCE có vấn đề **high variance** — gradient estimate dao động rất lớn. Dùng baseline (một sample thứ 2 độc lập) giúp:
> - Nếu `sample` tốt hơn `baseline` → reward > 0 → tăng xác suất action đó
> - Nếu `sample` tệ hơn `baseline` → reward < 0 → giảm xác suất action đó
> - Expected value của baseline không làm bias gradient (vì $\mathbb{E}[b] = \mathbb{E}[R]$)

Đây là kỹ thuật **Self-Critical Sequence Training** phổ biến trong NLP (Rennie et al., 2017), được ứng dụng sang CV.

### 6.6 Vấn Đề Observed (từ kết quả thực nghiệm)

| Objective | F1 | Dice |
|:-:|:-:|:-:|
| Cross-entropy | **0.7729** | 0.7585 |
| F1 (REINFORCE, train from scratch) | 0.4152 | 0.4369 |
| Cross-entropy → F1 (fine-tune) | 0.7615 | **0.7611** |

**Nhận xét**:
- Train REINFORCE từ đầu: **kết quả rất tệ** (F1 = 0.41) — nguyên nhân là REINFORCE có high variance cao, khó hội tụ từ random init
- Fine-tune từ CE checkpoint: **cho kết quả cân bằng tốt** giữa F1 và Dice
- CE baseline: tốt về F1 nhưng Dice thấp hơn fine-tune version

**Kết luận thực nghiệm**: REINFORCE hiệu quả nhất khi dùng làm **giai đoạn fine-tune sau khi đã pre-train với CE**. Đây là pattern phổ biến: warm-start với MLE, sau đó fine-tune với RL reward.

---

## 7. Callback Visualization

> **Dẫn chứng** — `src/callbacks.py`:

```python
class SegmentationImageGridSampler(Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Lưu grid gồm: input image | predicted mask | ground truth mask
        ...
        grid = torch.cat((inp_grid, pred_grid, target_grid), -1)
        save_image(grid, filename)
```

Callback này lưu ảnh visualization mỗi validation step, hỗ trợ cả CSVLogger và WandBLogger.

---

## 8. Framework & CLI

Project dùng **PyTorch Lightning CLI** với **jsonargparse**, cho phép cấu hình toàn bộ experiment qua command line mà không cần sửa code:

```python
class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.set_defaults({"trainer.logger": lazy_instance(CSVLogger, ...)})
        parser.add_lightning_class_args(SegmentationImageGridSampler, "image_sampler")
```

---

## 9. So Sánh Hai File Train

| Đặc điểm | `train_segmentation.py` | `train_segmentation_reinforce.py` |
|---|---|---|
| Loss | `F.binary_cross_entropy_with_logits` | REINFORCE: `-mean(log_prob * reward)` |
| Gradient | Trực tiếp qua BCE | Gián tiếp qua log-derivative trick |
| Import thêm | — | `Bernoulli`, `binary_f1_score` |
| Mục tiêu tối ưu | Cross-entropy | F1 Score |
| Hội tụ từ đầu | Ổn định | Khó, variance cao |

---

## 10. Tóm Tắt Luồng Hoạt Động

```
Input image (x) 
    ↓
U-Net (policy π_θ) → logit map (H×W)
    ↓
Bernoulli(logits=logit) → xác suất từng pixel
    /           \
sample()      sample()
(action a)    (baseline b)
    |               |
F1(a, y)      F1(b, y)
    \           /
     reward = F1(a,y) - F1(b,y)
         ↓
log_prob = mean(log π_θ(a))  [khả vi!]
         ↓
loss = -mean(log_prob * reward)
         ↓
Backprop → update θ
```

---

## 11. Hạn Chế và Mở Rộng

**Hạn chế hiện tại**:
- Chỉ có 1 task (binary segmentation) và 1 dataset (TNBC)
- REINFORCE train from scratch không ổn định
- Không có multi-GPU support được test
- Không có hyperparameter search

**Hướng mở rộng tiềm năng** (từ paper gốc):
- Áp dụng cho Object Detection (optimize mAP)
- Áp dụng cho Image Captioning (optimize CIDEr)
- Dùng PPO thay REINFORCE để ổn định hơn
- Thêm entropy regularization để tránh mode collapse

---

## 12. Tài Liệu Tham Khảo

1. **Paper** gốc: "Tuning computer vision models with task rewards" — [arXiv:2302.08242](https://arxiv.org/abs/2302.08242)
2. **REINFORCE**: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning.
3. **Self-Critical Sequence Training**: Rennie et al. (2017). Self-Critical Sequence Training for Image Captioning. CVPR.
4. **Dataset TNBC**: https://zenodo.org/record/1175282
5. **segmentation_models_pytorch**: https://github.com/qubvel/segmentation_models.pytorch
