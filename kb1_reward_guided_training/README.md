# KB1: Reward-guided YOLO Training

> Môi trường Python dùng chung gồm .venv, pyproject.toml, uv.lock, requirements.txt và .gitignore nằm tại root RL-YOLO. Chạy uv sync và kích hoạt .venv từ root. Guide và báo cáo KB1-B canonical nằm trong docs.

# YOLO-RL-Pest: UAV Pest Detection với RL Fine-tuning

So sánh **DP-YOLO vs YOLOv5s/v8n/v8s/v11n/v11s** trên bài toán phát hiện sâu bệnh cây trồng từ ảnh UAV,  
với **REINFORCE fine-tuning** để cải thiện recall trên vật thể nhỏ.

---

## Cấu trúc project

```
kb1_reward_guided_training/
├── configs/
│   ├── pest.yaml          # dataset config (nc=28, trỏ đến pre-data/data/v2i_cleanned)
│   ├── hyp.pest.yaml      # hyperparams supervised (YOLOv5 format)
│   └── hyp.rl.yaml        # hyperparams RL fine-tuning
├── models/
│   └── dp_yolo/
│       ├── dp_yolo.yaml   # architecture YAML (nc=28, P2+P3+P4 head)
│       ├── modules.py     # D2C3, D3C3, PTCSP, C3Ghost, DCNv2, DCNv3
│       ├── loss.py        # W3F_MPDIoU + patch_loss()
│       ├── psa.py         # PSA label assignment + patch_psa()
│       └── patch_yolov5.py  # đăng ký tất cả patches vào YOLOv5 runtime
├── adapters/
│   ├── yolov5_adapter.py        # YOLOv5 + DP-YOLO (giữ gradient qua confidence)
│   └── ultralytics_adapter.py   # YOLOv8/v11
├── dataloader.py          # PestDataset + Albumentations augmentation
├── reward.py              # recall_reward, small_object_recall_reward, composite_reward
├── train_supervised.py    # Giai đoạn 1: supervised (200 epochs, early stopping)
├── train_rl.py            # Giai đoạn 2: REINFORCE fine-tune (30k steps)
├── evaluate.py            # Giai đoạn 3: mAP/APs/recall/FPS comparison
├── dp_yolo_train.py       # Wrapper train DP-YOLO (apply patches trước khi train)
├── pyproject.toml         # uv / hatchling project config
└── yolov5/                # clone riêng (git clone ultralytics/yolov5)
```

**Dataset** được đặt ngoài project tại: `../pre-data/data/v2i_cleanned/`

```
v2i_cleanned/
├── train/images/ & labels/
├── valid/images/ & labels/
└── test/images/  & labels/
```

---

## 28 Class sâu bệnh

Dataset dùng **Plant Disease Detection** với 28 loại lá và bệnh cây trồng:

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0  | Apple Scab Leaf | 14 | Raspberry leaf |
| 1  | Apple leaf | 15 | Soyabean leaf |
| 2  | Apple rust leaf | 16 | Squash Powdery mildew leaf |
| 3  | Bell_pepper leaf | 17 | Strawberry leaf |
| 4  | Bell_pepper leaf spot | 18 | Tomato Early blight leaf |
| 5  | Blueberry leaf | 19 | Tomato Septoria leaf spot |
| 6  | Cherry leaf | 20 | Tomato leaf |
| 7  | Corn Gray leaf spot | 21 | Tomato leaf bacterial spot |
| 8  | Corn leaf blight | 22 | Tomato leaf late blight |
| 9  | Corn rust leaf | 23 | Tomato leaf mosaic virus |
| 10 | Peach leaf | 24 | Tomato leaf yellow virus |
| 11 | Potato leaf | 25 | Tomato mold leaf |
| 12 | Potato leaf early blight | 26 | grape leaf |
| 13 | Potato leaf late blight | 27 | grape leaf black rot |

---

## Cài đặt

> **Yêu cầu:** Python ≥ 3.10, [uv](https://docs.astral.sh/uv/) đã cài.  
> Cài uv nếu chưa có: `pip install uv` hoặc `winget install astral-sh.uv`

### 1. Tạo môi trường ảo và cài dependencies

```bash
# Tạo venv và cài tất cả dependencies từ pyproject.toml
uv sync

# Kích hoạt venv (Windows)
.venv\Scripts\activate

# Kích hoạt venv (Linux/macOS)
source .venv/bin/activate
```

> `uv sync` đọc `pyproject.toml` và cài đúng phiên bản (torch từ PyTorch index cu121).  
> File lock `uv.lock` đảm bảo reproducibility.

### 2. Cài YOLOv5 (cần cho DP-YOLO và YOLOv5s baseline)

```bash
# Clone YOLOv5 vào kb1_reward_guided_training/yolov5/
git clone https://github.com/ultralytics/yolov5.git

# Cài thêm requirements của YOLOv5 vào venv hiện tại
uv pip install -r yolov5/requirements.txt
```

### 3. Kiểm tra patch (tuỳ chọn)

```bash
python models/dp_yolo/patch_yolov5.py
```

Kết quả mong đợi:

```
DP-YOLO patch_yolov5:
  ✓ Custom modules registered: D2C3, D3C3, PTCSP, C3Ghost, ...
  ✓ parse_model patched for D2C3, D3C3, PTCSP
  -> W3F_MPDIoU loss patched -> utils.loss.bbox_iou (CIoU path)
  ✓ W3F_MPDIoU loss patched
  ✓ PSA label assignment patched -> ComputeLoss.build_targets (radius=1.0)
DP-YOLO patch complete.
```

> **Lưu ý kỹ thuật:** Patches được apply tự động khi train DP-YOLO qua `dp_yolo_train.py`.  
> Patch dùng `models.__path__` injection để tránh xung đột namespace giữa `kb1_reward_guided_training/models/` và `yolov5/models/`.

---

## Quy trình Training

### Giai đoạn 1 – Supervised Training

Train các YOLO model trên dataset (200 epochs, early stopping `patience=30`).  
Checkpoint được lưu tại `checkpoints/<model>/weights/best.pt`.

```bash
# Train tất cả model tuần tự
python train_supervised.py

# Hoặc chỉ train 1 model
python train_supervised.py --model yolov8n
python train_supervised.py --model yolov11n
python train_supervised.py --model dp_yolo
```

**Batch size theo model (tối ưu cho RTX 4060 8GB VRAM):**

| Model    | Batch | Framework    | Ghi chú                    |
|----------|-------|--------------|----------------------------|
| yolov5s  | 16    | YOLOv5       | Anchor-based baseline      |
| yolov8n  | 16    | Ultralytics  | Anchor-free nhẹ            |
| yolov8s  | 8     | Ultralytics  | Anchor-free nặng hơn       |
| yolov11n | 16    | Ultralytics  | Anchor-free mới nhất       |
| yolov11s | 8     | Ultralytics  | YOLOv11 lớn hơn            |
| dp_yolo  | 16    | YOLOv5+patch | **Main model** (D2C3/D3C3) |

> **DP-YOLO:** `dp_yolo_train.py` tự động apply patches (W3F_MPDIoU, PSA, custom modules)  
> trước khi gọi `yolov5/train.py`. Không cần cấu hình thêm.

Theo dõi training:

```bash
tensorboard --logdir checkpoints
```

---

### Giai đoạn 2 – RL Fine-tuning

Fine-tune từ checkpoint supervised bằng thuật toán **REINFORCE** với EMA baseline.  
Reward mặc định: `R = 0.6 × R_recall + 0.4 × R_small_object`

```bash
# Fine-tune tất cả model
python train_rl.py

# Chỉ fine-tune 1 model
python train_rl.py --model dp_yolo
python train_rl.py --model yolov8n

# Freeze backbone (tránh catastrophic forgetting, tiết kiệm VRAM)
python train_rl.py --model dp_yolo --freeze

# Override số bước và learning rate
python train_rl.py --model dp_yolo --steps 20000 --lr 5e-7

# Dùng config RL tuỳ chỉnh
python train_rl.py --cfg configs/hyp.rl.yaml
```

Script tự động:
- Load checkpoint từ `checkpoints/<model>/weights/best.pt`
- Lưu best checkpoint tại `rl_checkpoints/<model>_rl_best.pt` (theo rolling avg 200 bước)
- Evaluate trên val set mỗi 3,000 bước
- Log TensorBoard: reward, loss, baseline, val mAP50

```bash
tensorboard --logdir results/tensorboard
```

> **Nếu OOM:** giảm `batch_size: 4` trong `configs/hyp.rl.yaml` hoặc thêm `--freeze`.

---

### Giai đoạn 3 – Evaluation & So sánh

```bash
# So sánh tất cả model (supervised vs RL fine-tuned)
python evaluate.py

# Chỉ evaluate 1 model
python evaluate.py --model DP-YOLO

# Dùng tập test thay vì val
python evaluate.py --split test
```

Kết quả được lưu tại:
- `results/tables/results_full.csv`  – metrics đầy đủ (mAP50, mAP50-95, APs, APm, recall, FPS)
- `results/tables/results_delta.csv` – delta RL vs supervised

---

## Ma trận thí nghiệm

| Exp | Model    | Stage        | Mục tiêu                        |
|-----|----------|--------------|---------------------------------|
| E01 | YOLOv5s  | Supervised   | Baseline anchor-based           |
| E02 | YOLOv8n  | Supervised   | Anchor-free nhẹ                 |
| E03 | YOLOv8s  | Supervised   | Anchor-free nặng hơn            |
| E04 | YOLOv11n | Supervised   | Kiến trúc mới nhất              |
| E05 | YOLOv11s | Supervised   | YOLOv11 lớn hơn                 |
| E06 | DP-YOLO  | Supervised   | **Main model**                  |
| E07 | YOLOv5s  | RL Fine-tune | RL trên baseline                |
| E08 | YOLOv8n  | RL Fine-tune | RL trên anchor-free             |
| E09 | YOLOv11n | RL Fine-tune | RL trên latest                  |
| E10 | DP-YOLO  | RL Fine-tune | **Main contribution**           |

---

## Kiến trúc DP-YOLO

```
Input (640×640)
│
Backbone:
  P1/2   – Conv stem
  P2/4   – Conv + D2C3×3   (DCNv2 deformable, stage 1)
  P3/8   – Conv + D2C3×6   (DCNv2 deformable, stage 2)
  P4/16  – Conv + D2C3×9   (DCNv2 deformable, stage 3)
  P5/32  – Conv + D3C3×3   (DCNv3 "3+1" strategy, stage 4)
           + SPPF
│
Neck (PAN):
  Top-down:   P5 → C3Ghost → P4 → C3Ghost → P3 → PTCSP → P2
  Bottom-up:  P2 → C3Ghost → P3 → C3Ghost → P4
│
Head:
  Detect([P2/4, P3/8, P4/16], nc=28, 3 anchors per scale)
  [Bỏ P5, thêm P2 để bắt object rất nhỏ trên ảnh UAV]
│
Loss:     W3F_MPDIoU = MPDIoU + Focaler-IoU + WIoU v3
Label:    PSA (Petal-like Sample Amplification, radius=1 grid)
```

**Scale theo `depth_multiple=0.33`, `width_multiple=0.50`** (tương đương YOLOv5s về kích thước).

---

## Hyperparameters

| Param          | Supervised  | RL Fine-tune |
|----------------|-------------|--------------|
| Epochs/Steps   | 200         | 30,000       |
| Learning Rate  | 0.01 (SGD)  | 1e-6 (Adam)  |
| Batch Size     | 8–16 *      | 8            |
| Workers        | 8           | 4            |
| Early Stopping | patience=30 | –            |
| EMA Baseline α | –           | 0.99         |
| Conf Threshold | 0.25        | 0.20         |
| Reward α       | –           | 0.60 recall + 0.40 small-object |
| Grad clip      | –           | 1.0          |

> \* Batch size tuỳ model: 16 cho model nhẹ (v5s, v8n, v11n, dp_yolo), 8 cho model nặng (v8s, v11s).

---

## Yêu cầu hệ thống

| Thành phần      | Phiên bản yêu cầu                          |
|-----------------|---------------------------------------------|
| Python          | ≥ 3.10                                      |
| PyTorch         | ≥ 2.0.1 + CUDA 12.1                         |
| torchvision     | ≥ 0.15.2 (cần `DeformConv2d` cho DCNv2/v3) |
| ultralytics     | ≥ 8.0.0 (YOLOv8/v11)                        |
| albumentations  | ≥ 2.0 (API: `fill=`, `std_range=`)          |
| uv              | ≥ 0.4 (package manager)                     |
| GPU             | RTX 4060 8GB VRAM (khuyến nghị)             |

---

## Trích dẫn

```bibtex
@article{dpyolo2023,
  title   = {DP-YOLO: Improving YOLOv5 for Small Object Detection
             via Deformable Convolution and Parallel Transformer},
  journal = {Applied Sciences},
  year    = {2023}
}
@misc{yanivnik2021,
  title  = {Tuning CV Models with Reinforcement Learning},
  author = {Yanivnik et al.},
  year   = {2021},
  url    = {https://github.com/yanivnik/tuning_cv_models_with_rl_torch}
}
@misc{bwconrad2023,
  title  = {Fine-tuning Computer Vision Models with RL},
  author = {bwconrad},
  year   = {2023},
  url    = {https://github.com/bwconrad/cv-rl}
}
```
