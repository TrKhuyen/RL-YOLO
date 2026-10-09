# KB1: Reward-guided YOLO Training

## Chạy toàn bộ KB1 bằng một lệnh

Từ root workspace trong Git Bash:

```bash
bash kb1_reward_guided_training/run_all_kb1.sh
```

Script chạy audit dữ liệu → supervised sáu model → preflight và screening hai nhánh → đánh giá test → tạo `kb1_reward_guided_training/docs/KB1_FINAL_CLEAN_REPORT.md`. Checkpoint supervised hợp lệ sẽ được dùng lại; run thiếu hoặc sai manifest làm script dừng, không tự ghi đè. Để chỉ kiểm tra dataset mà không train:

```bash
bash kb1_reward_guided_training/run_all_kb1.sh --check-only
```

Trên PowerShell khi `bash` chưa có trong PATH: `& 'C:/Program Files/Git/bin/bash.exe' kb1_reward_guided_training/run_all_kb1.sh`.


## Quy trình KB1 dùng cho lần chạy sạch

Dữ liệu chính là `../pre-data/data/v2i_cleanned`; augmentation theo hậu tố flip/rotation đã biết chỉ nằm trong train. Kết quả nghiên cứu dùng `run_screening.py` và `run_verified.py`; phần hướng dẫn `train_rl.py` phía dưới là đường chạy cũ, không dùng để tổng hợp kết quả chính.

1. Từ root workspace: `python kb1_reward_guided_training/audit_splits.py`.
2. Từ `kb1_reward_guided_training`: `python train_supervised.py`. Cần đủ sáu `checkpoint_based/<model>/weights/best.pt` kèm `supervised_manifest.json`.
3. Cùng thư mục: `python run_screening.py`. Script chạy preflight rồi guided/native-only; checkpoint ở `checkpoint_reward_guide_trainning/screening_seed42_v1/<model>/<mode>/best.pt`.
4. Sau khi đã khóa checkpoint bằng validation: `python evaluate.py --checkpoint-source screening --split test`.

Metric chính là mAP50_95; ứng viên xếp hạng là best guided của sáu model. Stage hai sẽ dừng nếu checkpoint supervised hoặc dữ liệu không trùng manifest.


> Môi trường Python dùng chung gồm .venv, pyproject.toml, uv.lock, requirements.txt và .gitignore nằm tại root RL-YOLO. Chạy uv sync và kích hoạt .venv từ root. Guide và báo cáo KB1-B canonical nằm trong docs.

# YOLO-RL-Pest: UAV Pest Detection với RL Fine-tuning

So sánh **DP-YOLO vs YOLOv5s/v8n/v8s/v11n/v11s** trên bài toán phát hiện sâu bệnh cây trồng từ ảnh UAV,  
với **reward-guided surrogate fine-tuning** để cải thiện recall trên vật thể nhỏ.

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
├── train_rl.py            # Giai đoạn 2: reward-guided surrogate fine-tune (10k steps)
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
Checkpoint được lưu tại `checkpoint_based/<model>/weights/best.pt`.

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
| yolo26n  | 16    | Ultralytics  | YOLO26 nano, regression không dùng DFL |
| dp_yolo  | 4     | YOLOv5+patch | **Main model** (D2C3/D3C3) |

> **DP-YOLO:** `dp_yolo_train.py` tự động apply patches (W3F_MPDIoU, PSA, custom modules)  
> trước khi gọi `yolov5/train.py`. Không cần cấu hình thêm.

Theo dõi training:

```bash
tensorboard --logdir checkpoint_based
```

---

### Giai đoạn 2 – Reward-guided fine-tuning

Chạy `python run_screening.py` sau khi đã có sáu checkpoint supervised kèm manifest. Runner chạy preflight 2 step cho từng model/mode, rồi `native_only` và `kb1b` từ cùng supervised best. Objective guided gồm native detection loss, reward-guided confidence surrogate, TP/FP/FN proxy và L2-SP. Chỉ Detect head được cập nhật.

Checkpoint được lưu tại `checkpoint_reward_guide_trainning/screening_seed42_v1/<model>/<mode>/best.pt`. Validation mAP50_95 chọn best; step 0 cũng là một ứng viên. Xem trạng thái và chênh lệch từng model trong `docs/KB1_SCREENING_CLEAN_REPORT.md`. Khi preflight lỗi, xem `console.log` trong thư mục preflight tương ứng.

`train_rl.py` vẫn có thể dùng cho thử nghiệm cũ nhưng không tạo bộ kết quả chính của báo cáo KB1.

---

### Giai đoạn 3 – Đánh giá cuối

Sau khi đã khóa best theo validation, chạy `python evaluate.py --checkpoint-source screening --split test`. So sánh guided với supervised và native-only cho từng model. Dùng test để báo cáo, không dùng để chọn lại checkpoint hoặc hệ số loss.

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
| Epochs/Steps   | 200         | 10,000       |
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
