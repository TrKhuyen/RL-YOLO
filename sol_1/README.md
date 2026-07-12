# YOLO-RL-Pest: UAV Pest Detection với RL Fine-tuning

So sánh **DP-YOLO vs YOLOv5s/v8n/v8s/v11n/v11s** trên bài toán phát hiện sâu bệnh lúa từ UAV,  
với **REINFORCE fine-tuning** để cải thiện recall trên vật thể nhỏ.

---

## Cấu trúc project

```
yolo-rl-pest/
├── configs/
│   ├── pest.yaml          # dataset config (nc=10)
│   ├── hyp.pest.yaml      # hyperparams supervised
│   └── hyp.rl.yaml        # hyperparams RL
├── models/
│   └── dp_yolo/
│       ├── dp_yolo.yaml   # architecture YAML
│       ├── modules.py     # D2C3, D3C3, PTCSP, C3Ghost, DCNv2, DCNv3
│       └── patch_yolov5.py
├── adapters/
│   ├── yolov5_adapter.py  # YOLOv5 + DP-YOLO (gradient qua confidence)
│   └── ultralytics_adapter.py  # YOLOv8/v11
├── dataloader.py          # PestDataset + collate
├── reward.py              # recall_reward, composite_reward
├── train_supervised.py    # Giai đoạn 1: supervised (300 epochs)
├── train_rl.py            # Giai đoạn 2: REINFORCE fine-tune (50k steps)
├── evaluate.py            # Giai đoạn 3: mAP/APs/recall/FPS comparison
├── requirements.txt
└── data/
    └── pest/
        ├── train/
        │   ├── images/
        │   └── labels/    # YOLO format (class cx cy w h, normalized)
        ├── val/
        └── test/
```

---

## Cài đặt

```bash
pip install -r requirements.txt

# Cài YOLOv5 (dùng cho DP-YOLO và YOLOv5s baseline)
git clone https://github.com/ultralytics/yolov5.git
pip install -r yolov5/requirements.txt
```

---

## 10 Class sâu bệnh

| ID | Class          | Mô tả                     |
|----|----------------|---------------------------|
| 0  | sau_duc_than   | Sâu đục thân              |
| 1  | sau_cuon_la    | Sâu cuốn lá               |
| 2  | benh_dom_nau   | Bệnh đốm nâu              |
| 3  | benh_dao_on    | Bệnh đạo ôn               |
| 4  | benh_kho_van   | Bệnh khô vằn              |
| 5  | ran_xanh       | Rầy xanh                  |
| 6  | bo_tri         | Bọ trĩ                    |
| 7  | ran_nau        | Rầy nâu                   |
| 8  | dom_la         | Đốm lá                    |
| 9  | healthy        | Lúa khỏe (không bệnh)     |

---

## Quy trình 3 giai đoạn

### Giai đoạn 1 – Supervised Training

```bash
# Train tất cả model (YOLOv5s, YOLOv8n/s, YOLOv11n/s, DP-YOLO)
python train_supervised.py

# Chỉ train DP-YOLO
python train_supervised.py --model dp_yolo

# Chỉ train YOLOv8n
python train_supervised.py --model yolov8n
```

> **Lưu ý DP-YOLO**: trước khi train, patch các module tùy chỉnh vào YOLOv5:
> ```bash
> python models/dp_yolo/patch_yolov5.py
> ```

### Giai đoạn 2 – RL Fine-tuning

```bash
# Fine-tune tất cả (từ checkpoints giai đoạn 1)
python train_rl.py

# Chỉ fine-tune DP-YOLO với 30k steps
python train_rl.py --model dp_yolo --steps 30000

# Fine-tune với backbone frozen (tránh catastrophic forgetting)
python train_rl.py --model yolov8n --freeze

# Override learning rate
python train_rl.py --model dp_yolo --lr 5e-7
```

**Reward function** (composite mặc định):
$$R = 0.6 \times R_{\text{recall}} + 0.4 \times R_{\text{small}}$$

### Giai đoạn 3 – Evaluation

```bash
# So sánh tất cả model (supervised vs RL)
python evaluate.py

# Chỉ evaluate DP-YOLO
python evaluate.py --model DP-YOLO

# Dùng tập test
python evaluate.py --split test
```

Kết quả được lưu tại:
- `results/tables/results_full.csv`
- `results/tables/results_delta.csv`
- `results/tensorboard/` (TensorBoard logs)

---

## Ma trận thí nghiệm

| Exp | Model      | Stage       | Mục tiêu                          |
|-----|-----------|-------------|-----------------------------------|
| E01 | YOLOv5s   | Supervised  | Baseline                          |
| E02 | YOLOv8n   | Supervised  | Anchor-free nhẹ                   |
| E03 | YOLOv8s   | Supervised  | Anchor-free nặng hơn              |
| E04 | YOLOv11n  | Supervised  | Kiến trúc mới nhất                |
| E05 | YOLOv11s  | Supervised  | YOLOv11 lớn hơn                   |
| E06 | DP-YOLO   | Supervised  | **Main model**                    |
| E07 | YOLOv5s   | RL Fine-tune | RL trên baseline                 |
| E08 | YOLOv8n   | RL Fine-tune | RL trên anchor-free               |
| E09 | YOLOv11n  | RL Fine-tune | RL trên latest                    |
| E10 | DP-YOLO   | RL Fine-tune | **Main contribution**             |

---

## Kiến trúc DP-YOLO

```
Backbone:
  Stage 1-3 → D2C3 (DCNv2 deformable conv)  
  Stage 4   → D3C3 (DCNv3, chiến lược "3+1")

Neck:
  C3Ghost (thay C3 thường) + PTCSP tại P2 (CNN ‖ Transformer)

Head:
  P2 (stride=4) + P3 (stride=8) + P4 (stride=16)
  [Bỏ P5, thêm P2 để bắt object nhỏ trên ảnh UAV]

Loss:
  W3F_MPDIoU = MPDIoU + Focaler-IoU + WIoU v3

Label Assignment:
  PSA (Petal-like Sample Amplification, r=1)
```

---

## Hyperparameters quan trọng

| Param           | Supervised | RL Fine-tune |
|-----------------|-----------|--------------|
| Epochs/Steps    | 300       | 50,000       |
| Learning Rate   | 0.01      | 1e-6         |
| Batch Size      | 32        | 16           |
| EMA Baseline α  | –         | 0.99         |
| Conf Threshold  | 0.25      | 0.20         |
| Reward α        | –         | 0.60 (recall) / 0.40 (small) |

---

## Yêu cầu

- Python ≥ 3.10
- PyTorch ≥ 2.0 + CUDA 11.8
- torchvision ≥ 0.15 (cần DeformConv2d cho DP-YOLO)
- ultralytics ≥ 8.0
- See `requirements.txt` for full list

---

## Trích dẫn

```bibtex
@article{dpyolo2024,
  title  = {DP-YOLO: A Lightweight YOLOv5-Based Object Detection Model},
  year   = {2024}
}
@article{yanivnik2021,
  title  = {Object Detection with RL},
  author = {Yanivnik et al.},
  year   = {2021}
}
@inproceedings{bwconrad2023,
  title  = {Fine-tuning Computer Vision Models with RL},
  year   = {2023}
}
```
