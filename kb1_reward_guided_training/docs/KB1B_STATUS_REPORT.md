# Báo cáo trạng thái KB1-B: Reward-guided Weight Tuning

Cập nhật: 2026-08-19

## 1. Mục tiêu

KB1-B fine-tune trực tiếp trọng số YOLO từ checkpoint supervised bằng một
objective lai. Native Ultralytics detection loss giữ năng lực định vị và phân
loại, trong khi reward TP/FP/FN hướng cập nhật tới precision, recall và IoU.

Đây là reward-guided weight tuning, không phải DPO và không phải REINFORCE
chuẩn vì YOLO không lấy mẫu action từ một policy distribution tường minh.

## 2. Bài toán và dữ liệu

- Bài toán: phát hiện và phân loại tình trạng sâu bệnh trên lá.
- Annotation: mỗi bounding box là một lá.
- Train: 2.072 ảnh.
- Validation: 592 ảnh, 2.247 instances.
- Input: RGB float trong miền [0, 1].
- Checkpoint khởi đầu: supervised best của YOLOv11n.
- Mất cân bằng lớp đã được giảm từ khoảng 9-10 lần xuống khoảng 3 lần.

Vì box biểu diễn toàn bộ lá, KB1-B không dùng small-object reward cho đốm bệnh.

## 3. Cấu hình KB1-B v3.1

Objective:

    L_total =
        0.25 * L_native_YOLO
      + 0.10 * L_reward
      + 0.25 * L_TP_FP_FN_proxy
      + 0.001 * L2-SP

Native YOLO loss gồm box regression, classification và DFL. Reward detection
gồm precision 0.25, recall 0.45 và matched IoU 0.30 tại các ngưỡng IoU
0.5, 0.6, 0.7 và 0.8.

Thông số chính:

- Learning rate cực đại: 5e-7.
- Learning rate cực tiểu: 1e-7.
- Warmup: 500 step.
- Scheduler: cosine.
- Batch size: 8.
- Gradient accumulation: 2, batch hiệu dụng 16.
- Chỉ cập nhật Detect head.
- Validation mỗi 500 step.
- Early-stopping patience: 4.
- Minimum validation improvement: 0.0005.
- Seed đánh giá: 42, 43 và 44.

Validation score dùng để chọn checkpoint:

    0.4 * mAP50 + 0.4 * mAP50-95 + 0.2 * recall

Reward-best được tách khỏi validation-best và không dùng làm kết quả cuối.

## 4. Thiết kế đối chứng

Ba nhóm được so sánh:

1. Supervised: checkpoint tốt nhất trước fine-tune.
2. Native-only: continued fine-tuning với native loss và L2-SP; reward và
   TP/FP/FN proxy có trọng số bằng 0.
3. KB1-B v3.1: native loss, reward, TP/FP/FN proxy và L2-SP.

Native-only và KB1-B dùng cùng checkpoint khởi đầu, dữ liệu, learning-rate
schedule, batch size, số step tối đa, validation interval và early stopping.

## 5. Kết quả validation

### 5.1. KB1-B theo seed

| Seed | Best step | mAP50 | mAP50-95 | Recall | Validation score |
|---:|---:|---:|---:|---:|---:|
| 42 | 7000 | 0.59347 | 0.46118 | 0.61043 | 0.54395 |
| 43 | 5000 | 0.59246 | 0.45986 | 0.60761 | 0.54245 |
| 44 | 7500 | 0.59406 | 0.46181 | 0.61151 | 0.54465 |

KB1-B mean ± sample standard deviation:

| Metric | Mean ± std |
|---|---:|
| mAP50 | 0.59333 ± 0.00081 |
| mAP50-95 | 0.46095 ± 0.00100 |
| Recall | 0.60985 ± 0.00201 |
| Validation score | 0.54368 ± 0.00112 |

### 5.2. Native-only theo seed

| Seed | Best step | mAP50 | mAP50-95 | Recall | Validation score |
|---:|---:|---:|---:|---:|---:|
| 42 | 3500 | 0.59004 | 0.45599 | 0.60855 | 0.54012 |
| 43 | 5000 | 0.58972 | 0.45576 | 0.60741 | 0.53968 |
| 44 | 5500 | 0.59030 | 0.45622 | 0.60950 | 0.54051 |

Native-only mean ± sample standard deviation:

| Metric | Mean ± std |
|---|---:|
| mAP50 | 0.59002 ± 0.00029 |
| mAP50-95 | 0.45599 ± 0.00023 |
| Recall | 0.60849 ± 0.00104 |
| Validation score | 0.54010 ± 0.00042 |

### 5.3. So sánh tổng hợp

| Phương pháp | mAP50 | mAP50-95 | Recall |
|---|---:|---:|---:|
| Supervised | 0.5882 | 0.4525 | 0.6065 |
| Native-only | 0.59002 ± 0.00029 | 0.45599 ± 0.00023 | 0.60849 ± 0.00104 |
| KB1-B v3.1 | 0.59333 ± 0.00081 | 0.46095 ± 0.00100 | 0.60985 ± 0.00201 |

Chênh lệch ghép cặp KB1-B trừ native-only:

| Metric | Mean difference ± std |
|---|---:|
| mAP50 | +0.00331 ± 0.00053 |
| mAP50-95 | +0.00496 ± 0.00077 |
| Recall | +0.00136 ± 0.00101 |
| Validation score | +0.00358 ± 0.00072 |

KB1-B vượt native-only về mAP50 và mAP50-95 ở cả ba seed. Kết quả này cho
thấy reward mang lại lợi ích bổ sung ngoài continued supervised fine-tuning.

## 6. Checkpoint

KB1-B validation-best:

- yolov11n_seed42_rl_best.pt
- yolov11n_seed43_rl_best.pt
- yolov11n_seed44_rl_best.pt

Native-only validation-best:

- yolov11n_seed42_native_only_best.pt
- yolov11n_seed43_native_only_best.pt
- yolov11n_seed44_native_only_best.pt

Tất cả nằm trong thư mục rl_checkpoints của kịch bản KB1.

## 7. Kết luận hiện tại

KB1-B v3.1 cho cải thiện nhất quán trên validation qua ba seed. Native-only
cũng cải thiện so với supervised, nhưng KB1-B tiếp tục tăng trung bình khoảng
0.33 điểm phần trăm mAP50 và 0.50 điểm phần trăm mAP50-95 so với native-only.

Cách diễn đạt phù hợp:

    KB1-B produced a consistent validation improvement over matched
    continued-supervised fine-tuning across three random seeds.

Không nên tuyên bố statistical significance chỉ từ ba seed.

## 8. Giới hạn và việc còn lại

- Các số trên dùng quick evaluator trong vòng train, chưa phải native test.
- Cần chạy native Ultralytics validator trên held-out test set cho 7
  checkpoint: supervised và sáu checkpoint fine-tuned.
- Cần báo cáo per-class AP, macro-F1, precision, latency và tham số/FLOPs.
- Cần kiểm tra class khó và confusion matrix.
- Không tiếp tục chỉnh hyperparameter theo validation hiện tại để tránh
  validation overfitting.
- YOLOv5 và DP-YOLO chưa có native ComputeLoss trong KB1-B v3.1.

## 9. Lệnh tái lập

KB1-B mặc định:

    python .\kb1_reward_guided_training\train_rl.py --model yolov11n --seed 42

Native-only:

    python .\kb1_reward_guided_training\train_rl.py --model yolov11n --seed 42 --mode native-only

Thay seed 42 lần lượt bằng 43 và 44 để tái lập đầy đủ.
