# KB2 — DPO cho toàn bộ kết quả detection

Thiết kế hiện hành từ **26/10/2026**: KB2 chỉ nghiên cứu DPO. KB1, KB2 và
KB3 độc lập, cùng lấy best checkpoint supervised của KB1 theo từng kiến trúc
làm điểm xuất phát và mốc so sánh. Không dùng checkpoint reward-guided/HPO
làm đầu vào KB2.

## Pipeline

```text
S0 supervised → policy và reference có cùng trọng số
Train image → policy phân phối class/background và bốn phía hộp DFL
            → lấy mẫu nhiều action hoàn chỉnh → decode/NMS cố định → A/B
Train GT → chấm toàn bộ A/B → chosen/rejected
Replay cùng ảnh, cùng action → native loss + λ DPO loss → cập nhật policy
Ordinary deterministic inference → canonical validation → best checkpoint
```

**Đã triển khai và hoàn tất run dài YOLOv8n seed 42/43/44 với cùng S0 và fixed preferences.** Phiên bản
`structured_detection_dpo_v1` học phân phối của *toàn bộ action có candidate
indices*, nhận preference từ chất lượng bộ detection. Nó không tuyên bố tính
likelihood biên của tập hộp không thứ tự sau NMS. DFL sampling là thiết kế mới
của dự án, không phải inference YOLO chuẩn hoặc bảo đảm bất định được hiệu chỉnh.

## Hỗ trợ và phạm vi

- YOLOv8n/s và YOLO11n/s có head DFL; chạy đầu tiên với YOLOv8n.
- YOLO26, YOLOv5, DP-YOLO chưa có policy tọa độ tương ứng trong bản này;
  CLI từ chối thay vì dùng confidence proxy.
- Chỉ train Detect head; backbone/neck, DFL integral và BN running stats cố định.
- Offline preferences chỉ trên train; input resize/pad xác định, không augmentation
  ngẫu nhiên khi replay. Native loss cũng dùng chính ảnh/GT đã replay đó.
- Reference đóng băng và chấm cùng batch với policy ở mỗi update/audit.
  Cached log-prob từ lúc sinh cặp chỉ dùng để kiểm tra replay/truy vết;
  không dùng làm mẫu số DPO khi train batch khác kích thước.
- Không hỗ trợ resume trong v1; checkpoint lưu optimizer/RNG phục vụ truy vết,
  không hứa exact resume khi chưa lưu/replay batch cursor.

## Chạy

Môi trường hiện có: `.venv-ubuntu` (Ultralytics 8.4.163). Các lệnh dưới dùng
GPU qua CUDA; runner cũng mặc định `--device cuda`. Từ repo root:

```bash
bash kb2_preference_optimization/run_all_kb2.sh --model yolov8n --device cuda --check-only

# Pilot: không phải kết quả full validation hoặc bằng chứng gain.
bash kb2_preference_optimization/run_all_kb2.sh \
  --model yolov8n --device cuda --output-root /tmp/kb2_pilot_NEW \
  --pilot-images 16 --image-size 192 --draws 4 \
  --steps 2 --batch-size 2 --eval-interval 1 --eval-limit 2

# Run nghiên cứu mới trên đầy đủ train/valid; chỉ chạy sau audit pilot.
bash kb2_preference_optimization/run_all_kb2.sh \
  --model yolov8n --device cuda \
  --output-root kb2_preference_optimization/outputs/yolov8n_s42_NEW \
  --steps 15000 --batch-size 4 --eval-interval 1000
```

Mỗi run cần thư mục mới, không ghi đè artifact. Không có `--variants` hoặc
`--level`: sampling/hybrid/candidate-pairwise không thuộc pipeline hiện tại.
Runner chạy một model mỗi lần để audit từng policy trước khi nhân rộng.

Các bước có thể chạy riêng, dùng `python -m` từ repo root:

```bash
.venv-ubuntu/bin/python -m kb2_preference_optimization.generate_preferences \
  --model yolov8n --device cuda --output /path/to/NEW_preferences

.venv-ubuntu/bin/python -m kb2_preference_optimization.train_dpo \
  --model yolov8n --device cuda --preferences /path/to/NEW_preferences \
  --output /path/to/NEW_training

.venv-ubuntu/bin/python -m kb2_preference_optimization.evaluate \
  --model yolov8n --device cuda --checkpoint /path/to/NEW_training/best.pt \
  --split valid --output /path/to/NEW_comparison.json
```

Chỉ chuyển `--split test` sau khi khóa phương pháp/checkpoint. Không dùng test
để chọn sampler, β, λ, quality weights hoặc confidence. Pilot bị chặn khỏi
evaluation test-final. Không đối chiếu trực tiếp AP lịch sử dùng NMS 0.45/cap100
với AP mới dùng canonical NMS 0.60/cap300; evaluator mới đo lại S0 và DPO.

## Tham số thử ban đầu

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| Class/box temperature | 1 / 1 | Thang categorical sampling, dùng cùng khi sample và score |
| Background logit | 0 | Lựa chọn background trong categorical C+1 |
| Draws/ảnh | 4 | Sinh nhiều output; lấy best/worst đủ gap |
| Minimum quality gap | 0.02 | Bỏ cặp gần hòa, không ép tạo pair |
| Detection/localization quality | 0.8 / 0.2 | Heuristic được version trong manifest |
| Native coefficient | 1 | Native detection loss |
| DPO weight λ / β | 0.01 / 0.1 | Giá trị pilot, **chưa được tối ưu** |
| AdamW LR / weight decay | 1e-6 / 5e-4 | Fine-tune head |
| Train batch / input | 4 / 640 | Full-size protocol |
| Clip norm | 1 | Toàn bộ tham số được học |
| Validation / patience | 1000 / 5 | Step 0 được tham gia chọn best; min delta 1e-4 |

Tổng log-prob là tổng trên action đầy đủ, không chia riêng theo số hộp chosen
hoặc rejected. Tăng độ phân giải có thể thay đổi thang log-ratio; cần audit β/λ.
Không suy λ=0.01 là DPO chiếm đúng 1% gradient. Pilot input640 cho thấy
λ=0.1 có weighted DPO/native gradient norm khoảng 1.87 ở batch đầu; vì vậy
mặc định thử được hạ xuống 0.01. Một batch chưa xác nhận strength toàn run.

## Metric và kiểm chứng

- Chọn checkpoint bằng **validation mAP50–95**.
- mAP50, mAP75, AR300, APs/m/l kèm area support, AP/AR/support theo lớp.
- P/R/F1 tại confidence 0.25, IoU 0.5; FP/ảnh, FN rate.
- Matched mean IoU kèm GT coverage và GT-normalized IoU; không bỏ qua FN.
- Fixed train audit: DPO loss, relative/absolute gap, chất lượng output lấy mẫu,
  số detection, empty rate. Đây không phải đánh giá holdout độc lập.
- Gradient norm/cosine native–DPO và tỷ lệ update bị clipping.
- `evaluation_wall_seconds` là thời gian evaluator, không phải FPS/latency thuần.
- Chạy nhiều seed continuation và báo paired delta với cùng S0 sau khi pilot đạt.

```bash
YOLO_CONFIG_DIR=/tmp MPLCONFIGDIR=/tmp/kb2-matplotlib \
  .venv-ubuntu/bin/python -m unittest discover -s kb2_preference_optimization/tests -v
```

Chi tiết xác suất, bộ chấm và giới hạn: [docs/DPO.md](docs/DPO.md).
Kết quả kiểm chứng kỹ thuật và phạm vi đã dọn:
[KB2_REDESIGN_20261026.md](../docs/report_final/KB2_REDESIGN_20261026.md).
Sửa reference cùng batch và kết quả pilot GPU:
[KB2_REFERENCE_BATCH_FIX_20261026.md](../docs/report_final/KB2_REFERENCE_BATCH_FIX_20261026.md).
Run dài YOLOv8n seed 42: early stop 14k, best 9k; validation mAP50–95
37,8084% → 38,2432%, precision tăng nhưng recall giảm. Đã đánh giá GPU lại
best/S0 trên đủ 479 ảnh; chưa đánh giá test. Xem
[KB2_FULL_RUN_REVIEW_20261026.md](../docs/report_final/KB2_FULL_RUN_REVIEW_20261026.md).
Ba continuation seed: best validation mAP50–95 trung bình **38,2523% ± 0,0429 điểm**,
gain **+0,4438 điểm** so S0. Precision/F1 tăng nhưng recall giảm ở cả ba seed;
lớp Corn Gray leaf spot (14 GT valid) đóng góp khoảng 95,33% net gain trung bình.
Đã đánh giá GPU lại cả ba best/S0. Xem
[KB2_MULTISEED_REVIEW_20261026.md](../docs/report_final/KB2_MULTISEED_REVIEW_20261026.md).
Sau sinh preference, thư mục `preferences/previews/` có tối đa 8 ảnh so sánh
GT/chosen/rejected để kiểm tra trực quan trước khi train dài.
Các artifact cũ đã dọn theo yêu cầu: [legacy/cleanup_manifest.json](legacy/cleanup_manifest.json).
Source và bảng số nhỏ cũ được giữ trong `legacy/`; không dùng chúng làm kết quả
của objective mới. Checkpoint KB1 và dữ liệu chung không bị dọn.
