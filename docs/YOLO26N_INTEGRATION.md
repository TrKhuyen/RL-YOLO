# Chạy YOLO26n cùng các model RL-YOLO

YOLO26n được đăng ký với tên `yolo26n` và trọng số COCO pretrained
`yolo26n.pt`. Các lệnh `--model all` của supervised training, KB1 và KB2
bao gồm model này; các runner chạy tuần tự. Chạy YOLO26n riêng nếu những
model khác đã có kết quả, vì supervised runner từ chối ghi đè run đã tồn tại.

Tài liệu tham khảo trong repo:
[Phân tích YOLO26n trên PlantDoc](Phan_tich_chi_tiet_YOLO26n_PlantDoc.md).
Tích hợp này dùng dữ liệu sạch của dự án tại `pre-data/data/v2i_cleanned`;
kết quả PlantDoc trong bài báo không phải kết quả của lần chạy này.

## Môi trường và head dự đoán

`pyproject.toml`, `requirements.txt` yêu cầu `ultralytics>=8.4.163`;
`uv.lock` khóa phiên bản 8.4.163 đã kiểm tra trong môi trường hiện tại.
Nếu cần đồng bộ môi trường, chạy `uv sync` từ thư mục gốc.

YOLO26 có head one-to-many với NMS và head one-to-one không dùng NMS.
Adapter KB1/KB2 chọn **one-to-many** để giữ tensor `[B, 4 + nc, anchors]`
với hộp `cxcywh` và xác suất lớp. KB1 dùng canonical NMS; KB2 dùng NMS
của Ultralytics và giữ gradient của các confidence được chọn. Pairwise/DPO
chấm các candidate trước NMS, dùng cùng head cho policy và reference.
Native loss vẫn giữ cả hai nhánh huấn luyện, không thay bằng loss YOLOv8.
Checkpoint đã fuse bỏ nhánh one-to-many không dùng được cho feedback.

YOLO26 không dùng DFL. Phiên bản đã kiểm tra trả thành phần regression
`l1_loss`; worker KB3 đọc cả `dfl_loss` của các model cũ và `l1_loss`
của YOLO26. Muốn benchmark NMS-free cần một thí nghiệm riêng với head,
ngưỡng và cách đo latency được ghi rõ.

Nguồn API: [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/).

## 1. Tạo supervised checkpoint

Chạy từ thư mục gốc trong PowerShell:

```powershell
.\.venv\Scripts\python.exe .\kb1_reward_guided_training\train_supervised.py --model yolo26n
```

Ultralytics tải `yolo26n.pt` nếu chưa có, nên lần đầu cần mạng.
Cấu hình giữ như baseline Ultralytics hiện tại: imgsz 640, tối đa 200 epoch,
patience 30, SGD, batch 16, GPU 0, cùng augmentation. Đây là cấu hình
so sánh của dự án; không phải tái lập recipe hay số liệu bài PlantDoc.
Batch 16 chưa được đo VRAM thực tế trên YOLO26n trong lần tích hợp này.

Checkpoint và provenance sau khi train thành công:

```text
kb1_reward_guided_training/checkpoint_based/yolo26n/weights/best.pt
```

## 2. Chạy KB1 đối chứng và reward-guided

Sau khi có supervised checkpoint, chạy hai mode với cùng seed và budget:

```powershell
.\.venv\Scripts\python.exe .\kb1_reward_guided_training\run_verified.py --model yolo26n --mode native_only --output .\kb1_reward_guided_training\checkpoint_reward_guide_trainning\screening_seed42_v1
.\.venv\Scripts\python.exe .\kb1_reward_guided_training\run_verified.py --model yolo26n --mode kb1b --output .\kb1_reward_guided_training\checkpoint_reward_guide_trainning\screening_seed42_v1
.\.venv\Scripts\python.exe .\kb1_reward_guided_training\evaluate.py --model YOLO26n --split val --checkpoint-source screening
```

CLI train dùng `yolo26n`; CLI evaluate dùng nhãn `YOLO26n`.
Các runner kiểm tra supervised provenance và cùng dataset trước khi train.
Chỉ dùng validation để chọn checkpoint; test dành cho đánh giá cuối.

## 3. Chạy KB2 feedback / pairwise / DPO

Trong Bash có sẵn (Git Bash hoặc WSL), từ thư mục gốc:

```bash
bash kb2_preference_optimization/run_all_kb2.sh --model yolo26n --check-only
bash kb2_preference_optimization/run_all_kb2.sh --model yolo26n --seed 42
```

Runner tạo feedback rồi chạy các biến thể baseline, sampling, hybrid,
pairwise và DPO. Phải tạo supervised checkpoint KB1 trước. Các CLI
`generate_feedback.py`, `train_feedback.py`, `run_ablation.py` và legacy
`train_rl.py` cũng nhận `--model yolo26n`.

## 4. Chạy KB3

Worker đã nhận đường dẫn model tùy ý; đặt `--model yolo26n.pt` hoặc đường
dẫn supervised `best.pt` vào `--trainer-command`. Các đối chứng trong cùng
phép so sánh phải dùng cùng checkpoint khởi tạo, seed và budget.

Ví dụ chuỗi worker trong PowerShell, dùng với CLI KB3 như hướng dẫn
[README KB3](../kb3_hyperparameter_optimization/README.md):

```powershell
$worker = '.\.venv\Scripts\python.exe -m kb3_hyperparameter_optimization.adapters.ultralytics_worker --model kb1_reward_guided_training/checkpoint_based/yolo26n/weights/best.pt --data kb1_reward_guided_training/configs/pest.yaml --total-epochs 100 --dataset-root pre-data/data/v2i_cleanned --canonical-eval'
```

## Kiểm chứng tích hợp

Test tạo kiến trúc `yolo26n.yaml` thật với trọng số ngẫu nhiên, ảnh 64×64
và nhãn tổng hợp, không tải pretrained hoặc train toàn bộ dataset.
Kiểm tra chuyển NMS-free sang one-to-many, kích thước output, gradient
confidence sau NMS, native loss hữu hạn và gradient của cả hai nhánh.
Test KB3 xác nhận cộng đúng DFL/L1 và báo lỗi nếu thiếu loss regression.

Cập nhật 07/10/2026: đã có supervised checkpoint YOLO26n sau 142 epoch.
Cả 7 baseline đã được đánh giá lại trên validation và test bằng canonical
evaluator; xem [báo cáo so sánh baseline](YOLO26N_BASELINE_COMPARISON.md).
Lần đánh giá này đo throughput FP32 batch 16 với NMS; chưa đo VRAM lúc
train hoặc latency triển khai NMS-free/batch 1.
