# BÁO CÁO KỊCH BẢN 1: HUẤN LUYỆN YOLO HAI GIAI ĐOẠN

> **Cập nhật sau rà soát code:** xem [Báo cáo kiểm chứng và screening seed 42](KB1_SCREENING_REPORT.md) để đọc trạng thái và kết quả mới nhất. Các bảng bên dưới là kết quả lịch sử. Evaluator mới sửa clipping nhãn/letterbox và dùng AR300; không so trực tiếp điểm cũ với điểm mới để tính mức tăng do train. Việc mở khóa DFL cũng đã được sửa. Chưa đủ metadata để khẳng định checkpoint DP-YOLO baseline từng bật W3F/PSA; screening mới ghi rõ hai cờ này OFF. Tập test sẽ dùng sau khi chốt cấu hình, với cùng canonical evaluator.

## Supervised Baseline Training và Reward-Guided Fine-Tuning

> **Phiên bản rà soát:** 15/09/2026  
> **Phạm vi:** toàn bộ phần triển khai riêng trong `kb1_reward_guided_training`  
> **Trạng thái:** đã hoàn thành huấn luyện supervised cho 6 mô hình; đã hoàn thành thí nghiệm KB1-B trên YOLOv11n với 3 seed ở tập validation; chưa có bảng đánh giá native trên tập test cho các checkpoint KB1-B.

## 1. Tóm tắt điều hành

Kịch bản 1 là **một quy trình thực nghiệm nối tiếp gồm hai giai đoạn**:

1. **KB1-A — Supervised Baseline Training:** huấn luyện có giám sát năm baseline YOLOv5s, YOLOv8n, YOLOv8s, YOLOv11n, YOLOv11s và mô hình đề xuất DP-YOLO. Mỗi mô hình tạo ra một `best.pt` riêng.
2. **KB1-B — Reward-Guided Fine-Tuning:** lấy **từng best checkpoint tương ứng của KB1-A** làm điểm khởi đầu, sau đó tiếp tục tối ưu trọng số bằng objective có hướng dẫn bởi reward.

Thiết kế đúng không phải là chọn một mô hình thắng cuộc duy nhất sau KB1-A. Mỗi mô hình tạo thành một cặp so sánh độc lập:

```text
Model M sau KB1-A: best_supervised(M)
                    │
                    └── KB1-B(M) → best_reward_guided(M)

So sánh chính: best_supervised(M) ↔ best_reward_guided(M)
```

Sau khi hoàn thành KB1-B cho toàn bộ mô hình, kết quả cuối phải trả lời hai câu hỏi: reward-guided fine-tuning cải thiện từng mô hình bao nhiêu, và mô hình nào đạt kết quả cuối tốt nhất sau cả hai giai đoạn.

Kết quả validation hiện có cho thấy KB1-B v3.1 tốt hơn đối chứng `native-only` trên cả ba seed 42, 43 và 44. Trung bình, KB1-B tăng **0,00331 mAP50**, **0,00496 mAP50-95** và **0,00136 recall** so với continued supervised fine-tuning có cùng ngân sách tính toán.

Kết luận này chỉ áp dụng cho **YOLOv11n trên tập validation**. Chưa đủ bằng chứng để khẳng định hiệu quả trên tập test, trên toàn bộ sáu kiến trúc hoặc có ý nghĩa thống kê rộng hơn.

### 1.1. Quy ước tên của KB1

Tên cũ “Reward-guided YOLO Training” dễ làm người đọc hiểu rằng toàn bộ KB1 chỉ là RL hoặc KB1-A và KB1-B là hai nghiên cứu tách rời. Báo cáo này sử dụng tên rõ nghĩa hơn:

- **Tên toàn bộ KB1:** *Two-Stage YOLO Training: Supervised Baselines and Reward-Guided Fine-Tuning* — *Huấn luyện YOLO hai giai đoạn: baseline có giám sát và tinh chỉnh có hướng dẫn bởi phần thưởng*.
- **KB1-A:** *Supervised Baseline Training* — *Huấn luyện có giám sát và xác lập baseline*.
- **KB1-B:** *Per-Model Reward-Guided Fine-Tuning* — *Tinh chỉnh từng mô hình có hướng dẫn bởi phần thưởng*.

Không nên đặt tên KB1-B là DPO. Cũng không nên gọi ngắn gọn là “REINFORCE fine-tuning” mà không kèm giới hạn phương pháp, vì implementation dùng confidence-based policy surrogate và prediction xác định sau NMS. Cách gọi phù hợp nhất trong báo cáo là **reward-guided fine-tuning (RL-inspired)**.

## 2. Nguồn thông tin và nguyên tắc đối chiếu

Bản báo cáo này ưu tiên nguồn theo thứ tự:

1. mã nguồn và cấu hình đang chạy;
2. metadata trong checkpoint và CSV huấn luyện;
3. báo cáo trạng thái KB1-B;
4. các guide và tài liệu phân tích cũ.

Nguyên tắc trên là cần thiết vì tài liệu cũ còn một số mô tả không khớp phiên bản hiện tại, chẳng hạn: 10 lớp thay vì 28 lớp, chưa có dữ liệu, DP-YOLO chỉ dùng ba detection head, hoặc gọi toàn bộ quy trình là REINFORCE thuần.

## 3. Bài toán và dữ liệu

### 3.1. Bài toán

Mỗi ảnh có thể chứa nhiều lá. Mỗi bounding box bao quanh một lá và nhãn lớp biểu diễn loại lá hoặc tình trạng bệnh. Vì annotation nằm ở mức **toàn bộ lá**, kích thước đốm bệnh bên trong lá không đồng nghĩa với kích thước bounding box nhỏ.

Hệ quả đối với thiết kế KB1-B:

- reward chính không dùng `small-object recall`;
- reward mặc định là `detection_composite`, cân bằng precision, recall và chất lượng định vị;
- không nên diễn giải kết quả như một thí nghiệm phát hiện trực tiếp từng đốm bệnh nhỏ.

### 3.2. Cấu trúc và quy mô dữ liệu

Dataset được cấu hình tại `pre-data/data/v2i_cleanned` và dùng định dạng YOLO: `class_id cx cy width height`, với tọa độ chuẩn hóa về `[0, 1]`.

| Split | Số ảnh | Số tệp nhãn | Ghi chú |
|---|---:|---:|---|
| Train | 2.072 | 2.072 | dùng cho supervised training và KB1-B |
| Validation | 592 | 592 | báo cáo cũ ghi nhận 2.247 instances |
| Test | 296 | 296 | đã tồn tại nhưng chưa có bảng native test cuối cho KB1-B |

Dataset có **28 lớp**, gồm các nhóm Apple, Bell pepper, Blueberry, Cherry, Corn, Peach, Potato, Raspberry, Soyabean, Squash, Strawberry, Tomato và Grape. Danh sách chính xác được khai báo trong `configs/pest.yaml`.

### 3.3. Tiền xử lý cho KB1-B

Custom dataloader thực hiện:

- giữ đúng RGB;
- resize theo cạnh dài và pad về `640 × 640`;
- chuẩn hóa pixel về `[0, 1]`, không dùng ImageNet mean/std;
- biến đổi đồng thời ảnh và bounding box;
- augmentation train gồm thay đổi sáng/tương phản, hue-saturation, noise, motion blur, flip, rotate và CLAHE;
- validation chỉ resize, pad và chuẩn hóa.

Đầu ra là tensor ảnh `B × 3 × H × W` và danh sách target dạng `boxes xyxy` tuyệt đối cùng `labels`.

## 4. Quy trình thực nghiệm tổng thể

```text
Dataset 28 lớp
   │
   ├── KB1-A: Supervised Baseline Training
   │      ├── YOLOv5s  → best.pt ──→ KB1-B(YOLOv5s)
   │      ├── YOLOv8n  → best.pt ──→ KB1-B(YOLOv8n)
   │      ├── YOLOv8s  → best.pt ──→ KB1-B(YOLOv8s)
   │      ├── YOLOv11n → best.pt ──→ KB1-B(YOLOv11n)
   │      ├── YOLOv11s → best.pt ──→ KB1-B(YOLOv11s)
   │      └── DP-YOLO  → best.pt ──→ KB1-B(DP-YOLO)
   │
   └── Đánh giá cuối bằng cùng evaluator và cùng test split
          ├── so sánh ghép cặp: sau KB1-B − sau KB1-A
          ├── đối chứng native-only với cùng compute budget
          └── xếp hạng kết quả cuối giữa sáu mô hình
```

## 5. KB1-A: Supervised Baseline Training

### 5.1. Thiết lập chung

Năm baseline chuẩn và DP-YOLO được huấn luyện độc lập ở kích thước ảnh 640, tối đa 200 epoch, patience 30 và optimizer SGD. Mỗi mô hình lưu checkpoint supervised tốt nhất của chính nó tại `checkpoints/<model>/weights/best.pt`. Sáu checkpoint này là sáu đầu vào tương ứng cho KB1-B; chúng không được dùng để chọn duy nhất một mô hình đi tiếp.

| Mô hình | Batch size | Framework |
|---|---:|---|
| YOLOv5s | 16 | YOLOv5 |
| YOLOv8n | 16 | Ultralytics |
| YOLOv8s | 8 | Ultralytics |
| YOLOv11n | 16 | Ultralytics |
| YOLOv11s | 8 | Ultralytics |
| DP-YOLO | 4 | YOLOv5 + runtime patch |

Trên Windows, `workers=0`; trên hệ điều hành khác, `workers=4`.

### 5.2. Kiến trúc DP-YOLO hiện tại

DP-YOLO dùng `depth_multiple=0.33`, `width_multiple=0.50` và có bốn detection scale:

- **P2/4:** hỗ trợ đặc trưng chi tiết;
- **P3/8** và **P4/16:** đặc trưng trung gian;
- **P5/32:** được giữ lại vì dataset có nhiều bounding box lá lớn.

Đây là điểm quan trọng: cấu hình hiện tại dùng `Detect([P2, P3, P4, P5])`, không phải cấu hình ba head P2–P4 được mô tả trong một số tài liệu cũ.

Các thành phần chính:

- `D2C3` sử dụng DCNv2 trong ba stage đầu của backbone;
- `D3C3` dùng chiến lược “3+1” ở stage sâu;
- `PTCSP` kết hợp nhánh CNN và Transformer;
- `C3Ghost` giảm chi phí ở neck;
- PAN top-down/bottom-up tổng hợp đặc trưng đa tỉ lệ.

`DCNv3` trong repository là phiên bản giản lược dựa trên `torchvision.ops.DeformConv2d`, dùng shared offset thay vì per-group offset như DCNv3 nguyên bản. Vì vậy, trong luận văn nên gọi rõ là **simplified DCNv3 with shared offset**.

### 5.3. W3F_MPDIoU

DP-YOLO thay gradient hồi quy box của nhánh CIoU bằng W3F_MPDIoU:

```text
L_W3F = r × R_WIoU × L_F_MPDIoU

L_MPDIoU   = 1 − IoU + (d_top-left² + d_bottom-right²) / c²
L_F_MPDIoU = L_MPDIoU + IoU − IoU^γ, với γ = 0,5
```

Trong runtime patch, giá trị forward vẫn là CIoU để objectness target của YOLOv5 giữ miền giá trị phù hợp, còn gradient backward đi qua W3F surrogate. Cách triển khai này tránh dùng giá trị W3F không bị chặn làm objectness quality target.

### 5.4. PSA label assignment

PSA thay `ComputeLoss.build_targets` của YOLOv5. Với mỗi GT, thuật toán xét lưới `3 × 3` quanh cell chứa tâm GT và giữ các cell có tâm nằm trong bán kính 1 grid unit, đồng thời vẫn áp dụng anchor-ratio test và kiểm tra biên.

Mục tiêu là mở rộng tập positive candidate so với label assignment mặc định. Con số “tăng khoảng 5% positive” là mô tả tham khảo từ tài liệu DP-YOLO, chưa phải thống kê đã đo lại trên dataset này.

### 5.5. Kết quả supervised đã ghi log

Bảng dưới lấy hàng có `mAP50-95` cao nhất trong `results.csv` của từng run. Đây là **validation metric do framework ghi trong quá trình supervised training**, không phải held-out test.

| Mô hình | Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv5s | 162 | 0,65223 | 0,60629 | 0,63608 | 0,48774 |
| YOLOv8n | 183 | 0,63692 | 0,60160 | 0,62875 | 0,49550 |
| YOLOv8s | 190 | 0,62778 | 0,61346 | 0,62192 | 0,50940 |
| YOLOv11n | 155 | 0,57823 | 0,63262 | 0,62734 | 0,49002 |
| YOLOv11s | 190 | 0,70274 | 0,57479 | 0,62960 | **0,51498** |
| DP-YOLO | 199 | 0,56718 | 0,58646 | 0,58916 | 0,40238 |

Trong các run supervised đã có, YOLOv11s đạt mAP50-95 cao nhất. DP-YOLO chưa vượt các baseline. Vì chưa có ablation riêng cho từng module, chưa thể xác định thành phần nào của DP-YOLO có lợi hoặc gây suy giảm.

## 6. KB1-B: Per-Model Reward-Guided Fine-Tuning

### 6.1. Phạm vi và cách gọi đúng

Với mỗi model `M`, KB1-B nạp `checkpoints/M/weights/best.pt` của KB1-A và cập nhật tiếp trọng số của model đó. Đầu ra được so sánh với đúng checkpoint khởi đầu cùng model. Đây là **reward-guided fine-tuning**, không phải DPO và cũng không phải REINFORCE chuẩn theo nghĩa chặt.

Ma trận so sánh mục tiêu của toàn bộ KB1:

| Mô hình | Kết quả sau KB1-A | Kết quả sau KB1-B | So sánh ghép cặp cần báo cáo |
|---|---|---|---|
| YOLOv5s | `yolov5s/.../best.pt` | `yolov5s_*_rl_best.pt` | KB1-B − KB1-A của YOLOv5s |
| YOLOv8n | `yolov8n/.../best.pt` | `yolov8n_*_rl_best.pt` | KB1-B − KB1-A của YOLOv8n |
| YOLOv8s | `yolov8s/.../best.pt` | `yolov8s_*_rl_best.pt` | KB1-B − KB1-A của YOLOv8s |
| YOLOv11n | `yolov11n/.../best.pt` | `yolov11n_*_rl_best.pt` | KB1-B − KB1-A của YOLOv11n |
| YOLOv11s | `yolov11s/.../best.pt` | `yolov11s_*_rl_best.pt` | KB1-B − KB1-A của YOLOv11s |
| DP-YOLO | `dp_yolo/.../best.pt` | `dp_yolo_*_rl_best.pt` | KB1-B − KB1-A của DP-YOLO |

Dấu `*` biểu diễn seed. Khi báo cáo nhiều seed, cần lấy mean ± sample standard deviation của delta ghép cặp cho từng mô hình.

Lý do:

- YOLO sinh dự đoán xác định sau threshold và NMS;
- không có policy distribution tường minh để lấy mẫu action;
- box, label và quyết định NMS bị detach;
- gradient reward chủ yếu đi qua confidence/class scores.

Do đó, `log(confidence)` chỉ là một policy proxy khả vi. Báo cáo không nên tuyên bố rằng KB1-B trực tiếp tối ưu tọa độ box bằng policy gradient.

### 6.2. Reward detection

Prediction được greedy-match một-một với GT cùng lớp tại bốn ngưỡng IoU: 0,50; 0,60; 0,70; 0,80.

Tại mỗi ngưỡng:

```text
R_t = 0,25 × precision_t
    + 0,45 × recall_t
    + 0,30 × mean_matched_IoU_t

R = mean(R_t) qua bốn ngưỡng IoU
```

Recall có trọng số cao nhất vì bỏ sót lá bệnh là lỗi cần hạn chế. Reward được detach trước khi đi vào objective.

### 6.3. Confidence surrogate theo TP/FP/FN

Sau matching cùng lớp tại IoU 0,5:

- TP được khuyến khích tăng confidence qua `log(score)`;
- FP được khuyến khích giảm confidence qua `log(1 − score)`;
- GT bị bỏ sót tạo tín hiệu FN từ candidate score mạnh nhất;
- trọng số mặc định là `TP=1,0`, `FP=0,5`, `FN=1,5`.

Surrogate này tác động vào confidence ranking. Nó không thay thế native box regression.

### 6.4. Objective KB1-B v3.1

Với YOLOv8/YOLOv11, objective hiện hành là:

```text
L_total = 0,25 × L_native_YOLO
        + 0,10 × L_reward
        + 0,25 × L_TP/FP/FN_proxy
        + 0,001 × L_L2-SP
```

Trong đó:

- `L_native_YOLO` gồm box regression, classification và DFL;
- `L_reward = −mean(log_proxy × stop_gradient(advantage))`;
- advantage được chuẩn hóa bằng EMA, `alpha=0,99`, rồi clip trong `[-3, 3]`;
- `L_L2-SP` phạt độ lệch của tham số trainable so với checkpoint supervised ban đầu.

YOLOv5 và DP-YOLO chưa có native `ComputeLoss` trong adapter KB1-B. Hai họ này dùng nhánh v2 với proxy thay thế; vì vậy không nên gộp kết quả của chúng với v3.1 nếu chưa ghi rõ khác biệt objective.

### 6.5. Giao thức huấn luyện an toàn

| Tham số | Giá trị |
|---|---:|
| Số bước tối đa | 10.000 |
| Learning rate cực đại | 5e-7 |
| Learning rate cực tiểu | 1e-7 |
| Warmup | 500 bước |
| Scheduler | cosine |
| Batch size | 8 |
| Gradient accumulation | 2 |
| Batch hiệu dụng | 16 |
| Gradient clipping | 0,5 |
| Phạm vi cập nhật | Chỉ Detect head |
| Chu kỳ validation | 500 bước |
| Early-stopping patience | 4 lần validation |
| Minimum improvement | 0,0005 |

Tại step 0, pipeline đánh giá và lưu trạng thái supervised làm checkpoint an toàn. Checkpoint validation-best chỉ được thay khi score sau tăng đủ `min_delta`:

```text
validation_score = 0,4 × mAP50
                 + 0,4 × mAP50-95
                 + 0,2 × recall
```

Checkpoint reward-best được lưu riêng và không được dùng làm kết quả cuối.

Phạm vi trainable này được thực thi bởi `freeze_except_detection_head()` trong adapter.

### 6.6. Phương pháp triển khai đối chứng native-only

#### 6.6.1. Mục đích

`native-only` là **đối chứng tinh chỉnh bằng loss gốc của mô hình**. Nhánh này trả lời câu hỏi: mức tăng sau KB1-B đến từ reward hay chỉ đến từ việc tiếp tục train checkpoint supervised thêm một số bước?

Đối chứng bắt đầu từ đúng `best.pt` của KB1-A và dùng cùng ngân sách với nhánh reward-guided. Khác biệt có chủ đích duy nhất trong objective là tắt reward loss và TP/FP/FN confidence proxy.

#### 6.6.2. Checkpoint đầu vào và phạm vi cập nhật

Với mỗi model `M`, đầu vào dự kiến là:

```text
checkpoints/M/weights/best.pt
```

Sau khi nạp checkpoint:

1. toàn bộ backbone và neck bị đóng băng;
2. chỉ module `Detect` cuối cùng có `requires_grad=True`;
3. mô hình được giữ ở evaluation mode trong lúc tính gradient, nhằm tránh làm thay đổi BatchNorm running statistics;
4. một bản sao tham số Detect head ban đầu được giữ lại làm mốc cho L2-SP.

Như vậy, native-only đo hiệu quả của việc tiếp tục điều chỉnh head bằng native detection loss, thay vì huấn luyện lại toàn mạng.

#### 6.6.3. Native detection loss

Target từ custom dataloader có box dạng `xyxy` tuyệt đối. Adapter Ultralytics chuyển target về `cx, cy, width, height` chuẩn hóa theo kích thước ảnh, ghép `batch_idx`, `cls` và `bboxes`, sau đó gọi trực tiếp criterion gốc qua `model.loss(native_batch)`.

`L_native_YOLO` giữ đầy đủ ba thành phần huấn luyện detector của YOLOv8/YOLOv11:

- box regression loss;
- classification loss;
- Distribution Focal Loss (DFL).

Objective thực sự dùng trong mode `native-only` là:

```text
L_native-only = 0,25 × L_native_YOLO
              + 0,001 × L_L2-SP
```

với:

```text
L_L2-SP = mean_j mean((θ_j − θ_j,KB1-A)²)
```

Trong mã, `reward_loss_weight` và `supervised_loss_weight` được đặt bằng 0 khi `mode == native-only`. Pipeline vẫn tính reward, advantage và TP/FP/FN proxy để giữ cùng luồng xử lý và phục vụ theo dõi, nhưng các đại lượng này **không đóng góp vào loss và không tạo gradient cập nhật** cho native-only.

#### 6.6.4. Lịch tối ưu và kiểm soát công bằng

Native-only dùng cùng cấu hình với KB1-B:

| Thành phần | Thiết lập |
|---|---:|
| Optimizer thực tế trong mã | Adam |
| Learning rate cực đại | 5e-7 |
| Learning rate cực tiểu | 1e-7 |
| Warmup | 500 bước |
| Scheduler | cosine |
| Số bước tối đa | 10.000 |
| Batch size | 8 |
| Gradient accumulation | 2 |
| Gradient clipping | 0,5 |
| Validation interval | 500 bước |
| Early-stopping patience | 4 lần validation |
| Minimum validation improvement | 0,0005 |
| Seed | 42, 43, 44 |

Hai nhánh phải dùng cùng:

- checkpoint KB1-A ban đầu;
- train/validation split và augmentation;
- seed;
- số bước tối đa và lịch learning rate;
- phạm vi Detect head được cập nhật;
- evaluator, validation interval và quy tắc early stopping.

Nhờ đó, chênh lệch `KB1-B − native-only` phản ánh đóng góp bổ sung của reward-guided objective tốt hơn chênh lệch đơn giản `KB1-B − KB1-A`.

#### 6.6.5. Chọn và lưu checkpoint

Tại step 0, checkpoint supervised được đánh giá và lưu làm phương án an toàn. Cứ mỗi 500 bước, native-only được đánh giá bằng:

```text
validation_score = 0,4 × mAP50
                 + 0,4 × mAP50-95
                 + 0,2 × recall
```

Checkpoint chỉ được thay khi score tăng ít nhất 0,0005. Tên file validation-best:

```text
rl_checkpoints/<model>_seed<seed>_native_only_best.pt
```

Native-only không lưu `reward_best` làm kết quả cuối. Trường `method` trong checkpoint là `native_only_control`, giúp phân biệt với checkpoint KB1-B.

Lệnh chạy:

```powershell
python .\kb1_reward_guided_training\train_rl.py --model yolov11n --seed 42 --mode native-only
```

#### 6.6.6. Giới hạn hiện tại

Mode native-only hiện chỉ chạy với YOLOv8 và YOLOv11 vì `native_detection_loss()` mới được triển khai trong `UltralyticsAdapter`. `YOLOv5Adapter` dùng cho YOLOv5s và DP-YOLO chưa tích hợp native `ComputeLoss`; chương trình sẽ dừng với lỗi nếu chạy `--mode native-only` cho hai mô hình này.

Đây là phần còn thiếu quan trọng đối với thiết kế so sánh đủ sáu mô hình. Trước khi lập bảng kết quả cuối, cần bổ sung native-loss control tương đương cho YOLOv5s và DP-YOLO hoặc ghi rõ rằng đối chứng native-only chỉ áp dụng cho nhóm Ultralytics.

### 6.7. Thiết kế đối chứng đã chạy

Ba mốc được dùng để diễn giải kết quả:

1. **Supervised:** checkpoint trước fine-tune.
2. **Native-only:** continued fine-tuning chỉ với native loss và L2-SP.
3. **KB1-B v3.1:** native loss + reward + TP/FP/FN proxy + L2-SP.

Native-only và KB1-B dùng cùng checkpoint khởi đầu, dữ liệu, lịch learning rate, batch size, số bước tối đa, chu kỳ validation và early stopping.

## 7. Kết quả KB1-B hiện có: YOLOv11n

Theo thiết kế đầy đủ, mục này cuối cùng phải chứa sáu nhóm kết quả ghép cặp. Hiện repository mới có thí nghiệm KB1-B nhiều seed hoàn chỉnh cho YOLOv11n; vì vậy đây là **kết quả bộ phận**, chưa phải bảng so sánh cuối của toàn KB1.

### 7.1. Kết quả theo seed

| Phương pháp | Seed | Best step | mAP50 | mAP50-95 | Recall | Validation score |
|---|---:|---:|---:|---:|---:|---:|
| Native-only | 42 | 3.500 | 0,59004 | 0,45599 | 0,60855 | 0,54012 |
| Native-only | 43 | 5.000 | 0,58972 | 0,45576 | 0,60741 | 0,53968 |
| Native-only | 44 | 5.500 | 0,59030 | 0,45622 | 0,60950 | 0,54051 |
| KB1-B v3.1 | 42 | 7.000 | 0,59347 | 0,46118 | 0,61043 | 0,54395 |
| KB1-B v3.1 | 43 | 5.000 | 0,59246 | 0,45986 | 0,60761 | 0,54245 |
| KB1-B v3.1 | 44 | 7.500 | 0,59406 | 0,46181 | 0,61151 | 0,54465 |

Các số trên đã được đối chiếu trực tiếp với metadata trong sáu checkpoint validation-best.

### 7.2. Trung bình ba seed

| Phương pháp | mAP50 | mAP50-95 | Recall | Validation score |
|---|---:|---:|---:|---:|
| Supervised step 0 | 0,58820 | 0,45250 | 0,60650 | — |
| Native-only | 0,59002 ± 0,00029 | 0,45599 ± 0,00023 | 0,60849 ± 0,00104 | 0,54010 ± 0,00042 |
| KB1-B v3.1 | **0,59333 ± 0,00081** | **0,46095 ± 0,00100** | **0,60985 ± 0,00201** | **0,54368 ± 0,00112** |

### 7.3. Chênh lệch ghép cặp KB1-B trừ native-only

| Chỉ số | Mean difference ± sample std |
|---|---:|
| mAP50 | +0,00331 ± 0,00053 |
| mAP50-95 | +0,00496 ± 0,00077 |
| Recall | +0,00136 ± 0,00101 |
| Validation score | +0,00358 ± 0,00072 |

KB1-B vượt native-only về mAP50 và mAP50-95 ở cả ba seed. Điều này cho thấy reward-guided objective tạo thêm lợi ích ngoài continued supervised fine-tuning trong phạm vi validation đã chạy.

### 7.4. Cách diễn giải thận trọng

Có thể kết luận:

> KB1-B v3.1 tạo cải thiện validation nhất quán so với continued supervised fine-tuning có cùng ngân sách trên YOLOv11n qua ba random seed.

Chưa nên kết luận:

- kết quả có statistical significance chỉ từ ba seed;
- cải thiện chắc chắn giữ nguyên trên tập test;
- KB1-B hiệu quả với mọi họ YOLO;
- reward trực tiếp cải thiện box coordinates;
- DP-YOLO tốt hơn baseline từ kết quả hiện tại.

## 8. Hai hệ metric không được trộn trực tiếp

Repository hiện có hai nguồn metric validation:

1. **Native/framework metric** trong CSV supervised training.
2. **Quick evaluator** dùng `torchmetrics` trong vòng KB1-B.

Ví dụ, YOLOv11n có mAP50 0,62734 tại epoch có mAP50-95 cao nhất trong CSV native, trong khi step-0 quick evaluation của KB1-B ghi mAP50 khoảng 0,5882. Chênh lệch này có thể đến từ preprocessing, inference/NMS hoặc evaluator khác nhau.

Vì vậy:

- chỉ so KB1-B với native-only khi cả hai dùng cùng quick evaluator;
- chỉ so sáu mô hình supervised khi dùng cùng loại native metric;
- bảng cuối của luận văn phải đánh giá lại toàn bộ checkpoint bằng cùng một native evaluator trên cùng split.

## 9. Trạng thái artefact

### 9.1. Đã có

- sáu checkpoint supervised tốt nhất;
- CSV, curve và confusion matrix của supervised runs;
- checkpoint KB1-B validation-best cho YOLOv11n seed 42, 43, 44;
- checkpoint native-only validation-best cho YOLOv11n seed 42, 43, 44;
- checkpoint periodic/reward-best liên quan;
- mã huấn luyện, đánh giá, adapters, reward và custom dataloader;
- test suite `tests/test_fixes.py`.

Test suite đã được chạy lại khi rà soát báo cáo và đạt **19/19 test**.

### 9.2. Chưa có hoặc chưa hoàn tất

- `results/tables/results_full.csv` và `results_delta.csv` cho bộ thí nghiệm cuối;
- native held-out test metrics cho 7 checkpoint: supervised YOLOv11n và 6 checkpoint fine-tuned;
- precision, macro-F1, per-class AP và confusion matrix cho so sánh KB1-B cuối;
- latency/FPS, số tham số và FLOPs trong cùng giao thức;
- multi-seed KB1-B cho các mô hình ngoài YOLOv11n;
- ablation DP-YOLO theo từng module;
- native ComputeLoss cho YOLOv5/DP-YOLO trong KB1-B v3.1.

Ngoài ra, `evaluate.py` hiện ánh xạ tên checkpoint RL kiểu cũ không có seed. Muốn dùng script này cho bảng multi-seed cuối cần cập nhật mapping hoặc bổ sung tham số checkpoint/seed.

## 10. Lệnh tái lập chính

Chạy supervised cho một mô hình:

```powershell
python .\kb1_reward_guided_training\train_supervised.py --model yolov11n
```

Chạy KB1-B v3.1:

```powershell
python .\kb1_reward_guided_training\train_rl.py --model yolov11n --seed 42 --mode kb1b
```

Chạy đối chứng native-only:

```powershell
python .\kb1_reward_guided_training\train_rl.py --model yolov11n --seed 42 --mode native-only
```

Thay seed 42 bằng 43 và 44 để tái lập ba seed.

Đánh giá theo script hiện tại:

```powershell
python .\kb1_reward_guided_training\evaluate.py --model YOLOv11n --split test
```

Lệnh cuối chỉ cho kết quả đúng checkpoint mong muốn sau khi mapping RL trong `evaluate.py` được cập nhật theo tên checkpoint có seed.

## 11. Kết luận

Kịch bản 1 được định nghĩa là pipeline hai giai đoạn áp dụng theo từng mô hình: KB1-A tạo best supervised checkpoint riêng cho sáu mô hình; KB1-B tiếp tục reward-guided fine-tuning từ từng checkpoint tương ứng; cuối cùng đánh giá delta trước–sau trong từng cặp và so sánh kết quả cuối giữa các mô hình.

Hiện KB1-A đã có đủ sáu checkpoint, nhưng KB1-B mới có bằng chứng nhiều seed hoàn chỉnh cho YOLOv11n. Trên mô hình này, reward-guided objective vượt đối chứng native-only một cách nhất quán trên validation ở ba seed, đặc biệt ở mAP50-95. Vì năm cặp còn lại chưa hoàn thành, chưa thể xếp hạng kết quả sau KB1-B của toàn bộ sáu mô hình.

Tuy nhiên, kết luận cuối cùng vẫn cần native evaluation trên held-out test set. Đây là bước bắt buộc để loại bỏ sai khác evaluator và xác nhận rằng mức tăng validation không đến từ việc chọn checkpoint hoặc thích nghi quá mức với validation.

## 12. Checklist để người đọc kiểm tra

- [ ] Xác nhận mô tả dataset 28 lớp và annotation toàn bộ lá.
- [ ] Xác nhận DP-YOLO dùng bốn head P2, P3, P4, P5.
- [ ] Xác nhận KB1 là một pipeline hai giai đoạn, không phải hai nghiên cứu độc lập.
- [ ] Xác nhận mỗi best checkpoint của KB1-A đi vào một run KB1-B cùng model.
- [ ] Dùng tên “reward-guided fine-tuning (RL-inspired)”, không gọi DPO/REINFORCE thuần.
- [ ] Xác nhận objective và trọng số KB1-B v3.1.
- [ ] Xác nhận checkpoint và best step theo từng model, từng seed.
- [ ] Không trộn native metric với quick-evaluator metric.
- [ ] Chạy native test trước khi chốt bảng kết quả luận văn.
- [ ] Chỉ tuyên bố trong phạm vi YOLOv11n và validation cho đến khi có thêm thí nghiệm.
