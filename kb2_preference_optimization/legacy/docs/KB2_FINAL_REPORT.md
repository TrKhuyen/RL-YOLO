# Báo cáo tổng kết KB2: Fine-tune YOLO bằng học có phản hồi

## 1. Tóm tắt

KB2 nghiên cứu cách đưa tư tưởng của Direct Preference Optimization (DPO) vào
fine-tune mô hình phát hiện đối tượng YOLO. Mục tiêu không phải sao chép nguyên
DPO dành cho mô hình ngôn ngữ, mà là dùng phản hồi từ prediction và ground truth
để xác định mô hình đang sai ở đâu và điều chỉnh quá trình tối ưu.

Toàn bộ thử nghiệm giữ nguyên ảnh, nhãn và cách chia `train/valid/test`. Không có
ảnh mới, pseudo-label, crop replay hay thay đổi dữ liệu gốc. Native YOLO detection
loss luôn là mục tiêu chính.

Kết quả cuối cùng: pipeline feedback đã được triển khai, kiểm thử và đánh giá qua
ba seed, nhưng chưa có biến thể nào cải thiện `mAP50-95` trung bình so với
supervised continuation baseline. Tất cả biến thể chỉ thắng ở seed 42 và thua ở
seed 43, 44. Vì vậy KB2 không tuyên bố cải thiện hiệu suất tại thời điểm báo cáo.

## 2. Mục tiêu và câu hỏi nghiên cứu

Các câu hỏi chính của KB2:

1. Có thể dùng prediction và ground truth làm phản hồi tự động để fine-tune YOLO
   hay không?
2. Feedback cấp ảnh, cấp object hoặc feedback động có cải thiện validation mAP
   hay không?
3. Tư tưởng preference learning có thể được chuyển thành sampling, auxiliary
   loss hoặc gradient control phù hợp với detection hay không?
4. Kết quả có ổn định qua nhiều random seed hay chỉ là nhiễu của một lần chạy?

Tiêu chí chấp nhận là cải thiện `mAP50-95` trên validation, không làm giảm
AP-small, và phải ổn định qua tối thiểu ba seed.

## 3. Dữ liệu và checkpoint ban đầu

- Dataset: PlantDoc theo định dạng YOLO.
- Train: 1.722 ảnh.
- Validation: 495 ảnh.
- Số lớp: 28.
- Mô hình chính: YOLOv8n.
- Checkpoint đầu vào:
  `kb1_reward_guided_training/checkpoints/yolov8n/weights/best.pt`.
- SHA-256 checkpoint dùng sinh feedback:
  `87bd39af33c34d6cf9459c51cf1471aca9c9d9cd3c883431d6770d7e773fb838`.

KB2 chỉ dùng train để sinh feedback. Validation chỉ dùng lựa chọn checkpoint và
đánh giá. Test không được dùng để tuning và không chạy kết quả cuối vì không có
phương pháp nào vượt acceptance gate trên validation đa seed.

## 4. Protocol đánh giá

Protocol được cố định cho mọi phương pháp:

- Metric chọn checkpoint: `mAP50-95`.
- Confidence threshold: `0.25`.
- NMS IoU threshold: `0.45`.
- Validation: đủ 495 ảnh, streaming metric với backend `faster_coco_eval`.
- Batch size fine-tune: 2.
- Learning rate: `1e-6`.
- Gradient clipping: `1.0`.
- BatchNorm running statistics được khóa khi fine-tune batch nhỏ.
- Đánh giá step 0 trước khi train.
- Best checkpoint chỉ được cập nhật khi validation `mAP50-95` tăng.
- Early stopping sau các lần validation liên tiếp không cải thiện.
- Multi-seed: 42, 43, 44.

Evaluator mới đã được đối chiếu với baseline cũ:

| Nguồn | mAP50-95 | mAP50 | AP-small | Recall |
|---|---:|---:|---:|---:|
| Baseline đã ghi trước | 0.66797 | 0.80467 | 0.17172 | 0.71683 |
| Evaluator KB2 mới | 0.66741 | 0.80470 | 0.17172 | 0.71609 |

Sai khác `mAP50-95` khoảng `0.00056`; AP-small khớp và các metric còn lại rất
gần nhau.

## 5. Feedback schema và bộ sinh feedback

Mỗi ảnh train có một record JSONL version `1.0`, gồm:

- `matched`: đúng class và IoU đạt ngưỡng.
- `wrong_class`: IoU đạt ngưỡng nhưng sai class.
- `bad_localization`: có overlap nhưng IoU chưa đạt ngưỡng match.
- `false_positive`: prediction không match ground truth.
- `duplicate`: nhiều prediction match cùng một GT.
- `missed`: GT không được match.
- `preferences`: chosen/rejected prediction khi có pair hợp lệ.

Ngưỡng tạo feedback:

| Điều kiện | Loại feedback |
|---|---|
| IoU >= 0.5, đúng class | matched |
| IoU >= 0.5, sai class | wrong_class |
| 0.1 <= IoU < 0.5 | bad_localization |
| IoU < 0.1 | false_positive |
| GT không được match | missed |
| Prediction thứ hai match GT đã có prediction | duplicate |

Kết quả trên 1.722 ảnh train:

| Loại | Số lượng |
|---|---:|
| matched | 6.050 |
| wrong_class | 497 |
| bad_localization | 676 |
| false_positive | 1.558 |
| duplicate | 4 |
| missed | 1.124 |
| preference pairs | 5.543 |

Kiểm tra dữ liệu feedback:

- 1.722 record và 1.722 `image_id` duy nhất.
- Không có validation/test leakage.
- Không có box/score NaN hoặc Inf.
- Sinh file nguyên tử qua `.tmp` rồi mới replace.
- Có checkpoint hash và metadata để tái lập.
- Đã sinh 50 ảnh trực quan hóa feedback.

Artifact chính:

- `feedback_data/yolov8n_train.jsonl`.
- `feedback_data/yolov8n_train.summary.json`.
- `feedback_data/quality/feedback_quality_report.json`.

## 6. FeedbackDataset và sampling

`FeedbackDataset` gắn vào mỗi target:

- Vector số lượng sáu loại feedback.
- Điểm difficulty cấp ảnh.
- Preference pairs.
- Mã lỗi object-level.
- `gt_indices` để giữ ánh xạ GT qua augmentation.

Sampling weight ban đầu:

```text
weight = 1 + sampling_strength * normalized_feedback_difficulty
```

Weight được chặn trên để tránh một số ảnh khó chi phối toàn bộ quá trình học.
Với cấu hình mặc định:

```text
min  = 1.0000
mean = 1.3646
max  = 5.0000
```

RNG của Albumentations và `WeightedRandomSampler` được khóa riêng. Hai run cùng
seed sinh cùng image ID và pixel tensor. Việc này đặc biệt quan trọng để sampling
và hybrid loss được so sánh trên đúng cùng batch.

## 7. Các phương pháp đã thử

### 7.1 Anchor-DPO thử nghiệm ban đầu

Phiên bản ban đầu tạo positive/negative anchor và tối ưu DPO-style margin với
frozen reference model. Kết quả full validation:

| Mô hình | mAP50 | mAP50-95 | AP-small | Recall |
|---|---:|---:|---:|---:|
| Supervised baseline | 0.80467 | 0.66797 | 0.17172 | 0.71683 |
| Anchor-DPO best | 0.80116 | 0.65164 | 0.12172 | 0.71099 |

Anchor-DPO bị loại vì giảm rõ rệt mAP và AP-small. Đây là lý do KB2 được thiết
kế lại theo explicit feedback thay vì cố mô phỏng DPO nguyên bản.

### 7.2 Feedback-guided image sampling

Ảnh có missed, wrong-class hoặc bad-localization được lấy mẫu thường xuyên hơn.
Native YOLO loss vẫn học từ GT sau augmentation.

Tuning trên seed 42:

| Sampling strength | mAP50-95 |
|---:|---:|
| 0.25 | 0.668934 |
| 0.50 | 0.667917 |
| 1.50 | 0.667362 |
| 2.00 | 0.667674 |

Strength `0.25` được chọn để kiểm chứng đa seed.

### 7.3 Image-level hybrid loss

Loss được thử:

```text
L_total = (1 - alpha) * L_native_batch
        + alpha * L_native_feedback_weighted
```

Các alpha `0.01`, `0.03`, `0.05`, `0.10` được thử tại sampling strength `0.25`.
Không alpha nào vượt sampling-only một cách có ý nghĩa. Auxiliary image-level
loss bị loại.

### 7.4 Static object-level feedback

GT được theo dõi qua augmentation bằng `gt_indices`. Mỗi object nhận mã:

- 0: none/matched.
- 1: wrong_class.
- 2: bad_localization.
- 3: missed.

Object auxiliary loss:

```text
wrong_class      -> classification loss
bad_localization -> IoU box loss
missed           -> classification + IoU box loss
```

Static object feedback dùng lỗi sinh từ checkpoint ban đầu. Cấu hình tốt nhất
trên seed 42 là `object_alpha=0.05`.

### 7.5 Dynamic object feedback

Ở mỗi step, raw prediction hiện tại được dùng để phân loại lại từng GT thành
matched, wrong-class, bad-localization hoặc missed. Feedback không còn bị cố định
theo checkpoint ban đầu. Raw YOLOv8 output đã được xác minh có shape
`(B, 4 + 28, 8400)`, box decoded theo `xywh` và class score ở miền probability.

Các alpha `0.001`, `0.003`, `0.01` được thử. Tốt nhất trên seed 42 là `0.001`.

### 7.6 Gradient-conflict control (PCGrad-style)

Native gradient và dynamic feedback gradient được tính riêng. Nếu tích vô hướng
âm, thành phần feedback ngược hướng native gradient bị chiếu bỏ:

```text
g_feedback_projected = g_feedback
                     - min(0, dot(g_native, g_feedback) / ||g_native||^2)
                       * g_native

g_total = g_native + alpha * g_feedback_projected
```

Native gradient luôn được giữ nguyên. Các alpha `0.001`, `0.01`, `0.05` được thử.
Tốt nhất trên seed 42 là `0.001`.

## 8. Lỗi kỹ thuật quan trọng đã phát hiện

### 8.1 Encoding YAML trên Windows

Đọc YAML bằng encoding mặc định CP1252 gây `UnicodeDecodeError`. Các file YAML
được mở tường minh bằng UTF-8.

### 8.2 NMS time-limit

NMS bị gọi lặp theo từng image/view, gây warning time limit. Pipeline được đổi
sang NMS theo batch và giới hạn số candidate phù hợp.

### 8.3 Empty prediction tensor

Adapter cũ trả empty score sai kích thước. YOLOv5 và Ultralytics adapter được sửa
để trả tensor score dài 0 và giữ đúng device.

### 8.4 BatchNorm drift

Fine-tune batch size 2 làm BatchNorm running statistics thay đổi mạnh. Trước sửa,
baseline giảm từ khoảng `0.667` xuống `0.645` sau 25 step. Sau khi khóa running
statistics, step 25 đạt `0.66594`, và checkpoint step 0 luôn được giữ nếu tốt hơn.

### 8.5 Ablation không tái lập

Albumentations giữ RNG nội bộ và sampler chưa có generator riêng. Sampling và
hybrid nhận batch khác nhau dù cùng seed. Sau sửa, cả image ID, pixel tensor và
native loss trùng chính xác; ví dụ native loss của hai nhánh cùng là `3.70665`.

### 8.6 Checkpoint selection sai mục tiêu

Phiên bản cũ chọn checkpoint theo training reward/loss. Trainer mới đánh giá
validation step 0, chọn best theo `mAP50-95`, lưu optimizer/early-stopping state
và khôi phục đầy đủ khi resume.

## 9. Ablation chính thức

Ablation 100 step, seed 42, validation mỗi 10 step, patience 3:

| Variant | Best step | mAP50-95 | mAP50 | AP-small | Recall |
|---|---:|---:|---:|---:|---:|
| Native baseline | 10 | **0.667985** | **0.804698** | 0.171723 | **0.716522** |
| Feedback sampling, strength 1.0 | 10 | 0.667624 | 0.804321 | 0.171723 | 0.716289 |
| Sampling + hybrid alpha 0.1 | 10 | 0.667623 | 0.804307 | 0.171723 | 0.716350 |

Cả ba dừng ở step 40. Hybrid alpha `0.1` không tạo lợi ích.

## 10. Kết quả multi-seed

### 10.1 Feedback sampling tốt nhất

Cấu hình: sampling strength `0.25`, image/object alpha `0`.

| Seed | Baseline | Feedback sampling | Delta |
|---:|---:|---:|---:|
| 42 | 0.667985 | 0.668934 | +0.000949 |
| 43 | 0.668200 | 0.667646 | -0.000553 |
| 44 | 0.668517 | 0.667197 | -0.001320 |

```text
Baseline mean: 0.668234 +/- 0.000268
Feedback mean: 0.667926 +/- 0.000901
Paired delta: -0.000308 +/- 0.001154
Wins: 1/3
```

### 10.2 Static object-level feedback

| Seed | Baseline | Static object | Delta |
|---:|---:|---:|---:|
| 42 | 0.667985 | 0.668985 | +0.001000 |
| 43 | 0.668200 | 0.667498 | -0.000702 |
| 44 | 0.668517 | 0.666985 | -0.001532 |

```text
Object mean: 0.667823
Paired delta mean: -0.000411
Wins: 1/3
```

### 10.3 Dynamic object feedback

| Seed | Baseline | Dynamic object | Delta |
|---:|---:|---:|---:|
| 42 | 0.667985 | 0.668974 | +0.000989 |
| 43 | 0.668200 | 0.667665 | -0.000534 |
| 44 | 0.668517 | 0.667393 | -0.001124 |

```text
Dynamic mean: 0.668011
Paired delta mean: -0.000223
Wins: 1/3
```

Dynamic feedback có mean tốt nhất trong các phương pháp feedback, nhưng vẫn thấp
hơn baseline.

### 10.4 Gradient-conflict control

| Seed | Baseline | PCGrad-style | Delta |
|---:|---:|---:|---:|
| 42 | 0.667985 | 0.669000 | +0.001016 |
| 43 | 0.668200 | 0.667426 | -0.000774 |
| 44 | 0.668517 | 0.667154 | -0.001363 |

```text
PCGrad mean: 0.667860
Paired delta mean: -0.000374
Wins: 1/3
```

PCGrad cho kết quả single-seed cao nhất nhưng không ổn định đa seed.

## 11. So sánh tổng hợp

| Phương pháp | Mean mAP50-95 | Delta so với baseline | Wins |
|---|---:|---:|---:|
| Native baseline | **0.668234** | 0 | - |
| Image feedback sampling | 0.667926 | -0.000308 | 1/3 |
| Static object feedback | 0.667823 | -0.000411 | 1/3 |
| Dynamic object feedback | 0.668011 | -0.000223 | 1/3 |
| PCGrad-style control | 0.667860 | -0.000374 | 1/3 |

Không phương pháp feedback nào vượt baseline trung bình. Chênh lệch đều nhỏ hơn
độ dao động theo seed. Kết quả tốt ở seed 42 không tái lập ở seed 43 và 44.

## 12. Kiểm thử và độ tin cậy triển khai

Tại thời điểm kết thúc:

- Compile toàn bộ KB2: pass.
- Unit/regression tests: 26/26 pass.
- Feedback schema và malformed input tests: pass.
- 1.722 real-record coverage test: pass.
- Augmented GT index alignment: pass.
- Deterministic augmentation/sampler: pass.
- Native/hybrid/object/dynamic loss gradient tests: pass.
- PCGrad projection mathematical tests: pass.
- GPU forward/backward: pass.
- Gradient finite checks: pass.
- Atomic checkpoint save: pass.
- Strict model reload và optimizer resume: pass.
- Validation parity: pass.
- Multi-seed report validation: pass.

## 13. Kết luận

KB2 đã đạt mục tiêu kỹ thuật: xây dựng được pipeline học có phản hồi cho YOLO từ
khâu sinh feedback, kiểm định dữ liệu, sampling, loss cấp ảnh/object, feedback
động, gradient control, validation, early stopping, checkpoint và multi-seed
evaluation.

Tuy nhiên giả thuyết hiệu quả chưa được xác nhận. Trên dataset và checkpoint hiện
tại, supervised baseline đã ở vùng tối ưu ổn định; fine-tune thêm tạo dao động
khoảng `0.001 mAP`, lớn hơn lợi ích quan sát được. Tuning trên một seed dẫn đến
overfit seed 42 và không tổng quát sang seed 43, 44.

Do đó kết luận chính thức là:

> Các cơ chế feedback đã thử không cải thiện mAP50-95 trung bình của YOLOv8n trên
> ba seed khi giữ nguyên dữ liệu. Không có checkpoint feedback nào đủ điều kiện
> thay thế supervised baseline.

Đây là kết quả âm nhưng có giá trị: nó loại bỏ nhiều thiết kế tưởng như hợp lý,
chứng minh tầm quan trọng của step-0 baseline, BatchNorm control, deterministic
ablation và multi-seed validation.

## 14. Hạn chế

- Chỉ đánh giá chính trên YOLOv8n.
- Pilot feedback dùng batch size 2 và 10 step cho multi-seed sau khi best step
  được xác định ở ablation.
- Feedback chưa được tích hợp trực tiếp vào Ultralytics task assigner/DFL native
  theo trọng số từng GT.
- Không chạy test set cuối vì không có cấu hình nào vượt validation acceptance
  gate; điều này tránh dùng test để chọn phương pháp.
- Không thay đổi hoặc bổ sung dữ liệu theo yêu cầu của đề tài.

## 15. Hướng tiếp theo nếu tiếp tục nghiên cứu

Trong điều kiện giữ nguyên dữ liệu, các hướng còn hợp lý là:

1. Tích hợp object weight trực tiếp vào native task assigner và DFL thay vì dùng
   head auxiliary loss.
2. Dùng optimizer/scheduler chính thức của Ultralytics để giảm khác biệt với
   supervised training recipe.
3. Thử trên checkpoint chưa hội tụ hoàn toàn để đo liệu feedback hỗ trợ tốc độ
   hội tụ, thay vì cố cải thiện một checkpoint đã gần tối ưu.
4. Chốt hyperparameter trên nhiều seed đồng thời hoặc nested validation, không
   chọn theo seed 42.
5. Đánh giá thêm YOLOv8s/YOLOv11n trước khi kết luận feedback không phù hợp với
   mọi kiến trúc.

Không nên tiếp tục quét alpha trên cùng validation vì nguy cơ overfit tăng lên và
các thử nghiệm hiện tại đã cho thấy lợi ích nhỏ hơn phương sai theo seed.

## 16. File và artifact chính

### Code

- `feedback.py`: schema và matching.
- `generate_feedback.py`: sinh feedback train.
- `validate_feedback.py`: kiểm định và trực quan hóa.
- `feedback_dataset.py`: dataset, difficulty, sampler và object mapping.
- `feedback_loss.py`: hybrid, object, dynamic và PCGrad-style loss.
- `feedback_validation.py`: evaluator và early stopping.
- `train_feedback.py`: trainer chính.
- `run_ablation.py`: orchestration ablation.
- `tests/`: test suite KB2.

### Báo cáo máy đọc

- `ablation_100_seed42_bnfixed/ablation_results.json`.
- `tuning_seed42/tuning_results.json`.
- `multiseed/multiseed_results.json`.
- `object_multiseed/object_feedback_results.json`.
- `dynamic_feedback/dynamic_feedback_results.json`.
- `pcgrad/pcgrad_results.json`.

### Checkpoint tốt nhất theo từng thử nghiệm

- Baseline: `ablation_100_seed42_bnfixed/baseline_last_best.pt`.
- Sampling seed 42: `tuning_seed42/selected_sampling_last_best.pt`.
- Static object seed 42: `object_tuning_seed42/object_a0.05_last_best.pt`.
- Dynamic seed 42: `dynamic_feedback/dynamic_a0.001_last_best.pt`.
- PCGrad seed 42: `pcgrad/pcgrad_a0.001_last_best.pt`.

Các checkpoint feedback trên chỉ là artifact nghiên cứu, không được khuyến nghị
thay thế supervised baseline do không vượt qua kiểm chứng ba seed.
