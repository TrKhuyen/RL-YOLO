# KB2: Kế hoạch fine-tune YOLO bằng phản hồi

## 1. Mục tiêu

KB2 sử dụng ý tưởng preference learning từ DPO để fine-tune YOLO bằng phản
hồi. Phương pháp không bắt buộc sao chép nguyên DPO dành cho LLM.

Mục tiêu chính:

- Dùng dữ liệu train và ground truth làm nguồn phản hồi tự động.
- Không chỉnh sửa ảnh trong giai đoạn đầu.
- Giữ native YOLO detection loss làm thành phần chính.
- Dùng preference và reference constraint như regularizer.
- Chỉ chấp nhận mô hình mới khi validation mAP thực sự cải thiện.

## 2. Kiến trúc mục tiêu

```text
Train images + ground truth
          |
          v
Supervised YOLO predictions
          |
          v
Feedback generator
          |
          v
Versioned feedback records
          |
          v
FeedbackDataset / hard-example sampler
          |
          v
Native YOLO loss
+ weighted feedback losses
+ lightweight preference/reference loss
          |
          v
Streaming validation
          |
          v
Best checkpoint selected by validation metrics
```

## 3. Giai đoạn 1: Chuẩn hóa lại KB2

- Bỏ `G` và group augmentation khỏi Level 2.
- Không gọi Level 2 là GRPO.
- Đổi Level 2 thành `feedback_finetune`.
- Tạm tắt Level 3 DAPO khỏi pipeline chính.
- Cập nhật README, CLI, config và tên checkpoint.
- Giữ checkpoint và kết quả cũ để đối chiếu.

### Tiêu chí hoàn thành

- Code, CLI, config và tài liệu mô tả cùng một thuật toán.
- Không còn tham số không được sử dụng.
- Pipeline cũ không ghi đè checkpoint mới.

## 4. Giai đoạn 2: Thiết kế feedback schema

Mỗi ảnh train tạo một feedback record:

```json
{
  image_id: image_001,
  checkpoint: supervised_best.pt,
  matches: [],
  wrong_class: [],
  bad_localization: [],
  false_positives: [],
  duplicates: [],
  missed_ground_truths: []
}
```

Mỗi detection lưu:

- Bounding box và class dự đoán.
- Confidence.
- Ground truth tương ứng, nếu có.
- IoU.
- Loại phản hồi.
- Trọng số phản hồi.
- Chosen/rejected ID nếu tạo preference pair.

Ngưỡng khởi đầu:

| Điều kiện | Feedback |
|---|---|
| IoU >= 0.5 và đúng class | `matched` |
| 0.1 <= IoU < 0.5 | `bad_localization` |
| IoU >= 0.5 và sai class | `wrong_class` |
| IoU < 0.1 và confidence cao | `false_positive` |
| Ground truth không được match | `missed` |
| Nhiều prediction cùng match một GT | `duplicate` |

## 5. Giai đoạn 3: Feedback generator

Tạo script:

```text
kb2_preference_optimization/generate_feedback.py
```

Nhiệm vụ:

1. Chỉ đọc split `train`.
2. Chạy checkpoint supervised ở chế độ evaluation.
3. Ghép prediction với ground truth theo IoU và class.
4. Phân loại lỗi và sinh preference pairs.
5. Lưu feedback theo JSONL hoặc định dạng tensor có version.
6. Lưu checkpoint hash, config và timestamp.
7. Xuất thống kê tổng hợp theo class và loại feedback.

Không dùng validation hoặc test để sinh feedback.

### Tiêu chí hoàn thành

- 100% ảnh train được xử lý.
- Không có box hoặc score NaN/Inf.
- Feedback có thể tái lập từ cùng checkpoint và config.
- Có thống kê coverage theo ảnh và class.

## 6. Giai đoạn 4: Kiểm tra chất lượng feedback

Kiểm tra:

- Tỷ lệ ảnh có missed detection.
- False positives theo class.
- Phân bố IoU và confidence.
- Số preference pair hợp lệ.
- Class bị thiếu hoặc dư feedback.
- Một prediction có bị gán sai cho nhiều GT hay không.

Xuất tối thiểu 50 ảnh trực quan với màu riêng cho:

- Matched.
- Missed.
- False positive.
- Wrong class.
- Bad localization.

## 7. Giai đoạn 5: FeedbackDataset

Dataset trả về:

```python
images, targets, feedback
```

Ưu tiên sampling:

- Ảnh có missed object.
- Vật thể nhỏ.
- Class hiếm.
- False positive confidence cao.
- Box có IoU gần ngưỡng quyết định.

Giữ khoảng 30-50% batch là dữ liệu bình thường để hạn chế catastrophic
forgetting.

## 8. Giai đoạn 6: Hybrid loss

Objective dự kiến:

```text
L_total =
    L_YOLO
  + lambda_missed * L_missed
  + lambda_fp * L_hard_negative
  + lambda_box * L_bad_localization
  + lambda_pref * L_preference
  + lambda_ref * L_reference
```

Cấu hình khởi đầu:

```yaml
supervised_weight: 1.0
missed_weight: 0.25
false_positive_weight: 0.10
localization_weight: 0.25
preference_weight: 0.01
reference_weight: 0.01
```

Preference loss phải:

- Dùng chosen/rejected của cùng một ảnh.
- Áp dụng cho tất cả ground truth phù hợp.
- Bao gồm class probability và box quality/DFL.
- Tính trước NMS.
- Không dùng `log(mean confidence)`.

## 9. Giai đoạn 7: Validation và checkpoint

Chạy streaming validation mỗi 50-100 bước trên validation subset cố định.
Đánh giá đầy đủ khi có checkpoint ứng viên.

Điểm chọn checkpoint:

```text
selection_score =
    mAP50-95
  + 0.25 * AP-small
  + 0.10 * Recall
```

Quy tắc an toàn:

- Không chọn best chỉ dựa trên training loss hoặc training reward.
- Early stop nếu mAP50-95 giảm liên tiếp.
- Rollback nếu Recall hoặc AP-small giảm quá ngưỡng.
- Luôn dùng cùng evaluator và threshold với supervised baseline.

## 10. Giai đoạn 8: Ablation

Mỗi cấu hình chạy 500 bước:

| Thử nghiệm | Thành phần |
|---|---|
| A | Supervised continuation |
| B | Supervised + feedback weighting |
| C | B + preference loss |
| D | C + reference regularization |

Chỉ giữ C hoặc D nếu tốt hơn B. Phép so sánh này xác định cải thiện đến từ
feedback, preference hay chỉ do tiếp tục supervised training.

## 11. Giai đoạn 9: Thực nghiệm chính thức

Với cấu hình thắng:

- Chạy tối thiểu ba seed.
- Báo cáo mean và standard deviation.
- Đánh giá đầy đủ trên validation.
- Chốt config trước khi chạy test.
- Chỉ chạy test một lần cho kết quả cuối.
- Báo cáo mAP50, mAP50-95, AP-small, Precision, Recall và FPS.

## 12. Thứ tự triển khai

1. Chuẩn hóa lại KB2.
2. Hoàn thiện feedback schema.
3. Xây feedback generator.
4. Kiểm tra và trực quan hóa feedback.
5. Xây FeedbackDataset và sampler.
6. Triển khai hybrid feedback loss.
7. Tích hợp streaming validation và early stopping.
8. Chạy ablation 500 bước.
9. Chạy ba seed cho cấu hình tốt nhất.
10. Viết báo cáo kết quả KB2.

Không triển khai loss mới trước khi feedback generator được kiểm tra và chứng
minh rằng dữ liệu phản hồi đúng, đủ và không chứa validation/test leakage.
