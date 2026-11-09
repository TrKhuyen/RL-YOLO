# KB2: thử giảm LR sau500 thay vì1.000 update — 06/11/2026

## Kế hoạch đã khóa

Giả thuyết: checkpoint tốt nhất của lượt trước xuất hiện ở500 update, trước
mốc giảm LR1k. Thử giảm LR ngay sau500 để xem các cập nhật nhẹ hơn có nâng
mAP vượt38,7133% hay không. Không giả định trước kết quả sẽ tốt hơn.

Campaign:
`kb2_preference_optimization/outputs/yolov8n_hybrid_drop500_s42_20261106_trial1/`.
`plan.json` đã được ghi trước khi huấn luyện. So với lượt drop1k hoàn tất,
chỉ đổi `lr_schedule.drop_after_update`; bank, code trainer, mask, loss,
optimizer, seed, thứ tự ảnh/cặp và lịch đánh giá giữ nguyên.

| Thành phần | Drop1k trước đó | Drop500 mới |
|---|---|---|
| Khởi tạo/reference | Best TPE YOLOv8n trial2 | Giữ nguyên |
| Ngân sách/seed | 15.000 update /42 | Giữ nguyên |
| λ /beta /background | 0,03 /0,1 /0,5 | Giữ nguyên |
| LR ban đầu | 1e-6 | Giữ nguyên |
| Thời điểm giảm LR | Sau update1.000 | **Sau update500** |
| LR sau giảm | 1e-7 | Giữ nguyên |
| Eval | Mỗi250 tới2k, sau đó mỗi1k | Giữ nguyên |
| Chọn best | Validation mAP50–95, min_delta1e-4 fraction | Giữ nguyên |

Model vẫn train trên CUDA. Giữ runtime đã giúp lượt trước hoàn tất:
`CUDA_LAUNCH_BLOCKING=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, affinity riêng tiến trình CPU16–23. Không thay cấu
hình hệ thống hoặc dữ liệu. Chỉ chạy pipeline KB2 DPO; không thêm native-only.

Mốc so: S0, TPE, best/last của drop1k và drop500. Hai run đều có22 mốc0,
250,500,…,2k,3k,…,15k; không thay mật độ đánh giá trong phép so này.
First500 weights/optimizer phải khớp. Sau update501, LR hai run khác nhau.

## Thực thi

Lượt mới đã hoàn tất15k trong34,01 phút train wall, gồm periodic eval nhưng
chưa tính đầy đủ setup và hậu kiểm; không gặp segmentation fault. Lượt drop1k
trước mất32,84 phút, cùng runtime affinity/thread nhưng tải nền có thể khác;
không dùng chênh thời gian để kết luận việc đổi milestone làm train chậm hơn.
Không dùng test, không chạy thêm seed trong lượt này.

Đã hậu kiểm bằng cách nạp lại best/last và chấm full validation479 ảnh/1.887GT.

Mốc cần vượt: best drop1k ở500, mAP50–95 **38,7133%**.
Lượt cũ: [KB2_HYBRID_SCHEDULE_REVIEW_20261106.md](KB2_HYBRID_SCHEDULE_REVIEW_20261106.md).

## Kết quả selected best

Tỷ lệ tính theo%; FP/FN tại confidence0,25. Hai run đều chọn best500 và
raw mAP cao nhất quan sát được cũng ở500; không có mốc cao hơn bị bỏ qua do
min_delta. Best không phải weights sau15k.

| Model | Best step | mAP50–95 | AP50 | AP75 | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | — | 37,8218 | 49,1225 | 42,9680 | 50,9227 | 54,1070 | 52,4666 | 984 | 866 |
| TPE | 0 | 38,5097 | 50,4306 | 43,4945 | 50,8763 | 55,3789 | 53,0322 | 1.009 | 842 |
| KB2 drop1k | 500 | **38,7133** | 50,6287 | 43,7080 | 52,6208 | 54,2660 | 53,4307 | 922 | 863 |
| KB2 drop500 | 500 | **38,7133** | 50,6287 | 43,7080 | 52,6208 | 54,2660 | 53,4307 | 922 | 863 |

**Đổi milestone1k→500 chưa nâng được selected best trong seed42.** Tất cả
quality metric của best đều bằng nhau. Hai run giống hệt tới hết500 update;
chỉ từ501, drop500 mới có LR thấp hơn. Việc model tốt nhất xuất hiện trước
milestone cũ chỉ gợi ý giả thuyết thử, không bảo đảm giảm LR sớm sẽ tốt hơn.

## So cùng15.000 update

| Metric (%) trừ số đếm | Drop1k | Drop500 |
|---|---:|---:|
| mAP50–95 | 38,5167 | **38,6300** |
| AP50 | 50,3684 | **50,5850** |
| AP75 | 43,5123 | **43,6386** |
| Precision | **53,8916** | 53,8791 |
| Recall | 53,2061 | **53,3651** |
| F1 | 53,5467 | **53,6209** |
| Macro F1 | 51,8866 | **51,9331** |
| Macro recall | 53,9998 | **54,1406** |
| AR300 | 57,2631 | **57,6738** |
| APsmall | **4,3578** | 4,1322 |
| APmedium | **18,2398** | 17,9973 |
| APlarge | 43,7640 | **43,8945** |
| TP | 1.004 | **1.007** |
| FP ↓ | **859** | 862 |
| FN ↓ | 883 | **880** |

Drop500 cải thiện nhiều metric ở last: mAP**+0,1133 điểm phần trăm**,
AP50+0,2166, AP75+0,1263, recall+0,1590 và F1+0,0742 điểm. TP tăng3/FN giảm3,
nhưng FP tăng3 và precision giảm0,0125 điểm. APsmall/APmedium giảm tương ứng
0,2256/0,2425 điểm; không mô tả mọi metric đều tốt hơn.

Last mới vẫn thấp hơn best500**0,0833 điểm mAP**, nên không chọn last để thay
best theo tiêu chí đã khóa. Lợi ích cuối quỹ đạo không đồng nghĩa đã có model
được chọn tốt hơn.

GT identity so ordinary TPE:

| Model | GT được cứu | GT bị mất | TP thay đổi ròng |
|---|---:|---:|---:|
| Best500 của cả hai run | 3 | 24 | −21 |
| Drop1k last15k | 8 | 49 | −41 |
| Drop500 last15k | 8 | 46 | −38 |

Last mới giữ thêm3 GT nguồn, số GT được cứu vẫn8; không diễn giải phần tăng
TP này thành tăng số GT được cứu từ các trường hợp TPE bỏ sót.

## Phép so trên lịch1k chung

Nếu chỉ xét0,1k,2k,…,15k, giữ cùng rule min_delta:

| Run | Selected step trên lịch1k | mAP50–95 (%) |
|---|---:|---:|
| Drop1k | 5.000 | 38,6609 |
| Drop500 | 1.000 | 38,6788 |

Chênh selected best trên lịch1k là**+0,0179 điểm**. Tuy nhiên cả hai thực tế
đều dùng lịch dày và chọn500; không dùng phép so phụ này thay kết luận selected
best chính không đổi.

## Hậu kiểm

- 44 checkpoint eval có hash hợp lệ; selected best/last khớp weights và optimizer
  tại step được ghi. Mỗi run có đủ22 mốc.
- Stream15k giống hệt nhau; manifest/code/source/bank/mask/loss/batch/optimizer
  khớp. Chỉ khác `lr_schedule.drop_after_update` là1.000 hoặc500.
- Weights/optimizer tại500 khớp chính xác; LR thực trong optimizer đúng theo
  milestone từng run. Reference và toàn bộ phần model đóng băng không đổi.
- Đã nạp lại4 best/last và checkpoint drop500 selected trên lịch1k, chấm lại
  full validation trên GPU. Mọi quality metric, per-class và GT transitions khớp
  log. TPE và S0 cũng được chấm lại, tổng7 bộ metric.
- **State dict của best hai run giống hệt nhau** theo `torch.equal` trên từng
  tensor. Artifact mới có metadata lịch LR khác, nhưng không phải model inference
  tốt hơn hay một bộ weights mới tại selected best.
- Checkpoint nguồn, bank và các checkpoint cũ được bảo toàn.

![So sánh hai thời điểm giảm LR](assets/kb2_hybrid_drop500_20261106/learning_curves.png)

Artifact kiểm chứng:
[review.json](assets/kb2_hybrid_drop500_20261106/review.json),
[summary.csv](assets/kb2_hybrid_drop500_20261106/summary.csv),
[deltas.csv](assets/kb2_hybrid_drop500_20261106/deltas.csv),
[paired_steps.csv](assets/kb2_hybrid_drop500_20261106/paired_steps.csv),
[classes.csv](assets/kb2_hybrid_drop500_20261106/classes.csv),
[gt_transitions.csv](assets/kb2_hybrid_drop500_20261106/gt_transitions.csv).
Script:
[review_kb2_hybrid_drop500.py](../../scripts/review_kb2_hybrid_drop500.py).

Output mới:
`kb2_preference_optimization/outputs/yolov8n_hybrid_drop500_s42_20261106_trial1/run/dpo/`.

## Kết luận

**Giảm LR sau500 thay vì1.000 giúp nhiều metric ở last tốt hơn, nhưng chưa
nâng selected best.** Giữ best500 của lượt đã có làm ứng viên model hiện tại;
không cần thay model chỉ vì có output folder mới.

Đây là một continuation seed trên validation phát triển. Không kết luận thời
điểm500 luôn kém/tốt hơn1k ở seed hoặc dữ liệu khác. Chưa chạy thêm seed,
không dùng test và không sửa annotation trong phép thử này.
