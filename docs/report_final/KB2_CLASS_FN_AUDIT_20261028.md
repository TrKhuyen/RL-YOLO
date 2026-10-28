# KB2 — Bảng bỏ sót GT theo class, 28/10/2026

**Phạm vi:** YOLOv8n, checkpoint supervised S0 và best DPO của ba seed 42/43/44 đã chạy ngày 26/10. Giữ nguyên ảnh, annotation, checkpoint, scorer và cấu hình huấn luyện. Chỉ suy luận GPU và phân tích; không huấn luyện, không dùng test.

## Kết quả chính

- Validation: 479 ảnh / 1.887 GT / 28 class. Tỷ lệ bỏ sót tăng ở **22/28 class**, giữ nguyên ở 4, giảm ở 2.
- FN: **867 → 924**, tăng ròng **57 GT/run**. Trung bình mỗi run **mất thêm 70.33 GT** từng được S0 phát hiện và **cứu lại 13.33 GT** S0 bỏ sót. Không diễn giải 57 là số GT mới mất.
- Recall tổng: **54.05% → 51.03%**. Macro recall (28 class có GT, mỗi class ngang trọng số): **54.47% → 52.16%**.
- Train: tái hiện đúng 4 output/ảnh của S0 trên toàn bộ 2.411 ảnh. Ít nhất một trong bốn output phát hiện được **10,175/10,333 GT (98.47%)**; còn **158 GT** cả bốn output cùng bỏ sót.
- Trong 2.357 cặp đã dùng train, **1735 cặp (73.61%)** có chosen giữ nguyên số TP nhưng giảm FP; **266 cặp (11.29%)** tăng số TP; **20 cặp** giảm số TP và giảm FP. Đây là đặc điểm tín hiệu preference hiện có, chưa phải bằng chứng nhân quả giải thích toàn bộ giảm recall.

## 1. Validation: tỷ lệ bỏ sót theo class

Confidence 0,25; matching đúng class, một-một, IoU ≥ 0,5, theo thứ tự confidence của evaluator chuẩn. Trước đó inference dùng confidence 0,001, NMS IoU 0,60, tối đa 300 detection. S0 và DPO dùng cùng preprocessing, image size 640 và batch size 8.

**Bỏ sót (%) = FN / số GT × 100.** DPO, cứu lại và mất thêm là trung bình ba seed; số GT trung bình có thể là số lẻ. Δ dương nghĩa bỏ sót tăng, đơn vị điểm phần trăm. Sắp theo tỷ lệ bỏ sót DPO giảm dần. Recall = 100% − tỷ lệ bỏ sót; CSV có cả recall.

| ID | Class | GT | FN S0 | FN DPO TB | Bỏ sót S0 (%) | Bỏ sót DPO (%) | Δ (điểm) | Cứu lại TB | Mất thêm TB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | Potato leaf | 41 | 38 | 38.00 | 92.68 | 92.68 | +0.00 | 0.00 | 0.00 |
| 21 | Tomato leaf bacterial spot | 60 | 52 | 50.00 | 86.67 | 83.33 | -3.33 | 2.00 | 0.00 |
| 18 | Tomato Early blight leaf | 40 | 28 | 29.00 | 70.00 | 72.50 | +2.50 | 0.00 | 1.00 |
| 12 | Potato leaf early blight | 172 | 120 | 124.33 | 69.77 | 72.29 | +2.52 | 1.00 | 5.33 |
| 24 | Tomato leaf yellow virus | 164 | 105 | 107.00 | 64.02 | 65.24 | +1.22 | 2.00 | 4.00 |
| 13 | Potato leaf late blight | 62 | 36 | 40.33 | 58.06 | 65.05 | +6.99 | 0.00 | 4.33 |
| 25 | Tomato mold leaf | 50 | 32 | 32.00 | 64.00 | 64.00 | +0.00 | 1.00 | 1.00 |
| 4 | Bell_pepper leaf spot | 49 | 30 | 30.00 | 61.22 | 61.22 | +0.00 | 1.00 | 1.00 |
| 23 | Tomato leaf mosaic virus | 41 | 24 | 24.67 | 58.54 | 60.16 | +1.63 | 0.33 | 1.00 |
| 6 | Cherry leaf | 70 | 36 | 40.67 | 51.43 | 58.10 | +6.67 | 0.00 | 4.67 |
| 20 | Tomato leaf | 125 | 57 | 67.00 | 45.60 | 53.60 | +8.00 | 0.00 | 10.00 |
| 0 | Apple Scab Leaf | 32 | 16 | 17.00 | 50.00 | 53.12 | +3.12 | 0.00 | 1.00 |
| 19 | Tomato Septoria leaf spot | 60 | 30 | 31.00 | 50.00 | 51.67 | +1.67 | 1.00 | 2.00 |
| 15 | Soyabean leaf | 66 | 30 | 32.00 | 45.45 | 48.48 | +3.03 | 0.00 | 2.00 |
| 22 | Tomato leaf late blight | 49 | 20 | 23.00 | 40.82 | 46.94 | +6.12 | 0.00 | 3.00 |
| 10 | Peach leaf | 70 | 24 | 30.00 | 34.29 | 42.86 | +8.57 | 0.00 | 6.00 |
| 16 | Squash Powdery mildew leaf | 54 | 20 | 21.00 | 37.04 | 38.89 | +1.85 | 0.00 | 1.00 |
| 3 | Bell_pepper leaf | 29 | 10 | 11.00 | 34.48 | 37.93 | +3.45 | 0.00 | 1.00 |
| 27 | grape leaf black rot | 14 | 5 | 5.00 | 35.71 | 35.71 | +0.00 | 0.00 | 0.00 |
| 2 | Apple rust leaf | 46 | 14 | 16.00 | 30.43 | 34.78 | +4.35 | 0.00 | 2.00 |
| 5 | Blueberry leaf | 158 | 52 | 54.67 | 32.91 | 34.60 | +1.69 | 2.33 | 5.00 |
| 7 | Corn Gray leaf spot | 14 | 6 | 4.00 | 42.86 | 28.57 | -14.29 | 2.00 | 0.00 |
| 9 | Corn rust leaf | 39 | 10 | 11.00 | 25.64 | 28.21 | +2.56 | 0.00 | 1.00 |
| 17 | Strawberry leaf | 93 | 20 | 25.00 | 21.51 | 26.88 | +5.38 | 0.00 | 5.00 |
| 8 | Corn leaf blight | 56 | 12 | 15.00 | 21.43 | 26.79 | +5.36 | 0.00 | 3.00 |
| 26 | grape leaf | 57 | 14 | 14.33 | 24.56 | 25.15 | +0.58 | 0.67 | 1.00 |
| 14 | Raspberry leaf | 130 | 22 | 26.00 | 16.92 | 20.00 | +3.08 | 0.00 | 4.00 |
| 1 | Apple leaf | 46 | 4 | 5.00 | 8.70 | 10.87 | +2.17 | 0.00 | 1.00 |

Các class ít GT cần đọc cùng support: Corn Gray leaf spot và grape leaf black rot chỉ có 14 GT/class; một object tương đương 7,14 điểm phần trăm. Các tỷ lệ ở đây mô tả split hiện tại, chưa có khoảng tin cậy hoặc kết luận tổng quát ngoài split.

## 2. Chuyển trạng thái GT trên validation

- **Cứu lại:** S0 bỏ sót, DPO phát hiện được cùng GT.
- **Mất thêm:** S0 phát hiện được, DPO bỏ sót cùng GT.
- **Vẫn bỏ sót:** cả S0 và DPO bỏ sót cùng GT.
- Identity của GT dùng image path và thứ tự GT sau transform, trên cùng dataset/preprocessing; không ghép bằng tọa độ giữa hai checkpoint.
- Matching một-một ở ảnh đông object có thể đổi GT được gán khi prediction thay đổi. Các con số là chuyển trạng thái theo matcher chuẩn, chưa chứng minh box hoàn toàn biến mất khỏi dense output.

| Seed | Best step | Cứu lại | Mất thêm | Vẫn bỏ sót | FN cuối | FN tăng ròng |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 9000 | 14 | 67 | 853 | 920 | +53 |
| 43 | 9000 | 13 | 69 | 854 | 923 | +56 |
| 44 | 14000 | 13 | 75 | 854 | 929 | +62 |

Đã kiểm tra từng class/từng seed: `FN_DPO − FN_S0 = mất thêm − cứu lại`. Bảng chi tiết mỗi class và seed nằm trong CSV.

## 3. Train: class nào có tín hiệu cứu GT trong preference?

Bảng này dùng matcher của scorer: matching đúng class, một-một, tối đa số match rồi tối đa IoU, tại IoU 0,5. Output được lấy mẫu quyết định có xuất detection; không áp confidence 0,25. Vì vậy đây là phân tích preference trên train, không phải recall inference thông thường.

- **Cặp có cứu GT:** chosen phát hiện ít nhất một GT của class mà rejected bỏ sót. Một cặp có thể đồng thời cứu một GT và mất một GT khác; hai cột không phải hai nhóm loại trừ nhau.
- **Phủ bởi 4 output:** tỷ lệ GT được ít nhất một trong bốn output phát hiện, tính trên toàn bộ train, kể cả ảnh không giữ được cặp. Đây là giới hạn cơ hội chọn output trong pool đã sinh, không phải output duy nhất có thể phát hiện tất cả GT đó.
- Số cặp theo class tính các ảnh có GT của class; ảnh có nhiều class có thể được tính ở nhiều hàng. Tổng số cặp unique vẫn là 2.357.

| ID | Class | GT train | Cặp chứa class | Cặp có cứu GT | Cặp có mất GT | Phủ bởi 4 output (%) | GT cả 4 cùng bỏ sót |
|---:|---|---:|---:|---:|---:|---:|---:|
| 23 | Tomato leaf mosaic virus | 351 | 41 | 11 | 1 | 96.30 | 13 |
| 25 | Tomato mold leaf | 325 | 70 | 15 | 5 | 96.31 | 12 |
| 21 | Tomato leaf bacterial spot | 352 | 91 | 26 | 3 | 96.59 | 12 |
| 18 | Tomato Early blight leaf | 354 | 74 | 20 | 2 | 96.61 | 12 |
| 24 | Tomato leaf yellow virus | 523 | 43 | 28 | 8 | 96.75 | 17 |
| 19 | Tomato Septoria leaf spot | 389 | 107 | 23 | 2 | 97.17 | 11 |
| 8 | Corn leaf blight | 327 | 137 | 20 | 4 | 97.25 | 9 |
| 16 | Squash Powdery mildew leaf | 351 | 118 | 9 | 1 | 98.29 | 6 |
| 5 | Blueberry leaf | 598 | 70 | 9 | 2 | 98.33 | 10 |
| 13 | Potato leaf late blight | 361 | 81 | 17 | 2 | 98.34 | 6 |
| 12 | Potato leaf early blight | 371 | 92 | 12 | 0 | 98.38 | 6 |
| 14 | Raspberry leaf | 520 | 78 | 4 | 1 | 98.46 | 8 |
| 20 | Tomato leaf | 397 | 54 | 9 | 1 | 98.74 | 5 |
| 4 | Bell_pepper leaf spot | 320 | 64 | 12 | 2 | 98.75 | 4 |
| 10 | Peach leaf | 410 | 70 | 7 | 4 | 98.78 | 5 |
| 1 | Apple leaf | 336 | 76 | 0 | 0 | 98.81 | 4 |
| 11 | Potato leaf | 354 | 18 | 8 | 1 | 98.87 | 4 |
| 0 | Apple Scab Leaf | 319 | 100 | 4 | 0 | 99.06 | 3 |
| 15 | Soyabean leaf | 351 | 51 | 8 | 2 | 99.15 | 3 |
| 22 | Tomato leaf late blight | 353 | 98 | 5 | 0 | 99.43 | 2 |
| 17 | Strawberry leaf | 380 | 61 | 1 | 1 | 99.47 | 2 |
| 7 | Corn Gray leaf spot | 202 | 167 | 9 | 0 | 99.50 | 1 |
| 6 | Cherry leaf | 333 | 52 | 8 | 1 | 99.70 | 1 |
| 26 | grape leaf | 355 | 69 | 6 | 1 | 99.72 | 1 |
| 3 | Bell_pepper leaf | 369 | 40 | 6 | 3 | 99.73 | 1 |
| 2 | Apple rust leaf | 346 | 103 | 4 | 0 | 100.00 | 0 |
| 9 | Corn rust leaf | 336 | 281 | 0 | 0 | 100.00 | 0 |
| 27 | grape leaf black rot | 350 | 95 | 2 | 0 | 100.00 | 0 |

Trên 2.357 cặp unique, **281 cặp** cứu ít nhất một GT và **47 cặp** làm mất ít nhất một GT. Tổng số GT cứu/mất là **368/53**; số TP ròng chosen cao hơn rejected **315**. Có **213 GT** trong các ảnh có cặp bị cả chosen lẫn rejected bỏ sót.

Có **54 ảnh / 692 GT (6.70% GT train)** không giữ được cặp và không xuất hiện trong replay huấn luyện v1. Trong nhóm này, 4 output phủ được 677/692 GT; chỉ 15 GT cả bốn output cùng bỏ sót. Vì thế không được gọi toàn bộ 54 ảnh là các ảnh model không nhận diện được. Nếu thiết kế lại, có thể giữ các ảnh này cho native loss và chỉ bỏ DPO term khi không có cặp phù hợp.

## 4. Ý nghĩa cho bước tiếp theo

1. **Tách hai nhóm ưu tiên:** class vốn có tỷ lệ bỏ sót cao (Potato leaf, Tomato leaf bacterial spot, Tomato Early blight leaf, Potato leaf early blight, Tomato leaf yellow virus) và class suy giảm nhiều sau DPO (Peach leaf, Tomato leaf, Potato leaf late blight, Cherry leaf, Tomato leaf late blight). Đây là nhóm chẩn đoán từ validation, chưa dùng làm trọng số train.
2. **Scorer và chọn cặp là nơi cần thử trước:** phần lớn cặp hiện giữ nguyên TP và giảm FP. Có thể so F2 với scorer hiện tại, rồi mới thử chọn cặp ưu tiên cứu GT với mức FP được kiểm soát. Mỗi thử nghiệm giữ các thành phần khác cố định để đánh giá tác động.
3. **Không vội tăng số draw cho mọi ảnh:** pool 4 output đã phủ 96,30–100% GT train theo class. Vẫn còn 158 GT cần phân tích khả năng sinh output, nhưng thiếu phương án phát hiện trên train chưa giải thích được toàn bộ mất recall khi inference.
4. **Khoảng cách cần theo dõi:** sampled output, chất lượng sau chọn cặp và inference YOLO thông thường khác nhau. Không so trực tiếp 98,47% union coverage train với 51,03% recall deterministic validation để kết luận overfit hoặc khẳng định sampler giải quyết được recall.
5. **Chưa đặt trọng số class từ bảng validation này:** nếu thử class weighting hoặc cân bằng tần suất cặp, ước lượng và giới hạn trọng số trên train; theo dõi FP, macro recall và recall từng class trên valid. Không cần thay ảnh hoặc annotation.

## 5. Kiểm chứng và tệp kết quả

- Tái tính TP/FP/FN trên GPU: khớp chính xác mọi class của S0 và best seed 42/43/44 với ba `validation_comparison.json` đã lưu.
- Replay cùng seed 42, cùng sampler, cùng policy, cùng 4 draw: cả **4.714 action và output chosen/rejected** khớp chính xác artifact cũ. Bốn output được lấy mẫu lại để đo union coverage; không chỉ suy đoán từ hai output đã lưu.
- Hash checkpoint, pairs, dataset, nguồn huấn luyện và runtime được kiểm tra. State digest model giữ nguyên qua suy luận. Không sửa nguồn Python trong package KB2 để giữ provenance các run cũ.

- [CSV validation theo class](assets/kb2_20261028/validation_by_class.csv)
- [CSV validation theo class và seed](assets/kb2_20261028/validation_by_class_seed.csv)
- [CSV train preference theo class](assets/kb2_20261028/train_preferences_by_class.csv)
- [Snapshot JSON đầy đủ](assets/kb2_20261028/audit.json)
- [Danh sách GT và chuyển trạng thái validation](assets/kb2_20261028/validation_gt_transitions.jsonl)
- [Danh sách GT và độ phủ bốn output train](assets/kb2_20261028/train_gt_coverage.jsonl)
- [Script tái lập](../../scripts/audit_kb2_class_fn.py)

Lệnh chạy lại với thư mục output mới (script từ chối ghi đè):

```bash
YOLO_CONFIG_DIR=/tmp MPLCONFIGDIR=/tmp/kb2-matplotlib NO_ALBUMENTATIONS_UPDATE=1 .venv-ubuntu/bin/python scripts/audit_kb2_class_fn.py --device cuda --output docs/report_final/assets/kb2_class_fn_recheck
```
