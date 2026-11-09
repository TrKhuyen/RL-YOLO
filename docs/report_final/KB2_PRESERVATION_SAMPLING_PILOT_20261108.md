# KB2 — pilot tăng cơ hội học preservation, 08/11/2026

## Giả thuyết và phép thử đã khóa

Best KB2 trước thử nghiệm có mAP50–95 38,7133% tại step 500. So với TPE,
checkpoint này cứu 3 GT nhưng làm mất 24 GT tại confidence 0,25. Phép thử đặt
câu hỏi: tăng cơ hội học giữ detection đúng, đồng thời giảm cơ hội học loại
background/duplicate, có giúp cải thiện mAP và giảm mất GT không?

Đây là thử nghiệm sampler của **toàn bộ pipeline KB2 DPO**. Không thay công
thức loss, không thêm native-only và không chỉnh annotation.

| Thiết lập | Giá trị |
|---|---|
| Khởi tạo/reference | Best TPE YOLOv8n trial 2, không phải best KB2 |
| Ngân sách/seed | 2.000 update / 42 |
| λ / β | 0,03 / 0,1 |
| Native foreground/background | 1 / 0,5 |
| Optimizer | AdamW; weight decay 5e-4; clip norm 1 |
| LR | 1e-6 tới hết update 1.000; 1e-7 từ update 1.001 |
| Batch mỗi update | 2 preference decision; 4 native images |
| Phần được học | Classification cv3; giữ nguyên mask của lượt cũ |
| Eval | Full validation 479 ảnh / 1.887 GT, mỗi 250 update tới 2k |
| Chọn best | mAP50–95; min_delta 1e-4 fraction; TPE step 0 hợp lệ |
| Đối chiếu | Cùng chín mốc 0–2k của run drop1k đã hoàn tất |

Sampler mới đề xuất 35% preservation, 25% recovery, 20% background và 20%
duplicate, trước giới hạn xác suất theo ảnh. Trong mỗi nhóm, giữ nguyên
cân bằng class/target, class_mix 0,5 và giới hạn expected image exposure 5×.
Không dùng lỗi trên validation để chọn cặp train. Các tỷ lệ này là giả thuyết
thử nghiệm, không được coi là tỷ lệ tối ưu có sẵn.

**Phạm vi can thiệp:** hai ảnh uniform-native mỗi update giữ nguyên thứ tự.
Hai ảnh native còn lại lấy từ cặp preference, nên thay sampler cũng thay hai
ảnh này. Đây không phải phép đo tác động riêng của DPO loss trên native batch
cố định. Loss, mask, optimizer và lịch LR giữ nguyên.

## Audit bank trên train trước khi chạy

Bank đầy đủ: 2.411 ảnh train / 10.333 GT, 773 cặp. Loại 111 cặp probe khỏi
DPO train, còn 662 cặp trên 392 ảnh có cặp hợp lệ. Probe chỉ là ảnh train
không được lấy mẫu cho DPO; native uniform vẫn có thể dùng chúng, nên không
gọi đây là holdout độc lập.

| Nhóm | Cặp train | Xác suất cũ sau cap | Đề xuất mới | Xác suất mới sau cap |
|---|---:|---:|---:|---:|
| Preservation | 68 | 24,9402% | 35% | 34,9095% |
| Recovery | 149 | 24,9155% | 25% | 24,9206% |
| Background | 391 | 25,1397% | 20% | 20,1417% |
| Duplicate | 54 | 25,0046% | 20% | 20,0281% |

68 cặp preservation bao phủ 63 ảnh và 22/28 class. Đã replay GT judge cho
toàn bộ 68 cặp, evidence khớp bank. Confidence của chúng: min 0,2514,
median 0,3038, max 0,9298; 45 cặp nằm trong [0,25; 0,35), 20 cặp trong
[0,35; 0,50). Bank vì vậy có nhiều cặp preservation sát ngưỡng, nhưng không
bao phủ mọi class và mọi GT. Các class ID 0, 2, 9, 14, 22, 27 không có cặp
preservation train; đổi sampler không tự sinh thêm cặp cho các class này.

Với seed 42 và 4.000 quyết định của pilot, kế hoạch lấy mẫu đã khóa:

| Nhóm | Sampler cũ | Sampler mới |
|---|---:|---:|
| Preservation | 996 | 1.397 |
| Recovery | 970 | 983 |
| Background | 984 | 802 |
| Duplicate | 1.050 | 818 |

Giới hạn 5× áp dụng cho xác suất kỳ vọng theo ảnh, không phải giới hạn cứng
số lần lặp ở một lượt lấy mẫu hữu hạn. Tỷ lệ cặp trong bank, xác suất lấy mẫu
và số lần xuất hiện thực tế là ba đại lượng khác nhau.

## Artifact và kiểm chứng

Output riêng:
`kb2_preference_optimization/outputs/yolov8n_hybrid_sampling_s42_20261108_pilot1/`.

Đối chiếu:
`kb2_preference_optimization/outputs/yolov8n_hybrid_schedule_s42_20261106_trial3/run/`.

`plan.json` và `bank_sampling_audit.json` được ghi trước training. Code mới
nằm trong `hybrid_sampling/`, giữ nguyên byte các engine lịch sử để bảo toàn
fingerprint/checkpoint cũ. Chín kiểm tra sampler/lịch LR đạt, trong đó có:
chế độ 25% mỗi nhóm tái hiện sampler cũ; cap theo ảnh vẫn đúng; từ chối probe
và group thiếu; AST vòng cập nhật giống trainer đối chiếu.

Runtime dùng GPU CUDA với TF32 tắt, thread host bằng 1 và affinity CPU 16–23,
giữ thiết lập của lượt trước đã chạy thành công. Không sửa cấu hình hệ thống.

Chỉ dùng validation phát triển; không dùng test và không tuyên bố khả năng
tổng quát hóa từ một pilot seed 42.

## Kết quả

Đã hoàn tất 2.000 update trên CUDA, không gặp segmentation fault. Train wall
5,86 phút gồm periodic validation/probe, chưa gồm đầy đủ audit ban đầu và
hậu kiểm. Đã nạp lại best và checkpoint 2k của hai run, chấm lại full validation;
TPE và S0 cũng được chấm lại, tổng sáu bộ metric, khớp log chính xác.

### So selected best trên cùng lịch đánh giá 0–2k

Tất cả metric tỷ lệ dưới đây tính theo %. FP/FN tại confidence 0,25, matching
IoU 0,5. Đơn vị chênh lệch tỷ lệ là **điểm phần trăm**.

| Metric | Sampler cũ, best 500 | Sampler mới, best 1.000 | Mới − cũ |
|---|---:|---:|---:|
| **mAP50–95** | **38,7133** | 38,7046 | −0,0087 |
| AP50 | 50,6287 | **50,6606** | +0,0319 |
| AP75 | **43,7080** | 43,7062 | −0,0017 |
| Precision | 52,6208 | **52,8409** | +0,2201 |
| Recall | **54,2660** | 54,2130 | −0,0530 |
| F1 | 53,4307 | **53,5182** | +0,0874 |
| Macro F1 | 51,6693 | **51,8334** | +0,1641 |
| Macro recall | **55,0529** | 55,0460 | −0,0069 |
| AR300 | **58,0731** | 57,8888 | −0,1843 |
| APsmall | 4,1209 | **4,3096** | +0,1886 |
| APmedium | 18,3008 | **18,3701** | +0,0693 |
| APlarge | **44,0333** | 43,9967 | −0,0366 |
| TP | **1.024** | 1.023 | −1 |
| FP ↓ | 922 | **913** | −9 |
| FN ↓ | **863** | 864 | +1 |

Trong 12 metric tỷ lệ liệt kê, sáu tăng và sáu giảm. Một số chênh rất nhỏ;
không coi việc đếm số metric tăng là kiểm định ý nghĩa thống kê. Best mới gần
best cũ về mAP nhưng **chưa vượt best theo tiêu chí đã khóa**. Raw mAP cao
nhất của run mới cũng ở 1k, không có checkpoint cao hơn bị bỏ do min_delta.

| Model | mAP50–95 | AP50 | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | 37,8218 | 49,1225 | 50,9227 | 54,1070 | 52,4666 | 984 | 866 |
| TPE | 38,5097 | 50,4306 | 50,8763 | 55,3789 | 53,0322 | 1.009 | 842 |
| KB2 sampler cũ, best 500 | 38,7133 | 50,6287 | 52,6208 | 54,2660 | 53,4307 | 922 | 863 |
| KB2 sampler mới, best 1k | 38,7046 | 50,6606 | 52,8409 | 54,2130 | 53,5182 | 913 | 864 |

Best pilot mới vẫn cao hơn TPE 0,1949 điểm mAP và S0 0,8828 điểm. Đây là lợi
ích của pipeline KB2/hybrid tương ứng so nguồn, **không phải lợi ích tăng thêm
của sampler mới**, vì sampler cũ đã đạt kết quả tương đương hoặc cao hơn mAP.

### So tại cùng update và theo GT identity so TPE

| Step | Sampler | mAP50–95 | FP | FN | GT được cứu | GT bị mất |
|---|---|---:|---:|---:|---:|---:|
| 500 | Cũ | 38,7133 | 922 | 863 | 3 | 24 |
| 500 | Mới | 38,6515 | 924 | 863 | 3 | 24 |
| 1.000 | Cũ | 38,6314 | 904 | 869 | 6 | 33 |
| 1.000 | Mới | 38,7046 | 913 | 864 | 8 | 30 |
| 2.000 | Cũ | 38,6345 | 900 | 870 | 5 | 33 |
| 2.000 | Mới | 38,6271 | 903 | 867 | 8 | 33 |

Ở 1k, sampler mới có tín hiệu đúng hướng: cứu thêm 2 GT và mất ít hơn 3 GT,
TP tăng 5, nhưng FP tăng 9. Ở 2k, số GT bị mất bằng nhau (33); phần TP tăng
3 đến từ số GT được cứu tăng từ 5 lên 8. Vì vậy chưa có bằng chứng hiệu ứng
giữ GT tốt hơn duy trì đến cuối pilot.

So hai selected best ở **hai step khác nhau**, mới cứu 8/mất 30, cũ cứu 3/mất
24; TP ròng mới ít hơn một. Không dùng phép so khác step này để kết luận
riêng sampler gây mất thêm GT; bảng cùng step ở trên giúp phân biệt hai vấn đề.

Tại 2k, sampler mới có recall 54,0541% và F1 53,5433%, so cũ 53,8951% và
53,4700%; FN giảm 3 nhưng FP tăng 3. mAP thấp hơn 0,0074 điểm. Không chuyển
sang chọn checkpoint theo F1 hoặc giảm FN sau khi nhìn kết quả.

### Diễn giải và quyết định

Tăng preservation đã thay đổi cơ hội học đúng như kế hoạch: 1.397 thay vì
996 lượt cặp trong 4.000 quyết định; các số đếm thực tế khớp kế hoạch được
ghi trước training. Tuy nhiên tăng exposure không bảo đảm gradient sẽ cải
thiện mọi detection vì các candidate dùng chung tham số classification và
hai ảnh preference-native cũng thay đổi theo sampler.

Bank preservation chỉ bao phủ 68 GT/63 ảnh và thiếu sáu class; việc đổi trọng
số không bổ sung những quyết định mà bank chưa có. Đây là giới hạn phạm vi
của thử nghiệm, chưa xác định là nguyên nhân duy nhất của kết quả.

Raw box/backbone/BN/reference giữ nguyên; đây vẫn là tối ưu classification.
APsmall/APmedium tăng không chứng minh đã học chỉnh tọa độ hay cứu thêm GT
nhỏ ở confidence cố định: AP tích hợp theo confidence, còn số GT được phát
hiện trong bảng dùng ngưỡng 0,25.

**Giữ checkpoint ứng viên hiện tại: best 500 của sampler cũ.** Lưu pilot mới
như một phép thử có đánh đổi về metric, chưa thay recipe mặc định và chưa
khởi động lượt 15k cho sampler mới. Pilot một seed trên validation phát triển
không chứng minh cấu hình mới kém/tốt hơn một cách tổng quát; chênh mAP rất nhỏ.

## Hậu kiểm sau chạy và tài liệu tái hiện

- Hash 18 checkpoint eval của hai lịch 0–2k hợp lệ; best/last mới khớp weights
  và optimizer của step tương ứng; LR thực đúng tại mọi checkpoint eval.
- Weights và optimizer ban đầu giống hệt nhau; mask, loss, LR, optimizer,
  reference, bank và seed khớp. Khác biệt học tập là sampler; pilot chỉ cắt
  ngân sách ở 2k và dùng cùng prefix lịch eval của lượt cũ.
- Hai ảnh uniform-native mỗi update giống hệt; cặp và hai ảnh preference-native
  thay đổi như đã khai báo. Không mô tả là toàn bộ batch ảnh giống nhau.
- Đã nạp lại bốn checkpoint best/2k, cộng TPE/S0, chấm full validation; metric
  tổng hợp, per-class, GT transitions và threshold diagnostics khớp log.
- Code/nguồn/bank/checkpoint cũ nằm trong danh sách hash bảo vệ không đổi.
- Tổng 22 kiểm tra sampler, LR, entrypoint và hybrid source/update parity đạt.

![Đường học sampler cũ và mới](assets/kb2_preservation_sampling_20261108/learning_curves.png)

[review.json](assets/kb2_preservation_sampling_20261108/review.json),
[summary.csv](assets/kb2_preservation_sampling_20261108/summary.csv),
[deltas.csv](assets/kb2_preservation_sampling_20261108/deltas.csv),
[validation.csv](assets/kb2_preservation_sampling_20261108/validation.csv),
[exposure.csv](assets/kb2_preservation_sampling_20261108/exposure.csv),
[GT transitions](assets/kb2_preservation_sampling_20261108/gt_transitions.csv),
[per-class](assets/kb2_preservation_sampling_20261108/classes.csv),
[probe diagnostics](assets/kb2_preservation_sampling_20261108/probes.csv).

Code mới:
[sampler](../../kb2_preference_optimization/hybrid_sampling/sampler.py),
[trainer](../../kb2_preference_optimization/hybrid_sampling/train.py),
[runner](../../scripts/run_kb2_sampling_pilot.py),
[reviewer](../../scripts/review_kb2_sampling_pilot.py).

Lệnh đã dùng từ repo root, output phải là folder mới:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 YOLO_CONFIG_DIR=/tmp MPLCONFIGDIR=/tmp/kb2-matplotlib NO_ALBUMENTATIONS_UPDATE=1 PYTHONFAULTHANDLER=1 CUDA_LAUNCH_BLOCKING=1 taskset -c 16-23 .venv-ubuntu/bin/python -u -m scripts.run_kb2_sampling_pilot --output kb2_preference_optimization/outputs/yolov8n_hybrid_sampling_s42_20261108_pilot1
```
