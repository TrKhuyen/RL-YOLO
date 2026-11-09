# Đánh giá hybrid TPE → DPO so với YOLOv8n cũ — 06/11/2026

**Bổ sung lượt30k:** xem mục7. Lượt30k đã hoàn tất nhưng best mAP vẫn ở1k,
weights/optimizer best giống lượt15k; kéo dài không nâng best AP.

## 1. Kết luận và phạm vi

Hybrid hoàn tất **15.000 update**, seed42; checkpoint được chọn theo mAP50–95
validation nằm ở **bước1.000**. So KB2 cũ cùng seed42, hybrid best tăng mAP,
AP50/AP75, recall và micro F1; precision giảm nhẹ, FP tăng và macro F1 giảm.
So checkpoint TPE trước continuation, mAP chỉ tăng nhẹ trong khi recall giảm.
Kết quả là một cải thiện có đánh đổi trên validation, chưa phải tăng đồng thời mọi metric.

Run:
`kb2_preference_optimization/outputs/yolov8n_hybrid_tpe2_s42_20261105_180813_658387/`.

“YOLOv8n cũ” trong phép so chính là **S0 → KB2 DPO seed42**,
`yolov8n_strength_long_3seeds_20261103_095853_370249/s42/dpo/best.pt`, selected step12k.
S0 supervised được báo riêng để tránh nhập nhằng. Các checkpoint chọn best bằng
cùng quy tắc AP nhưng ở step khác nhau; đây là so selected models, không phải
đối chứng chỉ đổi objective hoặc cùng số update tại best.

## 2. Bảng so sánh best

Toàn bộ đo trên479 ảnh validation,1.887GT,28 class. AP/P/R/F1 tính theo%;
P/R/F1/FP/FN tại confidence0,25, IoU match0,5. Các mốc dùng cùng evaluator,
ordinary inference, input640, NMS0,6 và cap300. TPE được chấm lại, không dùng
số canonical lịch sử38,5071% thay kết quả runtime hiện tại38,5097%.

| Chỉ số | S0 supervised | KB2 cũ seed42 best12k | TPE trước DPO | Hybrid best1k |
|---|---:|---:|---:|---:|
| mAP50–95 | 37,8218 | 38,1363 | 38,5097 | **38,5962** |
| AP50 | 49,1225 | 49,5543 | 50,4306 | **50,5191** |
| AP75 | 42,9680 | 43,2303 | 43,4945 | **43,6132** |
| Precision | 50,9227 | 53,0063 | 50,8763 | 52,8037 |
| Recall | 54,1070 | 53,2591 | 55,3789 | 53,8951 |
| Micro F1 | 52,4666 | 53,1324 | 53,0322 | **53,3438** |
| Macro F1 | 51,3707 | 52,5780 | 51,2818 | 51,7057 |
| Macro recall | 54,5463 | 54,5880 | 55,9285 | 54,7504 |
| AR300 | 56,2044 | 56,1909 | 58,2612 | 57,7955 |
| AP small | 4,3965 | 4,0786 | 4,0997 | 4,2596 |
| AP medium | 17,9704 | 17,7595 | 18,4362 | 18,2450 |
| AP large | 42,5676 | 43,0132 | 43,8593 | **43,8808** |
| TP | 1.021 | 1.005 | 1.045 | 1.017 |
| FP | 984 | 891 | 1.009 | 909 |
| FN | 866 | 882 | 842 | 870 |

Nguồn: [summary.csv](assets/kb2_hybrid_review_20261106_v2/summary.csv),
[deltas.csv](assets/kb2_hybrid_review_20261106_v2/deltas.csv).

## 3. Cải thiện so KB2 cũ và S0 đến đâu?

So KB2 cũ cùng seed42, hybrid best:

- mAP tăng **0,4599 điểm phần trăm**, AP50 tăng0,9648 và AP75 tăng0,3829 điểm.
- Recall tăng0,6359 điểm, tương ứng thêm12TP/giảm12FN.
- Micro F1 tăng0,2114 điểm, nhưng precision giảm0,2026 điểm và thêm18FP.
- AP small/medium/large tăng lần lượt0,1810/0,4855/0,8676 điểm.
- Macro F1 giảm0,8723 điểm; per-class AP có14 class tăng và14 class giảm.

Vì vậy hybrid tốt hơn về AP tổng thể và recall so KB2 cũ, nhưng không trội hoàn
toàn về điểm vận hành hoặc phân bố class. Đây là thay cả initialization và bank,
không phải bằng chứng rằng DPO objective đã được cải tiến. So với checkpoint
KB2 cũ có AP cao nhất trong ba seed,38,2187% ởseed43, hybrid vẫn cao hơn0,3774 điểm;
phép so khác seed này chỉ có giá trị mô tả, không thay đối chứng cùng seed.

So S0, hybrid best tăng mAP **0,7744 điểm**, AP50 1,3966 điểm, AP75 0,6452 điểm,
precision1,8810 điểm, micro F1 0,8772 điểm; FP giảm75. Recall giảm0,2120 điểm
và FN tăng4; APsmall giảm0,1369 điểm. So S0, per-class AP có15 class tăng/13 giảm.

## 4. Giai đoạn KB2 bổ sung bao nhiêu sau TPE?

Ở best1k, hybrid−TPE:

- mAP **+0,0865 điểm**, AP50 +0,0886 và AP75 +0,1187 điểm.
- Precision +1,9274 điểm, micro F1 +0,3116 điểm, FP giảm100.
- Recall **−1,4838 điểm**, TP giảm ròng28 và FN tăng28.
- GT identity: cứu5GT TPE bỏ sót nhưng mất33GT TPE từng phát hiện.
- APsmall +0,1599 điểm, APmedium −0,1912 và APlarge +0,0216 điểm.
- Per-class AP có18 class tăng/10 giảm, nhưng gain macro AP tổng cộng vẫn nhỏ.

Trong tổng gain mAP so S0, **+0,6879 điểm** đã có ở TPE; continuation thêm
**+0,0865 điểm**, tạo tổng+0,7744 điểm. Phần lớn cải thiện AP của pipeline hiện
đến từ checkpoint trước giai đoạn KB2. Đây là phân rã số học giữa ba mốc;
không suy ra hiệu ứng nhân quả của TPE so với mọi phương pháp search khác.

Precision tăng vì giảm100FP trong khi TP giảm28. Model có ít detection sai hơn
nhưng cũng bỏ thêm object đúng tại0,25. Box regression đóng băng nên chưa có cơ sở
gọi đây là cải thiện tọa độ do DPO. Mean IoU trên common detected GT so TPE cũng
giảm nhẹ0,87025→0,86955; AP75 tăng có thể đến từ ranking/selection.

Lượt này **không có TPE → native-only**. Vì loss là native+DPO, gain+0,0865 điểm
đo tác dụng của giai đoạn continuation tổng thể, chưa cô lập auxiliary DPO.

## 5. Train dài có tốt hơn checkpoint1k không?

| Mốc | mAP50–95 | Precision | Recall | F1 | APsmall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| TPE step0 | 38,5097 | 50,8763 | 55,3789 | 53,0322 | 4,0997 | 1.009 | 842 |
| Hybrid best1k | **38,5962** | 52,8037 | 53,8951 | 53,3438 | 4,2596 | 909 | 870 |
| Hybrid11k | 38,3690 | **55,7854** | 52,8882 | **54,2982** | — | 791 | 889 |
| Hybrid last15k | 38,3230 | 55,6429 | 52,5172 | 54,0349 | **7,2676** | 790 | 896 |

Best AP ở1k; train đến15k làm mAP thấp hơn best0,2732 điểm và thấp hơn TPE0,1867 điểm.
Ngược lại, precision/F1 tăng và FP giảm tiếp. APsmall ở15k tăng nhưng recall
object nhỏ tại ngưỡng0,25 vẫn cần đọc riêng: so TPE, nhóm nhỏ không cứu thêm GT
nào và mất7GT. APsmall tăng không đồng nghĩa recall nhỏ ở0,25 tăng.

Mốc11k có micro F1 cao nhất quan sát trong16 mốc, nhưng không thay selector đã
khóa theo AP. Đây là phân tích sau chạy về đánh đổi. Không đủ dữ liệu để xác định
chính xác nguyên nhân suy giảm AP là overfitting hay dịch chuyển ranking/score.

![Diễn biến AP, precision và recall](assets/kb2_hybrid_review_20261106_v2/learning_curve.png)

Giữ `dpo/best.pt` ở1k làm ứng viên theo mục tiêu AP đã chốt; giữ last và các step
khác để phân tích, không mặc định15k tốt hơn. Chưa có căn cứ tăng thêm step với
recipe này để kỳ vọng AP tăng trở lại. Cần xác nhận thêm seed trước khi khẳng định
gain sau TPE lặp lại ổn định; giữ recipe/selector cố định trong phép xác nhận đó.

## 6. Kiểm chứng và giới hạn

[review.json](assets/kb2_hybrid_review_20261106_v2/review.json) lưu kiểm source/runtime,
bank hashes,16 checkpoint eval, tái dựng đủ15k stream từ seed/bank, đối chiếu
weights/optimizer best-last với step đã chọn, và nạp lại evaluation đầy đủ cho
S0, KB2 cũ seed42, TPE, hybrid best/last. Source và checkpoint được giữ nguyên.
Script: [review_kb2_hybrid_long.py](../../scripts/review_kb2_hybrid_long.py).

Bank TPE có773 cặp,662 train/111 probe; recovery167, preservation80,
background468, duplicate58. Raw pair counts không đồng nghĩa tần suất update:
sampler manifest cho xác suất bốn nhóm xấp xỉ25% mỗi nhóm. Bank khác bank S0 cũ,
nên không gọi hai lượt là cùng candidate bank. Train wall871,8 giây (~14,53 phút)
có periodic eval, chưa tính toàn bộ sinh bank/initial evaluation và giai đoạn KB3.

Chỉ một continuation seed, validation đã dùng phát triển và chọn checkpoint;
chưa có dữ liệu xác nhận độc lập. Mục tiêu nhiều metric tốt lên đã đạt ở một số
phép so, nhưng recall so TPE và macro F1 so KB2 cũ là các đánh đổi cần công bố.

## 7. Xác nhận kết quả chạy 30.000 bước

Run mới:
`kb2_preference_optimization/outputs/yolov8n_hybrid_tpe2_s42_20261106_030407_969341/`.

Lượt trước dừng ở quá trình sinh bank với segmentation fault và chưa train.
Lượt này dùng bank đầy đủ của run15k, cùng source TPE, seed42, λ0,1, β0,1,
native background0,5 và optimizer; đã hoàn tất30k, không có lỗi ghi nhận ở run này.
Train wall gồm periodic eval là **29,42 phút**, không gồm toàn bộ initial eval.

| Chỉ số (%) | TPE step0 | Hybrid best1k | Last15k | Last30k |
|---|---:|---:|---:|---:|
| mAP50–95 | 38,5097 | **38,5962** | 38,3230 | 38,2956 |
| AP50 | 50,4306 | **50,5191** | 50,3502 | 50,4246 |
| AP75 | 43,4945 | **43,6132** | 43,1832 | 43,1016 |
| Precision | 50,8763 | 52,8037 | 55,6429 | **56,3469** |
| Recall | **55,3789** | 53,8951 | 52,5172 | 51,9873 |
| Micro F1 | 53,0322 | 53,3438 | 54,0349 | **54,0794** |
| Macro F1 | 51,2818 | 51,7057 | **52,4395** | 52,2972 |
| Macro recall | **55,9285** | 54,7504 | 52,8784 | 52,1282 |
| APsmall | 4,0997 | 4,2596 | **7,2676** | 5,3752 |
| APmedium | 18,4362 | 18,2450 | 18,5040 | **18,8761** |
| APlarge | 43,8593 | **43,8808** | 43,3372 | 43,2285 |
| TP | 1.045 | 1.017 | 991 | 981 |
| FP ↓ | 1.009 | 909 | 790 | **760** |
| FN ↓ | **842** | 870 | 896 | 906 |

Nguồn: [summary.csv](assets/kb2_hybrid_30k_20261106/summary.csv).
Các dòng FP/FN/TP dùng số đếm, không phải%.

**So last30k với last15k:** precision tăng0,7040 điểm phần trăm và FP giảm30,
nhưng recall giảm0,5299 điểm, TP giảm10/FN tăng10. Micro F1 tăng rất ít0,0445 điểm;
macro F1 giảm0,1422 điểm. mAP giảm0,0274 điểm, AP75 giảm0,0816 điểm và APsmall giảm1,8924 điểm.
AP50 tăng0,0744 điểm, APmedium tăng0,3722 điểm. Các metric tiếp tục có đánh đổi;
không có cải thiện tổng thể nhờ tăng budget này.

**So TPE:** last30k thấp hơn0,2141 điểm mAP và3,3916 điểm recall, dù precision
cao hơn5,4706 điểm và micro F1 cao hơn1,0472 điểm. FP giảm249 nhưng FN tăng64.
Không nên thay best AP bằng last30k theo selector đã chốt.

**Best không thay đổi:** step1k ở hai lượt có weights và optimizer giống hệt nhau.
Step15k trong lượt30k cũng khớp weights/optimizer của last lượt15k; stream và
mọi validation metric ở16 mốc đầu khớp. Đây thực sự là tiếp tục khảo sát cùng
quỹ đạo tối ưu thêm15k update, dù launcher khởi tạo lại từ TPE ở đầu lượt.
Các file checkpoint chứa manifest khác budget nên không khẳng định bytes của
toàn bộ file `.pt` giống nhau.

Maxima quan sát trong31 mốc: mAP38,5962% ở1k; precision56,4117% ở29k;
F1 54,2982% vẫn ở11k; APsmall7,4018% ở24k và APmedium18,8786% ở26k.
Recall cao nhất là TPE step0,55,3789%. Các mốc này là diagnostic sau chạy,
không thay đổi quy tắc chọn checkpoint theo AP.

Đã kiểm hash31 checkpoint eval, best/last với saved steps, source/runtime/bank,
tái dựng đủ30k stream; nạp lại last30k trên GPU và xác nhận toàn bộ quality metrics
trên479 ảnh/1.887GT, đồng thời chấm lại S0 và TPE. Best mới được xác nhận bằng
state equality với best15k đã nạp lại trước đó. Bằng chứng:
[verification.json](assets/kb2_hybrid_30k_20261106/verification.json),
[reloaded_last.json](assets/kb2_hybrid_30k_20261106/reloaded_last.json).

Kết luận cập nhật: **giữ best1k nếu tối ưu mAP; tăng từ15k lên30k không nâng
best AP hoặc best F1 của recipe này**. Việc giảm FP và tăng precision đi kèm
mất thêm TP ở confidence0,25. Kết quả không ủng hộ tiếp tục tăng step đơn thuần
để giải quyết recall; chưa đủ để kết luận nguyên nhân cụ thể là overfitting.
Đây vẫn là một continuation seed và validation đã dùng phát triển.
