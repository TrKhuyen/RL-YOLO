# KB2 Conditional DPO V0 — review pilot 2.000 update, 31/10/2026

> Audit tiếp theo đã hoàn tất: [native/DPO tại GT, actual AdamW và interference](KB2_CONDITIONAL_GRADIENT_AUDIT_20261031.md). DPO kéo đúng target nhưng chưa bù được tác động cập nhật chung; xem full-bank results và giới hạn phép đo.

**Pilot đã hoàn tất cả native-only và native+DPO; cấu hình hiện tại chưa đạt mục
tiêu giảm bỏ sót. Chưa nên chuyển sang run dài với nguyên cấu hình này.**

DPO và native-only cùng FN911/FP820 sau2.000 update, trong khi S0 FN866/FP984.
Cả hai `best.pt` vẫn chứa đúng trọng số S0 step0; `last.pt` là trọng số sau2.000
update. Không coi AP tăng nhẹ hoặc FP giảm là thành công khi mục tiêu FN xấu đi.

## Hoàn tất và provenance

Run: `kb2_preference_optimization/outputs/yolov8n_conditional_s42_20261030_180851_313339/`.

- Objective `conditional_detection_dpo_v0`, seed42, CUDA,2.000 update/arm.
- Bank đầy đủ784 cặp,663 cặp dùng DPO train, offline từ S0; native batch4, preference batch2;
  LR1e-6, beta0,1, lambda0,01; chỉ train classification cv3.
- Cảhai có `completed.json`, `best.pt`, `last.pt`; eval0/1k/2k trên đủ479 ảnh/
  1.887 GT validation. Không có partial bank/evaluation; không chạy test hoặc train thêm.
- Đã kiểm SHA checkpoint best/last, provenance S0/data/code/bank, state_dict best
  bằng S0, frozen state ở last không đổi. Reference không đổi theo kiểm cuối trainer.
- Đã tái dựng đủ2.000 batch từ seed: cả2 ảnh uniform và2 cặp được lấy mẫu đúng;
  native/DPO có stream byte-identical và exposure giống nhau. Stream SHA:
  `377802ba46eaa91b79aae70ff0881ffed8e72b97c1435084ae819b78269edbe0`.
- Review dựa trên logs full validation đã hoàn tất và kiểm checkpoint trên CPU;
  lượt review này không chạy lại inference GPU. Không sửa checkpoint hay dữ liệu.

[Kiểm chứng và dữ liệu đầy đủ](assets/kb2_conditional_pilot_20261031/review.json).

## Kết quả ordinary validation

| Model | Step | mAP50–95 (%) | Micro recall (%) | Macro recall (%) | FN | FP |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0 | 37,8218 | 54,1070 | 54,5463 | 866 | 984 |
| Native-only | 1.000 | 37,9083 | 52,3052 | 53,0388 | 900 | 840 |
| Native+DPO | 1.000 | 37,9060 | 52,3052 | 53,0388 | 900 | 840 |
| Native-only | 2.000 | 37,8882 | 51,7223 | 52,4885 | 911 | 820 |
| Native+DPO | 2.000 | 37,8863 | 51,7223 | 52,4885 | 911 | 820 |

So S0, DPO last tăng45FN, giảm164FP; micro recall giảm2,3847 điểm phần trăm,
macro recall giảm2,0577 điểm. AP tăng0,0645 điểm nhưng thấp hơn native-only
0,00193 điểm. Những chênh AP nhỏ này không chứng minh lợi ích DPO.

So native-only, DPO không thay đổi tổng TP/FP/FN hoặc FN từng class ở cả1k/2k.
Không suy rằng weights hoặc mọi prediction giống hệt nhau: các score và AP vẫn
có khác biệt nhỏ. Kết luận đúng phạm vi là **chưa có lợi ích thêm về recall/FN
được quan sát trong pilot seed42 này**.

Tại cả1k/2k, AP và FP vẫn qua giới hạn đã khóa, nhưng micro recall thấp hơn S0
nên checkpoint không đủ điều kiện. `best.pt` giữ S0 là hành vi đúng của selector,
không phải trainer không lưu được trọng số mới. `last.pt` chứa model đã fine-tune.

S0 của protocol V0 tắt TF32 có FN866; không dùng FN867 trong protocol số học cũ
làm mốc cho pilot này. [validation.csv](assets/kb2_conditional_pilot_20261031/validation.csv).

## GT thực sự được cứu và mất

| Checkpoint DPO | GT cứu so S0 | GT S0 từng đúng bị mất | Net recovery |
|---|---:|---:|---:|
| Step1.000 | 1 | 35 | -34 |
| Step2.000 | 4 | 49 | -45 |

Ở2k,972GT S0 từng đúng còn được giữ,862GT vẫn bị cảhai bỏ sót. Native-only
có cùng số transitions tổng và các bảng class/density/size/overlap được lưu.
Đây là thống kê theo GT identity; không lấy TP ròng để che số lá đã bị mất.

| Mật độ ảnh | GT cứu | GT mất | Net |
|---|---:|---:|---:|
| 1–2 object | 1 | 3 | -2 |
| 3–5 object | 2 | 6 | -4 |
| >=6 object | 1 | 40 | -39 |

40/49 GT mất thuộc ảnh>=6 object. Nhóm small mất4, cứu0; medium mất27, cứu0;
large mất18, cứu4. 18class tăngFN,9class giữ nguyên và1class giảmFN: Tomato mold
leaf giảm32→31. Ví dụ Tomato leaf yellow virus105→111FN; Peach leaf24→29;
Tomato leaf57→62; Cherry leaf36→40. Các class có support khác nhau, không suy
mỗi thay đổi ít mẫu là một hiệu ứng ổn định.

[classes.csv](assets/kb2_conditional_pilot_20261031/classes.csv).

## DPO đang tác động vào GT mục tiêu thế nào?

Fixed probes gồm8 cặp mỗi nhóm/cohort, không phải toàn bộ bank hay holdout độc lập.
Trên8 recovery probes thuộc DPO train:

| Model | Mean score tại slot nguồn | Ordinary target được phát hiện | Mean margin so S0 |
|---|---:|---:|---:|
| S0 | 0,128454 | 0/8 | 0 |
| Native-only last | 0,071440 | 0/8 | -0,706662 |
| Native+DPO last | 0,071540 | 0/8 | -0,705148 |

DPO đỡ giảm score so native-only khoảng0,000100, nhưng score vẫn giảm khoảng44,3%
so S0 và thấp xa confidence0,25. Vì vậy không thể gọi đây là đã học cứu GT.

Trên8 preservation probes DPO train, mean score S0 là0,296807; native-only last
0,258832; DPO last0,259319. Ordinary detection chỉ còn3/8 target, so S0 là8/8.
Nhóm8 recovery probes bỏ khỏi DPO cũng không cứu target nào; score DPO
0,139997→0,075038. Các ảnh này vẫn có thể được native dùng và đã được S0 học,
nên không gọi là bằng chứng tổng quát hóa độc lập.

Background và duplicate có margin dương, còn recovery/preservation margin âm.
Cân bằng số cặp không bảo đảm cân bằng gradient hoặc bảo vệ được các lá đúng.
V0 đã giới hạn likelihood vào quyết định được sửa; trong kết quả này bản thân
logit tại các positive target vẫn đi xuống dưới tác động cập nhật chung.

[probes.csv](assets/kb2_conditional_pilot_20261031/probes.csv).

## Exposure, gradient và geometry

- 4.000 lượt cặp/arm: recovery986, preservation1.026, background1.018, duplicate970.
 Keep=50,3%; có371/378 ảnh và634/663 cặp khác nhau thực sự được dùng. Không phải
 vì recovery pairs không đi vào training. Ảnh lặp nhiều nhất60 lượt, target56;
 cap của sampler giới hạn xác suất kỳ vọng, không phải hard count mỗi run.
- Trên21 batch có gradient diagnostics, weighted DPO/native norm ratio trung bình
 1,0608%, khoảng0,2317%–2,2637%; cosine trung bình0,2248, thấp nhất-0,4372.
 Không suy tỷ lệ norm là phần trăm tác động AdamW. Các logs không tách gradient
 theo positive/negative hoặc từng GT, nên chưa xác định đầy đủ nguyên nhân.
- Regression/backbone/BN frozen state ở checkpoint không đổi; raw-box delta
 policy/reference trong logs/probes là0. Chưa có lỗi geometry drift quan sát được.
- Mean matched IoU tăng0,869785→0,872398 nhưng GT coverage giảm. Trên GT cùng được
 cảhai tìm được, IoU S0=0,872971 và DPO=0,872841. Không coi đây là học box khít hơn:
 raw boxes đóng băng; thay đổi score/argmax/NMS có thể chọn slot khác.

## Có phải chỉ cần hạ confidence?

Coarse sweep validation cho DPO last: tại confidence0,20, recall53,7361%, FP951;
S0 tại0,25 có recall54,1070%, FP984. Trên grid đã khóa, recall cao nhất của DPO
trong FP budget984 vẫn thấp hơn S0. Đây là grid thưa, chưa phải PR-envelope chính
xác ở matched FP và chưa chứng minh mọi threshold đều thua. Không tune trên test.

## Ý nghĩa và bước tiếp theo

Pilot có ý nghĩa vì có đối chứng cùng ảnh/mask/budget: sự giảm recall đã xuất hiện
ở native-only, DPO với strength hiện tại chỉ tạo thay đổi nhỏ và chưa đảo chiều.
Không quy toàn bộ45FN tăng cho DPO. Cũng chưa được kết luận DPO nói chung thất bại
hoặc chỉ tăng lambda chắc chắn sẽ sửa được vấn đề.

Ưu tiên tiếp theo là audit các positive decisions trong **batch hỗn hợp thật**:
so native-only, DPO-only và total/actual AdamW tại đúng slot GT; đo cả lớp/score,
ordinary recovery và GT khác bị mất. Từ đó thử một thay đổi strength/loss có đối
chứng và ngân sách nhỏ, giữ bank/mask/stream để đọc được tác động. Chưa thêm
Gaussian/NMS groups hoặc chạy dài nguyên recipe. Nếu tune lambda/beta/native
weight, khóa từng cấu hình trước run và kiểm lại giới hạn FP/AP/recall; không
chỉ chọn cấu hình cho train margin đẹp.

Pilot chỉ có1 seed; không có kiểm định test hoặc ngoài nguồn chụp. Kết quả này đủ
để chưa tăng ngân sách cho recipe hiện tại, chưa đủ để kết luận về mọi thiết kế DPO.
