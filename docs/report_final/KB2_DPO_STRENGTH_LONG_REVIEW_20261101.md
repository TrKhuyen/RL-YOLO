# KB2 — đánh giá lượt dài DPO λ=0,1, 01/11/2026

> Báo cáo phương pháp và toàn bộ quá trình phát triển: [Báo cáo tổng hợp KB2 DPO,03/11/2026](Report_KB2_DPO_20261103.md).

> **Cập nhật03/11/2026: đã hoàn tất và kiểm chứng6 lượt mới, seed42/43/44 × native/DPO.** Best mAP DPO trung bình **38,1707 ± 0,0429%**, native **38,1196 ± 0,0608%**, S0 **37,8218%**. DPO hơn native về mAP/AP50/AP75 ở cả3 seed; precision/recall vẫn có đánh đổi. Native seed42 đã được chạy lại đủ15k và tái hiện chính xác lượt cũ. Kết quả đầy đủ nằm trong phần **Bổ sung kết quả ba seed — 03/11/2026** ở cuối tài liệu. Các phần trước đó mô tả lượt seed42 ban đầu ngày01/11.

**Đã hoàn tất 15.000 update trên GPU. Best_AP nằm ở step12.000, thực sự là model đã train.** So S0, checkpoint này tăng mAP50–95, AP50, AP75, precision và F1; FP giảm93 nhưng FN tăng16 tại confidence0,25. So native-only cùng recipe, DPO có lợi thế AP nhỏ ở cuối lượt chạy, đồng thời có một số đánh đổi. Đây là cải thiện trên validation dùng phát triển, chưa phải kết quả trên test hoặc nhiều seed.

## Run và cấu hình

- Output: `kb2_preference_optimization/outputs/yolov8n_strength_long_lambda010_s42_20261101_180757_848916/`.
- Bắt đầu mới từ S0 KB1 supervised và optimizer mới; không resume pilot, không ghi đè checkpoint cũ.
- YOLOv8n, CUDA, seed42,15.000 update, LR1e-6, AdamW WD5e-4, clip1, input640, FP32/TF32 off.
- Tổng loss: `native_background_0.5 + 0.1 * DPO`, beta0,1; native foreground giữ hệ số1.
- Chỉ classification cv3 được train. Regression/backbone/neck/BN statistics đóng băng.
- Native4 ảnh/update =2 uniform full-train +2 ảnh preference; DPO2 quyết định/update. Giữ nguyên bank784 cặp/663 cặp DPO train.
- Full validation479 ảnh/1.887GT tại step0 và mỗi1.000 update. AP từ confidence0,001; precision/recall/FP/FN tại confidence0,25, matching IoU0,5, NMS0,6, cap300.
- Best theo mAP50–95, min_delta0,0001; S0 eligible, không có recall gate. Lưu riêng mọi checkpoint eval; Pareto mAP/P/R chỉ là diagnostic.
- Trainer ghi1.064,35 giây, khoảng17,74 phút; không bao gồm toàn bộ setup/initial evaluation.

Metadata `pilot=True` được kế thừa từ trainer, nhưng `config.steps=15000`, log, stream và checkpoint đều xác nhận ngân sách dài15k.

## So sánh S0 và các mốc chính

Các số AP/P/R/F1/AR dưới đây tính theo %. Delta của các tỷ lệ là **điểm phần trăm**, không phải phần trăm tương đối.

| Metric | S0 | DPO0,1 /1k | DPO0,1 /12k — best_AP | DPO0,1 /15k — last | Delta best−S0 |
|---|---:|---:|---:|---:|---:|
| mAP50–95 | 37,8218 | 38,0617 | **38,1363** | 38,1254 | **+0,3145** |
| AP50 | 49,1225 | 49,4282 | **49,5543** | 49,5123 | +0,4318 |
| AP75 | 42,9680 | 43,1986 | 43,2303 | **43,3429** | +0,2623 |
| Precision | 50,9227 | 51,2601 | **53,0063** | 53,1599 | +2,0836 |
| Recall | 54,1070 | 53,8951 | **53,2591** | 53,0472 | −0,8479 |
| Macro recall | 54,5463 | 54,4958 | 54,5880 | 54,2938 | +0,0417 |
| F1 | 52,4666 | 52,5446 | **53,1324** | 53,1034 | +0,6658 |
| Macro F1 | 51,3707 | 51,6247 | 52,5780 | 52,4964 | +1,2073 |
| AR300 | 56,2044 | 56,9128 | 56,1909 | 55,9898 | −0,0134 |
| AP small | 4,3965 | 4,3051 | 4,0786 | 4,0372 | −0,3180 |
| AP medium | 17,9704 | 17,8559 | 17,7595 | 18,1217 | −0,2109 |
| AP large | 42,5676 | 42,8888 | 43,0132 | 42,9583 | +0,4456 |
| TP | 1.021 | 1.017 | 1.005 | 1.001 | −16 |
| FP | 984 | 967 | **891** | 882 | **−93** |
| FN | 866 | 870 | **882** | 886 | **+16** |

Theo mục tiêu cải thiện model trên nhiều chỉ số, run này có kết quả tích cực: AP, precision và F1 cùng tăng. Không nên phủ nhận kết quả chỉ vì recall giảm. Đồng thời, không cộng precision/FP hoặc recall/FN như những bằng chứng độc lập; các chỉ số này liên hệ trực tiếp với nhau.

So pilot1k của chính λ0,1, train dài nâng mAP thêm0,0746 điểm %, precision thêm1,7462 điểm % và F1 thêm0,5879 điểm %, nhưng recall giảm0,6359 điểm %. Step15k có AP75/precision/APmedium tốt hơn12k; step12k giữ mAP50–95/AP50/F1/recall tốt hơn. Best_AP là một lựa chọn có quy tắc rõ ràng, không có nghĩa nó tốt nhất ở mọi metric.

## Phần đóng góp thêm của DPO so native-only

Tái sử dụng đối chứng native và DPOλ0,01 từ run dài background0,5:
`outputs/yolov8n_native_background_long_s42_20261031_020350_005551/`.
Đã kiểm cùng S0/data/bank, trainable mask, loss background, optimizer/config và toàn bộ15.000 batch/pair trong stream. Selector cũ khác nhưng chỉ quyết định lưu weights, không đổi update; không dùng `best.pt` cũ vì file đó là S0 do recall gate.

| Cấu hình | Step | mAP50–95 (%) | AP75 (%) | Precision (%) | Recall (%) | F1 (%) | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native-only — AP cao nhất trong log | 1.000 | 38,0625 | 43,2058 | 51,2355 | 53,8421 | 52,5065 | 967 | 871 |
| Native-only | 12.000 | 38,0537 | 43,1469 | 53,1184 | 53,2591 | 53,1887 | 887 | 882 |
| DPOλ0,01 | 12.000 | 38,0336 | 43,1346 | 53,1465 | 53,2591 | 53,2028 | 886 | 882 |
| **DPOλ0,1** | **12.000** | **38,1363** | **43,2303** | 53,0063 | 53,2591 | 53,1324 | 891 | 882 |
| Native-only | 15.000 | 38,0099 | 43,1656 | 53,2730 | 53,0472 | 53,1599 | 878 | 886 |
| DPOλ0,01 | 15.000 | 38,0072 | 43,1623 | 53,1881 | 53,0472 | 53,1175 | 881 | 886 |
| **DPOλ0,1** | **15.000** | **38,1254** | **43,3429** | 53,1599 | 53,0472 | 53,1034 | 882 | 886 |

- Tại12k, λ0,1 hơn native0,0826 điểm mAP và0,0834 điểm AP75; cùngFN882 nhưng thêm4FP, precision thấp hơn0,1121 điểm và F1 thấp hơn0,0562 điểm.
- Tại15k, λ0,1 hơn native0,1155 điểm mAP và0,1774 điểm AP75; cùngFN886 nhưng thêm4FP.
- So AP cao nhất của cả lịch sử native38,0625% ở1k, best DPO38,1363% cao hơn0,0738 điểm %. Weights native1k có sẵn trong pilot strength và được nạp lại để kiểm.
- Cả7 mốc9k–15k có mAP của λ0,1 cao hơn native cùng step. Đây là chuỗi đo trên cùng một run, **không phải7 thí nghiệm độc lập**.

Lợi thế DPO về AP đã rõ hơn pilot2k vốn gần như trùng native, nhưng còn nhỏ và chưa đồng thời cải thiện precision/recall so native.

## So cùng FP984 — kiểm tra phụ trên validation

Quét chính xác global confidence cutoff theo score thực, giữ cả nhóm bằng điểm và xác nhận lại canonical matching; không nội suy. Mỗi checkpoint có cutoff riêng, vì vậy bảng này không thay cho kết quả chính ở confidence0,25.

| Model | Step | Cutoff | FP | TP | FN | Recall (%) |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0 | 0,25028 | 984 | 1.021 | 866 | 54,1070 |
| Native-only — AP cao nhất | 1.000 | 0,24197 | 984 | 1.023 | 864 | 54,2130 |
| DPOλ0,1 — best_AP | 12.000 | 0,21378 | 984 | 1.028 | **859** | **54,4780** |
| Native-only | 15.000 | 0,21276 | 984 | 1.027 | 860 | 54,4250 |
| DPOλ0,01 | 15.000 | 0,21185 | 984 | 1.028 | **859** | **54,4780** |
| DPOλ0,1 | 15.000 | 0,21447 | 984 | 1.028 | **859** | **54,4780** |

Best12k cải thiện7TP so S0 ở cùngFP984; hơn native15k1TP và bằng λ0,01/15k. Hai model khác step trong phép đối chiếu best12k/native15k được ghi rõ; không có weights native12k để nạp lại/quét matched-FP tại chính mốc12k. Lợi ích về recall ở điểm này vẫn nhỏ; chưa chứng minh toàn bộ đường precision–recall trội hơn native. DPOλ0,1/5k có FN858 ở cùngFP984, nhưng không chọn lại best theo diagnostic này sau khi đã khóa selector AP.

Các cutoff được tìm trên validation dùng phát triển; không tự đổi threshold chính hoặc xem đây là kết quả test.

## Cải thiện phân bố theo class và GT như thế nào?

Ở best12k,14/28 class tăng AP và14/28 class giảm AP so S0. Không có cải thiện đồng đều toàn bộ class.

| Class | GT validation | Delta AP best12k−S0 (điểm %) |
|---|---:|---:|
| Corn Gray leaf spot | 14 | +16,2891 |
| Bell_pepper leaf spot | 49 | +2,9732 |
| Soyabean leaf | 66 | +2,7310 |
| Tomato mold leaf | 50 | +2,5827 |
| Blueberry leaf | 158 | +2,4565 |
| Apple Scab Leaf | 32 | −2,9462 |
| Corn rust leaf | 39 | −3,0886 |
| Potato leaf late blight | 62 | −3,6952 |
| grape leaf black rot | 14 | −6,8747 |

Corn Gray có ít GT nhưng đóng góp khoảng+0,5818 điểm vào macro mAP;27 class còn lại cộng ròng khoảng−0,2672 điểm. Vì vậy gain tổng+0,3145 điểm không đại diện cho mức tăng phổ biến ở mọi class. Riêng so native12k, lợi thế AP tập trung nhiều ở Corn Gray và Tomato mold; cần xác nhận trên seed/holdout khác trước khi kết luận ổn định.

Tại confidence0,25, best12k cứu30GT S0 bỏ sót nhưng làm mất46GT S0 đã bắt: ròng−16TP. Nhóm ảnh6+GT cứu18/mất31; object medium cứu6/mất22, small cứu0/mất4, large cứu24/mất20. Điều này phù hợp với APsmall/APmedium và recall giảm, trong khi APlarge tăng.

Matched mean IoU toàn bộ matched subset tăng86,9785→87,2284%, nhưng trên cùng975GT được cảS0 và best12k phát hiện, mean IoU87,2619→87,2226% giảm nhẹ. Regression đóng băng: các thay đổi AP75/IoU là do score, candidate selection và matching, không phải đã học sửa tọa độ box.

## Kiểm chứng và dữ liệu đối chiếu

Script kiểm tra: [review_kb2_strength_long.py](../../scripts/review_kb2_strength_long.py).
Các bảng đầy đủ, kết quả nạp lại weights, matched-FP và learning curves được lưu ở
[assets/kb2_strength_long_20261101/](assets/kb2_strength_long_20261101/).

- Đã kiểm hash `best.pt`, `last.pt` và toàn bộ16 checkpoint eval của lượt mới; nạp lại đủ16 weights step0–15k trên GPU. Mọi numeric metrics, bao gồm từng class và GT transitions, khớp log; chỉ bỏ metadata selector và thời gian evaluator khi so sánh.
- Nạp lại thêm native1k/native15k và DPOλ0,01/15k: full metrics khớp. Tổng19 lần kiểm checkpoint, ngoài lượt đánh giá S0 tham chiếu ban đầu.
- Best weights bằng step12k; last bằng step15k. Step0 bằng S0, mọi weights đã train khác S0; frozen state không đổi.
- Hash S0/data/bank/code/provenance và checkpoint đối chứng hợp lệ. Toàn bộ15.000 stream entries byte-identical với native/λ0,01. Prefix2k của λ0,1 tái hiện chính xác metrics/probes/train logs của pilot.
- Mean norm(weighted DPO)/norm(native) qua151 điểm diagnostic khoảng16,67%; đây là tỷ lệ norm gradient, không phải phần trăm đóng góp chính xác vào update AdamW.
- [review.json](assets/kb2_strength_long_20261101/review.json), [summary.csv](assets/kb2_strength_long_20261101/summary.csv), [classes.csv](assets/kb2_strength_long_20261101/classes.csv), [exact_fp_budget.csv](assets/kb2_strength_long_20261101/exact_fp_budget.csv) chứa số liệu kiểm chứng.

**Kiểm tra bổ sung03/11 trước lượt xác nhận ba seed:** vòng update của trainer background cũ và trainer strength mới giống nhau khi so AST. Native2k đã chạy bằng cả hai trainer; kiểm lại355 state tensors bằng nhau chính xác và optimizer state bằng nhau, không chỉ metric giống nhau. Các thay đổi selector/lưu checkpoint không đổi loss/update; tăng λ cũng không đổi native vì hệ số DPO thực tế của arm native luôn bằng0. Native đối chứng được dùng là bản background0,5, không phải bản background1. Bằng chứng ở [native_reuse_verification.json](assets/kb2_strength_long_20261101/native_reuse_verification.json). Khi kiểm tra này được thực hiện, native mới chỉ được tái hiện trực tiếp2k. **Lượt ba seed bổ sung bên dưới đã chạy lại native đủ15k và xác nhận trực tiếp weights/optimizer bằng lượt cũ.** Không so trực tiếp hai file `best.pt` có selector khác nhau.

![Diễn biến metric qua15k update](assets/kb2_strength_long_20261101/learning_curves.png)

Checkpoint nên giữ để đối chiếu chính:

- `.../yolov8n_strength_long_lambda010_s42_20261101_180757_848916/best.pt`: step12k, tốt nhất theo mAP50–95.
- `.../yolov8n_strength_long_lambda010_s42_20261101_180757_848916/last.pt`: step15k, AP75 cao hơn best12k nhưng AP tổng thấp hơn.
- Mọi `step_*.pt` vẫn được giữ; không cần chạy lại để lấy weights ở mốc trung gian.

## Hướng tiếp theo ở thời điểm01/11, trước lượt xác nhận ba seed

Giữ λ0,1/best12k làm ứng viên KB2 mới cho recipe hiện tại. Chưa có lý do kết luận cứ tăng step hoặc λ sẽ tiếp tục tăng chất lượng. Bước kiểm chứng hữu ích tiếp theo là lặp recipe đã khóa ở seed khác với native đối chứng, hoặc khóa checkpoint/quy tắc lựa chọn rồi đánh giá holdout chưa dùng phát triển. Không cần quay lại train bank candidate-swap ít cặp chỉ để tăng số thí nghiệm.

## Bổ sung kết quả ba seed — 03/11/2026

**Đã hoàn tất6/6 lượt, mỗi lượt15.000 update trên CUDA.** Đây là các lượt chạy mới từ cùng S0 và optimizer mới, gồm cả native seed42; không tiếp tục từ best12k và không gộp trọng số giữa các seed.

Root mới: `kb2_preference_optimization/outputs/yolov8n_strength_long_3seeds_20261103_095853_370249/`, tách `s42/native`, `s42/dpo`, `s43/native`, `s43/dpo`, `s44/native`, `s44/dpo`.

Giữ nguyên bank/data, head mask, background0,5, LR1e-6, beta0,1, batch4/2,15k và eval mỗi1k. Native có hệ số DPO thực tế0; nhánh DPO cóλ0,1. Cùng một seed dùng cùng15.000 batch/pair, còn các seed khác nhau có chuỗi lấy mẫu khác nhau. Best được chọn bằng cùng quy tắc mAP50–95/min_delta1e-4, không recall gate. Ba seed kiểm tra biến thiên của fine-tuning từ **một S0 và một bank cố định**, không phải ba lần supervised độc lập.

### Best checkpoint của từng lượt

Precision/recall/F1/FP/FN vẫn dùng confidence0,25. Mỗi lượt được chọn best độc lập theo cùng quy tắc đã khóa; vì vậy step giữa native và DPO có thể khác nhau.

| Seed | Cấu hình | Best step | mAP50–95 (%) | AP50 (%) | AP75 (%) | Precision (%) | Recall (%) | F1 (%) | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| — | S0 | 0 | 37,8218 | 49,1225 | 42,9680 | 50,9227 | 54,1070 | 52,4666 | 984 | 866 |
| 42 | Native | 1.000 | 38,0625 | 49,3843 | 43,2058 | 51,2355 | 53,8421 | 52,5065 | 967 | 871 |
| 42 | DPO | 12.000 | **38,1363** | 49,5543 | 43,2303 | 53,0063 | 53,2591 | 53,1324 | 891 | 882 |
| 43 | Native | 12.000 | 38,1835 | 49,5626 | 43,3513 | 53,1599 | 53,0472 | 53,1034 | 882 | 886 |
| 43 | DPO | 15.000 | **38,2187** | 49,6316 | 43,4666 | 53,3298 | 53,0472 | 53,1881 | 876 | 886 |
| 44 | Native | 13.000 | 38,1128 | 49,4829 | 43,2382 | 53,1915 | 52,9942 | 53,0926 | 880 | 887 |
| 44 | DPO | 15.000 | **38,1570** | 49,5567 | 43,3754 | 53,1383 | 52,9412 | 53,0396 | 881 | 888 |

Best DPO−best native lần lượt **+0,0738/+0,0353/+0,0441 điểm mAP** cho seed42/43/44. DPO có mAP, AP50 và AP75 cao hơn ở3/3 seed. Riêng seed44, precision/F1 thấp hơn native và FP/FN đều thêm1: không có model trội native trên mọi metric.


### Trung bình ± độ lệch chuẩn của best checkpoint

Độ lệch chuẩn mẫu, `ddof=1`, `n=3`. Các tỷ lệ dùng đơn vị%; delta dùng điểm phần trăm. FP/FN là số đếm trung bình, có thể có phần thập phân. S0 là một model cố định nên không gán độ lệch chuẩn giữa ba seed cho nó.

| Metric | S0 | Native best, mean ± SD | DPO best, mean ± SD | Mean DPO−S0 |
|---|---:|---:|---:|---:|
| mAP50–95 (%) | 37,8218 | 38,1196 ± 0,0608 | **38,1707 ± 0,0429** | **+0,3489** |
| AP50 (%) | 49,1225 | 49,4766 ± 0,0893 | **49,5809 ± 0,0439** | +0,4583 |
| AP75 (%) | 42,9680 | 43,2651 ± 0,0764 | **43,3574 ± 0,1191** | +0,3894 |
| Precision (%) | 50,9227 | 52,5289 ± 1,1203 | **53,1581 ± 0,1626** | +2,2354 |
| Recall (%) | 54,1070 | 53,2945 ± 0,4750 | 53,0825 ± 0,1619 | −1,0246 |
| Macro recall (%) | 54,5463 | 54,4189 ± 0,0784 | 54,5256 ± 0,1606 | −0,0207 |
| F1 (%) | 52,4666 | 52,9009 ± 0,3416 | **53,1200 ± 0,0750** | +0,6534 |
| Macro F1 (%) | 51,3707 | 52,2238 ± 0,5742 | **52,6405 ± 0,1284** | +1,2699 |
| AR300 (%) | 56,2044 | 56,3711 ± 0,4766 | 56,0177 ± 0,1503 | −0,1867 |
| AP small (%) | 4,3965 | 4,1360 ± 0,1486 | 4,0525 ± 0,0253 | −0,3441 |
| AP medium (%) | 17,9704 | 17,8119 ± 0,0462 | 18,0267 ± 0,2365 | +0,0563 |
| AP large (%) | 42,5676 | 42,9588 ± 0,0608 | **43,0127 ± 0,0166** | +0,4451 |
| FP | 984 | 909,67 ± 49,66 | **882,67 ± 7,64** | −101,33 |
| FN | 866 | 881,33 ± 8,96 | 885,33 ± 3,06 | +19,33 |

Paired mean DPO−native về mAP là **+0,0511 ± 0,0202 điểm %**, AP50 **+0,1042 ± 0,0570**, AP75 **+0,0923 ± 0,0597**. SD của delta được tính từ ba chênh lệch theo seed, không cộng/trừ SD hai nhánh. Gain so S0 hiện lặp lại ở cảba seed, nhưng phần tăng thêm do DPO nhỏ hơn nhiều phần tăng chung do tiếp tục tối ưu native/background0,5. Không diễn giải sự khác nhau về precision/F1 của các best khác step như tác động riêng ở cùng số update.

### Đối chiếu cùng ngân sách15.000 update

Đây là bảng `last.pt`, tách khỏi bảng best để kiểm tra ảnh hưởng của việc mỗi arm chọn checkpoint khác step.

| Seed | Native mAP (%) | DPO mAP (%) | DPO−native (điểm %) | Native FP/FN | DPO FP/FN |
|---|---:|---:|---:|---|---|
| 42 | 38,0099 | **38,1254** | +0,1155 | 878/886 | 882/886 |
| 43 | 38,0823 | **38,2187** | +0,1365 | 877/887 | 876/886 |
| 44 | 38,0864 | **38,1570** | +0,0706 | 879/887 | 881/888 |
| Mean ± SD | 38,0595 ± 0,0430 | **38,1670 ± 0,0475** | **+0,1075 ± 0,0337** | 878,00/886,67 | 879,67/886,67 |

Ở mốc15k, AP50 trung bình tăng thêm0,1381 điểm và AP75 tăng thêm0,1805 điểm so native, đều tăng ở3/3 seed. Recall trung bình của hai nhánh bằng nhau53,0118%, còn precision DPO53,2093% thấp hơn native53,2564%; F1 DPO53,1104% thấp hơn native53,1339%. DPO hơn về AP nhưng không tạo thêm lợi thế nhất quán về FP/FN tại confidence0,25.

![So mAP theo cặp native/DPO của ba seed](assets/kb2_strength_multiseed_20261103/paired_ap.png)

### CùngFP984 — diagnostic phụ

Mỗi model được quét confidence riêng trên validation, giữ nguyên NMS/cap và canonical matching. Không dùng bảng này để thay selector AP hoặc tự đổi threshold chính.

| Seed | Native best FN | DPO best FN | Native last15k FN | DPO last15k FN |
|---|---:|---:|---:|---:|
| 42 | 864 | **859** | 860 | **859** |
| 43 | 859 | **855** | 856 | **855** |
| 44 | 860 | **858** | 862 | **858** |
| Mean ± SD | 861,00 ± 2,65 | **857,33 ± 2,08** | 859,33 ± 3,06 | **857,33 ± 2,08** |

S0 có FN866 ở cùngFP984. Best DPO giảm trung bình8,67FN so S0 và3,67FN so native best. So cùng15k, DPO giảm1/1/4FN so native, trung bình2GT. Đây là lợi thế nhỏ nhưng cùng chiều ở cảba seed tại điểmFP này; không chứng minh toàn bộ PR curve tốt hơn hoặc kết quả test.

### Phân bố cải thiện và giới hạn

- AP theo class của DPO best tăng so S0 ở14/28 class với seed42,14/28 với seed43,12/28 với seed44. Nếu lấy mean AP từng class qua ba seed,12 class tăng và16 class giảm. Do đó kết quả không phải cải thiện đồng đều giữa các class.
- Corn Gray leaf spot có14GT: AP tăng trung bình17,8458 điểm %, đóng góp khoảng+0,6373 điểm vào macro mAP;27 class khác cộng ròng−0,2884 điểm, cho tổnggain+0,3489 điểm. Điều này làm kết quả nhạy với class ít support dù gain đã lặp lại theo seed.
- Mean AP tăng còn thấy ở Bell_pepper leaf spot+2,9520, Soyabean+2,8441, Blueberry+2,5310, Tomato mold+2,4370 điểm %. Các class giảm nhiều gồm grape black rot−7,3778, Potato late blight−3,7702, Apple Scab−3,1759 và Corn rust−3,1158 điểm %.
- DPO best cứu trung bình30,67GT S0 bỏ sót, đồng thời mất50GT S0 từng bắt: ròng−19,33TP ở confidence0,25. FP giảm và AP/F1 tăng cần được ghi nhận cùng đánh đổi recall này; không đếm precision/FP hoặc recall/FN thành những bằng chứng độc lập.
- Regression vẫn đóng băng; tăng AP75/matched IoU phản ánh score/candidate selection/matching, không phải đã train sửa tọa độ.
- Ba seed dùng cùng S0/bank/data và cùng479 ảnh validation đã dùng phát triển. Chỉ có3 đơn vị lặp stochastic fine-tuning, không phải96 thí nghiệm độc lập từ các mốc eval. Chưa dùng test, chưa có kiểm chứng trên dữ liệu mới hoặc khẳng định ý nghĩa thống kê.

### Kiểm chứng trọng số và tái hiện seed42

- Kiểm hash đủ96 checkpoint eval (`6 × 16`), cùng12 file best/last; mọi file không thay đổi sau khi review. Kiểm cả6 weights step0 bằng S0.
- Nạp best và last của cả6 arm trên GPU, tổng10 trạng thái train khác nhau vì DPO43/44 có best=last15k. Mọi full metric, từng class và GT transitions khớp log; chỉ bỏ thời gian eval và metadata selector khi so. Best/last bằng checkpoint step tương ứng, frozen state giữ nguyên.
- Cùng provenance S0/data/bank/runtime/code; độc lập dựng lại đủ15.000 batch/pair cho từng seed và xác nhận native/DPO cùng stream trong seed, khác stream giữa seed.
- **Native seed42 chạy lại đủ15k:** toàn bộ16 mốc validation/probes khớp lượt native cũ; state model và optimizer tại15k bằng nhau chính xác. **DPO seed42 cũng tái hiện đủ16 mốc và state/optimizer15k của run01/11.** Vì vậy thắc mắc về tác động của thay selector/code đã được kiểm trực tiếp ở đủ ngân sách15k, không còn chỉ dựa vào pilot2k.
- Script tái kiểm: [review_kb2_strength_multiseed.py](../../scripts/review_kb2_strength_multiseed.py). Không train thêm hoặc thay checkpoint/data trong quá trình review.

Số liệu đầy đủ ở [review.json](assets/kb2_strength_multiseed_20261103/review.json), [summary.csv](assets/kb2_strength_multiseed_20261103/summary.csv), [statistics.csv](assets/kb2_strength_multiseed_20261103/statistics.csv), [paired_deltas.csv](assets/kb2_strength_multiseed_20261103/paired_deltas.csv), [class_statistics.csv](assets/kb2_strength_multiseed_20261103/class_statistics.csv), [exact_fp_budget.csv](assets/kb2_strength_multiseed_20261103/exact_fp_budget.csv). [Learning curves](assets/kb2_strength_multiseed_20261103/learning_curves.png) và [paired AP](assets/kb2_strength_multiseed_20261103/paired_ap.svg) có bản ảnh/vector độc lập.

### Đánh giá sau ba seed

Theo mục tiêu cải thiện model, recipe DPOλ0,1/background0,5 có cải thiện AP/P/F1 so S0 lặp lại ở cảba seed. So native đối chứng, lợi thế mAP/AP50/AP75 cũng lặp lại3/3 seed, kể cả ở cùng step15k; đây là bằng chứng tốt hơn kết luận chỉ từ seed42. Tuy nhiên mức tăng DPO riêng còn nhỏ, class coverage chưa tốt và recall ở threshold chính vẫn giảm so S0.

Checkpoint AP cao nhất quan sát được trong sáu lượt là `s43/dpo/best.pt` ở15k, mAP38,2187%. Đây là ứng viên chọn theo validation; số liệu báo cáo phương pháp vẫn nên dùng **mean ± SD cảba seed**, không thay bằng riêng seed43. Giữ nguyên kết quả các seed để truy vết.

Bước kế tiếp có ích hơn việc tự động tăng step/λ là khóa recipe và quy tắc chọn checkpoint để kiểm tra holdout chưa dùng phát triển, hoặc triển khai cùng thiết kế trên các model còn lại của đề tài với S0/bank/đối chứng tương ứng. Chưa có cơ sở khẳng định chạy dài hơn sẽ khắc phục recall hoặc class giảm.
