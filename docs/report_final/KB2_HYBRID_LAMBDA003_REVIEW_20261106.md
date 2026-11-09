# Hybrid KB3 → KB2: kiểm tra DPO λ = 0,03 — 06/11/2026

## Giả thuyết và kế hoạch khóa trước khi chạy

Đối chứng seed42 trước đó cho best mAP50–95 của native-only/background0,5 là
38,6304%, nhỉnh hơn native+DPO λ0,1/background0,5 đạt38,5962%. Chênh lệch nhỏ
0,0342 điểm phần trăm chưa chứng minh DPO gây hại. Phép thử này kiểm tra giả
thuyết sức ép preference cần nhẹ hơn sau TPE, với **λ0,03** thay cho0,1.
Giá trị0,03 là mức thăm dò thấp hơn khoảng ba lần, không có bảo đảm tối ưu.

Loss vẫn là `L_native + λ L_DPO`, beta giữ0,1. Tại cùng model/batch, hệ số
gradient DPO củaλ0,03 bằng30% củaλ0,1; tỷ lệ này không mô tả tuyến tính toàn
quỹ đạo hoặc chất lượng model sau huấn luyện.

Giữ nguyên:

- YOLOv8n, khởi tạo/reference từ KB3 TPE trial2; mỗi lượt bắt đầu mới từ TPE.
- Background0,5, foreground1; input640, native batch4/preference batch2.
- LR1e-6, AdamW weight decay5e-4, clip1; seed42,15.000 update, eval mỗi1.000.
- Cùng bank đầy đủ và cùng stream ảnh/cặp với native-only và DPOλ0,1 cũ.
- Chỉ tối ưu nhánh classification `cv3`; đóng băng regression/backbone/neck,
  thống kê BN và reference như các đối chứng.
- Best theo validation mAP50–95, min_delta1e-4 theo đơn vị fraction;
  TPE step0 vẫn có thể được chọn. Không chọn lại best theo F1 hoặc precision.
- Full validation479 ảnh/1.887GT; không đánh giá test.

Lượt λ0,03 lưu tại:
`kb2_preference_optimization/outputs/yolov8n_hybrid_lambda003_s42_20261106_trial2/`.
Lượt `trial1` gặp segmentation fault khi import thư viện trước huấn luyện;
log và trạng thái failed được giữ riêng. Lượt chạy lại giữ nguyên recipe,
với `CUDA_LAUNCH_BLOCKING=1` như lượt DPO/background0,25 hoàn tất trước đó.
Chưa xác định nguyên nhân gốc của lỗi; không sửa trainer hoặc package.

Launcher: [run_kb2_hybrid_lambda_trial.py](../../scripts/run_kb2_hybrid_lambda_trial.py).
Script đối chiếu: [review_kb2_hybrid_lambda_trial.py](../../scripts/review_kb2_hybrid_lambda_trial.py).

## Kết quả hoàn tất

Lượt chạy lại hoàn tất15.000 update, train wall18,58 phút gồm periodic eval;
chưa tính đầy đủ setup/eval đầu lượt và chấm lại hậu kiểm. Chế độ CUDA đồng bộ
khác baseline cũ, nên không dùng wall time này để suy ra overhead riêng của DPO.

### Selected best theo mAP50–95

Các tỷ lệ dưới đây tính theo%; FP/FN là số đếm tại confidence0,25 theo evaluator
KB2 thống nhất. Ba continuation đều chọn best ở1.000 update.

| Cấu hình | Best step | mAP50–95 | AP50 | AP75 | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 supervised | — | 37,8218 | 49,1225 | 42,9680 | 50,9227 | 54,1070 | 52,4666 | 984 | 866 |
| TPE trước continuation | 0 | 38,5097 | 50,4306 | 43,4945 | 50,8763 | 55,3789 | 53,0322 | 1.009 | 842 |
| Native-only, λ0 | 1.000 | 38,6304 | **50,5586** | **43,6553** | **53,0208** | **53,9481** | **53,4804** | **902** | **869** |
| Native+DPO, λ0,1 | 1.000 | 38,5962 | 50,5191 | 43,6132 | 52,8037 | 53,8951 | 53,3438 | 909 | 870 |
| Native+DPO, λ0,03 | 1.000 | **38,6314** | 50,5556 | 43,6522 | 52,9657 | **53,9481** | 53,4523 | 904 | **869** |

In đậm chỉ so trong ba continuation, không hàm ý khác biệt có ý nghĩa thống kê.

So với **DPOλ0,1**, bestλ0,03 tăng mAP**0,0352 điểm phần trăm**, AP50**0,0365**,
AP75**0,0390**, precision**0,1619**, recall**0,0530** và F1**0,1085** điểm;
giảm5FP và1FN. Như vậy phép thử có cải thiện nhỏ trên các chỉ số chính so với
recipe DPOλ0,1 trong seed42.

So với **native-only**, bestλ0,03 chỉ hơn mAP**0,0011 điểm phần trăm**; recall
và TP/FN bằng nhau, nhưng FP nhiều hơn2 và precision/F1 hơi thấp hơn.
AP50/AP75 cũng hơi thấp hơn native-only. Không thể từ chênh lệch rất nhỏ này
khẳng định auxiliary DPO tạo lợi ích thực tiễn hoặc ổn định. Mức chênh mAP giữa
hai run còn nhỏ hơn min_delta0,01 điểm dùng cho selector trong từng run;
min_delta không phải sai số đo hoặc ngưỡng kiểm định thống kê.

So **TPE**, bestλ0,03 tăng mAP**0,1218 điểm**, precision**2,0893**, F1**0,4201**
điểm và giảm105FP; recall giảm**1,4308 điểm**, FN tăng27. Đây là tác động của
toàn bộ continuation native+DPO sau TPE, không phải gain riêng DPO.

So **S0**, bestλ0,03 tăng mAP**0,8097 điểm**, AP50**1,4331**, AP75**0,6842**,
precision**2,0430** và F1**0,9858** điểm; FP giảm80. Recall giảm**0,1590** điểm
và FN tăng3. Cần công bố cả phần tăng lẫn đánh đổi, không mô tả mọi metric đều tăng.

### Cùng ngân sách15.000 update

| Metric (%) trừ số đếm | Native λ0 | DPO λ0,1 | DPO λ0,03 |
|---|---:|---:|---:|
| mAP50–95 | 38,3144 | 38,3230 | **38,3601** |
| AP50 | **50,4219** | 50,3502 | 50,4056 |
| AP75 | 43,0837 | **43,1832** | 43,1646 |
| Precision | 55,6429 | 55,6429 | **55,7119** |
| Recall | **52,5172** | **52,5172** | 52,4642 |
| F1 | 54,0349 | 54,0349 | **54,0393** |
| Macro F1 | 52,3103 | **52,4395** | 52,4106 |
| Macro recall | 52,8127 | **52,8784** | 52,7465 |
| AR300 | 56,3786 | 56,4617 | **56,4631** |
| APsmall | 7,2584 | **7,2676** | 7,2645 |
| APmedium | **18,5985** | 18,5040 | 18,5405 |
| APlarge | 43,3519 | 43,3372 | **43,4157** |
| TP | **991** | **991** | 990 |
| FP ↓ | 790 | 790 | **787** |
| FN ↓ | **896** | **896** | 897 |

Ở cùng15k, λ0,03 hơn λ0,1**0,0372 điểm mAP**, giảm3FP nhưng tăng1FN; F1 chỉ
tăng0,0044 điểm. Không có cải thiện đồng đều ở mọi metric. Lastλ0,03 vẫn thấp hơn
TPE**0,1496 điểm mAP** và thấp hơn best của chính nó**0,2713 điểm**, nên giữ best1k.

### GT identity và checkpoint F1 cao nhất

GT recovered/lost đều so với ordinary TPE, không so với supervised S0:

| Cấu hình | Best: cứu/mất GT | Best: TP ròng | Last15k: cứu/mất GT | Last: TP ròng |
|---|---:|---:|---:|---:|
| Native λ0 | 5 /32 | −27 | 22 /76 | −54 |
| DPO λ0,1 | 5 /33 | −28 | 22 /76 | −54 |
| DPO λ0,03 | 6 /33 | −27 | 22 /77 | −55 |

Giảm λ không khắc phục xu hướng mất GT khi train lâu. TP/FN bằng nhau giữa
hai model cũng không có nghĩa chúng phát hiện đúng cùng một tập GT.

Vì người dùng quan tâm nhiều metric, ghi thêm F1 cao nhất trong16 mốc đánh giá
của mỗi run, nhưng **không dùng các mốc này thay selector mAP**:

| Cấu hình | Step có F1 cao nhất | F1 (%) | mAP50–95 tại step đó (%) |
|---|---:|---:|---:|
| Native λ0 | 10.000 | 54,3036 | 38,3379 |
| DPO λ0,1 | 11.000 | 54,2982 | 38,3690 |
| DPO λ0,03 | 10.000 | **54,3874** | 38,3693 |

λ0,03 có peak F1 cao hơn λ0,1**0,0892 điểm** và native**0,0838 điểm** trong
quỹ đạo này. Đây là chẩn đoán trên validation đã xem, không phải xác nhận độc lập
hay bằng chứng nên chọn lại checkpoint theo F1.

![So sánh quỹ đạo của ba mức lambda](assets/kb2_hybrid_lambda003_20261106/learning_curves.png)

## Kiểm tính tương đương và nạp lại checkpoint

Đã hoàn tất các kiểm tra sau:

- Toàn bộ15.000 dòng stream ảnh/cặp giống nhau giữa ba run. Bank đầy đủ, cùng
  TPE initialization/reference, seed và ngân sách.
- Toàn bộ trường manifest ngoài config giống nhau, gồm fingerprint code,
  trainable names, frozen/reference state và bank. Config chỉ khác arm/trọng số
  DPO; background, foreground, beta, LR và các thiết lập khác khớp.
- Ở update đầu, native gradient norm giữ3,82548; weighted DPO gradient norm đổi
  từ0,276375 ởλ0,1 sang0,082913 ởλ0,03, đúng hệ số0,3 trong sai số số học.
- Hash48 checkpoint eval của ba run và hash best/last hợp lệ; weights/optimizer
  best và last khớp với step được chọn, đủ16 mốc mỗi run từ0 tới15k.
- Nạp lại6 checkpoint best/last trên GPU và chấm lại TPE/S0 trên full validation
 479 ảnh/1.887GT: toàn bộ quality metric, per-class và GT transitions khớp log.
- Reference, frozen state, checkpoint nguồn, bank và checkpoint đối chứng cũ
  được bảo toàn. Không sửa dữ liệu hoặc trainer trong quá trình chạy.

Artifact: [review.json](assets/kb2_hybrid_lambda003_20261106/review.json),
[summary.csv](assets/kb2_hybrid_lambda003_20261106/summary.csv),
[deltas.csv](assets/kb2_hybrid_lambda003_20261106/deltas.csv),
[classes.csv](assets/kb2_hybrid_lambda003_20261106/classes.csv),
[probes.csv](assets/kb2_hybrid_lambda003_20261106/probes.csv),
[validation.csv](assets/kb2_hybrid_lambda003_20261106/validation.csv).

Checkpoint mới:
`kb2_preference_optimization/outputs/yolov8n_hybrid_lambda003_s42_20261106_trial2/run/dpo/best.pt`.
Đây là checkpoint state_dict của KB2, dùng evaluator/adapter hybrid để nạp,
không mặc định tương đương định dạng model pickle của Ultralytics.

## Kết luận

**λ0,03 cải thiện nhỏ so với λ0,1 trong seed42, nhưng best mAP gần như ngang
native-only.** Có thể giữλ0,03 là cấu hình DPO ứng viên mới cho bước xác nhận,
đồng thời giữ nguyên cả recipeλ0,1 và native-only làm mốc. Chưa đủ bằng chứng
để khẳng định DPO có lợi ích riêng rõ ràng sau TPE hoặc λ0,03 là tối ưu.

Nếu tiếp tục, nên xác nhận theo cặp trên seed43/44 với cùng native control,
thay vì chỉ chọn thêm nhiều λ theo một validation. Nếu vẫn không tách khỏi
native-only, cần xem lại tín hiệu/cách chọn preference hoặc objective đối chứng.
Các lượt bổ sung này **chưa chạy** trong thí nghiệm hiện tại.

## Giới hạn diễn giải

Đây là một continuation seed từ một TPE initialization, trên validation đã
được dùng phát triển. Một mức λ bổ sung không xác lập λ tối ưu và chưa đánh giá
khả năng tổng quát trên dữ liệu xác nhận độc lập. Cần phân biệt gain toàn pipeline
so S0 với phần đóng góp riêng DPO so native-only cùng điều kiện.
