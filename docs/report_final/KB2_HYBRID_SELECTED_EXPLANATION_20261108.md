# KB2 hybrid λ = 0,03 — kết quả chốt và giải thích có căn cứ

**Làm rõ ngày09/11:** phần17.1–17.8 của
[báo cáo KB2](Report_KB2_DPO_20261103.md#171-model-thực-sự-học-gì-từ-một-cặp-keepdrop)
đã bổ sung ví dụ score vượt threshold, ví dụ AP75 tăng dù box không đổi,
phân tích native background bằng log batch đầu, công thức P/R/F1 thay số và
case thật theo class. Các ví dụ tự đặt được đánh dấu riêng với số liệu đã đo;
[artifact tính lại](assets/kb2_hybrid_selected_20261108/clarification_examples_20261109.json)
kiểm ví dụ AP75 bằng cùng loại backend evaluator, không chạy thêm model.

**Ngày chốt: 08/11/2026.** Kết quả được chọn là YOLOv8n sau **KB3 best TPE →
KB2 DPO**, λ = 0,03, native background weight = 0,5, seed 42; checkpoint best
tại **update 500** trong một lượt có ngân sách 15.000 update.

**Phân biệt mức độ bằng chứng:** toán học giải thích hướng tác động của loss
và cách tính metric; paper cung cấp nền tảng phương pháp. Không có định lý
trong các nguồn này bảo đảm KB2 cụ thể tăng mAP hoặc λ = 0,03 tối ưu. Mức
tăng của KB2 được xác nhận bằng so sánh thực nghiệm trên validation phát triển.
Các giá trị siêu tham số được giữ sau quan sát thử nghiệm, không suy ra từ paper.

## 1. Kết quả nào được chốt?

Checkpoint gốc được giữ tại:

`kb2_preference_optimization/outputs/yolov8n_hybrid_schedule_s42_20261106_trial3/run/dpo/best.pt`

SHA256:
`7b371f3313f1fe8fd5de157d16c961f076874c7659a6060e6947b4b8f31a80a4`.

Khởi tạo **và frozen reference** đều là best TPE trial 2 của KB3. Không phải
tiếp tục từ best KB2 cũ. S0 là mốc supervised chung. Trong báo cáo này, “DPO”
chỉ toàn bộ pipeline KB2 gồm bank, GT judge, sampling và native + DPO loss;
gain so TPE không được quy toàn bộ cho riêng một hạng loss.

Chỉ classification `cv3` được học, gồm 24 tensor tham số. Backbone/neck,
regression và BN running statistics giữ nguyên. Model nhận ảnh khi inference;
không cần GT, preference judge hoặc reference để dự đoán.

| Thiết lập | Giá trị đã dùng |
|---|---|
| Input | 640×640 |
| Preference bank | 773 cặp train-source; 662 dùng DPO, 111 probe không lấy mẫu DPO |
| Nhóm preference | Recovery/preservation/background/duplicate; sampler cũ gần 25% mỗi nhóm sau cap |
| Native batch | 4 ảnh: 2 uniform train + 2 ảnh preference |
| DPO batch | 2 quyết định mỗi update |
| Tổng loss | Native detection loss + 0,03 × DPO loss |
| Native foreground/background | 1 / 0,5; giữ assignment và mẫu số native |
| β | 0,1, khác λ = 0,03 |
| AdamW | LR 1e-6; weight decay 5e-4; clip norm 1 |
| LR sau 1k | 1e-7 từ update 1.001; chưa tác động lên best 500 |
| Eval | Mỗi 250 update tới 2k, sau đó mỗi 1k |
| Chọn best | Validation mAP50–95; min_delta 1e-4 fraction; TPE step 0 hợp lệ |

Chốt kết quả không thay default launcher lịch sử, không ghi đè/copy lại
checkpoint và không phát sinh training. Hồ sơ máy đọc được:
[selection.json](assets/kb2_hybrid_selected_20261108/selection.json).

## 2. Minh chứng thực nghiệm: tốt hơn so với mốc nào?

Cả ba model được chấm cùng 479 ảnh validation / 1.887 GT / 28 class. AP dùng
confidence tối thiểu 0,001, NMS IoU 0,60, max_det 300. Precision/recall/F1 và
FP/FN dùng confidence 0,25, matching class-aware IoU 0,5. Tỷ lệ dưới đây là %.

| Metric | S0 | Best TPE | Hybrid chốt | Δ so TPE, điểm % |
|---|---:|---:|---:|---:|
| **mAP50–95** | 37,8218 | 38,5097 | **38,7133** | **+0,2036** |
| AP50 | 49,1225 | 50,4306 | **50,6287** | +0,1981 |
| AP75 | 42,9680 | 43,4945 | **43,7080** | +0,2135 |
| Precision | 50,9227 | 50,8763 | **52,6208** | +1,7444 |
| Recall | 54,1070 | **55,3789** | 54,2660 | −1,1129 |
| F1 | 52,4666 | 53,0322 | **53,4307** | +0,3985 |
| Macro F1 | 51,3707 | 51,2818 | **51,6693** | +0,3875 |
| Macro recall | 54,5463 | **55,9285** | 55,0529 | −0,8756 |
| AR300 | 56,2044 | **58,2612** | 58,0731 | −0,1881 |
| APsmall | **4,3965** | 4,0997 | 4,1209 | +0,0212 |
| APmedium | 17,9704 | **18,4362** | 18,3008 | −0,1354 |
| APlarge | 42,5676 | 43,8593 | **44,0333** | +0,1741 |
| TP | 1.021 | **1.045** | 1.024 | −21 GT |
| FP ↓ | 984 | 1.009 | **922** | −87 detection |
| FN ↓ | 866 | **842** | 863 | +21 GT |

Theo mAP50–95, phân rã số học của hai giai đoạn là:

- S0 → TPE: **+0,6879 điểm phần trăm**.
- TPE → KB2: **+0,2036 điểm phần trăm**.
- S0 → hybrid cuối: **+0,8915 điểm phần trăm**.

Đây là chênh lệch giữa các checkpoint trong pipeline đã chạy, không phải phép
đo độc lập đóng góp nhân quả của từng thành phần. Đặc biệt, không gọi toàn bộ
+0,8915 điểm là gain của KB2. So S0, AP/P/R/F1 đều tăng, FP giảm 62 và FN giảm
3; so TPE, mô hình đánh đổi một phần recall lấy precision/AP/F1.

Nguồn: [summary.csv](assets/kb2_hybrid_schedule_20261106/summary.csv) và
[review.json](assets/kb2_hybrid_schedule_20261106/review.json), đã nạp lại
checkpoint để đối chiếu metric. Phân tích mới kiểm SHA best và đối chiếu lại
hai bộ kết quả nạp checkpoint đã hoàn tất; không chạy thêm model.

## 3. Cơ chế DPO: có thể chứng minh hướng học, không bảo đảm tăng AP

Với cùng ảnh, A/B chỉ khác giữ hoặc bỏ một candidate; box/class/score nguồn
và phần context còn lại dùng chung trước NMS. GT judge chấm hậu quả trên toàn
ảnh. Keep phải cứu/giữ GT mục tiêu mà không làm mất GT khác hoặc tăng FP;
drop phải giảm FP mà không mất GT đã phát hiện. Đây là các điều kiện chính
xác trong [pairs.py](../../kb2_preference_optimization/conditional/pairs.py).

DPO gốc dùng chênh log-probability chosen/rejected so với reference. Phân tích
gradient cho thấy objective khuyến khích tăng xác suất tương đối của phương
án ưu tiên. Nền tảng này đến từ [Rafailov et al., DPO, mục 4](https://arxiv.org/html/2305.18290v3#S4).

Với policy Bernoulli hiện thực trong KB2, đặt `z` là class logit, `z_ref` là
logit TPE cố định, và s = +1 cho keep / −1 cho drop:

$$
L_{DPO}=\operatorname{softplus}\!\left[-\beta s(z-z_{ref})\right],\qquad
\frac{\partial L_{DPO}}{\partial z}
=-\beta s\,\sigma\!\left[-\beta s(z-z_{ref})\right].
$$

Đạo hàm tại output có dấu âm cho keep, dấu dương cho drop. Nếu chỉ tối ưu
riêng biến z bằng gradient descent, nó tăng logit keep và giảm logit drop.
Đây là **kết quả đại số** cho loss trong
[policy.py](../../kb2_preference_optimization/conditional/policy.py).

Trong model thật, tham số được chia sẻ giữa nhiều candidate, cộng thêm native
gradient, AdamW và clipping. Vì thế không được suy “mỗi cặp keep chắc chắn
làm score tại GT đó tăng” hoặc “DPO loss giảm thì AP chắc chắn tăng”.

Reference xác định mốc so sánh logit của TPE. Objective nhị phân còn viết được
bằng `BCEWithLogits(beta*(z-z_ref), chosen_keep)`. Không có hạng KL tường minh
hoặc ràng buộc cứng giữ model gần TPE trong code này. Giới hạn mask/LR/budget
là lựa chọn thực nghiệm hỗ trợ cập nhật thận trọng, không phải định lý bảo
toàn kiến thức. Paper DPO không chứng minh reference này vượt mọi BCE khác
hoặc bảo đảm mAP tăng trên detector của đề tài.

## 4. Vì sao precision và F1 tăng dù recall so TPE giảm?

Đây là phần có thể chứng minh **chính xác từ số đếm đã đo**, không cần suy
đoán hành vi model. Gọi T = TP, F = FP, G = tổng GT:

$$
P=\frac{T}{T+F},\quad R=\frac{T}{G},\quad
F1=\frac{2T}{T+F+G}.
$$

TPE có T = 1.045, F = 1.009, G = 1.887. Hybrid có T' = 1.024, F' = 922:
TP giảm d = 21 và FP giảm f = 87. Do G cố định, recall giảm 21/1.887, tức
1,1129 điểm phần trăm. Precision tăng khi T×f > F×d:

$$1045\times87=90915>1009\times21=21189.$$

F1 tăng khi T×f > d×(F+G):

$$90915>21\times(1009+1887)=60816.$$

Như vậy mức giảm FP đủ bù mức giảm TP trong precision/F1, nhưng không bù
được recall. Không diễn giải kết quả này thành “cứu nhiều GT hơn TPE”. GT
identity audit ghi cứu **3**, mất **24**, TP thay đổi ròng **−21**.

Các bất đẳng thức giải thích metric đã đo, không dự báo chắc chắn lượt train
khác cũng sẽ có cùng FP/TP. So S0, TP tăng 3 và FP giảm 62 nên P/R/F1 đều tăng;
hai kết luận so TPE và so S0 không mâu thuẫn.

## 5. Vì sao mAP/AP75 tăng khi không học tọa độ?

COCO AP tính từ các detection được sắp theo confidence rồi tích lũy TP/FP
ở nhiều mức IoU. Vì vậy AP phụ thuộc score, class, thứ tự và tập detection
cuối, ngoài chất lượng tọa độ. Cơ chế được thể hiện trong
[COCOeval chính thức](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).
KB2 dùng TorchMetrics/faster-coco-eval với cap300, không phải mặc định cap100
của COCO; cùng cấu hình này được dùng cho mọi model trong bảng.

Giữ box tại từng raw slot cố định vẫn cho phép AP thay đổi: classification
thay đổi confidence, argmax class, candidate qua threshold và thứ tự NMS; một
box có sẵn nhưng phù hợp GT hơn có thể được giữ thay cho box khác. Điều này
giải thích **khả năng** tăng AP75 mà không trực tiếp học sửa geometry.

Minh chứng phù hợp với cơ chế: reference/frozen state được xác minh không đổi,
raw-box drift trên các kiểm tra train/probe bằng 0; AP50 tăng 0,1981 điểm và
AP75 tăng 0,2135 điểm so TPE. Vì không có phân rã counterfactual riêng score,
class và NMS, chưa xác định được mỗi cơ chế đóng góp bao nhiêu.

Chỉ đổi scale confidence đơn điệu mà giữ nguyên mọi thứ hạng/tập output
không tự bảo đảm AP tăng; và giảm FP ở riêng confidence0,25 cũng không suy ra
AP tăng. Ở đây AP tăng được xác nhận trực tiếp bởi evaluator. Không viết
“AP75 tăng chứng minh bounding box được học khít hơn”. Matched mean IoU tại
ngưỡng vận hành còn giảm nhẹ so TPE (0,870426 → 0,869820), trên tập match có
thành phần thay đổi; số này cũng không phải phép so tọa độ trên cùng GT.

## 6. Vì sao chọn λ = 0,03? Phần nào là hiệu chỉnh thực nghiệm?

Trong tổng loss `native + lambda*DPO`, với cùng model/batch, giảm λ từ 0,1
xuống 0,03 làm riêng gradient DPO có trọng số còn **30%**. Điều này đúng
đại số trước khi cộng gradient và qua clipping/AdamW, không phải nói toàn bộ
update hay độ lệch model giảm còn30%.

Log batch đầu cho phép kiểm trực tiếp vì hai run cùng initialization và ảnh:

| Batch đầu | λ = 0,1 | λ = 0,03 |
|---|---:|---:|
| Native gradient norm | 3,825481 | 3,825481 |
| Weighted DPO gradient norm | 0,276375 | 0,082913 |
| Tỷ số norm weighted-DPO/native | 7,2246% | 2,1674% |

Tỷ lệ hai weighted norm đo được là 0,29999997. Đây là minh chứng đã giảm
sức can thiệp của preference ở batch đầu, không chứng minh tỷ lệ gradient
giữ nguyên cả run hay nguyên nhân duy nhất của cải thiện AP.

Trên **cùng lịch eval1k**, best λ0,1 đạt38,5962%, λ0,03 đạt38,6314%, chênh
**+0,0352 điểm**; cùng bank, image/pair stream và các thiết lập khác đã được
kiểm. Kết quả phù hợp giả thuyết cần điều chỉnh nhẹ khi bắt đầu từ TPE tốt,
nhưng chênh nhỏ và một seed nên chưa chứng minh λ0,03 tối ưu phổ quát.

Best38,7133% của cấu hình chốt được tìm thấy khi đánh giá thêm mốc500. Không
có checkpoint500 tương ứng của run λ0,1 trên cùng lịch dày để kết luận phần
tăng tổng cộng0,1171 điểm so λ0,1 đều do λ. **Chọn0,03 là hiệu chỉnh sau thử
nghiệm**, không phải hằng số được paper DPO đề xuất cho YOLO.

## 7. Các thành phần khác: cơ sở có sẵn và phần tự thiết kế

| Thành phần | Cơ sở giải thích | Mức kết luận cho KB2 |
|---|---|---|
| GT judge toàn ảnh | Kiểm tác động cặp bằng matching, threshold/NMS và ràng buộc không mất GT khác | Đúng theo quy tắc/code đã đặt; nhãn tốt offline chưa bảo đảm ordinary output tốt lên |
| Native foreground1/background0,5 | Giảm tổng sức ép từ unassigned background slot, giữ foreground và mẫu số native | Có thể chứng minh gradient background bị nhân0,5 ở cùng logits/assignment; giá trị0,5 là thực nghiệm |
| Cân bằng nhóm/class/target và cap ảnh | Giảm việc nhóm nhiều cặp hoặc một ảnh chiếm quá nhiều lượt | Kiểm được exposure; không chứng minh cân bằng gradient hoặc AP theo class |
| Freeze backbone/box/BN, LR nhỏ | Giới hạn phần model có thể thay đổi trong continuation | Kiểm được frozen-state; chưa có đối chứng chứng minh mask này tối ưu |
| Batch4, β0,1, LR1e-6, WD5e-4, budget15k | Các giá trị recipe đã chọn và ghi manifest | Lựa chọn kỹ thuật/thực nghiệm; không có định lý xác lập bộ giá trị này |
| Best500 | mAP cao nhất trên các mốc đã đánh giá | Quan sát trên validation của run này; không phải số update tối ưu cho mọi lần chạy |

Về native background, nghiên cứu [Focal Loss, Lin et al.](https://arxiv.org/abs/1708.02002)
chỉ ra vấn đề foreground/background imbalance ở dense detection. Đây là cơ
sở liên quan cho hướng giảm ảnh hưởng background. KB2 dùng hệ số slot0,5 cố
định, **không triển khai Focal Loss** và paper không xác lập con số0,5 cho KB2.

Về sampling, [Chang et al., ICML2021](https://proceedings.mlr.press/v139/chang21c.html)
phân biệt cân bằng ảnh và object trong ảnh nhiều class. Điều này phù hợp
động cơ theo dõi class/target thay vì chỉ đếm ảnh. Sampler KB2 là thiết kế
riêng, không phải triển khai phương pháp memory-bank của paper và chưa có
ablation độc lập xác lập đóng góp của sampler lên best đang chọn.

Không gán mức tăng AP cụ thể cho riêng background0,5/reference/sampler hoặc
mask vì chưa tách từng yếu tố trong phép so hybrid này. Các đối chứng lịch sử
vẫn được lưu, nhưng không thay thế phép kiểm tương ứng trên nguồn TPE.

## 8. Đánh giá dày và những lần thử sau best

Best500 nằm trước mốc giảm LR1k. Weights/optimizer tới1k của run scheduled
đã khớp chính xác run constant-LR λ0,03. Vì vậy tăng selected mAP từ38,6314%
lên38,7133% khi thêm eval250/500/... là **tìm được checkpoint sớm trên cùng
quỹ đạo**, không phải giảm LR đã tạo model tốt hơn ở step500.

Trên lịch1k chung, run LR giảm chọn5k/mAP38,6609%, hơn constant-LR38,6314%
0,0295 điểm; đây là phép so khác, không giải thích selected best500. Lịch
đánh giá dày cũng tăng số cơ hội chọn trên validation phát triển, nên không
coi chênh selected best là bằng chứng tổng quát hóa độc lập.

Hai phép thử tiếp theo đã hoàn tất:

- Giảm LR sau500: best vẫn38,7133%, state dict giống hệt best đang có; last
  tốt hơn nhưng không vượt best. [Review](KB2_HYBRID_DROP500_REVIEW_20261106.md).
- Sampler preservation35%: pilot2k đạt best38,7046%; P/F1/AP50/APsmall/APmedium
  tăng, FP giảm9 nhưng FN tăng1 so best cũ. Không vượt best mAP và chưa giảm
  mất GT ổn định. [Review](KB2_PRESERVATION_SAMPLING_PILOT_20261108.md).

Những kết quả này là lý do giữ recipe/checkpoint cũ trong các phương án đã
thử, không chứng minh đã đạt trần tối ưu hoặc mọi thay đổi khác sẽ thất bại.

## 9. Cải thiện có trải rộng theo class không?

So TPE, AP50–95 tăng ở **20/28 class**, giảm ở8 class. Đóng góp vào mAP là
delta AP mỗi class chia28, không nhân số lượng GT vì đây là macro-average.

| Ví dụ class | GT valid | Δ AP so TPE, điểm % | Đóng góp mAP, điểm % |
|---|---:|---:|---:|
| Tomato Septoria leaf spot | 60 | +0,9838 | +0,0351 |
| Tomato mold leaf | 50 | +0,9230 | +0,0330 |
| Corn leaf blight | 56 | +0,7785 | +0,0278 |
| grape leaf black rot | 14 | +0,7654 | +0,0273 |
| Squash Powdery mildew leaf | 54 | +0,6510 | +0,0233 |
| Tomato leaf yellow virus | 164 | −0,3281 | −0,0117 |
| Tomato leaf mosaic virus | 41 | −1,0552 | −0,0377 |

Đây là bằng chứng lợi ích AP của giai đoạn KB2 không nằm ở duy nhất một
class, nhưng không đồng nghĩa mọi class cải thiện hoặc đã được xác nhận
ổn định. So S0, hybrid có15 class tăng/13 giảm; phần thay đổi lớn ở vài class
còn bao gồm giai đoạn KB3, nên không gán cho riêng KB2.

Object nhỏ vẫn là hạn chế: APsmall hybrid4,1209% chỉ tăng0,0212 điểm so TPE
và thấp hơn S0 0,2756 điểm; support94 GT nhỏ trên ảnh đã resize/pad. So TPE,
APmedium còn giảm0,1354 điểm. Không viết “hybrid cải thiện tất cả kích thước”.

Nguồn đầy đủ 28 class và kiểm tổng đóng góp khớp mAP trong sai số float32:
[class_contributions.csv](assets/kb2_hybrid_selected_20261108/class_contributions.csv).

## 10. Phát biểu kết luận dùng trong báo cáo/bảo vệ

> Trên validation phát triển của đề tài, hybrid KB3 TPE → KB2 DPO với λ = 0,03
> đạt mAP50–95 38,7133%, tăng0,2036 điểm phần trăm so checkpoint TPE khởi tạo
> và0,8915 điểm so S0. AP50/AP75, precision và F1 tăng so TPE, trong khi recall
> giảm. Phân tích loss cho thấy tín hiệu preference có hướng tăng/giảm logit
> phù hợp quyết định keep/drop được GT đánh giá; thay đổi classification có thể
> cải thiện score, class và lựa chọn detection khi geometry đóng băng. Đây là
> giải thích cơ chế phù hợp với kết quả, không phải định lý bảo đảm tăng mAP.
> λ = 0,03 và các hệ số/lịch huấn luyện là các lựa chọn hiệu chỉnh thực nghiệm.
> Kết quả tăng cụ thể được xác nhận bằng so sánh checkpoint thực tế dưới cùng
> protocol đánh giá; chưa phải xác nhận độc lập trên dữ liệu mới.

Phạm vi xác nhận hiện tại: một continuation seed42 từ một TPE winner trên
validation đã dùng phát triển. Các kết quả ba seed từ S0 lịch sử là recipe
và nguồn khởi tạo khác, không được dùng thay cho ba seed của hybrid này. Chưa
có CI/significance test hoặc holdout mới; test chung từng được KB1 xem nhiều
lần không được mặc nhiên gọi là tập xác nhận độc lập. Việc chốt ở đây là chốt
kết quả để báo cáo, không nâng mức bằng chứng thực nghiệm.

Artifact:
[selection.json](assets/kb2_hybrid_selected_20261108/selection.json),
[mechanism_evidence.json](assets/kb2_hybrid_selected_20261108/mechanism_evidence.json),
[metrics.csv](assets/kb2_hybrid_selected_20261108/metrics.csv),
[script tái tính](../../scripts/explain_kb2_hybrid_selected.py).
