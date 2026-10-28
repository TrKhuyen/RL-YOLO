# Báo cáo review logic KB2 và hướng khắc phục

## 1. Phạm vi review

KB2 nghiên cứu cách fine-tune YOLO bằng phản hồi sinh từ prediction và ground
truth. Ý tưởng được lấy cảm hứng từ DPO, nhưng không yêu cầu sao chép nguyên DPO
dành cho mô hình ngôn ngữ.

Pipeline hiện tại gồm:

```text
Train images + ground truth
        |
        v
YOLO predictions
        |
        v
Feedback matching và phân loại lỗi
        |
        v
Image sampling / object loss / dynamic feedback / PCGrad
        |
        v
Native YOLO fine-tuning
        |
        v
Validation mAP và checkpoint selection
```

Review này tập trung vào tính đúng đắn của:

- Ghép prediction với ground truth.
- Phân loại feedback.
- Sampling weight.
- Object-level và dynamic feedback loss.
- Gradient-conflict control.
- Evaluator và checkpoint selection.
- Resume, reproducibility và ablation protocol.

Review không tự động phủ định toàn bộ kết quả cũ. Tuy nhiên một số lỗi được phát
hiện có thể làm sai tín hiệu feedback và ảnh hưởng kết luận thực nghiệm. Vì vậy
kết quả hiện tại nên được xem là kết quả của implementation hiện tại, chưa phải
kết luận cuối cùng về khả năng áp dụng feedback learning cho YOLO.

## 2. Tóm tắt phát hiện

| ID | Mức độ | Vấn đề | Tác động chính |
|---|---|---|---|
| L1 | Critical | Một GT đã matched vẫn có thể bị gắn wrong-class/bad-localization | Object loss học từ lỗi không tồn tại |
| L2 | Critical | Một lỗi bị đếm chồng với missed | Sampling weight bị phóng đại |
| L3 | High | Matching nearest-GT không one-to-one và không class-aware | Sai cặp prediction–GT |
| L4 | High | mAP được tính sau khi lọc confidence 0.25 | AP không theo protocol chuẩn, có thể đổi ranking checkpoint |
| L5 | High | Dynamic feedback chọn dense anchor chỉ theo max IoU | Feedback không phản ánh detection cuối |
| L6 | Medium-high | Wrong-class loss chỉ tăng class đúng | Không trực tiếp giảm class sai |
| L7 | Medium-high | Resume không lưu RNG/sampler/dataloader state | Resume không tương đương run liên tục |
| L8 | Medium | Adapter train/eval mode không nhất quán | API dễ crash hoặc trả output sai dạng |
| L9 | Medium | Preference rejected có thể là detection đúng của GT khác | Preference pair không hợp lệ |
| L10 | Medium | Augmentation fallback nuốt mọi exception | Lỗi dữ liệu/transform bị che giấu |
| L11 | Medium | Hyperparameter chọn trên một seed rồi báo best | Nguy cơ overfit seed/validation |
| L12 | Low-medium | Tài liệu mô tả raw logits trong khi output eval là probability | Dễ xây loss sai trong thay đổi sau |

## 3. Lỗi L1: GT đã detect đúng vẫn có thể bị gắn lỗi

### 3.1 Logic hiện tại

Trong `feedback.py`, mỗi prediction được phân loại độc lập. Một GT có thể đồng
thời nhận:

- Một prediction đúng class, IoU đủ cao, được ghi là `matched`.
- Một prediction khác sai class, được ghi là `wrong_class`.
- Một prediction khác có IoU thấp, được ghi là `bad_localization`.

Trong `feedback_dataset.py`, object code được gán theo thứ tự ghi đè:

```text
missed -> bad_localization -> wrong_class
```

Hàm này không loại các GT đã có trong `matched`. Do đó GT đã được phát hiện đúng
vẫn có thể nhận object code `wrong_class` hoặc `bad_localization`.

### 3.2 Tác động

- Static object loss tối ưu một lỗi không tồn tại ở output cuối.
- Số lượng wrong-class/bad-localization bị thổi phồng.
- Một detection phụ kém chất lượng có thể ghi đè trạng thái detection đúng.
- Kết quả static object feedback và sampling dựa trên feedback tĩnh bị nhiễu.

### 3.3 Hướng sửa

Phản hồi cấp GT phải loại trừ lẫn nhau:

```text
if có matched prediction đúng:
    status = matched
elif có prediction IoU cao nhưng sai class:
    status = wrong_class
elif có prediction overlap nhưng IoU thấp:
    status = bad_localization
else:
    status = missed
```

`false_positive` và `duplicate` là trạng thái cấp prediction, không được dùng để
ghi đè trạng thái cấp GT.

### 3.4 Test bắt buộc

- Một GT có cả prediction đúng và prediction sai class phải là `matched`.
- Một GT có prediction đúng và prediction định vị kém phải là `matched`.
- Mỗi GT chỉ có đúng một trạng thái cuối.
- Tổng số trạng thái GT phải bằng tổng số GT.

## 4. Lỗi L2: Đếm chồng wrong-class/bad-localization với missed

### 4.1 Logic hiện tại

`matched_gt` chỉ chứa GT có prediction đúng class và IoU đạt ngưỡng. Vì vậy GT có
wrong-class hoặc bad-localization vẫn được thêm vào danh sách `missed`.

`feedback_difficulty()` sau đó cộng số lượng của mọi loại feedback:

```text
difficulty =
    wrong_class_weight * wrong_class_count
  + bad_localization_weight * bad_localization_count
  + missed_weight * missed_count
  + ...
```

Một object có thể đóng góp hai lần vào cùng điểm difficulty.

### 4.2 Tác động

- Ảnh có wrong-class/mislocalized object bị oversample mạnh ngoài dự kiến.
- Sampling distribution lệch và có variance cao.
- Khó diễn giải ý nghĩa của `sampling_strength`.
- Kết quả image-level feedback sampling có thể phản ánh lỗi double-count thay vì
  hiệu quả của curriculum learning.

### 4.3 Hướng sửa

Tách hai nhóm thống kê:

```text
GT feedback: matched | wrong_class | bad_localization | missed
Prediction feedback: true_positive | false_positive | duplicate
```

Difficulty cấp ảnh chỉ cộng một trạng thái cho mỗi GT. Nếu cần thêm penalty cho
false-positive hoặc duplicate thì cộng riêng theo prediction.

### 4.4 Test bắt buộc

- Một wrong-class GT không đồng thời được tính missed.
- Một bad-localization GT không đồng thời được tính missed.
- Difficulty của một object không thay đổi do có thêm alias của cùng lỗi.
- Có test công thức difficulty trên record nhiều GT và nhiều prediction.

## 5. Lỗi L3: Matching không one-to-one và không class-aware

### 5.1 Logic hiện tại

Mỗi prediction được gán cho GT có IoU lớn nhất độc lập với các prediction khác và
không xét class trước khi chọn GT.

Trong ảnh có nhiều object gần nhau, prediction đúng cho GT A có thể có IoU cao
hơn một chút với GT B. Khi đó:

- Prediction bị gán cho B.
- B có thể bị ghi wrong-class.
- A có thể bị ghi missed.
- Những prediction tiếp theo tiếp tục dùng cùng quy tắc, không tối ưu matching
  toàn cục.

### 5.2 Hướng sửa đề xuất

Thực hiện matching theo các bước:

1. Lọc prediction hợp lệ và sắp theo confidence.
2. Tạo ma trận IoU prediction–GT.
3. Ghép one-to-one các cặp đúng class có IoU >= match threshold.
4. Với GT chưa matched, ghép one-to-one prediction sai class có IoU cao để xác
   định `wrong_class`.
5. Với GT vẫn chưa matched, tìm prediction overlap trong khoảng localization để
   xác định `bad_localization`.
6. GT còn lại là `missed`.
7. Prediction chưa dùng là `false_positive`.
8. Prediction thừa trùng với GT đã matched đúng là `duplicate`.

Có thể dùng greedy matching theo IoU giảm dần hoặc Hungarian matching. Greedy đủ
đơn giản nếu có deterministic tie-breaking và test đầy đủ.

### 5.3 Test bắt buộc

- Hai GT gần nhau, hai prediction đúng: matching không đổi chéo class.
- Hai prediction tranh một GT: chỉ một primary match, prediction còn lại duplicate.
- Tie IoU phải cho kết quả deterministic.
- Prediction đúng class được ưu tiên trước prediction sai class.
- Không một prediction nào được primary-match nhiều GT.

## 6. Lỗi L4: Protocol mAP dùng confidence threshold quá cao

### 6.1 Logic hiện tại

Evaluator gọi NMS với confidence `0.25` trước khi đưa prediction vào COCO mAP.
Điều này loại bỏ phần lớn prediction score thấp trước khi metric xây
precision–recall curve.

Trong thử nghiệm đối chứng:

```text
conf = 0.001 -> mAP50-95 khoảng 0.6921
conf = 0.25  -> mAP50-95 khoảng 0.6674
```

### 6.2 Tác động

- Đây không phải AP protocol tiêu chuẩn.
- Thay đổi confidence calibration có thể bị che giấu.
- Ranking checkpoint có thể thay đổi.
- Kết quả khó so sánh với Ultralytics/COCO report.

### 6.3 Hướng sửa

Dùng hai nhóm metric riêng:

```text
Ranking metrics:
    mAP50-95, mAP50, AP-small với conf khoảng 0.001

Operating-point metrics:
    precision, recall, F1 với conf 0.25
```

NMS IoU và `max_det` phải khớp protocol của baseline. Sau khi đổi protocol phải
đánh giá lại checkpoint supervised từ đầu; không được so trực tiếp số AP mới với
bảng cũ dùng conf 0.25.

### 6.4 Test bắt buộc

- So evaluator KB2 với `YOLO.val()` trên cùng checkpoint/split.
- Chênh lệch metric phải nằm trong tolerance được định nghĩa trước.
- Test ảnh không prediction, ảnh nhiều prediction và ảnh nhiều hơn `max_det`.

## 7. Lỗi L5: Dynamic feedback không phản ánh output detection cuối

### 7.1 Logic hiện tại

Dynamic feedback duyệt 8.400 dense candidates và chọn candidate có IoU lớn nhất
với mỗi GT. Sau đó class/confidence của riêng candidate này quyết định trạng thái.

Candidate IoU cao nhất có thể:

- Có confidence thấp và không qua NMS.
- Không phải candidate được task assigner chọn.
- Kém hơn một candidate khác có IoU thấp hơn rất ít nhưng class score cao.

Do đó một GT có detection hợp lệ sau NMS vẫn có thể bị phân loại `missed` hoặc
`bad_localization`.

### 7.2 Hướng sửa

Tách feedback decision và differentiable loss:

1. Dùng prediction sau NMS để xác định trạng thái lỗi hiện tại.
2. Ánh xạ lỗi đó về candidate pre-NMS phù hợp để truyền gradient.
3. Candidate pre-NMS nên được chọn theo quality kết hợp, ví dụ:

```text
quality = IoU^alpha * p(true_class)^beta
```

4. Tốt hơn nữa là tích hợp object weight vào native Ultralytics task assigner,
   thay vì tự tạo một assignment thứ hai bên ngoài.

### 7.3 Test bắt buộc

- Max-IoU anchor confidence thấp nhưng anchor thứ hai được NMS giữ.
- Detection đúng sau NMS phải cho trạng thái matched.
- Dynamic status phải khớp feedback generator trên cùng output NMS.
- Candidate nhận gradient phải thuộc vùng/level phù hợp với GT.

## 8. Lỗi L6: Wrong-class loss không phạt class sai

### 8.1 Logic hiện tại

Object classification loss chỉ dùng:

```text
-log(p_true_class)
```

YOLO dùng independent class probabilities, nên tăng class đúng không đảm bảo
class sai giảm.

### 8.2 Hướng sửa

Với wrong-class feedback, dùng BCE vector hoặc ít nhất:

```text
L_wrong = -log(p_true) - lambda_wrong * log(1 - p_predicted_wrong)
```

Nếu tích hợp vào native task assigner, dùng target class vector có trọng số thay
vì tạo loss classification ngoài.

### 8.3 Test bắt buộc

- Tăng `p_true` phải giảm loss.
- Tăng `p_wrong` phải tăng loss.
- Gradient của true class và wrong class phải có dấu mong đợi.

## 9. Lỗi L7: Resume không tái lập run liên tục

### 9.1 Trạng thái được lưu hiện tại

- Model state.
- Optimizer state.
- Step.
- Early-stopping state.

### 9.2 Trạng thái chưa được lưu

- Python random state.
- NumPy RNG state.
- PyTorch CPU RNG state.
- CUDA RNG states.
- WeightedRandomSampler generator state.
- Albumentations RNG state.
- Vị trí hiện tại trong epoch/DataLoader.

Khi resume, sampler bắt đầu lại từ seed ban đầu và có thể lặp lại các batch đã
học. Config trong checkpoint cũng được đọc nhưng không đối chiếu với CLI hiện tại.

### 9.3 Hướng sửa

Checkpoint cần lưu:

```python
python_rng
numpy_rng
torch_rng
cuda_rng_all
sampler_generator_state
epoch
batch_in_epoch
training_config_hash
feedback_sha256
```

Khi resume:

- Validate model, dataset, feedback hash và các hyperparameter quan trọng.
- Restore RNG trước khi tạo iterator.
- Skip chính xác số batch đã xử lý hoặc dùng stateful sampler.
- Cho phép override chỉ với cờ tường minh.

### 9.4 Test bắt buộc

So sánh:

```text
run liên tục 20 step
vs
run 10 step + save + resume 10 step
```

Model parameters, optimizer state, sampled image IDs và loss history phải giống
nhau trong tolerance xác định.

## 10. Lỗi L8: Adapter mode không nhất quán

### 10.1 Hiện trạng

`train_mode()` gọi `model.eval()`. `forward_with_grad()` lại giả định model đang
ở eval mode nhưng không tự thiết lập. Nếu gọi ngay sau khi load, model có thể trả
dict train-output thay vì tensor inference-output và gây lỗi shape.

### 10.2 Hướng sửa

Đổi interface thành các mode có tên rõ ràng:

```python
set_inference_mode()
set_feedback_train_mode(freeze_bn=True)
set_native_train_mode(freeze_bn=True)
```

`raw_predictions()` và `forward_with_grad()` phải đảm bảo mode/output contract.
Không nên phụ thuộc vào hàm nào vừa được gọi trước đó.

### 10.3 Test bắt buộc

- Gọi `forward_with_grad()` ngay sau constructor.
- Gọi sau `supervised_loss()`.
- Gọi sau validation.
- Output type và shape phải giống nhau trong mọi trường hợp.
- BatchNorm running mean/variance không đổi khi `freeze_bn=True`.

## 11. Lỗi L9: Preference rejected có thể là detection đúng

### 11.1 Logic hiện tại

Với mỗi chosen prediction, rejected được chọn là prediction confidence cao nhất
bất kỳ còn lại. Prediction đó có thể là detection đúng của một GT khác.

### 11.2 Tác động

Nếu preference loss được bật lại, mô hình có thể bị yêu cầu giảm score của một
detection hợp lệ.

### 11.3 Hướng sửa

Rejected phải là candidate cạnh tranh cho cùng GT:

- Cùng vùng hoặc cùng assignment neighborhood.
- Không primary-match GT khác.
- Có cùng class hoặc là wrong-class candidate của chính GT đó.
- Chosen phải có quality cao hơn rejected theo tiêu chí rõ ràng.

Nếu không tạo được pair hợp lệ thì bỏ pair, không ép tạo.

## 12. Lỗi L10: Augmentation fallback che giấu lỗi

### 12.1 Logic hiện tại

Dataset bắt toàn bộ `Exception` trong transform và âm thầm resize ảnh sạch.

### 12.2 Tác động

- Lỗi bbox hoặc label-field có thể tồn tại lâu mà không được phát hiện.
- Một phần dataset dùng distribution augmentation khác.
- Không biết có bao nhiêu ảnh fallback.
- Khó tái lập và debug.

### 12.3 Hướng sửa

- Chỉ bắt exception đã biết.
- Log image path và exception type.
- Có counter `augmentation_failures`.
- Fail-fast trong test/debug mode.
- Trong training, dừng nếu tỷ lệ fallback vượt ngưỡng định trước.

## 13. Lỗi L11: Nguy cơ overfit seed và validation

Nhiều alpha/strength được quét trên seed 42. Cấu hình tốt nhất sau đó chỉ thắng
seed 42 và thua seed 43, 44. Đây là dấu hiệu chọn hyperparameter theo nhiễu của
một seed.

### Hướng sửa

- Dùng hai seed development để chọn cấu hình và một seed confirmatory.
- Hoặc chọn cấu hình theo mean của ba seed development với budget nhỏ.
- Chốt cấu hình trước khi chạy evaluation dài.
- Không tiếp tục quét alpha sau khi thấy một cải thiện rất nhỏ trên cùng validation.
- Định nghĩa minimum effect size trước, ví dụ `+0.001 mAP50-95` và không giảm
  AP-small.

## 14. Lỗi L12: Contract raw output được mô tả sai

Tài liệu adapter ghi class output là logits chưa sigmoid, trong khi output eval
đã quan sát nằm trong miền probability. Code hiện tại phần lớn xử lý nó như
probability, nhưng comment sai dễ dẫn đến sigmoid hai lần hoặc loss sai sau này.

### Hướng sửa

- Định nghĩa rõ `raw_train_output` và `decoded_eval_output`.
- Assert shape/range ở adapter boundary.
- Không dùng cùng tên `raw_predictions` cho hai representation khác nhau.

## 15. Đánh giá lại các kết quả KB2 hiện tại

Các kết quả đã chạy vẫn có giá trị để mô tả hành vi của implementation cũ:

| Phương pháp | Mean mAP50-95 | Delta so baseline | Wins |
|---|---:|---:|---:|
| Baseline | 0.668234 | 0 | - |
| Image feedback sampling | 0.667926 | -0.000308 | 1/3 |
| Static object feedback | 0.667823 | -0.000411 | 1/3 |
| Dynamic object feedback | 0.668011 | -0.000223 | 1/3 |
| PCGrad-style | 0.667860 | -0.000374 | 1/3 |

Tuy nhiên không nên diễn giải bảng này thành bằng chứng rằng feedback learning
không phù hợp với YOLO, vì:

1. Feedback tĩnh có lỗi trạng thái mâu thuẫn và double-count.
2. Matching chưa one-to-one.
3. Dynamic assignment chưa phản ánh detection sau NMS.
4. AP được tính ở confidence threshold cao.

Kết luận phù hợp hơn:

> Với implementation và protocol hiện tại, các biến thể feedback chưa vượt
> supervised continuation baseline qua ba seed.

## 16. Kế hoạch khắc phục theo thứ tự

### Giai đoạn A: Sửa nền tảng feedback

1. Viết lại matching thành one-to-one, ưu tiên đúng class.
2. Tạo một trạng thái loại trừ lẫn nhau cho mỗi GT.
3. Tách trạng thái GT và trạng thái prediction.
4. Tính lại difficulty không double-count.
5. Sinh lại toàn bộ feedback JSONL.

Điều kiện qua cổng:

- Unit test matching đầy đủ.
- Tổng GT status bằng tổng GT.
- Không GT nào vừa matched vừa có error status.
- Trực quan kiểm tra lại tối thiểu 50 ảnh.

### Giai đoạn B: Chuẩn hóa evaluator

1. Dùng confidence thấp cho COCO AP.
2. Đối chiếu với Ultralytics `val()`.
3. Báo riêng metric tại operating threshold 0.25.
4. Đánh giá lại supervised checkpoint và cố định protocol mới.

Điều kiện qua cổng:

- Metric parity đạt tolerance định trước.
- Không còn thay protocol giữa các experiment.

### Giai đoạn C: Sửa adapter và resume

1. Chuẩn hóa train/eval/feedback mode.
2. Thêm output contract và shape assertions.
3. Lưu/khôi phục toàn bộ RNG và sampler state.
4. Test continuous run so với resume run.

### Giai đoạn D: Xây lại phương pháp feedback

Thứ tự thử:

1. Corrected image-level sampling.
2. Object weighting tích hợp native assignment/loss.
3. Dynamic feedback dựa trên post-NMS status và pre-NMS gradient target.
4. Gradient conflict control chỉ sau khi object objective được chứng minh đúng.

Không nên thử lại preference loss trước khi pair generation được sửa.

### Giai đoạn E: Chạy lại thực nghiệm

1. Baseline và feedback dùng cùng protocol, seed và budget.
2. Pilot nhỏ có step-0 validation.
3. Chỉ giữ cấu hình vượt minimum effect size.
4. Chạy ít nhất ba seed.
5. Chỉ chạy test set sau khi khóa cấu hình.

## 17. Ma trận test cần bổ sung

| Nhóm | Test |
|---|---|
| Matching | one-to-one, class-aware, tie, duplicate, no prediction, no GT |
| Feedback | exclusive GT status, no double-count, prediction status separation |
| Dataset | GT index qua augmentation, malformed labels, fallback counter |
| Adapter | output contract ở mọi mode, BN frozen, score mapping sau NMS |
| Dynamic | post-NMS status parity, candidate mapping, confidence edge cases |
| Loss | true/wrong class gradient signs, IoU gradient, empty error batch |
| PCGrad | aligned/conflicting/zero gradients, finite norm, clipping interaction |
| Evaluator | parity với Ultralytics, low-conf AP, max-det edge case |
| Resume | continuous-vs-resume equivalence, config/hash mismatch |
| Experiment | deterministic same-seed run, paired multi-seed aggregation |

## 18. Ưu tiên sửa

### Bắt buộc trước khi chạy lại

1. L1: trạng thái GT mâu thuẫn.
2. L2: difficulty double-count.
3. L3: matching không one-to-one.
4. L4: AP protocol confidence 0.25.

### Bắt buộc trước khi dùng dynamic/object feedback

5. L5: dynamic candidate selection.
6. L6: wrong-class objective.
7. L8: adapter mode contract.

### Bắt buộc trước experiment dài hoặc chạy gián đoạn

8. L7: exact resume.
9. L10: augmentation failure visibility.

### Chỉ cần nếu bật lại preference learning

10. L9: valid chosen/rejected construction.

## 19. Kết luận review

KB2 đang đúng về mục tiêu nghiên cứu: dùng phản hồi từ lỗi của mô hình để điều
khiển fine-tuning YOLO mà không thay đổi dữ liệu. Pipeline cũng đã có nhiều thành
phần kỹ thuật tốt như versioned feedback, deterministic sampler, validation
checkpointing, early stopping và multi-seed evaluation.

Tuy nhiên lõi feedback matching hiện có lỗi logic đủ lớn để ảnh hưởng sampling và
object-level loss. Protocol AP cũng chưa theo cách tính COCO AP chuẩn. Vì vậy cần
sửa nền tảng feedback và evaluator trước khi tiếp tục tối ưu alpha hoặc kết luận
về hiệu quả của phương pháp.

Thứ tự hành động được khuyến nghị:

```text
Correct matching
    -> exclusive feedback
    -> corrected difficulty
    -> standard AP evaluator
    -> adapter/resume hardening
    -> rerun baseline
    -> rerun feedback ablation
    -> multi-seed confirmation
```

Sau các sửa đổi này, nếu feedback vẫn không vượt baseline qua nhiều seed thì mới
có đủ cơ sở để kết luận phương pháp không mang lại lợi ích trên cấu hình dữ liệu
và mô hình hiện tại.
