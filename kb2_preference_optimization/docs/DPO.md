> Từ31/10, runner chính dùng **conditional_detection_dpo_v0**: Bernoulli keep/drop,
> likelihood tại một quyết định, freeze regression, native+DPO có đối chứng.
> Xem [README hiện tại](../README.md) và [thiết kế đầy đủ](../../docs/KB2_DPO_DESIGN_BRAINSTORM_20261031.md).
> Phần dưới mô tả objective whole-action lịch sử, không phải likelihood của V0.

# Structured whole-output DPO cho YOLO — thiết kế 26/10/2026

**Objective:** `structured_detection_dpo_v1`. Đây là implementation mới; các
run candidate-DPO v1/v2 ngày 07/10 không chứng minh hiệu quả của phương pháp này.

## 1. Ánh xạ

| DPO LLM | KB2 |
|---|---|
| Prompt | Một ảnh train đã resize/pad xác định |
| Response | Toàn bộ action indexed-grid: background/class và bin tọa độ |
| Hai responses | Hai action khác nhau, render thành hai bộ detection A/B |
| Preference annotator | Matching toàn bộ output với GT train |
| SFT | Best checkpoint supervised chung từ KB1 |
| Policy/reference | Hai bản S0; reference không update |

Hai output dùng cùng ảnh, trọng số và quy tắc sampling. Reference không phải
model chuyên sinh rejected. Không đưa GT vào sampler; không dùng hai ảnh
augmentation khác nhau thay cho hai responses của cùng input.

## 2. Policy chuẩn hóa và full action likelihood

Với mỗi vị trí grid i, class logits g_i của YOLO tạo categorical:

```text
p_i = softmax([background_logit, g_i,1, ..., g_i,C] / T_class)
```

Category 0 là background. Cách chuẩn hóa này là lựa chọn riêng của KB2, không
phải xác suất objectness native của YOLOv8. Background logit và temperature
được khóa, lưu trong manifest và dùng y hệt khi sample/chấm likelihood.

Với mỗi cạnh s thuộc l/t/r/b, raw DFL logits tạo:

```text
r_i,s = softmax(raw_DFL_i,s / T_box)
```

Một action a gồm category ở mọi grid slot; bốn DFL bins nếu category khác
background. Lưu foreground indices/classes/bins, background còn lại được suy
ra **nhưng vẫn tính trong likelihood**:

```text
log π(a|x) = sum_i log p_i(category_i)
            + sum_foreground_i sum_s log r_i,s(bin_i,s)
```

Không có tọa độ random ngoài phân phối được chấm; không dùng log(mean confidence),
âm native loss hoặc score chất lượng làm log-probability. Không chia theo số
detection từng output. Tổng xác suất các complete action bằng một trong policy
factorized này. Các độc lập giữa slot/cạnh là giả thiết mô hình hóa.

Native DFL decoder dùng kỳ vọng; bộ lấy mẫu dùng bins. Vì vậy gradient DPO
đi vào raw regression logits khi chấm bins đã lưu, còn native loss tiếp tục
học kỳ vọng tọa độ. Chưa có bảo đảm sampling tốt hơn đồng nghĩa deterministic
inference tốt hơn; evaluation chính luôn đo inference YOLO thường.

## 3. Renderer cố định và giới hạn set likelihood

Giải mã bins theo anchor/grid/stride cố định, clip hộp vào ảnh, bỏ hộp suy biến,
class-aware NMS/cap300. NMS ưu tiên score từ S0 đóng băng, không dùng score/box
của policy đang thay đổi để vẽ lại action cũ. Ties dùng thứ tự grid xác định.
Không confidence-filter lần hai: category đã quyết định xuất/không xuất hộp.

Chất lượng preference được chấm trên output sau renderer. Log-probability
được tính trên **action đầy đủ trước renderer**, gồm cả quyết định bị NMS bỏ.
Nhiều action có thể tạo cùng unordered set; không cộng biên tất cả preimages.
Vì thế tên chính xác là **DPO trên structured detection actions với whole-output
GT preferences**, không tuyên bố exact likelihood của unordered post-NMS set.

## 4. Chấm GT và xây preference

Matching cùng class, one-to-one; tối đa hóa số match trước rồi tổng IoU. Đây
là matcher cho preference/geometry, khác matcher confidence-ordered của COCO.

```text
Q = 0.8 * mean(F2@IoU=0.50:0.05:0.95)
    + 0.2 * sum(matched IoU@0.5)/number_of_GT
```

Từ 28/10/2026 dùng `F2 = 5TP/(5TP + 4FN + FP)` thay F1 để thử ưu tiên recall.
Trọng số quality 0.8/0.2, matching, renderer và ngưỡng gap giữ nguyên. Manifest
ghi `quality.f_beta=2` và `quality.version=whole_set_fbeta_iou_v2`; `--f-beta 1`
cho phép chấm F1 trong một run đối chứng mới. Diagnostic lưu cả mean F1/F2.
Scorer beta=2 độc lập với DPO beta=0.1. Không suy từ lựa chọn F2 rằng recall
inference chắc chắn tăng; kết quả cần so trên cùng validation/protocol.
Thử nghiệm ba seed đã hoàn tất: giảm trung bình 0,67 FN/run so F1,
macro recall không tăng; chưa đạt mục tiêu cải thiện bỏ sót. Xem
[báo cáo F2](../../docs/report_final/KB2_F2_EXPERIMENT_20261028.md).

Ảnh không GT dùng Q=1/(1+number_of_predictions). Score confidence không trực
tiếp tham gia Q. Class sai không match nên gây FP và GT có thể còn FN; duplicate
không nhận thêm TP. Chọn best/worst trong 4 draws, bỏ nếu Q-gap<0.02 hoặc hai
rendered sets giống nhau. Các hằng số là heuristic pilot, chưa được xác nhận.

Preferences chỉ có train và deterministic transform. Mọi lần replay dùng đúng
ảnh, index, class, bins đã lưu. Không augment ảnh ngẫu nhiên với action cũ.
JSONL lưu cả rendered A/B, quality breakdown, cached reference log-probs.

## 5. Objective và training

```text
Δ = (log πθ(a+|x) - log πθ(a-|x))
    - (log πref(a+|x) - log πref(a-|x))
L_DPO = mean(softplus(-β Δ))
L_total = L_native + λ L_DPO
```

Reference frozen/no_grad, forward trên **cùng tensor ảnh, cùng batch size** với
policy mỗi update. Audit cũng chấm policy/reference cùng batch. Cached likelihood
batch1 từ lúc sinh cặp chỉ dùng replay/truy vết, không đi vào DPO objective.
Source/data/transform/code/runtime fingerprints vẫn phải khớp. Kiểm tra replay
cache trước train và log chênh reference gap cùng batch so với cache để phát
hiện sai lệch số học CUDA. Config ghi `reference_scoring=same_batch_reference_v1`.
Native loss/BN và DPO đều dùng cùng policy head; backbone/neck và DFL integral
cố định. Không lấy đạo hàm xuyên sampling/NMS/matching; gradient đi qua
likelihood của action. Khi policy=reference, loss=log2 và gradient có thể khác0.

λ=0.01, β=0.1 là mặc định thử, không kế thừa strength từ candidate-DPO cũ.
Pilot input640 cho thấy λ=0.1 tạo weighted gradient norm khoảng 1.87 lần
native ở một batch, nên giảm λ thử xuống0.01; cần đo lại nhiều batch/model.
Log norm/cosine của native và λ*DPO để phát hiện quá yếu/xung đột. Chưa thêm
projection, weighted sampling, image hybrid, PCGrad hoặc GRPO vào phương pháp.

## 6. Đánh giá và acceptance gates

1. Probability mass=1 trên bài toán nhỏ; gradient đúng hướng, regression/class
   nhận signal; background và tọa độ được tính; reference/BN immutable.
2. Train sampler sinh bộ output khác nhau có chất lượng/coverage hữu ích, đủ
   preference không gần hòa; kiểm tra trực quan các A/B trước run dài.
3. Fixed train audit theo dõi ranking/quality; không chọn checkpoint bằng Q
   hoặc win-rate train. Ground-truth preference đúng theo định nghĩa không
   đồng nghĩa policy đã học.
4. Canonical deterministic S0/DPO dùng cùng resize/pad, confidence .001, NMS .60,
   cap300. Metric chính mAP50–95; thêm AP75/per-class/area-support/P/R/F1/
   FP/FN/IoU+coverage. Area APs/m/l tính trong không gian ảnh đã transform.
5. Step0 luôn là ứng viên best. Pilot/partial validation có cờ riêng, không
   được báo như final test. Khóa method trước test; không train/test tự động.
6. Chỉ sau full validation/multi-seed mới kết luận gain. So S0 đo tác dụng
   quy trình native+DPO tổng thể, chưa cô lập DPO khỏi continued native learning.
   Native-only ablation có thể bổ sung khi cần, không phải hướng KB2 riêng.

## 7. Nguồn và trạng thái

- [DPO gốc, Eq.7](https://arxiv.org/html/2305.18290v3): objective policy/reference;
  không chứng minh chất lượng detector với cách ánh xạ mới của dự án.
- [Ultralytics DFL decoder](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py):
  cơ sở lấy raw bin distributions; implementation thực chạy cần khóa package.
- [COCOeval](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py):
  định nghĩa metric; maxDet300 là contract dự án, khác default maxDet100.

V1 hỗ trợ DFL YOLOv8/YOLO11; chưa hỗ trợ YOLO26 DFL-free. Run dài YOLOv8n
seed 42 đã hoàn tất: best 9k có validation mAP50–95 tăng 0,4347 điểm so S0,
recall giảm 2,8087 điểm. Bổ sung seed 43/44 cùng fixed preferences: gain mean
0,4438 ± 0,0429 điểm qua ba continuation seed, recall đều giảm và gain tập
trung ở một lớp chỉ có 14 GT validation. Chưa xác nhận gain riêng của DPO
hoặc độ vững ngoài S0/preferences/split hiện tại. Xem
[review đa seed](../../docs/report_final/KB2_MULTISEED_REVIEW_20261026.md).
Xem [review run dài](../../docs/report_final/KB2_FULL_RUN_REVIEW_20261026.md).
Source/artifact lịch sử ở [../legacy](../legacy).
