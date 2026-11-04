# KB2 — audit cơ hội ưu tiên box tốt hơn, 01/11/2026

**Đã audit toàn bộ train; chưa train nhánh candidate ranking.** Với cấu hình
3 proposal/GT, chỉ20 cặp qua gate,15 cặp thuộc DPO train. Kiểm thêm mọi proposal
đủ điều kiện, giữ nguyên toàn bộ gate, chỉ tăng lên28 cặp/22 train. Độ phủ
này vẫn thấp cho pilot2k; không nới gate chất lượng sau khi xem kết quả.

## Cấu hình cố định

- Dùng S0 YOLOv8n và GT train hiện có, không sửa ảnh/annotation.
- Tận dụng complete pre-NMS contexts trong bank V0 đã audit, chứa mọi candidate
  active có score>=0,25, cả những candidate bị NMS loại. Không suy đoán các
  class logits còn thiếu, không chèn box GT vào output, không sửa confidence.
- Source bank/code/dataset/runtime/checksums được xác nhận bởi Bank; contexts
  này từng được đối chiếu actual GPU ordinary prediction lúc sinh bank.
- Chỉ xét GT đã được S0 phát hiện có IoU<=0,85, và candidate cùng class bị
  suppress, score>=0,25, score không cao hơn source, tăng IoU ít nhất0,1,
  overlap với source candidate vượt NMS0,6. Thử tối đa3 candidate/GT.
- A giữ candidate tốt/bỏ candidate kém; B ngược lại. B phải render đúng output
  ordinary S0. Common context còn lại không thay đổi giữa A/B.
- A phải dùng đúng candidate tốt để match GT mục tiêu, tăng IoU mục tiêu>=0,1,
  tăng tổng matched IoU>=0,05, không tăng FP/không mất GT, đồng thời không mất
  GT ở các mức IoU0,50:0,05:0,95. Tối đa4 cặp hợp lệ/ảnh.

Giới hạn score0,25 có chủ ý: thử cơ hội chọn box tốt đã có trong vùng ordinary
inference, tránh phải bypass confidence như các cặp recovery cũ. Audit này
không bao quát box có score thấp hơn0,25 hoặc class đang dự đoán sai.

## Kết quả

| Chỉ số | Số lượng |
|---|---:|
| Ảnh train | 2.411 |
| GT train | 10.333 |
| GT S0 phát hiện | 9.789 |
| GT S0 còn bỏ sót | 544 |
| GT đã đúng nhưng IoU<=0,85 | 2.100 |
| GT có candidate bị suppress và IoU tốt hơn theo cấu hình | 151 |
| Cặp đã thử | 260 |
| Cặp bị whole-image gate loại | 240 |
| Cặp hợp lệ trước/sau cap | 20/20 |
| Ảnh có cặp hợp lệ | 20 |
| Cặp DPO train / probe thuộc train | 15/5 |

15 cặp train trên15 ảnh,12 class; tổng20 cặp có13 class. Không có cặp bị mất
do cap4/ảnh. Probe là ảnh train loại khỏi DPO, không phải holdout độc lập.

## Vì sao bị loại?

Các lý do sau có thể đồng thời xảy ra trên một cặp, không cộng chúng thành260:

| Lý do | Lượt thử |
|---|---:|
| Candidate tốt không trở thành box match GT mục tiêu | 239 |
| IoU mục tiêu sau render không tăng đủ | 194 |
| Tổng matched IoU không tăng đủ | 128 |
| Mất GT tại ít nhất một mức IoU0,5–0,95 | 22 |
| Tăng FP | 13 |
| Mất GT tại IoU0,5 | 1 |

Trong239 trường hợp đầu: **237 candidate tốt không còn trong output sau NMS**,
2 candidate vẫn xuất hiện nhưng không match GT mục tiêu. Các output liên quan
không chạm cap300, nên237 trường hợp không phải bị cắt bởi max_det.

Điều này cho thấy bỏ một candidate kém chưa đủ: một candidate khác trong common
context vẫn có thể chặn box tốt. Chỉ xem raw IoU của hai box sẽ nhận nhầm nhiều
cặp là hữu ích, trong khi ordinary output sau NMS không có cải thiện tương ứng.

## Kiểm giới hạn số proposal, giữ nguyên gate

Để không quy sai độ phủ thấp cho giới hạn3 proposal/GT, đã xét hết các candidate
đủ điều kiện, bỏ cap số proposal và cap số cặp/ảnh, vẫn chỉ lấy một cặp hợp lệ
mỗi GT. Giữ mọi ngưỡng score/IoU/NMS và whole-image gate như cũ.

Kết quả:307 attempts,279 bị loại,28 cặp hợp lệ trên26 ảnh;22 cặp train/20 ảnh,
6 probe,14 class trong train. Đã replay129 ảnh chứa toàn bộ151 GT có candidate
tiềm năng; các ảnh/GT không có candidate theo cùng gate hình học không thể tăng
coverage chỉ bằng việc bỏ cap proposal.

Giới hạn proposal có bỏ lỡ8 cặp, nhưng không giải thích phần lớn độ phủ thấp.
Đây là sensitivity audit đọc-only, không thay cấu hình của bank gốc hoặc mở run
train theo một ngưỡng chất lượng được nới hậu nghiệm.

## Policy đã kiểm kỹ thuật, chưa train

Policy chọn đúng một trong hai candidate có xác suất
`p(good)=sigmoid(z_good-z_bad)`, tương đương chuẩn hóa hai khả năng của
Bernoulli độc lập có điều kiện đúng một candidate được giữ. DPO dùng contrast
`(z_good-z_bad)_policy - (z_good-z_bad)_S0`.

Đã kiểm công thức probability, dấu gradient tại good/bad, reference detach,
không thay box/score/context, slot IDs không liên tục của cache và trường hợp
candidate tốt bị lá bên cạnh suppress.5 unit tests đạt. Các kiểm tra này không
chứng minh parameter update hoặc model sau train sẽ cải thiện ordinary detection.

## Quyết định sau audit

Không train2k bằng15 cặp của cấu hình gốc; kể cả22 cặp khi xét hết proposal,
4.000 preference draws vẫn lặp trung bình khoảng182 lần/cặp, với chỉ20 ảnh train.
Không nhận đây là bằng chứng hướng quality
preference nói chung thất bại; chỉ là thiết kế hai-candidate tại cấu hình này
chưa cung cấp đủ cặp hữu ích trên S0/train.

Bước thiết kế tiếp có cơ sở là xem cụm các candidate cạnh tranh qua NMS và
kiểm probability/gradient để ordinary inference thực sự ưu tiên box tốt. Nếu
xóa thêm candidate trong context, phải định nghĩa rõ action và kiểm rằng loss
cũng học điều cần thay đổi; không chỉ làm renderer đẹp lên. Mở regression để
học tọa độ là hướng riêng, chưa triển khai trong lượt này.

## Artifact và tái hiện

Audit gốc: `kb2_preference_optimization/outputs/yolov8n_candidate_rank_audit_20261101/`.
Gồm completed.json, pairs.jsonl, contexts.jsonl, gt_audit.jsonl; các cặp mới không
trộn với bank V0. Source/code/context SHA nằm trong manifest.

- [Manifest](assets/kb2_candidate_rank_20261101/audit_manifest.json).
- [Phân tích gate](assets/kb2_candidate_rank_20261101/analysis.json).
- [Từng attempt và evidence](assets/kb2_candidate_rank_20261101/attempts.jsonl).
- [Tests](assets/kb2_candidate_rank_20261101/tests.txt).
- [Kiểm hết proposal](assets/kb2_candidate_rank_20261101/proposal_cap_sensitivity.json).
- [Pilot λ DPO](KB2_DPO_STRENGTH_PILOT_20261101.md).

Lệnh tái hiện audit, output phải chưa tồn tại:

```bash
YOLO_CONFIG_DIR=/tmp MPLCONFIGDIR=/tmp/kb2-matplotlib NO_ALBUMENTATIONS_UPDATE=1 .venv-ubuntu/bin/python -m scripts.audit_kb2_candidate_ranking --output /tmp/kb2_rank_audit_NEW --device cpu
```

CPU ở lệnh này chỉ tái chấm các prediction S0 đã có; các pilot train và fresh
checkpoint evaluation vẫn dùng GPU.
