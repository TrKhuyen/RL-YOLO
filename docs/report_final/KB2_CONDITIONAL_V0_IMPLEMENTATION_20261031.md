# KB2 Conditional DPO V0 — triển khai và kiểm kỹ thuật, 31/10/2026

> Cập nhật sau triển khai: pilot2.000 update đã hoàn tất, chưa cải thiện FN. Xem [review pilot](KB2_CONDITIONAL_PILOT_REVIEW_20261031.md). Phần dưới ghi trạng thái tại thời điểm triển khai/smoke.

Đã triển khai bước đầu của [brainstorm](../KB2_DPO_DESIGN_BRAINSTORM_20261031.md):
**DPO giữ/bỏ một candidate, chấm tác động trên toàn ảnh, regression đóng băng**.
Full-train bank và smoke GPU đã hoàn tất. Chưa chạy pilot2.000 update hoặc run dài;
chưa có bằng chứng cải thiện FN trên validation hay tổng quát hóa sang ảnh mới.

## Những gì đã đổi

- Code mới trong [conditional/](../../kb2_preference_optimization/conditional/), objective
  `conditional_detection_dpo_v0`. Code whole-action và recovery cũ giữ nguyên để truy vết.
- Launcher [run_all_kb2.sh](../../kb2_preference_optimization/run_all_kb2.sh) chuyển sang V0,
  mặc định CUDA. Launcher lịch sử là `run_structured_kb2.sh`.
- S0 supervised YOLOv8n khởi tạo policy/reference; reference không update. Chỉ24 tensors
  của classification `cv3` được học; cv2/regression, DFL integral, backbone/neck và BN
  running buffers đóng băng. Không thêm reward model hoặc head.
- A/B khác một active-bit giữ/bỏ; cùng slot/class/box/S0 score và mọi phần chung khác.
  Cặp lưu diff, context lưu một lần/ảnh. Model không nhận context như input phụ.
- Keep cho candidate xuất hiện dù score dưới0,25 để GT chấm cấu trúc, không nâng score
  hoặc thay bằng box GT. Ordinary inference lúc đánh giá vẫn dùng confidence0,25.
- GT gate dùng matching một-một sau NMS toàn ảnh. Keep phải cứu đúng ID GT, không mất
  GT khác/không tăng FP; drop phải giảm FP và không mất GT. Trường hợp nhập nhằng class/
  localization không tự động là negative-background. Duplicate phải thực sự giảm FP.
- Policy Bernoulli: `p(keep)=sigmoid(z)`; `y=+1` cho keep, `-1` cho drop.
  `DPO=mean(softplus(-0.1*y*(z_policy-z_reference)))`; tổng loss `native+0.01*DPO`.
  Đây là likelihood quyết định có điều kiện, không phải likelihood biên của detection set.
- Native batch4:2 ảnh full-train +2 ảnh preference, native học toàn bộ GT cả4 ảnh;
  DPO học2 quyết định. Reference/policy cùng input batch4, float32, TF32 tắt.
- Native-only, DPO và BCE là đối chứng của cùng thí nghiệm; stream/mask/optimizer/
  ngân sách giống nhau. Không khôi phục các nhánh sampling/hybrid cũ.
- Manifest khóa code/S0/data/context/pairs/transform và sampling distribution. Folder
  mới bắt buộc, không resume checkpoint objective cũ, không thay annotation.

## Cấu hình đã khóa

| Thành phần | Giá trị V0 |
|---|---:|
| Input/native batch/preference batch | 640 /4 /2 |
| AdamW LR/weight decay/clip | 1e-6 /5e-4 /1 |
| beta/lambda | 0,1 /0,01 |
| Recovery candidate | argmax đúng class, score>=0,05 và<0,25, IoU>=0,60 |
| Background candidate | score>=0,25, IoU<=0,10 với mọi GT, tâm ngoài mọi GT |
| Giới hạn sinh | Tối đa3 candidate/GT;4 cặp/loại/ảnh |
| NMS/cap | 0,60 /300 |
| Operating confidence/match IoU | 0,25 /0,50 |
| AP inference confidence | 0,001 |
| Sampler | Cân sign/loại/class/GT, rồi cap xác suất theo ảnh |
| Image probability cap | 5/N ảnh có cặp thuộc DPO train |

Giới hạn sampling là **kỳ vọng**, không phải số lần lặp cứng. Sau cap, expected keep
fraction là50,3755%; mỗi ảnh tối đa1,32275% probability trên378 ảnh DPO có cặp.
Không ép nhóm hiếm về đúng50/50 nếu vi phạm cap. Class không có cặp không thể được
cân bằng bằng cách sao chép một mẫu lớp khác. Exposure thực tế theo ảnh/GT/class/cặp
được lưu sau run.

Mặc định pilot chọn best **macro recall cao nhất** với điều kiện micro recall>=S0,
FP<=1,05*S0 và mAP50–95>=S0−0,001; min-delta macro recall1e-4. S0 luôn đủ điều kiện.
Đây là mặc định thử nghiệm đã lưu trước update, có thể thay trước run; không chỉnh
sau khi xem kết quả. Pilot mặc định2.000 update/arm, eval0/1k/2k.

## Audit toàn bộ train

| Nhóm cặp | Tổng | DPO train | Probe không vào DPO |
|---|---:|---:|---:|
| Recovery: giữ để cứu ordinary FN | 296 | 263 | 33 |
| Preservation: giữ GT S0 đã tìm được | 91 | 77 | 14 |
| Background: bỏ prediction nền | 353 | 292 | 61 |
| Duplicate: bỏ prediction dư | 44 | 31 | 13 |
| Tổng | 784 | 663 | 121 |

Đã duyệt2.411 ảnh/10.333 GT: S0 train FN544, FP3.393. Có435 ảnh tạo được cặp,
1.976 ảnh không có cặp hợp lệ. Có11.467 proposal bị gate loại; đây là số proposal,
không phải11.467 ảnh hoặc GT khác nhau. FN recovery coverage là296/544=54,41% trên
train, chưa phải296 GT đã được model học cứu thành công.

| Mật độ ảnh | Số ảnh | S0 FN | Cặp hợp lệ | GT FN có recovery pair |
|---|---:|---:|---:|---:|
| 1–2 object | 1.229 | 40 | 50 | 8 |
| 3–5 object | 592 | 139 | 164 | 66 |
| >=6 object | 590 | 365 | 570 | 222 |

Không loại toàn bộ ảnh đông lá:570/784 cặp đến từ ảnh>=6 object. Tuy nhiên coverage
GT chồng lấn IoU>=0,3 là8/23FN; object nhỏ là6/22FN, thấp hơn nhóm medium192/280FN.
Cần giữ các phân tầng này khi review pilot. Class9 không có FN train và không có
cặp GT hợp lệ trong bank; không tự ép đủ cặp cho mọi class.

Cả2.411 ảnh đã được đối chiếu ordinary S0 thật với baseline context, gồm boxes/
labels/counts/GT IDs/FP. **Replay lại đủ784 cặp cho logit error tối đa0**, GT judge
và evidence giữ nguyên. Bank/context/pairs có hash riêng, tách khỏi objective cũ.

Số đầy đủ: [bank_manifest.json](assets/kb2_conditional_20261031/bank_manifest.json).
Probe thuộc train, chỉ bị bỏ khỏi DPO; vẫn có thể vào native và đã được S0 học.
Không gọi đó là holdout độc lập. Không dùng test để sinh cặp, tune hoặc chọn best.

## Kiểm thử và smoke GPU

Bộ46 tests KB2 đều qua, gồm2 integration GPU. Kiểm được công thức Bernoulli/DPO,
gradient keep/drop, cache/reference cùng batch, GT identity, nhiều lá/NMS, negative
nhập nhằng, repeat cap, checkpoint gates, và raw box không đổi sau AdamW thật.

Smoke dùng bank đầy đủ, seed42, native/DPO/BCE **2 update mỗi arm**, eval step0/2
trên đủ479 ảnh validation/1.887 GT. Stream SHA giống nhau:
`e964bfd2ca19a311e394ceff9a5b54bb5046938e9dc0adddf8183cd42524b0fd`.

| Checkpoint | mAP50–95 (%) | Micro recall (%) | Macro recall (%) | FN | FP |
|---|---:|---:|---:|---:|---:|
| S0, precision V0 | 37,821755 | 54,107048 | 54,546291 | 866 | 984 |
| Native-only, step2 | 37,823460 | 54,107048 | 54,546291 | 866 | 983 |
| Native+DPO, step2 | 37,823460 | 54,107048 | 54,546291 | 866 | 983 |
| Native+BCE, step2 | 37,823933 | 54,107048 | 54,546291 | 866 | 984 |

Cả ba `best.pt` vẫn là S0 step0 vì macro recall không tăng; `last.pt` chứa model
sau2 update. Chênh lệch rất nhỏ không chứng minh phương pháp tốt hơn. FN của
S0 ở báo cáo cũ là867, còn lần này866: V0 khóa TF32 off và đo lại S0 cùng batch/
precision với các arm; không được tính1 FN chênh này thành lợi ích huấn luyện.
Threshold/NMS/matcher chính không đổi; precision được ghi riêng trong manifest.

Metric mới còn lưu recovered/lost/persistent-miss/retained theo ID GT, class/
density/size/overlap, IoU trên tập GT cùng được cảhai phát hiện, score/class-gap/
ordinary target detection ở fixed-slot probes, và coarse threshold sweep để so
recall tại FP budget S0. Sweep chỉ là diagnostic validation, chưa đo calibration.

[Số smoke](assets/kb2_conditional_20261031/smoke_completed.json) và
[manifest/metrics/probes đầy đủ](assets/kb2_conditional_20261031/smoke_details.json).

## Actual AdamW audit tại quyết định đã lấy mẫu

So `native` với `DPO` sau cùng2 update, trên cùng4 quyết định trong stream:

| Quyết định | Score S0 | Score native | Score DPO |
|---|---:|---:|---:|
| Bỏ nền, pair590 | 0,38024440 | 0,37974533 | 0,37970087 |
| Giữ cứu GT, pair625 | 0,08948883 | 0,09022286 | 0,09022646 |
| Bỏ nền, pair414 | 0,29040846 | 0,29006052 | 0,29005778 |
| Giữ GT đã đúng, pair287 | 0,25065851 | 0,25032118 | 0,25032586 |

Ở4 quyết định này, DPO dịch score theo hướng preference so native-only, mean
signed delta+1,3871e-5. Nhưng preservation score vẫn giảm so S0, và positive
probes khác còn có margin âm: Jacobian dùng chung/native vẫn ảnh hưởng các lá
khác. Hai update chưa xác nhận DPO khắc phục bỏ sót, kể cả trên train.

- Raw boxes policy/S0/native có max-abs delta **0**; frozen state/reference không đổi.
- Weighted DPO/native grad norm batch đầu~1,1886%, cosine~0,3841; cảhai có clip.
- L2 update native=0,0009600947; DPO=0,0009599273; L2 difference giữa hai model
  =0,0000439932. Không suy norm gradient là phần trăm tác động optimizer.

Chi tiết: [adam_audit.json](assets/kb2_conditional_20261031/adam_audit.json).
Script tái kiểm: `python -m scripts.audit_kb2_conditional_step --run <V0_run> --output <NEW.json> --device cuda`.

## Artifact và bước tiếp theo

- Bank: `kb2_preference_optimization/outputs/yolov8n_conditional_bank_20261031_v1/preferences/`.
- Smoke: `kb2_preference_optimization/outputs/yolov8n_conditional_smoke_20261031_v2/`.
- Checkpoint S0 và recovery-long best/last đã đối chiếu SHA với audit trước và giữ
  nguyên: [preserved_checkpoints.json](assets/kb2_conditional_20261031/preserved_checkpoints.json).
- Không sửa ảnh/annotation. Full data provenance được kiểm trước/sau generation/train.

Bước tiếp theo là matched pilot2.000 update/arm, chưa chạy trong lượt triển khai này.
Lệnh một dòng dùng bank đã audit, tự tạo output mới:

```bash
bash kb2_preference_optimization/run_all_kb2.sh --device cuda --preferences kb2_preference_optimization/outputs/yolov8n_conditional_bank_20261031_v1/preferences --steps 2000 --eval-interval 1000
```

Chỉ cân nhắc run dài khi ordinary validation cải thiện so native-only trong budget
FP/AP đã khóa. Không lấy train margin tăng hoặc4 score trên làm thay thế. BCE là
đối chứng đã có đường chạy; cùng lambda không đồng nghĩa strength tương đương,
cần ngân sách tuning có kiểm soát trước khi kết luận ưu thế của DPO.

Class repair, cụm NMS, Gaussian geometry, augmentation và refresh pair chưa thuộc
V0; không trộn vào pilot này. Single-candidate gate bỏ nhiều preservation proposals
vì có candidate khác thay thế giữ cùng GT: đây là giới hạn cần báo, không hạ gate
để làm tăng số cặp. Tổng quát hóa ngoài nguồn ảnh hiện tại vẫn chưa được kiểm chứng.
