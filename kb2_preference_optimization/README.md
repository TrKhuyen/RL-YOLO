# KB2 — Conditional keep/drop DPO V0 (31/10/2026)

> **Báo cáo tổng hợp KB2:** [Report_KB2_DPO_20261103.md](../docs/report_final/Report_KB2_DPO_20261103.md), gồm lý thuyết, ánh xạ, lịch sử chỉnh sửa, lý do chọn tham số, kết quả ba seed và định hướng tiếp theo.

> 03/11: [Đã chạy lại đủba seed42/43/44, native và DPO mỗi seed](../docs/report_final/KB2_DPO_STRENGTH_LONG_REVIEW_20261101.md). DPO best mAP38,1707 ± 0,0429% so native38,1196 ± 0,0608%, S0 37,8218%; AP cao hơn native3/3 seed. Đã nạp best/last và xác nhận native/DPO seed42 tái hiện chính xác weights/optimizer15k cũ. Recall tại0,25 vẫn giảm so S0; số liệu mới ở phần bổ sung03/11 cuối báo cáo.

> 01/11: [Lượt dài DPOλ0,1 đã hoàn tất15k](../docs/report_final/KB2_DPO_STRENGTH_LONG_REVIEW_20261101.md). Best_AP tại12k: mAP38,1363% so S0 37,8218%, precision53,0063%, recall53,2591%, FP891/FN882. So native cùng12k, mAP tăng0,0826 điểm %, cùngFN nhưng thêm4FP. Đã nạp lại mọi checkpoint eval; kết quả validation một seed, chưa dùng test.

> 01/11: Đã hoàn tất [pilot λ DPO0/0,01/0,03/0,1](../docs/report_final/KB2_DPO_STRENGTH_PILOT_20261101.md),2k mỗi nhánh, selector mới lưu mọi checkpoint eval và best_AP thực tế ở1k. λ0,1 có lợi thế1 GT so native ở1k nhưng chưa hơn rõ trên nhiều metric. [Audit candidate swap](../docs/report_final/KB2_CANDIDATE_RANK_AUDIT_20261101.md) chỉ20 cặp hợp lệ/15 train; xét hết proposal chỉ28/22 train, chưa train nhánh này.

> Run xác nhận15k đã hoàn tất: tại confidence0,25 native/DPO cùngFN924 so S0 FN866; cảhai best làS0. Ở cùngFP984 sau tune threshold, cảhai FN860. Chưa có lợi ích thêm rõ củaDPO. Xem [review dài](../docs/report_final/KB2_CONDITIONAL_LONG_REVIEW_20261031.md).

> Pilot mới giảm native background0,5 đã xong bốn cấu hình2.000 bước: DPO FN872/FP959 so recipe gốc FN911/FP820 và S0 FN866/FP984. Native positive giữ nguyên; best vẫnS0. Tại cùngFP984, DPO mới FN862/native-only FN864: tín hiệu nhỏ, chưa xác nhận ổn định. Xem [review background](../docs/report_final/KB2_NATIVE_BACKGROUND_PILOT_20261031.md).

> Xác nhận dài background0,5 đã xong15.000 bước/lượt: DPO FN886/FP881,
> native-only FN886/FP878, best vẫn S0. Tại cùng FP984, DPO FN859/native-only FN860
> so S0 FN866; phần thêm DPO chỉ1 GT. Xem [review dài background](../docs/report_final/KB2_NATIVE_BACKGROUND_LONG_REVIEW_20261031.md) và [plan/log](../docs/report_final/KB2_NATIVE_BACKGROUND_LONG_RUN_20261031.md).

Runner chính hiện dùng `conditional_detection_dpo_v0`, triển khai bước đầu của
[brainstorm mới](../docs/KB2_DPO_DESIGN_BRAINSTORM_20261031.md). Model đầu tiên là
YOLOv8n; policy/reference cùng bắt đầu từ **S0 supervised của KB1**. Chỉ nhánh
classification `cv3` được học; regression, backbone/neck và BN running statistics
đóng băng. V0 chưa học sửa tọa độ, class repair hoặc nhóm NMS.

Với cùng ảnh và candidate, A/B chỉ khác **giữ hoặc bỏ** candidate đó. Phần còn lại
có thể còn lỗi nhưng giống hệt nhau trước NMS. GT judge render toàn ảnh rồi matching
một-một: keep phải cứu đúng GT, không mất GT khác/không tăng FP; drop phải giảm FP
và giữ mọi GT đã phát hiện. Candidate cứu FN có thể có score dưới0,25: keep cho nó
xuất hiện để chấm cấu trúc, **không nâng score hoặc ghép box GT**.

Policy Bernoulli dùng raw class logit `z`: `p(keep)=sigmoid(z)`.
`L_DPO=mean(softplus(-beta*y*(z_policy-z_S0)))`, với `y=+1` nếu chọn keep,
`y=-1` nếu chọn drop. Tổng loss `native + 0.01*DPO`, beta0,1. Reference chấm
cùng batch4 với policy; native dùng cả4 ảnh (2 ảnh full-train +2 ảnh preference),
DPO chỉ dùng2 quyết định. Đây là likelihood quyết định có điều kiện, không phải
likelihood của toàn bộ bộ detection không thứ tự sau NMS.

Kiểm kỹ thuật đã qua46 tests, full bank784 cặp và smoke GPU2 update/arm;
Pilot2.000 update/arm đã hoàn tất: native và DPO cùngFN911, S0 FN866; cảhai best
vẫn làS0. Run xác nhận15k sau đó cũng đã hoàn tất, xem review dài ở đầu trang. Xem [review pilot](../docs/report_final/KB2_CONDITIONAL_PILOT_REVIEW_20261031.md)
và [báo cáo triển khai](../docs/report_final/KB2_CONDITIONAL_V0_IMPLEMENTATION_20261031.md).

Từ repo root, lệnh **GPU pilot2.000 update cho mỗi đối chứng native và DPO**, mỗi
lần tự tạo folder mới và checkpoint `best.pt`/`last.pt` riêng:

```bash
bash kb2_preference_optimization/run_all_kb2.sh --model yolov8n --device cuda --steps 2000 --eval-interval 1000
```

Native-only ở đây là đối chứng ngắn có cùng mask/ảnh/cặp/optimizer/seed để đo
phần đóng góp của DPO, không phải một nhánh nghiên cứu KB2 mới. Không ghi đè
checkpoint S0, run cũ hay sửa annotation. Không resume checkpoint whole-action.

```bash
# Kiểm CUDA và provenance, chưa train.
bash kb2_preference_optimization/run_all_kb2.sh --device cuda --check-only

# Sinh/audit full-train pair bank, chưa train.
bash kb2_preference_optimization/run_all_kb2.sh --device cuda --audit-only

# Smoke: chỉ kiểm kỹ thuật, không kết luận về chất lượng.
bash kb2_preference_optimization/run_all_kb2.sh --device cuda --limit-images 64 --steps 2 --eval-interval 1 --eval-limit 8 --audit-pairs 8

# Dùng lại bank V0 có hash/code/provenance khớp; vẫn cần output mới.
bash kb2_preference_optimization/run_all_kb2.sh --device cuda --preferences /path/to/V0/preferences --steps 2000
```

Cấu hình khóa trong manifest trước train: input640/batch4, LR1e-6, AdamW
weight decay5e-4, clip1; cặp recovery cần đúng argmax class, score>=0,05,
IoU>=0,60; nền cần IoU<=0,10 với **mọi GT**, tâm không nằm trong GT. Cặp
preservation và duplicate qua gate riêng; tối đa3 candidate/GT và4 cặp/nhóm/ảnh.
Sampler ưu tiên cân bằng keep/drop, loại cặp, class và GT; sau đó chặn xác suất
mỗi ảnh ở5/N ảnh hợp lệ. Đây là giới hạn kỳ vọng, không phải số lần lặp cứng;
manifest/log ghi tỷ lệ thực tế khi nhóm hiếm không thể đạt50/50.

Chọn best theo **macro recall cao nhất**, chỉ nhận checkpoint có micro recall
>=S0, FP<=1,05*S0 và mAP50–95>=S0−0,001. Đây là mặc định thử nghiệm có thể đổi
trước run qua CLI; S0 luôn là ứng viên best. Eval0/1k/2k trên validation;
checkpoint/metric chính vẫn dùng ordinary inference confidence0,25/IoU0,5,
NMS0,60/cap300, AP từ confidence0,001. Báo GT recovered/lost/persistent-miss,
class/density/overlap/size, score/class-gap/IoU tại slot và drift raw box.
Train probe bỏ khỏi DPO vẫn có thể vào native và đã được S0 học, không phải
holdout độc lập. Test chưa được dùng trong pipeline phát triển này.

Có thể chạy đối chứng local-BCE bằng `--arms native dpo bce`. Cùng hệ số loss
không bảo đảm cùng strength gradient; chưa dùng so sánh đó để tuyên bố DPO tốt
hơn hard-example supervision. Bộ sinh/audit/trainer mới ở [conditional/](conditional/).
Đánh giá riêng checkpoint V0 trên full validation:

```bash
.venv-ubuntu/bin/python -m kb2_preference_optimization.conditional.evaluate --checkpoint /path/to/V0/dpo/best.pt --device cuda --output /path/to/NEW_validation.json
```

### Thử nghiệm riêng: giảm native background

Implementation [conditional_bg/](conditional_bg/) giữ native positive/box/DFL,
chỉ nhân0,5 vào BCE ở các slot **không được native assigner gán cho object**.
Target bằng0 tại slot foreground vẫn giữ hệ số1; mẫu số chuẩn hóa không đổi.
Loss và parameter gradients với hệ số1 đã được kiểm tra trùng hoàn toàn bản gốc
trên GPU; hệ số0,5 giữ nguyên gradient foreground và giảm một nửa background.

Pilot đối chứng2×2: background1/0,5 × native-only/native+DPO,2.000 update mỗi
cấu hình, seed42, cùng chuỗi batch và cặp. Dùng full validation và selector V0;
không đổi runner chính. Lệnh một dòng từ repo root, tự tạo output/checkpoint mới:

```bash
YOLO_CONFIG_DIR=/tmp MPLCONFIGDIR=/tmp/kb2-matplotlib NO_ALBUMENTATIONS_UPDATE=1 .venv-ubuntu/bin/python -m kb2_preference_optimization.conditional_bg.run --preferences kb2_preference_optimization/outputs/yolov8n_conditional_bank_20261031_v1/preferences --device cuda --steps 2000 --eval-interval 1000 --background-weight 0.5
```

Thiết kế và kết quả ghi ở [báo cáo native background](../docs/report_final/KB2_NATIVE_BACKGROUND_PILOT_20261031.md).
Các kiểm tra loss mới nằm trong `tests/test_native_background.py`.
Run thực tế: `outputs/yolov8n_native_background_s42_20261031_001256_145730/`.
Mỗi arm có checkpoint mới; `last.pt` là weights2k, `best.pt` vẫnS0 vì chưa đủ
recall gate. Nạp lại checkpoint/full validation đã khớp log;51 tests đạt.

### Pilot sức mạnh DPO và chọn checkpoint theo nhiều metric — 01/11

Implementation mới ở `conditional_strength/`, giữ nguyên loss và fingerprint
của `conditional/`/`conditional_bg/`. Bốn λ cố định0/0,01/0,03/0,1, seed42,
background0,5, beta0,1,2k mỗi nhánh; cùng bank và input stream. Không có recall
gate khi chọn best_AP. Lưu step0/1k/2k riêng và ghi Pareto mAP/P/R để đối chiếu.

```bash
YOLO_CONFIG_DIR=/tmp MPLCONFIGDIR=/tmp/kb2-matplotlib NO_ALBUMENTATIONS_UPDATE=1 .venv-ubuntu/bin/python -m scripts.run_kb2_strength_pilot --device cuda
```

Run đã hoàn tất: `outputs/yolov8n_strength_s42_20261101_094532_307953/`.
Các nhánh `native`, `lambda001`, `lambda003`, `lambda010` có best.pt ở1k,
last.pt ở2k và toàn bộ step checkpoints. Đã nạp lại8 checkpoint train/S0 trên GPU,
mọi metric khớp log. Không đổi launcher V0 cũ hoặc ghi đè checkpoint trước đó.

`candidate_rank/` cung cấp audit/policy thử nghiệm chọn giữa hai candidate;
chưa dùng train vì chỉ15 cặp train hợp lệ ở config gốc,22 khi xét hết proposal
với cùng gate chất lượng. CPU audit tái sử dụng complete S0
pre-NMS contexts đã xác minh; không phải chuyển train về CPU.

Các thiết kế và lệnh bên dưới là **lịch sử whole-action26–30/10**. Launcher
`run_structured_kb2.sh` giữ nguyên để truy vết chúng; không dùng bank đó cho V0.

---

# KB2 — DPO cho toàn bộ kết quả detection

Thiết kế DPO từ **26/10/2026**, scorer F2 từ **28/10/2026**: KB2 chỉ nghiên cứu DPO. KB1, KB2 và
KB3 độc lập, cùng lấy best checkpoint supervised của KB1 theo từng kiến trúc
làm điểm xuất phát và mốc so sánh. Không dùng checkpoint reward-guided/HPO
làm đầu vào KB2.

## Pipeline

```text
S0 supervised → policy và reference có cùng trọng số
Train image → policy phân phối class/background và bốn phía hộp DFL
            → lấy mẫu nhiều action hoàn chỉnh → decode/NMS cố định → A/B
Train GT → chấm toàn bộ A/B → chosen/rejected
Replay cùng ảnh, cùng action → native loss + λ DPO loss → cập nhật policy
Ordinary deterministic inference → canonical validation → best checkpoint
```

**Đã triển khai và hoàn tất run dài YOLOv8n seed 42/43/44 với cùng S0 và fixed preferences.** Phiên bản
`structured_detection_dpo_v1` học phân phối của *toàn bộ action có candidate
indices*, nhận preference từ chất lượng bộ detection. Nó không tuyên bố tính
likelihood biên của tập hộp không thứ tự sau NMS. DFL sampling là thiết kế mới
của dự án, không phải inference YOLO chuẩn hoặc bảo đảm bất định được hiệu chỉnh.

## Hỗ trợ và phạm vi

- YOLOv8n/s và YOLO11n/s có head DFL; chạy đầu tiên với YOLOv8n.
- YOLO26, YOLOv5, DP-YOLO chưa có policy tọa độ tương ứng trong bản này;
  CLI từ chối thay vì dùng confidence proxy.
- Chỉ train Detect head; backbone/neck, DFL integral và BN running stats cố định.
- Offline preferences chỉ trên train; input resize/pad xác định, không augmentation
  ngẫu nhiên khi replay. Native loss cũng dùng chính ảnh/GT đã replay đó.
- Reference đóng băng và chấm cùng batch với policy ở mỗi update/audit.
  Cached log-prob từ lúc sinh cặp chỉ dùng để kiểm tra replay/truy vết;
  không dùng làm mẫu số DPO khi train batch khác kích thước.
- Không hỗ trợ resume trong v1; checkpoint lưu optimizer/RNG phục vụ truy vết,
  không hứa exact resume khi chưa lưu/replay batch cursor.

## Chạy

Môi trường hiện có: `.venv-ubuntu` (Ultralytics 8.4.163). Các lệnh dưới dùng
GPU qua CUDA; runner cũng mặc định `--device cuda`. Từ repo root:

```bash
bash kb2_preference_optimization/run_structured_kb2.sh --model yolov8n --device cuda --check-only

# Pilot: không phải kết quả full validation hoặc bằng chứng gain.
bash kb2_preference_optimization/run_structured_kb2.sh \
  --model yolov8n --device cuda --output-root /tmp/kb2_pilot_NEW \
  --pilot-images 16 --image-size 192 --draws 4 \
  --steps 2 --batch-size 2 --eval-interval 1 --eval-limit 2

# Run nghiên cứu mới trên đầy đủ train/valid; chỉ chạy sau audit pilot.
bash kb2_preference_optimization/run_structured_kb2.sh \
  --model yolov8n --device cuda \
  --output-root kb2_preference_optimization/outputs/yolov8n_s42_NEW \
  --steps 15000 --batch-size 4 --eval-interval 1000
```

Mỗi run cần thư mục mới, không ghi đè artifact. Không có `--variants` hoặc
`--level`: sampling/hybrid/candidate-pairwise không thuộc pipeline hiện tại.
Runner chạy một model mỗi lần để audit từng policy trước khi nhân rộng.

Các bước có thể chạy riêng, dùng `python -m` từ repo root:

```bash
.venv-ubuntu/bin/python -m kb2_preference_optimization.generate_preferences \
  --model yolov8n --device cuda --output /path/to/NEW_preferences

.venv-ubuntu/bin/python -m kb2_preference_optimization.train_dpo \
  --model yolov8n --device cuda --preferences /path/to/NEW_preferences \
  --output /path/to/NEW_training

.venv-ubuntu/bin/python -m kb2_preference_optimization.evaluate \
  --model yolov8n --device cuda --checkpoint /path/to/NEW_training/best.pt \
  --split valid --output /path/to/NEW_comparison.json
```

Chỉ chuyển `--split test` sau khi khóa phương pháp/checkpoint. Không dùng test
để chọn sampler, β, λ, quality weights hoặc confidence. Pilot bị chặn khỏi
evaluation test-final. Không đối chiếu trực tiếp AP lịch sử dùng NMS 0.45/cap100
với AP mới dùng canonical NMS 0.60/cap300; evaluator mới đo lại S0 và DPO.

## Tham số thử ban đầu

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| Class/box temperature | 1 / 1 | Thang categorical sampling, dùng cùng khi sample và score |
| Background logit | 0 | Lựa chọn background trong categorical C+1 |
| Draws/ảnh | 4 | Sinh nhiều output; lấy best/worst đủ gap |
| Minimum quality gap | 0.02 | Bỏ cặp gần hòa, không ép tạo pair |
| Detection/localization quality | 0.8 / 0.2 | Heuristic được version trong manifest |
| Quality F-beta | 2 | Ưu tiên recall; `--f-beta 1` dùng công thức F1 để đối chứng |
| Native coefficient | 1 | Native detection loss |
| DPO weight λ / β | 0.01 / 0.1 | Giá trị pilot, **chưa được tối ưu** |
| AdamW LR / weight decay | 1e-6 / 5e-4 | Fine-tune head |
| Train batch / input | 4 / 640 | Full-size protocol |
| Clip norm | 1 | Toàn bộ tham số được học |
| Validation / patience | 1000 / 5 | Step 0 được tham gia chọn best; min delta 1e-4 |

Tổng log-prob là tổng trên action đầy đủ, không chia riêng theo số hộp chosen
hoặc rejected. Tăng độ phân giải có thể thay đổi thang log-ratio; cần audit β/λ.
Không suy λ=0.01 là DPO chiếm đúng 1% gradient. Pilot input640 cho thấy
λ=0.1 có weighted DPO/native gradient norm khoảng 1.87 ở batch đầu; vì vậy
mặc định thử được hạ xuống 0.01. Một batch chưa xác nhận strength toàn run.

Scorer hiện tại là `0.8 * mean(F2@IoU=0.50:0.05:0.95) + 0.2 * GT-normalized IoU`.
Quality beta=2 khác DPO beta=0.1. Manifest ghi `quality.f_beta` và phiên bản
`whole_set_fbeta_iou_v2`; diagnostic vẫn lưu F1 và F2. Thay scorer cần sinh lại
preferences và chọn lại best/worst từ đủ bốn output, không dùng nhãn cặp F1 cũ.
Các run F1 ngày 26/10 được giữ làm đối chứng; provenance code của chúng khác
phiên bản hiện tại nên không đưa trực tiếp preferences cũ vào trainer mới.

## Metric và kiểm chứng

- Chọn checkpoint bằng **validation mAP50–95**.
- mAP50, mAP75, AR300, APs/m/l kèm area support, AP/AR/support theo lớp.
- P/R/F1 tại confidence 0.25, IoU 0.5; FP/ảnh, FN rate.
- Matched mean IoU kèm GT coverage và GT-normalized IoU; không bỏ qua FN.
- Fixed train audit: DPO loss, relative/absolute gap, chất lượng output lấy mẫu,
  số detection, empty rate. Đây không phải đánh giá holdout độc lập.
- Gradient norm/cosine native–DPO và tỷ lệ update bị clipping.
- `evaluation_wall_seconds` là thời gian evaluator, không phải FPS/latency thuần.
- Chạy nhiều seed continuation và báo paired delta với cùng S0 sau khi pilot đạt.

```bash
YOLO_CONFIG_DIR=/tmp MPLCONFIGDIR=/tmp/kb2-matplotlib \
  .venv-ubuntu/bin/python -m unittest discover -s kb2_preference_optimization/tests -v
```

Chi tiết xác suất, bộ chấm và giới hạn: [docs/DPO.md](docs/DPO.md).
Kết quả kiểm chứng kỹ thuật và phạm vi đã dọn:
[KB2_REDESIGN_20261026.md](../docs/report_final/KB2_REDESIGN_20261026.md).
Sửa reference cùng batch và kết quả pilot GPU:
[KB2_REFERENCE_BATCH_FIX_20261026.md](../docs/report_final/KB2_REFERENCE_BATCH_FIX_20261026.md).
Run dài YOLOv8n seed 42: early stop 14k, best 9k; validation mAP50–95
37,8084% → 38,2432%, precision tăng nhưng recall giảm. Đã đánh giá GPU lại
best/S0 trên đủ 479 ảnh; chưa đánh giá test. Xem
[KB2_FULL_RUN_REVIEW_20261026.md](../docs/report_final/KB2_FULL_RUN_REVIEW_20261026.md).
Ba continuation seed: best validation mAP50–95 trung bình **38,2523% ± 0,0429 điểm**,
gain **+0,4438 điểm** so S0. Precision/F1 tăng nhưng recall giảm ở cả ba seed;
lớp Corn Gray leaf spot (14 GT valid) đóng góp khoảng 95,33% net gain trung bình.
Đã đánh giá GPU lại cả ba best/S0. Xem
[KB2_MULTISEED_REVIEW_20261026.md](../docs/report_final/KB2_MULTISEED_REVIEW_20261026.md).
Thử scorer F2 ngày 28/10 trên ba seed đã hoàn tất: mAP50–95 **38,2361%**,
recall **51,0687%**, FN **923,33**; so F1, recall chỉ tăng **0,0353 điểm**
và FN giảm **0,67/run**. Macro recall không tăng; chưa xác nhận lợi ích giảm
bỏ sót. F2 là scorer mặc định của thí nghiệm hiện tại, chưa được coi là một
cải tiến đã kiểm chứng. Xem [KB2_F2_EXPERIMENT_20261028.md](../docs/report_final/KB2_F2_EXPERIMENT_20261028.md)
và [bảng FN trước thay scorer](../docs/report_final/KB2_CLASS_FN_AUDIT_20261028.md).
Pilot chọn/cân bằng cặp khôi phục ngày 29/10 nằm trong subpackage `recovery/`:
ba phương án seed42/2k có cùng FN916/FP785, dù nhóm cân bằng tăng lượt cặp
khôi phục từ8,03% lên46,37%. Native image stream được giữ giống nhau; chưa
chạy dài. Xem [KB2_RECOVERY_PILOT_20261029.md](../docs/report_final/KB2_RECOVERY_PILOT_20261029.md).
Sau sinh preference, thư mục `preferences/previews/` có tối đa 8 ảnh so sánh
GT/chosen/rejected để kiểm tra trực quan trước khi train dài.
Các artifact cũ đã dọn theo yêu cầu: [legacy/cleanup_manifest.json](legacy/cleanup_manifest.json).
Source và bảng số nhỏ cũ được giữ trong `legacy/`; không dùng chúng làm kết quả
của objective mới. Checkpoint KB1 và dữ liệu chung không bị dọn.
