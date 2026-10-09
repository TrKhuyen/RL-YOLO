# KB2 — Review feedback-guided, pairwise và candidate-selection DPO

## 1. Phạm vi và tên phương pháp

KB2 dùng prediction của checkpoint nền trên **tập train** cùng ground truth để tạo tín hiệu chọn mẫu và loss phụ cho YOLO. Native detection loss vẫn là objective chính. Nhánh **pairwise preference optimization** chọn hai candidate của mô hình hiện tại cho cùng GT và tối ưu điểm chosen cao hơn rejected. Baseline là native continuation; sampling và hybrid dùng feedback. Từ ngày 2026-10-07, nhánh **candidate-selection DPO** dùng reference đóng băng, policy categorical trên chỉ số candidate và objective policy/reference log-ratio. Phiên bản hiện tại **v2 chọn A/B bằng IoU prediction của policy đang train so với GT**; reference chỉ tính log-prob trên cùng chỉ số A/B. Các mục 8–9 ghi kết quả v1 lịch sử, mục 10 ghi sửa đổi và pilot v2. Đây là DPO online cho chọn candidate, chưa phải policy sinh toàn bộ detection set; định nghĩa và giới hạn ở [DPO.md](../../kb2_preference_optimization/docs/DPO.md).

## 2. Luồng thực thi

| Bước | Code | Vai trò |
|---|---|---|
| Sinh feedback | [generate_feedback.py](../../kb2_preference_optimization/generate_feedback.py), [feedback.py](../../kb2_preference_optimization/feedback.py) | Dự đoán trên train, ghép GT và ghi lỗi. |
| Gắn feedback | [feedback_dataset.py](../../kb2_preference_optimization/feedback_dataset.py) | Tính difficulty; nối trạng thái GT với box sau augmentation qua gt_indices. |
| Fine-tune | [feedback_loss.py](../../kb2_preference_optimization/feedback_loss.py), [train_feedback.py](../../kb2_preference_optimization/train_feedback.py) | Native loss, sampling, loss phụ theo feedback, pairwise ranking và nhánh candidate-selection DPO riêng. |
| Chọn checkpoint | [feedback_validation.py](../../kb2_preference_optimization/feedback_validation.py) | Đánh giá AP trên validation theo protocol cố định. |

Các nhánh phải dùng cùng checkpoint nền, split, augmentation, số update và evaluator để có thể diễn giải chênh lệch.

## 3. Matching và trạng thái feedback

**K2-L1–L3: bắt buộc, đã sửa trong code.** Trước đây từng prediction được xét độc lập, nên một GT có thể vừa matched vừa bị tính wrong-class, bad-localization hoặc missed. Difficulty có thể đếm trùng và object code có thể bị ghi đè. Với các GT gần nhau, prediction còn có thể bị gán sang GT sai.

[feedback.py](../../kb2_preference_optimization/feedback.py) hiện tạo **một trạng thái chính cho mỗi GT**: ưu tiên ghép đúng lớp, đủ IoU và one-to-one; với GT chưa matched, chọn wrong-class hoặc bad-localization; cuối cùng mới missed. Duplicate và false-positive thuộc cấp prediction. [feedback_dataset.py](../../kb2_preference_optimization/feedback_dataset.py) tính difficulty và object code từ trạng thái GT độc quyền. One-to-one ở đây chỉ áp dụng cho primary match của feedback; native YOLO task assignment không đổi.

**K2-L4: đã sửa generator; nhánh pairwise dùng cặp online riêng.** Pair trong file feedback thuộc output của checkpoint nền trước augmentation, nên không thể dùng trực tiếp chỉ số prediction đó để train mô hình hiện tại. [feedback_loss.py](../../kb2_preference_optimization/feedback_loss.py) tạo cặp từ dense prediction sau augmentation: chosen có IoU với GT từ 0,5 trở lên, rejected có IoU từ 0,1 đến dưới 0,5, cùng GT sở hữu candidate và cùng điểm lớp GT. Loss `softplus(margin - (logit(score_chosen) - logit(score_rejected)))` cộng vào native loss. Box dùng để chọn cặp được detach; gradient đi qua hai điểm lớp. Đây là pairwise ranking, không phải DPO. Cặp tĩnh trong feedback vẫn chỉ dùng để phân tích.

Schema feedback chuyển từ `1.0` sang `2.1`. Loader từ chối record cũ và kiểm tra trạng thái từng GT. Summary ghi hash tệp feedback và quy tắc matching; trainer đối chiếu trước khi chạy. **Phải sinh lại feedback**; các kết quả từ logic cũ không chứng minh hiệu quả của logic mới.

## 4. Objective và đánh giá

| ID | Mức cần thiết | Trạng thái |
|---|---|---|
| K2-L5 | Cao nếu dùng dynamic object feedback | Runtime đã lấy trạng thái lỗi từ detection sau NMS, rồi dùng dense prediction để tính gradient. Cần kiểm tra nhánh này với checkpoint thực và đo chi phí forward bổ sung. |
| K2-L6 | Cần kiểm chứng | Auxiliary loss wrong-class tăng xác suất lớp đúng; native classification loss có thể đã phạt lớp sai. Kiểm tra gradient và ablation trước khi thêm penalty để tránh đếm đôi tín hiệu. |
| K2-L7 | Bắt buộc, đã sửa mặc định | Ngưỡng confidence khi tính AP giảm từ `0.25` xuống `0.001`. Phải đánh giá lại các checkpoint trên cùng protocol. Chỉ số ở ngưỡng vận hành là phép đo riêng. |
| K2-L8 | Vừa, chưa sửa | Resume chưa lưu đủ RNG, sampler và vị trí batch; chưa thể coi là tái lập chính xác run liên tục. |
| K2-L9 | Vừa, đã sửa | Diagnostic projection nay ghi đúng feedback loss; không dùng diagnostic cũ để kết luận. |
| K2-L10 | Có điều kiện, cần test | Ultralytics adapter giữ BatchNorm ở eval khi tính native loss. Cần kiểm tra contract mode/output trên phiên bản thư viện và checkpoint thực dùng. |

## 5. Provenance dữ liệu

[KB1 supervised manifest](../../kb1_reward_guided_training/run_provenance.py) lưu hash checkpoint, hash toàn bộ ảnh/nhãn và cấu hình dữ liệu tại lúc train; preflight của [run_all_kb2.sh](../../kb2_preference_optimization/run_all_kb2.sh) đối chiếu các hash này với dữ liệu hiện tại. Preflight hiện **PASS** cho bốn model hỗ trợ; các source ID theo quy tắc bỏ hậu tố augmentation đã được kiểm tra không trùng giữa train/valid/test. Feedback KB2 được sinh từ train sau khi qua bước này. Điều này xác minh nguồn của checkpoint KB1 supervised trong phạm vi manifest đang có. Quy tắc source ID không phát hiện mọi trường hợp cùng ảnh nhưng đổi tên khác hẳn; nguồn dữ liệu của pretrained weights bên ngoài cũng không được manifest này chứng minh.

## 6. Điều kiện trước khi kết luận hiệu quả

1. Khóa split, xác minh provenance checkpoint nền và sinh lại feedback schema `2.1` chỉ từ train.
2. So native continuation với từng biến thể feedback trong cùng điều kiện. Kiểm tra gt_indices sau augmentation; nếu dùng dynamic feedback, kiểm tra post-NMS status và gradient.
3. Chọn checkpoint trên validation bằng AP protocol đã khóa rồi đánh giá lại. Không dùng số từ feedback hoặc evaluator cũ để kết luận.
4. Đánh giá nhánh pairwise riêng với baseline cùng shuffle sampling, checkpoint, seed, số update và evaluator. Thử nhiều trọng số loss và seed; chỉ kết luận sau khi kiểm tra validation và tập test khóa.

## 7. Kết quả nhánh pairwise

Nhánh pairwise dùng cùng checkpoint KB1, 2.411 ảnh train, 479 ảnh validation, batch 4, ảnh 640 px, learning rate `1e-6`, 1.000 update tối đa, seed 42, chọn checkpoint theo mAP50–95 mỗi 250 bước. Baseline và pairwise cùng sampling shuffle; khác biệt chính là loss xếp hạng candidate. Hệ số pairwise `0.01` được chọn từ pilot 100 bước; margin `0.1`. Đây là đánh giá **validation**, chưa phải test độc lập.

| Model | Baseline mAP50–95 | Pairwise mAP50–95 | Chênh lệch (điểm %) |
|---|---:|---:|---:|
| [YOLOv8n](../../kb2_preference_optimization/checkpoint_preference_optimization/pairwise_full_a001_s42/ablation_results.json) | 0,385598 | 0,385295 | −0,0303 |
| [YOLOv8s](../../kb2_preference_optimization/checkpoint_preference_optimization/pairwise_full_a001_s42_yolov8s/ablation_results.json) | 0,395656 | 0,395025 | −0,0632 |
| [YOLO11n](../../kb2_preference_optimization/checkpoint_preference_optimization/pairwise_full_a001_s42_yolov11n/ablation_results.json) | 0,409761 | 0,410275 | +0,0514 |
| [YOLO11s](../../kb2_preference_optimization/checkpoint_preference_optimization/pairwise_full_a001_s42_yolov11s/ablation_results.json) | 0,407687 | 0,407114 | −0,0574 |

Để kiểm tra độ ổn định theo seed, YOLOv8n được chạy đủ 1.000 update thêm hai lần, với baseline được chạy lại cùng seed:

| Seed | Baseline | Pairwise | Chênh lệch (điểm %) |
|---|---:|---:|---:|
| 42 | 0,385598 | 0,385295 | −0,0303 |
| [43](../../kb2_preference_optimization/checkpoint_preference_optimization/pairwise_full_a001_s43/ablation_results.json) | 0,387702 | 0,388343 | +0,0641 |
| [44](../../kb2_preference_optimization/checkpoint_preference_optimization/pairwise_full_a001_s44/ablation_results.json) | 0,386110 | 0,386696 | +0,0586 |

Pilot 100 bước đã thử `0.01`, `0.03`, `0.10` trên seed 42; `0.01` cao hơn baseline nhẹ và được kiểm tra thêm trên seed 43/44. Nhưng ở 1.000 bước, seed 42 đổi chiều. Nhánh pairwise thật sự nhận gradient và tạo hàng nghìn cặp, song **chưa có bằng chứng nó cải thiện mAP ổn định**: ba trong bốn model giảm ở seed 42, còn YOLOv8n cho kết quả khác dấu giữa các seed. Chênh lệch đều dưới 0,07 điểm phần trăm. Vì hệ số được chọn trên validation và hiệu quả chưa nhất quán, không dùng tập test khóa để chọn lại hệ số hoặc tuyên bố cải thiện. Nếu tiếp tục nghiên cứu, cần thử nhiều seed trên các model còn lại và đánh giá test một lần sau khi khóa toàn bộ cấu hình.

## 8. Pilot candidate-selection DPO v1 lịch sử — 2026-10-07

Mục này lưu kết quả v1 chọn cặp từ reference. Không dùng số dưới đây để kết luận hiệu quả của v2 đang chạy; không resume checkpoint này vào v2.

Nhánh mới dùng policy categorical chuẩn hóa từ class logits trên toàn bộ dense candidate của cùng lớp GT. Reference là bản sao đóng băng của checkpoint KB1 khởi đầu. Trên cùng ảnh train sau augmentation, reference chọn A/B theo IoU; policy và reference tính log-prob trên đúng cùng chỉ số candidate. Objective là `native_loss + alpha * mean(-logsigmoid(beta * (gap_policy - gap_reference)))`, với `alpha=0.1`, `beta=0.1`. Đây là cấu hình thử đầu tiên, chưa tuning. Định nghĩa action, công thức và giới hạn ở [DPO.md](../../kb2_preference_optimization/docs/DPO.md).

Pilot YOLOv8n chạy mới cả baseline, pairwise và DPO, cùng seed 42, checkpoint nền, shuffle, augmentation, batch 4, ảnh 640, AdamW lr `1e-6`, weight decay `5e-4`, **100 update**, validation mỗi 50 bước. Evaluator giữ confidence `0.001`, NMS IoU `0.45`, max_det 100; chọn best mAP50–95 gồm step 0. Cả ba nhánh có mAP50–95 step 0 bằng `0.374931`; bảng dưới ghi checkpoint tốt nhất của mỗi nhánh, đều tại step 100. Không so trực tiếp số này với bảng 1.000 update ở mục 7.

| Nhánh | mAP50–95 | mAP50 | Chênh mAP50–95 so baseline (điểm %) |
|---|---:|---:|---:|
| Baseline | 0,377923 | 0,493222 | 0 |
| Pairwise | 0,377707 | 0,493545 | −0,0216 |
| DPO | 0,377847 | 0,493325 | −0,0075 |

Nguồn: [ablation_results.json](../../kb2_preference_optimization/checkpoint_preference_optimization/dpo_pilot_20261007_s42_100/ablation_results.json), các checkpoint và metrics JSONL trong cùng thư mục. DPO xử lý **1.620 cặp trên 100/100 bước**. Batch cuối trước update có relative gap trung bình `0.135785`, loss DPO `0.686416`; 80% cặp trong riêng batch đó cải thiện gap so reference, nhưng chỉ 33,33% có A>B. Cải thiện gap so reference chưa đồng nghĩa đảo được ranking tuyệt đối hoặc tăng AP.

Kiểm chứng: 45 test tự động qua (test GPU thật được skip trong lượt discover), một test thật YOLOv8n/GPU qua riêng, Bash syntax và DPO-only provenance preflight qua. Test thật trên một batch augmentation có 24 cặp, loss ban đầu `log(2)=0.693147`, gradient DPO sau nhân alpha bằng 5,136% native, loss sau một update kết hợp là `0.692675`; reference không thay đổi.

Pilot xác minh DPO chạy, có gradient và dùng reference cố định. **Chưa thấy DPO vượt baseline ở mAP50–95 trong pilot này**; chênh lệch rất nhỏ. Một model, một seed, 100 update và một alpha/beta chưa đủ kết luận hiệu quả dài hạn. Nhánh DPO cũng thay nguồn chọn cặp từ policy sang reference, nên so với pairwise không cô lập riêng tác dụng reference. DPO mới chỉ tối ưu chọn candidate qua điểm lớp; tọa độ hộp, missed GT không có A hợp lệ và false positive xa GT vẫn chủ yếu do native loss xử lý. Resume kiểm tra hash nền/reference và objective, nhưng giới hạn RNG/sampler ở K2-L8 vẫn còn. Tập test khóa chưa được dùng.

## 9. DPO v1 lịch sử, YOLOv8n đủ 1.000 update — 2026-10-07

Run này dùng nguồn cặp `frozen_reference_same_gt_v1`, trước sửa đổi ở mục 10. Kết quả và nhận xét trong mục này chỉ áp dụng cho v1.

Run do người dùng chạy riêng nhánh `dpo` kết thúc đủ 1.000 bước, seed 42,
`alpha=0.1`, `beta=0.1`. Nguồn: [ablation_results.json](../../kb2_preference_optimization/checkpoint_preference_optimization/ablation/20261007T075619Z_354/yolov8n/ablation_results.json)
và metrics JSONL cùng thư mục. DPO hoạt động trên 1.000/1.000 bước, xử lý
17.306 cặp; hash reference trùng hash checkpoint KB1 nền.

`pairwise=0 pairs=0` trong log là các trường của loss ranking ở nhánh pairwise
riêng đang tắt. DPO vẫn học cặp A/B và ghi ở `dpo`/`dpo_pairs`. Với run này,
`total = native + 0.1 * dpo`; `feedback` là giá trị native được trả lại trong
diagnostic của nhánh feedback-alpha=0, không phải một loss cộng thêm lần nữa.
Ví dụ bước 1: `2.44858 + 0.1 * 0.69315 = 2.51789` sau làm tròn.

| Bước validation | mAP50–95 | mAP50 | AP_small | Recall |
|---|---:|---:|---:|---:|
| 0 | 0,374931 | 0,488979 | 0,041219 | 0,549668 |
| 250 | 0,383136 | 0,497734 | 0,053154 | 0,557746 |
| 500 | 0,380733 | 0,495729 | 0,056325 | 0,552194 |
| 750 | 0,384278 | 0,501334 | 0,059478 | 0,555242 |
| 1.000 | 0,385223 | 0,500910 | 0,060874 | 0,556928 |

Checkpoint tốt nhất theo mAP50–95 là step 1.000, tăng 1,0292 điểm phần trăm
so step 0. Mức tăng này là hiệu quả của cả native continuation và DPO;
không quy toàn bộ cho DPO. Mốc step 500 giảm rồi phục hồi, chưa đủ để kết
luận overfitting. mAP50 riêng cao nhất ở step 750, nhưng objective chọn
checkpoint đã khóa là mAP50–95.

So với các checkpoint tốt nhất của **run lịch sử** trong mục 7, đã đối chiếu
config trong checkpoint: cùng model, checkpoint nền, seed, 1.000 update tối
đa, shuffle, batch, ảnh, lr, weight decay, gradient clip và evaluator.
Baseline/pairwise không được chạy lại trong run DPO mới này.

| Nhánh | Best step | mAP50–95 | mAP50 | AP_small | Recall |
|---|---:|---:|---:|---:|---:|
| Baseline | 750 | 0,385598 | 0,502531 | 0,059616 | 0,556643 |
| Pairwise | 750 | 0,385295 | 0,501549 | 0,059343 | 0,556404 |
| DPO | 1.000 | 0,385223 | 0,500910 | 0,060874 | 0,556928 |

DPO thấp hơn baseline 0,0375 điểm phần trăm mAP50–95 và thấp hơn pairwise
0,0072 điểm phần trăm. AP_small cao hơn baseline 0,1258 điểm phần trăm;
Recall cao hơn 0,0284 điểm phần trăm. Chênh lệch nhỏ, kết quả một model/một
seed mang tính hỗn hợp và chưa chứng minh DPO cải thiện detection tổng thể.

Trên các batch được ghi log, mean DPO loss của 11 mốc trong bước 1–100 là
0,688016, còn 10 mốc trong bước 901–1.000 là 0,654021. Đây là các batch khác
nhau, không phải phép đánh giá cố định; mức giảm phù hợp với cải thiện
relative ranking nhưng chưa xác định phần đóng góp riêng của DPO so native.
Batch cuối có 7 cặp: 5/7 cải thiện gap so reference, nhưng chỉ 3/7 có A>B,
bằng số cặp A>B của reference. Điều này minh họa rằng cải thiện relative gap
chưa luôn đảo được thứ tự điểm. Không ngoại suy các tỷ lệ batch cuối cho
toàn bộ train/validation. `grad` trong log là norm trước clipping (ngưỡng
1.0), nên các mốc norm lớn không đồng nghĩa cập nhật dùng norm lớn đó.

Kết quả hiện hỗ trợ giữ DPO như một ablation thử nghiệm. Muốn đánh giá lợi
ích ổn định cần nhiều seed và cấu hình alpha/beta đã khóa trên validation;
tập test vẫn chưa được dùng để chọn cấu hình hoặc kết luận.

## 10. Sửa ưu tiên theo GT của policy hiện tại — DPO online v2

V1 dùng hộp reference để gán A/B. Khi hộp policy thay đổi, nhãn đó không bảo đảm A vẫn có IoU cao hơn B trong prediction hiện tại. V2 đáp ứng yêu cầu **hộp đang dự đoán khớp GT hơn được ưu tiên**:

1. Sau augmentation, detach hộp policy chỉ để tính IoU và chọn cặp cho từng GT. Candidate thuộc GT có IoU cao nhất; không lấy candidate thuộc GT khác làm rejected.
2. A là candidate thuộc GT đó có IoU cao nhất, với IoU >= 0,5. B có IoU >= 0,1 và thấp hơn A với sai số `1e-6`; chọn B có điểm lớp GT cao nhất trong tập hợp này. A=0,9/B=0,6 được phép, không bắt buộc B dưới 0,5. IoU hòa nhau hoặc thiếu candidate hợp lệ thì bỏ cặp.
3. Policy và reference tính log-prob trên **cùng chỉ số A/B vừa chọn**. Reference đóng băng, không quyết định nhãn ưu tiên. Công thức loss vẫn là `native + alpha * mean(-logsigmoid(beta * (gap_policy - gap_reference)))`.
4. Bước train tiếp theo tạo lại cặp từ hộp policy mới. Checkpoint và ablation JSON ghi `dpo_pair_source=online_policy_gt_iou_v2`, `dpo_objective_version=candidate_selection_online_dpo_v2`; resume từ chối trộn với v1.

Đây là cách áp dụng công thức [DPO gốc, Eq. 7](https://arxiv.org/html/2305.18290v3) vào policy categorical chọn candidate, với preference được tạo **online** bằng GT. Bài gốc dùng preference dataset offline; không mặc định chuyển mọi bảo đảm lý thuyết sang quy tắc online này. Gradient phụ đi qua class scores, native loss vẫn học tọa độ và classification. Chưa trực tiếp tối ưu số lượng hộp hoặc toàn bộ output sau NMS. IoU đúng hơn không đồng nghĩa chắc chắn mAP tăng.

**Kiểm chứng sau sửa:** 53 test trong lượt discover: 52 qua, một test GPU skip và chạy riêng thành công. Test kiểm tra đảo A/B khi chất lượng hộp policy đảo, geometry/confidence reference không gán nhãn, gradient tăng điểm chosen/giảm rejected, không dùng positive của GT khác, cặp cùng trên IoU 0,5, và chặn resume v1. Test thật YOLOv8n có 24 cặp, loss khởi đầu `0,693147`, weighted DPO gradient bằng `3,5336%` native, loss sau một update kết hợp `0,692895`, IoU gap trung bình `0,049637`; reference không nhận gradient và không đổi tham số/buffer.

**Pilot mới:** chạy lại cả baseline, pairwise và DPO v2 từ cùng checkpoint KB1, seed 42, 100 update, validation mỗi 50 bước, batch 4, ảnh 640, lr `1e-6`; protocol AP như trên. Các checkpoint tốt nhất đều ở bước 100.

| Nhánh | mAP50–95 | mAP50 | Chênh mAP50–95 so baseline (điểm %) |
|---|---:|---:|---:|
| Baseline | 0,377736 | 0,493192 | 0 |
| Pairwise | 0,377869 | 0,493585 | +0,0133 |
| DPO online v2 | 0,377642 | 0,493445 | −0,0094 |

Nguồn: [ablation_results.json](../../kb2_preference_optimization/checkpoint_preference_optimization/dpo_online_v2_20261007_s42_100/ablation_results.json) và `run.log`/metrics JSONL cùng thư mục. DPO v2 xử lý 1.620 cặp trên 100/100 bước. Batch cuối có chosen IoU trung bình `0,939944`, rejected `0,913634`: cả hai vượt 0,5 nhưng A tốt hơn B theo GT. 10/15 cặp cải thiện relative gap so reference; 0/15 có điểm A>B tuyệt đối. Các con số batch này chỉ là diagnostic train, không thay AP validation.

Pilot xác minh nhãn ưu tiên v2 và gradient hoạt động; **chưa chứng minh lợi ích mAP hoặc hội tụ**. Không so trực tiếp chênh v2 với v1 để quy toàn bộ cho sửa thuật toán: đây là lượt chạy mới trong môi trường hiện tại. Pairwise giữ quy tắc B dưới 0,5, nên so với DPO v2 còn thay đổi miền chọn B, chưa cô lập tác dụng reference. Cần chạy mới ngân sách dài hơn và nhiều seed để đánh giá hiệu quả; tập test khóa chưa được dùng. Giới hạn resume ở K2-L8 vẫn còn.

### Ngân sách cho lượt train dài tiếp theo

Sau pilot, mặc định của runner và trainer được chốt theo yêu cầu người dùng ở tối đa **15.000 update**, patience **5**. Với 2.411 ảnh train, batch 4, loader shuffle có 602 batch/lượt (bỏ batch cuối không đủ), tương đương khoảng 24,9 lượt nếu chạy đủ. Validation mỗi 1.000 update; early stopping sau 5 lần liên tiếp không cải thiện best mAP50–95 quá `1e-4`, tức khoảng 5.000 update. Checkpoint trung gian mỗi 2.000 update. Learning rate và trọng số loss giữ nguyên.

Đây là cấu hình cho run mới, **chưa có kết quả 15.000 bước**. Số liệu các mục trên giữ ngân sách lịch sử. Tăng ngân sách cho phép theo dõi plateau lâu hơn, không bảo đảm hội tụ hoặc cải thiện mAP. Khi so các nhánh, giữ cùng cấu hình và ghi nhận số bước thực chạy vì early stopping có thể khác nhau.
