# KB2 — Review feedback-guided và pairwise YOLO fine-tuning

## 1. Phạm vi và tên phương pháp

KB2 dùng prediction của checkpoint nền trên **tập train** cùng ground truth để tạo tín hiệu chọn mẫu và loss phụ cho YOLO. Native detection loss vẫn là objective chính. Bản hiện tại có thêm một nhánh **pairwise preference optimization**: trên mỗi ảnh train đã augmentation, nhánh này chọn hai candidate của mô hình hiện tại cho cùng GT và tối ưu điểm chosen cao hơn rejected. Các nhánh baseline, sampling và hybrid vẫn là feedback-guided fine-tuning. KB2 chưa dùng DPO vì không có reference model hay policy log-ratio.

## 2. Luồng thực thi

| Bước | Code | Vai trò |
|---|---|---|
| Sinh feedback | [generate_feedback.py](../../kb2_preference_optimization/generate_feedback.py), [feedback.py](../../kb2_preference_optimization/feedback.py) | Dự đoán trên train, ghép GT và ghi lỗi. |
| Gắn feedback | [feedback_dataset.py](../../kb2_preference_optimization/feedback_dataset.py) | Tính difficulty; nối trạng thái GT với box sau augmentation qua gt_indices. |
| Fine-tune | [feedback_loss.py](../../kb2_preference_optimization/feedback_loss.py), [train_feedback.py](../../kb2_preference_optimization/train_feedback.py) | Native loss, sampling, loss phụ theo feedback và nhánh pairwise ranking riêng. |
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
