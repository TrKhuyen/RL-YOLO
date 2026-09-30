# KB2 — Review feedback-guided YOLO fine-tuning

## 1. Phạm vi và tên phương pháp

KB2 dùng prediction của checkpoint nền trên **tập train** cùng ground truth để tạo tín hiệu chọn mẫu và loss phụ cho YOLO. Native detection loss vẫn là objective chính. Trainer hiện không tối ưu cặp chosen/rejected và không có reference model cùng DPO log-ratio. Vì vậy tên đúng là **feedback-guided YOLO fine-tuning lấy cảm hứng từ preference learning**. Pairwise optimization cần là một thí nghiệm riêng.

## 2. Luồng thực thi

| Bước | Code | Vai trò |
|---|---|---|
| Sinh feedback | [generate_feedback.py](../../kb2_preference_optimization/generate_feedback.py), [feedback.py](../../kb2_preference_optimization/feedback.py) | Dự đoán trên train, ghép GT và ghi lỗi. |
| Gắn feedback | [feedback_dataset.py](../../kb2_preference_optimization/feedback_dataset.py) | Tính difficulty; nối trạng thái GT với box sau augmentation qua gt_indices. |
| Fine-tune | [feedback_loss.py](../../kb2_preference_optimization/feedback_loss.py), [train_feedback.py](../../kb2_preference_optimization/train_feedback.py) | Native loss, sampling và loss phụ theo feedback. |
| Chọn checkpoint | [feedback_validation.py](../../kb2_preference_optimization/feedback_validation.py) | Đánh giá AP trên validation theo protocol cố định. |

Các nhánh phải dùng cùng checkpoint nền, split, augmentation, số update và evaluator để có thể diễn giải chênh lệch.

## 3. Matching và trạng thái feedback

**K2-L1–L3: bắt buộc, đã sửa trong code.** Trước đây từng prediction được xét độc lập, nên một GT có thể vừa matched vừa bị tính wrong-class, bad-localization hoặc missed. Difficulty có thể đếm trùng và object code có thể bị ghi đè. Với các GT gần nhau, prediction còn có thể bị gán sang GT sai.

[feedback.py](../../kb2_preference_optimization/feedback.py) hiện tạo **một trạng thái chính cho mỗi GT**: ưu tiên ghép đúng lớp, đủ IoU và one-to-one; với GT chưa matched, chọn wrong-class hoặc bad-localization; cuối cùng mới missed. Duplicate và false-positive thuộc cấp prediction. [feedback_dataset.py](../../kb2_preference_optimization/feedback_dataset.py) tính difficulty và object code từ trạng thái GT độc quyền. One-to-one ở đây chỉ áp dụng cho primary match của feedback; native YOLO task assignment không đổi.

**K2-L4: bắt buộc trước khi dùng pairwise loss, đã sửa generator.** Rejected phải cạnh tranh cho cùng GT, có chất lượng thấp hơn và không là prediction được chọn cho GT khác. Trainer hiện vẫn chưa dùng pair; phải có ablation riêng nếu bổ sung pairwise objective.

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

## 5. Provenance dữ liệu: chưa đóng

[Checkpoint nền mặc định](../../kb2_preference_optimization/train_rl.py) có nguồn từ KB1, còn [data root KB2](../../kb2_preference_optimization/train_feedback.py) được khai báo riêng. Hash checkpoint và feedback chứng minh đúng tệp đang dùng, nhưng **không chứng minh** checkpoint nền chưa học ảnh validation/test của KB2. Cần đối chiếu source ID theo manifest lịch sử hoặc train lại checkpoint với split chung đã khóa. Chưa có đủ bằng chứng để khẳng định có hay không có rò rỉ.

## 6. Điều kiện trước khi kết luận hiệu quả

1. Khóa split, xác minh provenance checkpoint nền và sinh lại feedback schema `2.1` chỉ từ train.
2. So native continuation với từng biến thể feedback trong cùng điều kiện. Kiểm tra gt_indices sau augmentation; nếu dùng dynamic feedback, kiểm tra post-NMS status và gradient.
3. Chọn checkpoint trên validation bằng AP protocol đã khóa rồi đánh giá lại. Không dùng số từ feedback hoặc evaluator cũ để kết luận.
4. Chỉ đánh giá pairwise objective sau khi có ablation riêng. Kiểm tra L6, resume và adapter theo nhánh thực nghiệm thực sự sử dụng.
