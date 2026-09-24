# KB2 — Review logic feedback/preference optimization cho YOLO

## 1. Hướng nghiên cứu

KB2 nghiên cứu cách dùng phản hồi sinh từ **prediction và ground truth của tập train** để điều chỉnh việc fine-tune YOLO. Ý tưởng preference learning có thể được chuyển sang detection bằng cách xác định prediction nào tốt hơn cho cùng một GT, hoặc bằng cách biến lỗi detection thành tín hiệu chọn mẫu và loss phụ. Người dùng không yêu cầu sao chép DPO cho mô hình ngôn ngữ; native YOLO detection loss có thể tiếp tục là objective chính.

Để phương pháp có thể kiểm chứng, cần tách rõ ba khái niệm:

- **Feedback-guided sampling/loss:** dùng nhãn lỗi và độ khó để thay đổi sampling hoặc trọng số loss.
- **Pairwise preference optimization:** thật sự dùng cặp chosen/rejected trong objective.
- **DPO chuẩn:** còn có reference model và công thức log-ratio tương ứng với policy xác suất.

[Trainer chính](../../kb2_preference_optimization/train_feedback.py#L63) hiện thuộc nhóm đầu. [Feedback generator](../../kb2_preference_optimization/feedback.py#L74) có tạo preference pairs nhưng [dataset](../../kb2_preference_optimization/feedback_dataset.py#L125) chỉ gắn chúng vào target; trainer chính chưa dùng các cặp này. Vì vậy tên đúng của implementation đang review là **feedback-guided YOLO fine-tuning lấy cảm hứng từ preference learning**. Có thể bổ sung pairwise objective sau khi sửa tính hợp lệ của cặp, nhưng không cần để hướng nghiên cứu hiện tại có giá trị.

## 2. Luồng logic hiện có

| Bước | Code | Nhận xét |
|---|---|---|
| Sinh feedback | [generate_feedback.py](../../kb2_preference_optimization/generate_feedback.py), [feedback.py](../../kb2_preference_optimization/feedback.py#L31) | Prediction được ghép với GT và phân loại thành các lỗi. Đây là trung tâm của phương pháp; lỗi matching sẽ lan sang mọi ablation. |
| Gắn feedback vào train | [feedback_dataset.py](../../kb2_preference_optimization/feedback_dataset.py#L99) | Record theo image ID; gắn difficulty, preference pairs và mã object. Mapping GT qua augmentation là ý tưởng đúng, nhưng mã object phải độc quyền và ổn định. |
| Fine-tune | [feedback_loss.py](../../kb2_preference_optimization/feedback_loss.py), [train_feedback.py](../../kb2_preference_optimization/train_feedback.py#L63) | Native loss vẫn là nền. Feedback thay sampling, loss theo ảnh/object hoặc gradient projection. Các nhánh cần cùng checkpoint, dữ liệu, augmentation, update budget và evaluator để diễn giải nhân quả. |
| Chọn checkpoint | [feedback_validation.py](../../kb2_preference_optimization/feedback_validation.py#L40), [train_feedback.py](../../kb2_preference_optimization/train_feedback.py#L113) | Có validation selection và step khởi đầu. Protocol AP hiện cần sửa trước khi dùng để kết luận. |

## 3. Lỗi logic trong feedback

### K2-L1 — Trạng thái GT không độc quyền

[feedback.py](../../kb2_preference_optimization/feedback.py#L42) xử lý từng prediction độc lập. Một GT có detection đúng có thể đồng thời nhận prediction sai class hoặc định vị kém. [object_feedback_codes](../../kb2_preference_optimization/feedback_dataset.py#L74) ghi đè mã theo các danh sách lỗi, nên GT đã được phát hiện đúng có thể bị xem là GT lỗi và đưa vào auxiliary loss.

**Sửa:** tạo một trạng thái chính cho mỗi GT theo thứ tự ưu tiên rõ ràng: matched; nếu chưa matched thì wrong-class; nếu chưa có thì bad-localization; cuối cùng missed. False-positive và duplicate là trạng thái **cấp prediction**, không ghi đè trạng thái GT. Test một GT có cả detection đúng và detection sai vẫn được gắn matched.

### K2-L2 — Một lỗi bị đếm nhiều lần

[feedback.py](../../kb2_preference_optimization/feedback.py#L66) đánh dấu missed cho mọi GT không có primary match đúng class. GT wrong-class hoặc bad-localization có thể vẫn vào missed. [feedback_difficulty](../../kb2_preference_optimization/feedback_dataset.py#L64) cộng các danh sách lỗi, làm cùng một GT đóng góp nhiều lần vào sampling weight.

**Sửa:** dùng trạng thái GT độc quyền cho difficulty; nếu muốn phạt prediction thừa, cộng false-positive/duplicate riêng. Test tổng số trạng thái GT bằng số GT và difficulty không đổi khi thêm alias của cùng một lỗi.

### K2-L3 — Matching chưa one-to-one và class-aware

[feedback.py](../../kb2_preference_optimization/feedback.py#L47) chọn GT có IoU cao nhất cho từng prediction mà chưa ưu tiên match đúng lớp, chưa ngăn nhiều prediction cùng tranh một GT. Với object gần nhau, prediction đúng của GT này có thể được gán sang GT khác, gây wrong-class/missed giả.

**Sửa:** ghép primary đúng lớp và đủ IoU theo one-to-one với tie-break xác định; sau đó phân loại các GT chưa matched và prediction chưa dùng. Test nhiều GT gần nhau, duplicate, wrong-class và thứ tự confidence hòa nhau.

### K2-L4 — Cặp preference có thể vô hiệu

[feedback.py](../../kb2_preference_optimization/feedback.py#L76) lấy rejected là prediction có confidence cao nhất khác chosen, bất kể prediction đó có thể là detection đúng của GT khác. Pair như vậy sẽ mâu thuẫn với GT nếu được dùng để train.

**Sửa:** rejected phải cạnh tranh với chosen cho **cùng GT**, không là primary match của GT khác và phải có quality thấp hơn theo quy tắc công bố trước. Không tạo pair nếu không có candidate hợp lệ. Hiện lỗi này chưa tác động trainer chính vì pair chưa được dùng, nhưng cần sửa trước khi thêm pairwise loss.

## 4. Lỗi logic trong objective và đánh giá

| ID | Ưu tiên | Phát hiện | Sửa/kiểm chứng |
|---|---|---|---|
| K2-L5 | Cao | [Dynamic feedback](../../kb2_preference_optimization/feedback_loss.py#L10) chọn dense candidate có IoU cao nhất; candidate đó có thể không qua confidence/NMS, nên trạng thái dynamic không phản ánh output detector cuối. | Quyết định trạng thái lỗi từ output post-NMS, rồi ánh xạ sang candidate có gradient hoặc tích hợp trọng số vào native task assignment. Test detection đúng sau NMS không bị gọi missed. |
| K2-L6 | Cao | [Object loss](../../kb2_preference_optimization/feedback_loss.py#L45) cho wrong-class chủ yếu tăng xác suất lớp đúng, chưa phạt trực tiếp lớp sai. | Dùng target classification vector hoặc penalty có kiểm soát cho lớp sai; test dấu gradient của cả lớp đúng và sai. |
| K2-L7 | Cao | [Evaluator](../../kb2_preference_optimization/feedback_validation.py#L40) lọc prediction ở confidence tương đối cao trước khi tính AP, nên đường precision–recall mất phần score thấp và ranking checkpoint có thể đổi. | Dùng ngưỡng rất thấp cho AP và ngưỡng vận hành riêng cho precision/recall/F1; đánh giá lại mọi checkpoint trên cùng protocol. |
| K2-L8 | Vừa | [Resume](../../kb2_preference_optimization/train_feedback.py#L83) chỉ phục hồi model, optimizer và early stopping; không phục hồi RNG, sampler và vị trí batch. | Lưu đầy đủ state và config/data/feedback hash; test run liên tục so với save+resume cho cùng số update. |
| K2-L9 | Vừa | [Gradient diagnostics](../../kb2_preference_optimization/feedback_loss.py#L141) ghi nhầm native loss vào trường feedback loss trong nhánh projection. | Ghi log loss và gradient đúng từng thành phần; test với hai loss khác nhau. |
| K2-L10 | Vừa | [Ultralytics adapter](../../kb2_preference_optimization/adapters/ultralytics_adapter.py) đặt train_mode thành eval và comment raw output không khớp với decoded inference output. | Định nghĩa contract mode/output rõ; test gọi forward trước và sau native loss/validation; xác minh BatchNorm được xử lý theo chủ ý. |

## 5. Provenance dữ liệu sau khi làm lại

[KB2 checkpoint mặc định](../../kb2_preference_optimization/train_rl.py#L625) lấy từ KB1, còn [data root KB2](../../kb2_preference_optimization/train_feedback.py#L205) được khai báo riêng. Hai đường dữ liệu này phải được đối chiếu theo **source ID mà checkpoint nền thực sự đã học**, không chỉ theo tên file hiện có. Nếu checkpoint đã học ảnh thuộc validation/test của KB2, phép đánh giá không độc lập; nếu không, điều kiện này đạt. Báo cáo này **không kết luận có rò rỉ trên dữ liệu mới** vì chưa có manifest lịch sử đầy đủ của checkpoint và chưa chạy lại.

Cách đóng vấn đề: dùng checkpoint nền được train chỉ trên train của split KB2, hoặc dùng split chung được khóa cho KB1/KB2; lưu manifest source ID và hash trong checkpoint. Sinh lại feedback từ checkpoint nền tương ứng với split đã khóa.

## 6. Điều kiện logic trước khi chạy lại

- Tạo feedback chỉ từ train; bảo đảm mỗi GT có đúng một trạng thái chính và primary matching one-to-one.
- Record feedback mới phải có schema version, checkpoint hash, split/data manifest và quy tắc matching; không trộn record của logic cũ với logic mới.
- Với feedback tĩnh, kiểm tra GT index còn đúng sau augmentation. Với feedback động, trạng thái lỗi phải phù hợp detection cuối và loss phụ phải có gradient hợp lệ.
- So native continuation với từng biến thể feedback trong cùng điều kiện; nếu dùng preference pair, thêm đối chứng pairwise riêng.
- Chọn checkpoint bằng một validation evaluator đã khóa; sau khi chạy lại mới đánh giá hiệu quả. Không dùng các số từ trước khi làm lại data để kết luận.

