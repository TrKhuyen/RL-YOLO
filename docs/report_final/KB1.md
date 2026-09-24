# KB1 — Review logic reward-guided YOLO fine-tuning

## 1. Hướng nghiên cứu cần giữ

Với từng kiến trúc YOLO, giai đoạn supervised tạo một checkpoint tốt nhất của **chính kiến trúc đó** theo validation. Giai đoạn reward-guided phải bắt đầu từ đúng checkpoint này và cập nhật trọng số detector bằng một objective có hướng dẫn từ reward. So sánh được ghép cặp trong từng model: cùng checkpoint khởi đầu, dữ liệu, thứ tự batch hoặc seed, ngân sách và evaluator.

Hai câu hỏi cuối của KB1 được định nghĩa trước khi chạy lại:

1. Với mỗi model, chênh lệch chất lượng giữa checkpoint reward-guided đã chọn và checkpoint supervised ban đầu là bao nhiêu? Báo cáo cả trường hợp chênh lệch âm hoặc checkpoint ban đầu vẫn là best.
2. Sau hai giai đoạn, model nào có checkpoint tốt nhất theo **một metric chính đã khóa trước**? Nêu rõ tập ứng viên là chỉ các checkpoint reward-guided hay toàn bộ checkpoint supervised và reward-guided.

Để biết reward có thêm giá trị so với train tiếp thông thường, cần nhánh **native-only continuation** bắt đầu từ cùng supervised checkpoint. Nếu muốn quy lợi ích cho *riêng reward*, còn cần tách proxy loss khỏi reward term bằng ablation. Không thể suy từ “reward-guided thắng supervised” rằng reward là nguyên nhân duy nhất.

## 2. Luồng code hiện tại

| Bước | Bằng chứng | Đánh giá logic |
|---|---|---|
| Supervised theo model | [train_supervised.py](../../kb1_reward_guided_training/train_supervised.py) | Mapping các kiến trúc và quy trình train riêng đã có. Cần ghi nguồn pretrained, data manifest và cấu hình cho từng checkpoint để bảo đảm stage sau dùng đúng nguồn. |
| Nạp checkpoint tương ứng | [train_rl.py](../../kb1_reward_guided_training/train_rl.py#L679), [run_verified.py](../../kb1_reward_guided_training/run_verified.py#L139) | Mapping một model tới một best checkpoint đúng ý tưởng. Runner verified lưu hash checkpoint nguồn nhưng chưa ràng buộc hash/split nguồn dữ liệu đã train checkpoint đó. |
| Fine-tune trọng số | [run_verified.py](../../kb1_reward_guided_training/run_verified.py#L211), [adapter](../../kb1_reward_guided_training/adapters/ultralytics_adapter.py) | Có native detection loss, reward term, TP/FP/FN proxy và regularization về trọng số ban đầu. Runner verified chỉ cập nhật Detect head; đây là fine-tune trọng số một phần model, cần mô tả đúng. |
| Reward và gradient | [reward.py](../../kb1_reward_guided_training/reward.py), [match_aware_objective](../../kb1_reward_guided_training/train_rl.py#L121) | Reward tính từ prediction so GT; gradient chủ yếu qua confidence proxy và native loss. Box/NMS/matching không tạo một stochastic policy có log-likelihood action tường minh. |
| Chọn checkpoint | [run_verified.py](../../kb1_reward_guided_training/run_verified.py#L255), [canonical_eval.py](../../kb1_reward_guided_training/canonical_eval.py#L69) | Có validation selection, step khởi đầu là candidate và kiểm tra reload. Cần dùng cùng metric/evaluator cho mọi nhánh và không dùng training reward để chọn kết quả cuối. |
| Đối chứng | [run_verified.py](../../kb1_reward_guided_training/run_verified.py#L211) | Native-only là đối chứng cần thiết, nhưng hiệu ứng reward-guided trừ native-only hiện là hiệu ứng **reward cộng proxy**, không cô lập reward. |

Cách gọi phương pháp phù hợp với implementation hiện tại là **reward-guided fine-tuning / RL-inspired training**. Gọi nó là REINFORCE chuẩn sẽ hàm ý prediction được lấy mẫu từ một policy có log-prob đúng; code hiện không làm việc đó.

## 3. Các lỗi logic cần sửa

| ID | Ưu tiên | Phát hiện | Cách sửa và tiêu chí kiểm tra |
|---|---|---|---|
| K1-L1 | Cao | Checkpoint stage supervised chưa gắn manifest dữ liệu/nhãn và source ID. Sau khi làm lại data, đường dẫn best checkpoint cũ vẫn có thể được nạp dù nguồn train khác split mới. | Tạo manifest bất biến cho split, hash ảnh/nhãn/config/pretrained và gắn vào checkpoint; fine-tune từ checkpoint chỉ khi manifest khớp protocol hiện hành. Nếu không chứng minh được nguồn, train lại stage supervised. |
| K1-L2 | Cao | [train_rl.py](../../kb1_reward_guided_training/train_rl.py#L564) mô tả objective là REINFORCE, nhưng log-confidence sau NMS/matching chỉ là surrogate. | Đổi mô tả nghiên cứu thành reward-guided surrogate objective; nếu muốn tuyên bố policy gradient, phải định nghĩa action sampling và log-prob hợp lệ, rồi kiểm chứng gradient estimator. |
| K1-L3 | Cao | Nhánh guided có cả reward term và supervised TP/FP/FN proxy, còn native-only thiếu cả hai. Chênh lệch giữa hai nhánh không đo riêng reward. | Chạy ablation native, native+proxy, native+reward, native+proxy+reward với cùng checkpoint/seed/budget; báo cáo thêm gradient norm từng term. |
| K1-L4 | Cao | Runner verified [đặt cứng seed](../../kb1_reward_guided_training/run_verified.py#L123) và [report screening](../../kb1_reward_guided_training/run_screening.py#L39) chỉ tổng hợp cùng seed đó. Logic đa seed chưa được xuyên suốt. | Truyền seed qua RNG, sampler, augmentation, manifest, output; kiểm tra hai seed tạo run tách biệt và một seed có thể tái lập. |
| K1-L5 | Cao | Metric chọn best theo composite score; câu hỏi cuối thường muốn xếp hạng bằng một metric chính. Hai ranking có thể khác nhau. | Khóa metric chính trước khi train; dùng cùng metric để chọn best và trả lời câu hỏi cuối, hoặc báo cáo rõ hai ranking mà không chọn lại theo test. |
| K1-L6 | Vừa | Early stopping làm số update thực tế giữa native-only và guided khác nhau. Nếu so không ghi cost, tác dụng objective có thể lẫn tác dụng budget. | Ghi update/epoch-equivalent/time thực; có đối chứng budget cố định hoặc phân tích quality theo compute. |
| K1-L7 | Vừa | [run_verified.py](../../kb1_reward_guided_training/run_verified.py#L129) vô hiệu hóa các patch W3F/PSA của DP-YOLO trong screening. Tên DP-YOLO tự nó không chứng minh hai thành phần này đã tham gia ở cả hai stage. | Lưu flags và kiểm tra patch thực trong manifest stage supervised/guided; đặt tên biến thể phù hợp với cấu hình thực. |
| K1-L8 | Vừa | [train_supervised.py](../../kb1_reward_guided_training/train_supervised.py#L62) cho phép dùng lại tên run. Điều này có thể làm khó truy vết checkpoint gốc của stage sau. | Mỗi run có ID/thư mục bất biến; không ghi đè checkpoint đã dùng làm mốc. |
| K1-L9 | Vừa | Cùng hệ số loss không bảo đảm cùng thang gradient giữa các họ YOLO. | Kiểm tra từng thành phần loss/gradient là hữu hạn, có tác động và không lấn hoàn toàn native loss; điều chỉnh bằng validation rồi khóa trước test. |

## 4. Điều kiện logic trước khi chạy lại

- Cố định một split đã làm lại và chứng minh checkpoint supervised chưa học validation/test của **split đó**.
- Mỗi model/seed có cùng supervised checkpoint khởi đầu cho guided và native-only; lưu lineage đến output checkpoint.
- Reward, proxy và native loss phải được log riêng, gradient đi tới đúng trainable weights; frozen weights và BatchNorm được kiểm tra theo thiết kế.
- Chọn checkpoint chỉ bằng validation; reward-best dùng để chẩn đoán, không tự động là best detection model.
- Cùng preprocessing, NMS, định nghĩa AP/recall, metric chọn best và test protocol cho tất cả model.
- Chỉ sau khi chạy lại đầy đủ mới điền bảng cải thiện từng model và tên model tốt nhất. Báo cáo này không đưa ra các con số đó.

