# KB1 — Review logic reward-guided YOLO fine-tuning

## 0. Trạng thái trước khi chạy

- Dataset: `pre-data/data/v2i_cleanned`. Audit cho thấy train 2.411 ảnh/nhãn, valid 479, test 239; theo hậu tố augment đã biết, train có 667 ảnh augment, valid/test có 0. Không thấy source ID trùng giữa các split. Kiểm tra theo source ID không thay thế kiểm tra ảnh gần trùng bằng nội dung.
- Hai thư mục checkpoint KB1 chưa có file `.pt`; cần train supervised lại. Mỗi checkpoint mới có manifest với fingerprint ảnh/nhãn/config, SHA256 checkpoint và cờ DP. Stage hai từ chối nguồn thiếu hoặc lệch manifest.
- Metric chính là **mAP50_95**, khóa trước khi chạy. Tập ứng viên xếp hạng gồm best reward-guided của sáu model. Validation chọn checkpoint; test chỉ báo cáo sau khi khóa lựa chọn.
- Đường chạy chính: `train_supervised.py` → `run_screening.py`/`run_verified.py` → `evaluate.py --checkpoint-source screening --split test`. `train_rl.py` là runner cũ.


## 1. Hướng nghiên cứu cần giữ

Với từng kiến trúc YOLO, giai đoạn supervised tạo một checkpoint tốt nhất của **chính kiến trúc đó** theo validation. Giai đoạn reward-guided phải bắt đầu từ đúng checkpoint này và cập nhật trọng số detector bằng một objective có hướng dẫn từ reward. So sánh được ghép cặp trong từng model: cùng checkpoint khởi đầu, dữ liệu, thứ tự batch hoặc seed, ngân sách và evaluator.

Hai câu hỏi cuối của KB1 được định nghĩa trước khi chạy lại:

1. Với mỗi model, chênh lệch chất lượng giữa checkpoint reward-guided đã chọn và checkpoint supervised ban đầu là bao nhiêu? Báo cáo cả trường hợp chênh lệch âm hoặc checkpoint ban đầu vẫn là best.
2. Sau hai giai đoạn, model nào có checkpoint tốt nhất theo **một metric chính đã khóa trước**? Tập ứng viên đã khóa là best reward-guided của từng model.

Để biết reward có thêm giá trị so với train tiếp thông thường, cần nhánh **native-only continuation** bắt đầu từ cùng supervised checkpoint. Nếu muốn quy lợi ích cho *riêng reward*, còn cần tách proxy loss khỏi reward term bằng ablation. Không thể suy từ “reward-guided thắng supervised” rằng reward là nguyên nhân duy nhất.

## 2. Luồng code hiện tại

| Bước | Bằng chứng | Đánh giá logic |
|---|---|---|
| Supervised theo model | [train_supervised.py](../../kb1_reward_guided_training/train_supervised.py), [run_provenance.py](../../kb1_reward_guided_training/run_provenance.py) | Mapping sáu kiến trúc đã có. Sau train, mỗi best checkpoint được gắn manifest dữ liệu sạch, nguồn pretrained nếu file còn, cờ DP và hash checkpoint. |
| Nạp checkpoint tương ứng | [run_verified.py](../../kb1_reward_guided_training/run_verified.py), [run_provenance.py](../../kb1_reward_guided_training/run_provenance.py) | Stage hai xác nhận model, đường dẫn, SHA256 checkpoint và fingerprint dữ liệu trùng manifest supervised trước khi bắt đầu. |
| Fine-tune trọng số | [run_verified.py](../../kb1_reward_guided_training/run_verified.py), [adapter](../../kb1_reward_guided_training/adapters/ultralytics_adapter.py) | Có native detection loss, reward term, TP/FP/FN proxy và regularization về trọng số ban đầu. Runner verified chỉ cập nhật Detect head; đây là fine-tune trọng số một phần model, cần mô tả đúng. |
| Reward và gradient | [reward.py](../../kb1_reward_guided_training/reward.py), [match_aware_objective](../../kb1_reward_guided_training/train_rl.py) | Reward tính từ prediction so GT; gradient chủ yếu qua confidence proxy và native loss. Box/NMS/matching không tạo một stochastic policy có log-likelihood action tường minh. |
| Chọn checkpoint | [run_verified.py](../../kb1_reward_guided_training/run_verified.py), [canonical_eval.py](../../kb1_reward_guided_training/canonical_eval.py) | Step 0 là candidate; validation mAP50_95 chọn best và kiểm tra reload. Test không dùng chọn lại checkpoint. |
| Đối chứng | [run_verified.py](../../kb1_reward_guided_training/run_verified.py) | Native-only là đối chứng cần thiết, nhưng hiệu ứng reward-guided trừ native-only hiện là hiệu ứng **reward cộng proxy**, không cô lập reward. |

Cách gọi phương pháp phù hợp với implementation hiện tại là **reward-guided fine-tuning / RL-inspired training**. Gọi nó là REINFORCE chuẩn sẽ hàm ý prediction được lấy mẫu từ một policy có log-prob đúng; code hiện không làm việc đó.

## 3. Các lỗi logic cần sửa

| ID | Ưu tiên | Phát hiện | Cách sửa và tiêu chí kiểm tra |
|---|---|---|---|
| K1-L1 | Đã sửa trong luồng chính | Checkpoint mới phải thuộc đúng dữ liệu sạch. | Supervised ghi fingerprint ảnh/nhãn/config và SHA256 best checkpoint; verified runner kiểm tra trước khi train. Cần train lại để sinh manifest. |
| K1-L2 | Cần giữ đúng tên gọi | Objective là surrogate qua confidence sau NMS/matching. | Gọi phương pháp chính là reward-guided fine-tuning hoặc RL-inspired objective; không tuyên bố REINFORCE chuẩn. |
| K1-L3 | Bổ sung khi cần quy riêng reward | Guided gồm reward và proxy; native-only thiếu cả hai. | Hai câu hỏi gốc cần supervised và native-only làm mốc. Muốn quy riêng tác dụng reward cần thêm proxy-only; đủ bốn nhánh chỉ khi phân tích tương tác. |
| K1-L4 | Sau pilot | Verified runner nhận `--seed`; orchestration screening hiện là seed 42. | Chạy pilot seed 42 trước; thêm seed độc lập trong output riêng trước kết luận về độ ổn định. |
| K1-L5 | Đã khóa | Cần thống nhất metric và ứng viên. | Chọn best và xếp hạng guided bằng validation mAP50_95; test không chọn lại. |
| K1-L6 | Ghi nhận cost | Early stopping có thể làm số update khác nhau. | `completed.json` ghi steps, optimizer updates, wall time của process; fixed-budget bổ sung nếu muốn cô lập objective khỏi compute. |
| K1-L7 | Đã sửa, cần preflight DP | W3F/PSA trước đây mặc định tắt nếu không đặt biến môi trường. | Supervised bật cả hai cho DP; verified runner dùng cờ từ manifest. Patch lỗi làm dừng run. |
| K1-L8 | Đã xử lý | Entrypoint supervised đã chặn thư mục run có nội dung. | Giữ run riêng và không ghi đè checkpoint đã manifest hóa. |
| K1-L9 | Theo dõi khi pilot | Hệ số loss bằng nhau không bảo đảm thang gradient giống nhau. | Runner kiểm tra gradient hữu hạn/khác 0 và cập nhật head. Nếu guided không có tác động, kiểm tra gradient từng term. |

## 4. Trình tự chạy và điều kiện dừng

Lệnh chính từ root workspace trong Git Bash: `bash kb1_reward_guided_training/run_all_kb1.sh`. Script chạy đủ các bước dưới đây, xác minh output từng giai đoạn và chỉ tạo báo cáo test khi có đủ 18 dòng model-stage. Dùng `--check-only` để audit dữ liệu mà không train.


1. Chạy `python kb1_reward_guided_training/audit_splits.py`. Tiếp tục nếu trạng thái là `no_known_source_overlap`; kiểm tra thêm kiểu augment ngoài mẫu hậu tố flip/rotation nếu có.
2. Từ `kb1_reward_guided_training`, chạy `python train_supervised.py` hoặc `--model ...`. Mỗi model tạo `checkpoint_based/<model>/weights/best.pt` và `supervised_manifest.json`. Bất cứ model lỗi nào cũng trả exit code khác 0.
3. Khi đủ sáu checkpoint, chạy `python run_screening.py`. Preflight 2 step chạy trước từng nhánh. Nếu lỗi, xem `checkpoint_reward_guide_trainning/preflight_seed42_v1/<model>/<mode>/console.log` rồi sửa trước run dài. Guided và native-only bắt đầu từ cùng checkpoint, seed và evaluator.
4. Screening chọn best và xếp hạng guided theo validation mAP50_95. Báo cáo guided − supervised, guided − native-only cho mỗi model, gồm cả chênh lệch âm, số update và cost thực tế. Không quy riêng cải thiện cho reward khi chưa có proxy-only.
5. Sau khi khóa lựa chọn, chạy `python evaluate.py --checkpoint-source screening --split test`. Test dùng để báo cáo, không chọn lại checkpoint hoặc hyperparameter.
6. Sau pilot seed 42, chạy thêm seed độc lập và tổng hợp biến thiên trước kết luận cuối về độ ổn định.

Chưa có kết quả hiệu năng mới sau khi làm lại dữ liệu.
