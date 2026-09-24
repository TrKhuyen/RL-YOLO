# Review logic cuối cho KB1, KB2 và KB3

## Phạm vi

Báo cáo này chỉ kiểm tra **logic nghiên cứu và logic triển khai hiện có**. Dữ liệu đã được làm lại nhưng các thí nghiệm chưa chạy lại, nên tài liệu này **không dùng số liệu từ các lượt chạy cũ, không đánh giá mức cải thiện và không xếp hạng mô hình**. Các hyperparameter trong code là giá trị triển khai hiện tại, chưa phải cấu hình thực nghiệm cuối đã được chọn và kiểm chứng.

Đọc phân tích riêng: [KB1](KB1.md), [KB2](KB2.md), [KB3](KB3.md).

## Kết luận logic

| Kịch bản | Hướng nghiên cứu | Kết luận review logic |
|---|---|---|
| KB1 | Train supervised từng YOLO, lấy best checkpoint tương ứng rồi fine-tune trọng số theo objective có reward; so trước/sau từng model và chọn model cuối tốt nhất | **Đúng hướng về quy trình hai giai đoạn.** Cần gọi đúng bản chất reward-guided, tách ảnh hưởng của reward khỏi proxy/continued training, và khóa provenance của checkpoint cùng protocol đánh giá. |
| KB2 | Dùng phản hồi tự động từ prediction và ground truth để fine-tune YOLO, lấy cảm hứng từ preference learning | **Đúng hướng về ý tưởng.** Code hiện là feedback-guided fine-tuning; preference pairs được sinh nhưng chưa được dùng trong trainer chính. Matching feedback và một số auxiliary loss có lỗi logic cần sửa trước khi chạy lại. |
| KB3-A | Chọn một cấu hình tuyệt đối cho mỗi trial và giữ cố định đến cuối run | **Đúng logic phương pháp.** Cần bảo đảm backend thực sự áp dụng đúng cấu hình và việc chọn best dựa trên metric nhất quán. |
| KB3-B | Agent thay đổi hyperparameter sau mỗi segment trong cùng episode | **Đúng ý tưởng adaptive RL, nhưng chưa đúng đầy đủ điều kiện continuity đã nêu.** Backend resume từ checkpoint giữa các segment; raw model và scheduler không được giữ nguyên như một trainer liên tục. |

## Những lỗi cần ưu tiên trước khi chạy lại

1. **KB1:** mô tả loss như REINFORCE chuẩn chưa phù hợp với detection xác định sau NMS và surrogate log-confidence. Đối chứng hiện tại chỉ đo tác dụng gộp của reward và proxy. Runner verified còn cố định seed, nên phải mở rộng trước khi làm đa seed.
2. **KB2:** một ground truth có thể nhận nhiều trạng thái phản hồi mâu thuẫn; wrong-class và bad-localization có thể bị đếm thêm là missed. Matching chưa one-to-one; dynamic feedback có thể khác output detection cuối. Đây là lỗi trong tín hiệu huấn luyện.
3. **KB2:** checkpoint nền được lấy từ KB1 còn data root của KB2 được cấu hình riêng. Sau khi làm lại dữ liệu, cần kiểm tra checkpoint nền đã học những source ID nào rồi mới dùng nó cho KB2. Đây là điều kiện logic của phép đánh giá độc lập, **không phải nhận định rằng dữ liệu mới đang bị rò rỉ**.
4. **KB3-B:** cần kiểm tra tính tương đương giữa train liên tục và train chia segment. Nếu giữ kiến trúc resume hiện tại, phải mô tả đúng việc nạp raw model từ EMA và tái dựng scheduler, đồng thời thêm đối chứng fixed schedule cũng chia segment.
5. **KB3:** comparator hiện có thể gộp file kết quả khác seed, split, model, evaluator hoặc giai đoạn search/evaluation mà không từ chối. Reward có thành phần AP-small nhưng backend không canonical trả giá trị cố định cho metric này.

## Trạng thái kết luận nghiên cứu

- Câu hỏi KB1 “reward-guided cải thiện từng mô hình bao nhiêu?” và “mô hình nào cuối cùng tốt nhất?” **chưa có đáp án** cho dữ liệu đã làm lại.
- KB2 **chưa có kết luận hiệu quả** sau khi sửa dữ liệu và các lỗi logic.
- KB3-A/B **chưa có kết luận so sánh thực nghiệm** sau khi kiểm chứng backend và chạy lại.

Báo cáo này là danh sách việc cần sửa và tiêu chí kiểm tra logic trước khi chạy thí nghiệm. Nó không đề xuất một bộ thông số tối ưu.

## Tài liệu phương pháp

- [DPO gốc](https://arxiv.org/abs/2305.18290): dùng cặp chosen/rejected và reference trong objective. KB2 chỉ lấy cảm hứng từ phản hồi ưu tiên, không cần sao chép DPO cho ngôn ngữ.
- [Ultralytics validation](https://docs.ultralytics.com/modes/val): cần cố định cách tạo prediction và định nghĩa metric khi so checkpoint.
- [PyTorch optimizer state](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.load_state_dict.html): continuity của training state cần được xác minh ở cấp state, không suy ra từ một cờ resume.

