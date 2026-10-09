# Thiết kế KB3 gồm hai kịch bản con

## KB3-A — Traditional HPO

### Câu hỏi nghiên cứu

Một bộ hyperparameter cố định được tối ưu trước khi train có cải thiện YOLO so với cấu hình mặc định không?

### Quy trình

```text
Chọn một cấu hình tuyệt đối
          ↓
Khởi tạo YOLO từ YAML kiến trúc, trọng số ngẫu nhiên theo seed
          ↓
Train toàn bộ số epoch, không thay đổi cấu hình
          ↓
Đánh giá validation
          ↓
Random Search hoặc TPE chọn trial tiếp theo
```

### Phương pháp

1. Default configuration.
2. Random Search với cấu hình cố định.
3. Optuna/TPE với cấu hình cố định.

Trong giai đoạn tìm kiếm, mọi trial dùng cùng detector seed để giảm nhiễu. Sau khi khóa cấu hình tốt nhất, cấu hình đó phải được huấn luyện lại trên ít nhất ba seed.

## KB3-B — Adaptive RL

### Câu hỏi nghiên cứu

Một policy quan sát trạng thái huấn luyện và điều chỉnh hyperparameter theo segment có tốt hơn cấu hình cố định đã được tối ưu không?

### Quy trình

```text
Quan sát state hiện tại
          ↓
Agent chọn tăng/giảm/giữ hyperparameter
          ↓
YOLO tiếp tục train K epoch với cùng model và optimizer
          ↓
Đánh giá validation và tính reward
          ↓
Cập nhật policy, rồi chuyển sang segment tiếp theo
```

### Phương pháp

1. Random Schedule: thay đổi ngẫu nhiên theo segment.
2. Bandit: học action nhưng chưa phụ thuộc state.
3. PPO: policy phụ thuộc state và reward.

## Nguyên tắc so sánh

Hai kịch bản phải dùng chung:

- Cùng kiến trúc YAML và cách khởi tạo từ đầu theo seed, không dùng pretrained.
- Dataset manifest và train/validation/test split.
- Miền giá trị của hyperparameter.
- Tổng ngân sách epoch hoặc GPU-hour.
- Canonical evaluator và quy tắc chọn checkpoint.
- Các seed đánh giá cuối.

Không so sánh các episode dùng để huấn luyện PPO với test run của KB3-A. Quy trình đúng là:

1. Dùng ngân sách search để tìm cấu hình tốt nhất của KB3-A.
2. Dùng ngân sách tương đương để học policy KB3-B.
3. Khóa cấu hình và policy.
4. Chạy lại cả hai trên cùng các seed chưa dùng trong search.
5. Chỉ sau đó đánh giá các checkpoint được khóa trên test set.

## Ma trận thí nghiệm cuối

| Mã | Kịch bản | Phương pháp | Tham số trong một run |
|---|---|---|---|
| A0 | KB3-A | Default | Cố định |
| A1 | KB3-A | Random Search best | Cố định |
| A2 | KB3-A | Optuna/TPE best | Cố định |
| B0 | KB3-B | Random Schedule | Thay đổi ngẫu nhiên theo segment |
| B1 | KB3-B | Bandit | Thay đổi theo reward, không dùng state |
| B2 | KB3-B | PPO | Thay đổi theo state và reward |

So sánh A2 với B2 trả lời câu hỏi chính: RL thích nghi có mang lại lợi ích so với một cấu hình cố định đã được tối ưu hay không?
