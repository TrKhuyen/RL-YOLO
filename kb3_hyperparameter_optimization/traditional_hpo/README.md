# KB3-A — Traditional HPO

Mỗi trial chọn một bộ hyperparameter tuyệt đối trước khi train và giữ cố định trong toàn bộ số epoch.

```text
Trial -> chọn {lr, weight decay, momentum, augmentation}
      -> train YOLO từ đầu đến cuối
      -> lấy validation mAP50-95
```

Các phương pháp:

- `default_baseline`: cấu hình khởi tạo, đánh giá nhiều seed.
- `random_search`: lấy mẫu ngẫu nhiên một cấu hình cố định cho mỗi trial.
- `optuna_search`: TPE đề xuất một cấu hình cố định cho mỗi trial.
- `evaluate_config`: khóa cấu hình tốt nhất rồi huấn luyện lại trên các seed mới.

Trong giai đoạn search, mọi trial dùng cùng detector seed để giảm nhiễu. Sau khi tìm được cấu hình tốt nhất, chạy:

```bash
python -m kb3_hyperparameter_optimization.traditional_hpo.evaluate_config \
  --hyperparameters-json runs/kb3/traditional_optuna.json \
  --seeds 101 102 103
```

File kết quả Random Search và Optuna đều chứa `best_params`, nên có thể truyền trực tiếp cho `evaluate_config`.
