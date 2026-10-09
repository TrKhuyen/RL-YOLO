# RL-YOLO Pest Detection

YOLO26n đã được thêm vào supervised training, KB1-B, KB2 feedback/DPO và
worker Ultralytics của KB3. Tên CLI là `yolo26n`, trọng số pretrained là
`yolo26n.pt`. Xem [hướng dẫn tích hợp và chạy](docs/YOLO26N_INTEGRATION.md).

Repository được tổ chức theo tên kịch bản:

- kb1_reward_guided_training: supervised baselines, DP-YOLO và KB1-B
  reward-guided weight tuning.
- kb2_preference_optimization: kịch bản preference optimization/DPO đang
  nghiên cứu và chưa được xác nhận thực nghiệm.
- kb3_hyperparameter_optimization: hiện chạy A–TPE bốn lượt từ recipe KB1,
  train scratch và so với checkpoint supervised KB1. Xem
  [protocol A–TPE](kb3_hyperparameter_optimization/docs/KB3_A_TPE_PROTOCOL.md)
  và [báo cáo B tương lai](kb3_hyperparameter_optimization/docs/KB3_B_FUTURE_RESEARCH_REPORT.md).

Môi trường Python dùng chung nằm tại root:

- .venv
- pyproject.toml
- uv.lock
- requirements.txt
- .gitignore

Ví dụ chạy KB1-B:

    .\.venv\Scripts\python.exe .\kb1_reward_guided_training\train_rl.py --model yolov11n --seed 42

Báo cáo KB1-B hiện tại nằm tại:

    kb1_reward_guided_training/docs/KB1B_STATUS_REPORT.md
