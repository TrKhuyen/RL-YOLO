# RL-YOLO Pest Detection

Repository được tổ chức theo tên kịch bản:

- kb1_reward_guided_training: supervised baselines, DP-YOLO và KB1-B
  reward-guided weight tuning.
- kb2_preference_optimization: kịch bản preference optimization/DPO đang
  nghiên cứu và chưa được xác nhận thực nghiệm.

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
