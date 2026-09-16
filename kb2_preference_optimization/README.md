# KB2: Feedback Fine-tuning for YOLO

KB2 fine-tune YOLO bằng phản hồi sinh từ prediction và ground truth trên tập
train. DPO cung cấp ý tưởng chosen/rejected và reference constraint; native
YOLO detection loss vẫn là objective chính.

Chạy thử:

    python generate_feedback.py --model yolov8n --device cuda

Môi trường Python dùng chung nằm tại root RL-YOLO:

- .venv
- pyproject.toml
- uv.lock
- requirements.txt
- .gitignore

Chạy script từ root bằng Python trong .venv hoặc chuyển vào thư mục kịch bản
theo hướng dẫn của từng script.
