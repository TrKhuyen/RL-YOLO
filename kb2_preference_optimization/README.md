# KB2: Feedback and Pairwise Fine-tuning for YOLO

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

## Pairwise preference branch

The `pairwise` ablation now optimizes a real chosen/rejected ranking loss
alongside native YOLO detection loss. On each augmented training image, decoded
dense candidates are assigned to their highest-IoU ground truth object. A
candidate with IoU >= 0.5 is chosen; a candidate for the same object with
0.1 <= IoU < 0.5 is rejected. The loss is

    native_loss + pairwise_alpha * softplus(
        pairwise_margin - (logit(score_chosen) - logit(score_rejected)))

The score is the current model probability for the GT class. Candidate and GT
selection use detached boxes; gradients flow through both class scores. The
native detection loss continues to train box coordinates and classification.
Pairs are rebuilt online because stored teacher prediction indices cannot be
mapped reliably after augmentation or model updates. Stored feedback pairs
remain diagnostic data and are not used by this branch. This is pairwise
preference optimization, not DPO: there is no reference model or policy
log-ratio.

The controlled runner includes `baseline`, `sampling`, `hybrid`, and
`pairwise`. The pairwise branch uses the same shuffle sampling as baseline,
so their difference isolates the ranking objective. It writes pair counts in
the checkpoint and ablation report. Example:

    bash kb2_preference_optimization/run_all_kb2.sh --model yolov8n --pairwise-alpha 0.01 --seed 42

For an additional seed or loss weight using existing schema-2.1 feedback:

    python kb2_preference_optimization/run_ablation.py --model yolov8n --feedback PATH_TO_SCHEMA_2_1_JSONL --variants baseline pairwise --seed 43 --pairwise-alpha 0.01 --output-dir PATH_TO_OUTPUT

Choose loss weight on validation only and reserve the locked test split for
final evaluation. Short pilots verify that the branch runs; they do not
establish a reliable mAP gain. The 1000-update four-model comparison
and three-seed YOLOv8n check are recorded in
[KB2 report](../docs/report_final/KB2.md); they do not show a
consistent validation mAP improvement.
