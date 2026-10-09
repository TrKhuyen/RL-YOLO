# KB2: Feedback, Pairwise and Candidate-selection DPO for YOLO

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

The controlled runner includes `baseline`, `sampling`, `hybrid`, `pairwise`,
and `dpo`. The pairwise branch uses the same shuffle sampling as baseline,
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

## Candidate-selection DPO branch

The `dpo` branch applies the reference-relative objective from
[the DPO paper](https://arxiv.org/abs/2305.18290) to a categorical policy for
selecting a dense candidate. It uses a frozen copy of the original KB1 checkpoint,
GT-derived chosen/rejected pairs from the CURRENT model on each augmented train
image, and the same selected candidate indices for policy and reference. A is
the best-IoU candidate for its GT; B has strictly lower current IoU and the
highest GT-class score among eligible inferior candidates, including those
above IoU 0.5. Reference geometry does not decide which candidate is preferred.
This is an online adaptation of the DPO loss. Native YOLO loss
remains the main objective. This policy covers candidate selection, rather than
the probability of an entire set of bounding boxes. See [DPO.md](docs/DPO.md)
for the policy definition, equation, limitations, and experiment results.

The new pair source is `online_policy_gt_iou_v2`, with objective version
`candidate_selection_online_dpo_v2`. Checkpoints using reference-selected
pairs (`frozen_reference_same_gt_v1`) cannot resume as v2. Their reported
metrics describe the historical method; v2 must be trained in a new output
directory from the original KB1 checkpoint.

Run only DPO (no baseline/pairwise retraining):

```bash
bash kb2_preference_optimization/run_all_kb2.sh --model yolov8n --variants dpo --steps 15000 --patience 5 --dpo-alpha 0.1 --dpo-beta 0.1
```

The script generates train feedback and writes a new timestamped output directory.
Training defaults to at most 15,000 updates (about 25 train passes at batch 4),
validation every 1,000 updates, patience 5, and intermediate checkpoints every
2,000 updates. This gives validation more time to settle; convergence still
requires inspecting the training and validation curves. Override these with
`--steps`, `--eval-interval`, `--patience`, and `--save-interval`.
The default now runs all five variants on all five supported models. Select
`--variants baseline pairwise dpo` for a focused comparison. To reuse existing
feedback, use `run_ablation.py` as shown in [DPO.md](docs/DPO.md).
