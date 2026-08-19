# KB1-B: Reward-Guided Weight Tuning for Leaf Disease Detection

## 1. Scope and claim

The dataset contains images with multiple leaves. Each bounding box covers one
leaf and its class is the pest/disease category. The difficult evidence may be a
small lesion inside a much larger leaf box.

KB1-B starts from the best supervised detector and updates only its final Detect
head. It uses a task reward to adjust confidence/class ranking while preserving
the supervised solution. It is named **reward-guided weight tuning**, not exact
REINFORCE: YOLO inference is deterministic after thresholding/NMS and
`log(mean confidence)` is only a differentiable confidence proxy.

KB1-B does not replace KB1-A hyperparameter optimization. It also does not claim
that localization coordinates are directly optimized: boxes and NMS decisions
are detached, so the primary trainable signal is objectness/class confidence.

## 2. Required input contract

All adapters receive RGB `float32` tensors in `[0, 1]`, exactly as native YOLO
inference expects. ImageNet mean/std normalization must not be used. Geometric
augmentation must transform boxes together with images.

Before training, step-0 quick validation must be reasonably consistent with the
native validator. A large discrepancy is a blocking pipeline error.

## 3. Task reward

For every image, predictions are greedily matched one-to-one to same-class GT
boxes at IoU thresholds `0.50, 0.60, 0.70, 0.80`. At each threshold:

```text
reward_t = 0.35 * precision_t
         + 0.35 * recall_t
         + 0.30 * mean_matched_iou_t
reward   = mean_t(reward_t)
```

This reward fits whole-leaf annotations. `small-object reward` is not used:
small disease signs inside a leaf do not make the annotated leaf box small.

## 4. Optimization objective

Let `q_i` be the differentiable confidence proxy for image `i`, `R_i` its
detached task reward and `b` an EMA baseline:

```text
L_reward = -mean(log(q_i) * stop_gradient(R_i - b))
```

To limit drift from the supervised solution, trainable Detect-head parameters
are regularized with L2-SP:

```text
L_stability = mean_j mean((theta_j - theta_j_supervised)^2)
L_total = reward_loss_weight * L_reward
        + stability_loss_weight * L_stability
```

KB1-B v2 adds a cross-family matched-confidence surrogate. Greedy class-aware
matching labels retained predictions as TP or FP. It raises TP confidence,
lowers FP confidence and uses the strongest candidate as an FN signal. This is
not the native YOLO detection loss; native YOLOv5 and Ultralytics losses remain
separate implementations and are reserved for the matched-compute S1 control.

KB1-B v3 uses the native Ultralytics detection criterion for YOLOv8/YOLOv11:
box regression, classification and DFL remain the supervised anchor, while the
TP/FP/FN-aware reward term directly guides model weights. YOLOv5 and DP-YOLO
continue to use v2 until their separate ComputeLoss integration is validated.

KB1-B v3.1 rebalances the hybrid objective after v3 showed that full-weight
native loss dominated the reward signal. Native loss is retained as a 0.25
anchor, while reward and the TP/FP/FN confidence surrogate receive meaningful
weights.

## 5. Safe training protocol

Default protocol:

```yaml
steps: 10000
lr: 5.0e-7
min_lr: 1.0e-7
warmup_steps: 500
scheduler: cosine
train_head_only: true
native_supervised_loss_weight: 0.25
reward_loss_weight: 0.10
supervised_loss_weight: 0.25
fallback_proxy_loss_weight: 1.0
stability_loss_weight: 0.001
gradient_accumulation_steps: 2
tp_weight: 1.0
fp_weight: 0.5
fn_weight: 1.5
normalize_advantage: true
advantage_clip: 3.0
eval_interval: 500
early_stopping_patience: 4
early_stopping_min_delta: 0.0005
```

At step 0, save the supervised state as the safe validation-best checkpoint.
After each evaluation, compute:

```text
validation_score = 0.4 * mAP50
                 + 0.4 * mAP50-95
                 + 0.2 * recall
```

Only replace validation-best when this score improves by `min_delta`. Training
reward-best is saved separately and must not be used as the final model.

## 6. Controls required for a paper

For each selected detector and at least three seeds, compare:

1. `S0`: supervised best checkpoint.
2. `S1`: continued native supervised fine-tuning with matched compute.
3. `R0`: reward-guided head-only, without L2-SP.
4. `R1`: reward-guided head-only with L2-SP (proposed KB1-B).

Use the same train/validation/test split, image size and native validator.
Report precision, recall, mAP50, mAP50-95, macro-F1, per-class AP, latency and
mean +/- standard deviation. KB1-B is successful only if it consistently beats
both `S0` and the matched-compute `S1` control on the held-out test set.

## 7. Commands

Run one model:

```powershell
python .\kb1_reward_guided_training\train_rl.py --model yolov8s
```

Run all supported models:

```powershell
python .\kb1_reward_guided_training\train_rl.py
```

For reproducibility, run separate seeds with --seed 42, --seed 43 and
--seed 44. The seed is included in every checkpoint filename.

The default mode remains KB1-B v3.1. The native-only matched control is run
with --mode native-only and is saved as
<model>_seed<seed>_native_only_best.pt. Omitting --mode is equivalent to
--mode kb1b and preserves the current KB1-B objective.

The canonical output is:

```text
kb1_reward_guided_training/rl_checkpoints/<model>_seed<seed>_rl_best.pt
```

If best step is `0`, KB1-B did not improve that model and the supervised
checkpoint remains the correct result.

## 8. Mandatory preflight checks

- Input range is within `[0, 1]`.
- Step-0 quick metrics are close to native metrics.
- Only the final Detect module has trainable parameters.
- Prediction boxes, labels and scores have matching lengths.
- Reward is finite and lies in `[0, 1]`.
- `L_reward`, `L_stability` and total loss are finite.
- Validation-best and reward-best are different files.
- Final paper tables use native validation/test metrics, not training reward.
