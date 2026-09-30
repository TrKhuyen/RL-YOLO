# KB1 final clean-data report

Primary metric: mAP50_95. Candidate set: best reward-guided checkpoint of each model.
Checkpoint selection and the validation leader were locked before this test evaluation.
Validation leader: YOLOv11n (40.655%).

| Model | Supervised test | Guided test | Guided - supervised | Native-only test | Guided - native-only |
|---|---:|---:|---:|---:|---:|
| YOLOv5s | 30.395 | 31.245 | +0.850 | 30.902 | +0.343 |
| YOLOv8n | 33.204 | 33.762 | +0.558 | 33.242 | +0.520 |
| YOLOv8s | 33.976 | 35.241 | +1.264 | 34.991 | +0.249 |
| YOLOv11n | 37.531 | 37.727 | +0.196 | 37.788 | -0.060 |
| YOLOv11s | 36.269 | 36.496 | +0.227 | 36.958 | -0.462 |
| DP-YOLO | 28.912 | 29.467 | +0.555 | 29.292 | +0.174 |

Highest guided test score: YOLOv11n. This is descriptive; test did not select checkpoints or hyperparameters.
The guided - native-only difference includes the reward and proxy terms together.

Full metrics: `results/canonical_clean/screening/test/results_full.csv`; paired deltas: `results/canonical_clean/screening/test/results_delta.csv`.
