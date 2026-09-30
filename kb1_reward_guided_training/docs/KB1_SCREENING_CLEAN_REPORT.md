# KB1 screening on clean dataset

Updated: 2026-09-28T07:19:15
Completed runs: 12/12; comparable pairs: 6/6.

Dataset: pre-data/data/v2i_cleanned. Stage 1: checkpoint_based/<model>/weights/best.pt.
Stage 2: checkpoint_reward_guide_trainning/screening_seed42_v1/<model>/<mode>/best.pt.
Only completed runs are included below. Validation selects checkpoints; test is reserved for final evaluation.

## Dataset audit

Status: no_known_source_overlap; train/valid/test images: {'train': 2411, 'valid': 479, 'test': 239}.
- train_valid: 0 overlapping source groups
- train_test: 0 overlapping source groups
- valid_test: 0 overlapping source groups

## Validation results

| Model | Stage | Best step | mAP50 | mAP50-95 | AR300 |
|---|---|---:|---:|---:|---:|
| yolov5s | supervised | 0 | 48.597 | 33.792 | 51.454 |
| yolov5s | native_only | 5000 | 48.497 | 34.688 | 52.515 |
| yolov5s | kb1b | 8500 | 48.712 | 35.029 | 52.660 |
| yolov8n | supervised | 0 | 49.124 | 37.818 | 56.206 |
| yolov8n | native_only | 2500 | 49.426 | 37.986 | 56.590 |
| yolov8n | kb1b | 5000 | 49.665 | 38.367 | 56.669 |
| yolov8s | supervised | 0 | 48.906 | 37.956 | 54.973 |
| yolov8s | native_only | 6000 | 49.454 | 38.589 | 55.488 |
| yolov8s | kb1b | 7500 | 49.880 | 38.972 | 55.197 |
| yolov11n | supervised | 0 | 52.566 | 40.531 | 57.043 |
| yolov11n | native_only | 1000 | 52.594 | 40.571 | 57.115 |
| yolov11n | kb1b | 1000 | 52.698 | 40.655 | 57.118 |
| yolov11s | supervised | 0 | 50.474 | 39.810 | 55.072 |
| yolov11s | native_only | 8500 | 50.925 | 40.258 | 55.596 |
| yolov11s | kb1b | 7000 | 51.039 | 40.412 | 55.554 |
| dp_yolo | supervised | 0 | 47.635 | 33.317 | 56.111 |
| dp_yolo | native_only | 6000 | 47.844 | 33.985 | 56.771 |
| dp_yolo | kb1b | 10000 | 47.975 | 34.109 | 56.894 |

| Model | Delta mAP50-95 vs supervised | Delta mAP50-95 vs native-only |
|---|---:|---:|
| yolov5s | +1.237 | +0.341 |
| yolov8n | +0.550 | +0.381 |
| yolov8s | +1.017 | +0.384 |
| yolov11n | +0.124 | +0.084 |
| yolov11s | +0.602 | +0.154 |
| dp_yolo | +0.792 | +0.124 |

## Validation ranking (locked before test)

Primary metric: mAP50_95. Candidate set: guided best checkpoint of each model.
Validation leader: yolov11n (40.655%).
Evaluate all locked checkpoints on test once; do not reselect hyperparameters from test.

## Run status

| Model | Mode | Status | Recorded steps | Optimizer updates | Training seconds |
|---|---|---|---:|---:|---:|
| yolov5s | native_only | complete | 5000 | 2500 | 514.2 |
| yolov5s | kb1b | complete | 9500 | 4750 | 1315.1 |
| yolov8n | native_only | complete | 3000 | 1500 | 310.5 |
| yolov8n | kb1b | complete | 5000 | 2500 | 585.9 |
| yolov8s | native_only | complete | 7000 | 3500 | 728.9 |
| yolov8s | kb1b | complete | 8000 | 4000 | 1454.9 |
| yolov11n | native_only | complete | 2000 | 1000 | 213.4 |
| yolov11n | kb1b | complete | 3000 | 1500 | 379.0 |
| yolov11s | native_only | complete | 8500 | 4250 | 887.1 |
| yolov11s | kb1b | complete | 7000 | 3500 | 1264.7 |
| dp_yolo | native_only | complete | 6500 | 3250 | 1058.8 |
| dp_yolo | kb1b | complete | 10000 | 5000 | 2805.6 |

A complete status requires completed.json in the new run directory.
Results from the former leaked split are excluded.
