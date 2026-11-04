# Kết quả S0 đầy đủ giữa bảy model

S0 là checkpoint supervised seed 0, không phải mô hình đã reward-guided. Các giá trị chất lượng dùng %, làm tròn ba chữ số. Lấy hàng S0 trong cohort continuation seed 43; đây chỉ là đánh giá lại S0 cố định, không phải S0 seed 43.

## Validation — 479 ảnh

| Model | mAP50–95 | mAP50 | P | R | F1 | Macro-F1 | AP nhỏ | AP vừa | AP lớn | AR300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5s | 33.792 | 48.596 | 52.420 | 54.531 | 53.455 | 51.281 | 5.267 | 16.247 | 39.091 | 51.454 |
| YOLOv8n | 37.819 | 49.128 | 50.873 | 54.054 | 52.415 | 51.297 | 4.396 | 17.942 | 42.577 | 56.206 |
| YOLOv8s | 37.958 | 48.911 | 51.445 | 53.789 | 52.591 | 50.609 | 5.290 | 19.368 | 43.581 | 54.973 |
| YOLOv11n | 40.537 | 52.570 | 53.806 | 54.690 | 54.244 | 52.471 | 4.057 | 20.962 | 45.729 | 57.043 |
| YOLOv11s | 39.813 | 50.476 | 56.742 | 53.524 | 55.086 | 53.715 | 5.184 | 21.327 | 45.179 | 55.072 |
| YOLO26n | 39.233 | 50.888 | 56.109 | 52.570 | 54.282 | 52.429 | 6.281 | 20.580 | 44.455 | 57.062 |
| DP-YOLO | 33.317 | 47.633 | 43.254 | 59.459 | 50.078 | 48.539 | 7.605 | 16.142 | 38.354 | 56.111 |

## Test — 239 ảnh

| Model | mAP50–95 | mAP50 | P | R | F1 | Macro-F1 | AP nhỏ | AP vừa | AP lớn | AR300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5s | 30.398 | 42.634 | 50.729 | 52.399 | 51.550 | 43.668 | 8.434 | 18.473 | 35.448 | 47.280 |
| YOLOv8n | 33.201 | 43.776 | 48.178 | 50.988 | 49.543 | 42.961 | 6.445 | 22.501 | 38.333 | 51.251 |
| YOLOv8s | 33.976 | 44.787 | 47.231 | 54.563 | 50.633 | 44.479 | 5.245 | 20.963 | 40.319 | 52.247 |
| YOLOv11n | 37.534 | 49.237 | 51.235 | 54.657 | 52.890 | 47.762 | 3.375 | 27.958 | 41.464 | 54.213 |
| YOLOv11s | 36.271 | 46.753 | 53.122 | 52.023 | 52.567 | 46.343 | 7.412 | 23.120 | 41.683 | 51.407 |
| YOLO26n | 33.357 | 43.290 | 51.090 | 50.706 | 50.897 | 44.140 | 8.914 | 19.903 | 39.461 | 49.907 |
| DP-YOLO | 28.913 | 40.807 | 37.649 | 56.350 | 45.139 | 40.182 | 5.728 | 18.446 | 33.917 | 50.025 |

P/R/F1 tại confidence 0.25, matching IoU 0.50. AP dùng confidence 0.001, NMS IoU 0.60, tối đa 300 detection/ảnh. AR300 khác operating recall R. AP nhỏ/vừa/lớn theo diện tích sau resize/pad 640. Cột recall legacy trong CSV là alias AR300 nên không trình bày trùng.

## Kích thước và tốc độ — benchmark riêng ngày 07/10/2026

| Model | Parameters | Checkpoint (MiB) | FPS FP32 batch 16 |
|---|---:|---:|---:|
| YOLOv5s | 7.085.641 | 13.862 | 55.89 |
| YOLOv8n | 3.016.308 | 5.972 | 85.31 |
| YOLOv8s | 11.146.436 | 21.485 | 44.17 |
| YOLOv11n | 2.595.300 | 5.230 | 80.79 |
| YOLOv11s | 9.438.628 | 18.303 | 39.66 |
| YOLO26n | 2.514.720 | 5.157 | 80.62 |
| DP-YOLO | 6.177.161 | 12.511 | 30.76 |

FPS: RTX 4060 Laptop GPU; FP32; batch 16; forward + canonical NMS, không gồm đọc ảnh/decode/resize. Đây là benchmark riêng, không phải latency website batch 1.

Bảng nguồn S0 chưa có AP75, TP/FP/FN tổng, FLOPs hoặc peak VRAM đồng nhất cho cả bảy model; không tự điền hoặc suy số từ giá trị đã làm tròn.

Nguồn: [CSV canonical](../../kb1_reward_guided_training/checkpoint_reward_guide_trainning/remaining_seed43_44_ablation_v1/results_all.csv); [Report KB1, §7 và §13.2](Report_KB1_15_10_2026.md).
