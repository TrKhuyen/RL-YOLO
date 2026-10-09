# So sánh supervised baseline YOLO26n với các model khác

Ngày đánh giá: 07/10/2026. Kết quả được chạy lại trên checkpoint `best.pt` có sẵn.

YOLO26n đứng thứ **4/7 trên test** theo mAP50–95 (**33.36%**), và thứ **3/7 trên validation** (**39.23%**). Model đứng đầu validation là **yolov11n**; test chỉ dùng để báo cáo chất lượng cuối.

## Điều kiện đánh giá

- Chỉ đánh giá supervised baseline, không dùng checkpoint reward-guided hoặc KB2.
- Cả 7 checkpoint đã qua kiểm tra hash checkpoint và fingerprint dữ liệu hiện tại.
- Dữ liệu: `pre-data/data/v2i_cleanned`, 2.411 ảnh train, 479 validation, 239 test, 28 lớp.
- Dataset SHA256: `cbd68ffc1da935ac300fba953ea5f9c97c831f53887e1b9391d500f9e40c0c95`.
- Chung evaluator `kb1_canonical_v2_ar300`: RGB float [0,1], letterbox 640, confidence 0,001; class-aware NMS IoU 0,60; max_det 300; AP tại IoU 0,50:0,05:0,95.
- P/R/F1 là chỉ số micro tại confidence 0,25 và matching IoU 0,50; không phải AR300.
- AP nhỏ dùng kích thước hộp trong ảnh đã resize/pad 640×640, không phải kích thước ảnh gốc.
- YOLO26n dùng head one-to-many với NMS, cùng đường inference mà KB1/KB2 đang dùng.
- Máy: NVIDIA GeForce RTX 4060 Laptop GPU, torch 2.5.1+cu121, Ultralytics 8.4.163.
- FPS đo FP32, batch 16, forward + canonical NMS, đồng bộ CUDA, bỏ batch đầu; không gồm đọc ảnh/tiền xử lý và không phải tốc độ NMS-free hoặc latency batch 1.

## Kết quả test

| Model    |   mAP50 (%) |   mAP50–95 (%) |   AP nhỏ (%) |   P (%) |   R (%) |   F1 (%) |   FPS |
|:---------|------------:|---------------:|-------------:|--------:|--------:|---------:|------:|
| yolov11n |       49.23 |          37.53 |         3.37 |   51.23 |   54.66 |    52.89 | 80.79 |
| yolov11s |       46.75 |          36.27 |         7.41 |   53.12 |   52.02 |    52.57 | 39.66 |
| yolov8s  |       44.79 |          33.98 |         5.24 |   47.23 |   54.56 |    50.63 | 44.17 |
| yolo26n  |       43.29 |          33.36 |         8.91 |   51.09 |   50.71 |    50.90 | 80.62 |
| yolov8n  |       43.78 |          33.20 |         6.45 |   48.18 |   50.99 |    49.54 | 85.31 |
| yolov5s  |       42.63 |          30.40 |         8.43 |   50.73 |   52.40 |    51.55 | 55.89 |
| dp_yolo  |       40.81 |          28.91 |         5.73 |   37.65 |   56.35 |    45.14 | 30.76 |

## Kết quả validation

| Model    |   mAP50 (%) |   mAP50–95 (%) |   AP nhỏ (%) |   P (%) |   R (%) |   F1 (%) |   FPS |
|:---------|------------:|---------------:|-------------:|--------:|--------:|---------:|------:|
| yolov11n |       52.57 |          40.53 |         4.06 |   53.81 |   54.69 |    54.24 | 82.67 |
| yolov11s |       50.47 |          39.81 |         5.18 |   56.74 |   53.52 |    55.09 | 41.84 |
| yolo26n  |       50.89 |          39.23 |         6.28 |   56.11 |   52.57 |    54.28 | 82.94 |
| yolov8s  |       48.91 |          37.96 |         5.29 |   51.44 |   53.79 |    52.59 | 47.20 |
| yolov8n  |       49.12 |          37.82 |         4.40 |   50.87 |   54.05 |    52.42 | 79.56 |
| yolov5s  |       48.60 |          33.79 |         5.27 |   52.42 |   54.53 |    53.45 | 57.27 |
| dp_yolo  |       47.63 |          33.32 |         7.61 |   43.25 |   59.46 |    50.08 | 29.86 |

## YOLO26n chênh lệch bao nhiêu?

| Model    |   Δ mAP50 test (điểm %) |   Δ mAP50–95 test (điểm %) |   Δ AP nhỏ test (điểm %) |
|:---------|------------------------:|---------------------------:|-------------------------:|
| yolov5s  |                   +0.66 |                      +2.96 |                    +0.48 |
| yolov8n  |                   -0.49 |                      +0.15 |                    +2.47 |
| yolov8s  |                   -1.50 |                      -0.62 |                    +3.67 |
| yolov11n |                   -5.94 |                      -4.17 |                    +5.54 |
| yolov11s |                   -3.46 |                      -2.91 |                    +1.50 |
| dp_yolo  |                   +2.48 |                      +4.44 |                    +3.19 |

- So với YOLOv8n, chênh lệch mAP50–95 là **+0.15 điểm phần trăm**. Mức chênh này nhỏ; một seed chưa đủ chứng minh ưu thế ổn định.
- So với YOLOv11n, chênh lệch mAP50–95 là **-4.17 điểm phần trăm**. YOLOv11n vẫn là baseline mạnh hơn về độ chính xác tổng thể trong lần chạy này.
- YOLO26n có AP nhỏ **8.91%**, đứng thứ **1/7**. Đây là điểm mạnh tương đối, nhưng mức AP nhỏ tuyệt đối vẫn thấp.
- mAP50–95 của YOLO26n giảm **5.88 điểm phần trăm** từ validation sang test. Mức giảm cho thấy kết quả trên validation không chuyển nguyên vẹn sang test; riêng số liệu này chưa xác định được nguyên nhân.
- Trong lần đo này, YOLO26n đạt 80.6 FPS và YOLOv11n đạt 80.8 FPS. Hai tốc độ gần nhau; chưa có lợi thế tốc độ rõ ràng của YOLO26n trong đường chạy có NMS này.

## Ngân sách huấn luyện đã thực hiện

Cấu hình baseline đều seed 0, SGD, tối đa 200 epoch, patience 30. Batch khác nhau theo model; early stopping khiến số epoch thực chạy khác nhau. Các model Ultralytics dùng cùng imgsz, learning rate và augmentation trong cấu hình dự án.

| Model    |   Completed_epochs |   Batch |   Seed | Training_seconds   |
|:---------|-------------------:|--------:|-------:|:-------------------|
| dp_yolo  |                200 |       4 |      0 | —                  |
| yolo26n  |                142 |      16 |      0 | 8151.78            |
| yolov11n |                161 |      16 |      0 | 8266.29            |
| yolov11s |                167 |       8 |      0 | 11075.4            |
| yolov5s  |                151 |      16 |      0 | —                  |
| yolov8n  |                166 |      16 |      0 | 8369.91            |
| yolov8s  |                126 |       8 |      0 | 8439.6             |

Thời gian train lấy từ cột `time` của log; YOLOv5s/DP-YOLO không có cột đó. Không dùng tổng thời gian khác số epoch để kết luận model nào train nhanh hơn. `Peak_native_epoch` trong training_summary.csv là epoch đạt mAP native cao nhất theo log, không tự khẳng định epoch của checkpoint `best.pt` theo mọi fitness.

## AP theo lớp: YOLO26n so với YOLOv11n

Năm lớp có chênh lệch cao nhất:

|   Class_ID | Class                    |   yolo26n |   yolov11n |   Δ (điểm %) |   Hộp test |
|-----------:|:-------------------------|----------:|-----------:|-------------:|-----------:|
|         25 | Tomato mold leaf         |     31.29 |       6.93 |        24.36 |          5 |
|          2 | Apple rust leaf          |     56.96 |      39.06 |        17.89 |         22 |
|          1 | Apple leaf               |     40.83 |      32.24 |         8.59 |         31 |
|         24 | Tomato leaf yellow virus |     30.62 |      25.40 |         5.23 |        124 |
|         10 | Peach leaf               |     34.64 |      31.64 |         3.01 |        178 |

Năm lớp có chênh lệch thấp nhất:

|   Class_ID | Class                    |   yolo26n |   yolov11n |   Δ (điểm %) |   Hộp test |
|-----------:|:-------------------------|----------:|-----------:|-------------:|-----------:|
|         15 | Soyabean leaf            |     52.33 |      83.48 |       -31.15 |         17 |
|          0 | Apple Scab Leaf          |     17.74 |      40.97 |       -23.23 |         18 |
|          7 | Corn Gray leaf spot      |     19.24 |      40.76 |       -21.52 |         14 |
|          3 | Bell_pepper leaf         |     28.58 |      43.65 |       -15.07 |         45 |
|         18 | Tomato Early blight leaf |      5.63 |      19.82 |       -14.18 |         10 |

Số hộp hỗ trợ giúp đọc mức độ đại diện của từng lớp; chưa có khoảng tin cậy AP theo lớp.

## Nhận định sử dụng trong luận văn

Giữ YOLO26n làm baseline nano bổ sung và ứng viên cho thí nghiệm cải thiện đối tượng nhỏ. Với độ chính xác tổng thể, YOLOv11n là lựa chọn dẫn đầu validation và cũng có kết quả test tốt hơn YOLO26n. Chưa có bằng chứng rằng YOLO26n thay thế tốt hơn toàn bộ baseline hiện tại.

Kết quả là một seed và cấu hình SGD của dự án; không phải kết luận về mọi cấu hình YOLO26 hay tái lập recipe mặc định của YOLO26. Cần nhiều seed và cùng budget để kiểm tra độ ổn định. Không thay checkpoint hoặc tuning sau khi xem test.

## Artifact

- [Kết quả đầy đủ](../kb1_reward_guided_training/results/baseline_comparison_yolo26n_20261007/results.csv)
- [AP/AR từng lớp](../kb1_reward_guided_training/results/baseline_comparison_yolo26n_20261007/per_class.csv)
- [Training summary](../kb1_reward_guided_training/results/baseline_comparison_yolo26n_20261007/training_summary.csv)
- [Provenance và protocol](../kb1_reward_guided_training/results/baseline_comparison_yolo26n_20261007/manifest.json)

![So sánh mAP baseline](../kb1_reward_guided_training/results/baseline_comparison_yolo26n_20261007/comparison.png)
