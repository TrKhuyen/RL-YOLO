# BÁO CÁO PHÂN TÍCH CHI TIẾT: DP-YOLO

## 1. Thông Tin Chung Về Bài Báo
* [cite_start]**Tên bài báo:** DP-YOLO: Effective Improvement Based on YOLO Detector[cite: 5].
* [cite_start]**Tác giả:** Chao Wang, Qijin Wang, Yu Qian, Yating Hu, Ying Xue và Hongqiang Wang[cite: 6, 7].
* [cite_start]**Đơn vị công tác:** Trường Kỹ thuật Điện tử và Thông tin (Đại học An Huy Kiến Trúc), Trường Dữ liệu lớn và Trí tuệ Nhân tạo (Đại học An Huy Tân Hoa), và Viện Máy Thông minh (Viện Hàn lâm Khoa học Trung Quốc)[cite: 20, 22, 23].
* [cite_start]**Nơi xuất bản:** Tạp chí Applied Sciences (MDPI), năm 2023[cite: 2, 3, 10].

---

## 2. Bối Cảnh Và Đặt Vấn Đề
* [cite_start]**Điểm mạnh của YOLOv5:** YOLOv5 là một trong những mô hình phát hiện thời gian thực được sử dụng rộng rãi nhất do sự cân bằng giữa độ chính xác, tốc độ, dễ huấn luyện và triển khai[cite: 25, 39].
* [cite_start]**Hạn chế tồn đọng:** * So với các bộ phát hiện mới hơn, chiến lược phân bổ nhãn (label assignment) của YOLOv5 còn nhiều điểm yếu và có không gian lớn để tối ưu hóa[cite: 26].
    * [cite_start]Mô hình gặp khó khăn khi nhận dạng các mục tiêu có hình dáng và tư thế biến đổi liên tục[cite: 27].
    * [cite_start]Khi áp dụng vào các tập dữ liệu đặc thù, YOLOv5 bộc lộ các hạn chế như: khả năng phát hiện mục tiêu nhỏ kém, không thích ứng tốt với nền phức tạp và khả năng xử lý biến dạng mục tiêu còn hạn chế[cite: 48].
* [cite_start]**Động lực cụ thể:** Bài báo lấy ví dụ về hình ảnh vi thể của cặn nước tiểu (urine sediment), nơi các đặc điểm không rõ ràng, mục tiêu nhỏ nhiều, dễ nhầm lẫn phân lớp và có sự khác biệt lớn về hình thái trong cùng một loại[cite: 51, 52].

---

## 3. Phương Pháp Đề Xuất (Cấu trúc DP-YOLO)
[cite_start]Nhóm nghiên cứu đề xuất **DP-YOLO** bằng cách tích hợp tích chập biến dạng (deformable convolution) vào mạng xương sống (backbone) và cải thiện chiến lược gán nhãn[cite: 29].

### 3.1. Chiến lược Petal-like Sample Amplification (PSA)
* [cite_start]**Khái niệm:** Khác với chiến lược tĩnh ban đầu của YOLOv5 (mở rộng hộp neo dựa trên lưới lân cận nếu tỷ lệ nằm trong ngưỡng 0.25-4), PSA tích hợp đặc tính của vùng cảm nhận hiệu quả (effective receptive field) vào quá trình chọn mẫu[cite: 144, 166].
* **Cách hoạt động:** Phương pháp vẽ một hình tròn lấy trọng tâm của mỗi lưới làm tâm. [cite_start]Nếu trọng tâm của nhãn gốc (ground truth) nằm trong hình tròn này, các hộp neo (anchors) liên kết của lưới đó sẽ được coi là mẫu dương tính[cite: 168, 169].
* [cite_start]**Kết quả:** Phạm vi mở rộng của các lưới lân cận xếp chồng lên nhau tạo thành hình dáng giống hai cánh hoa, nên được gọi là Petal-like Sample Amplification (PSA)[cite: 170].
* [cite_start]**Chi tiết thực nghiệm:** Bán kính mở rộng (đại diện bởi hệ số $r$) đạt hiệu quả phát hiện tốt nhất khi $r = 1$[cite: 367]. [cite_start]PSA giúp tổng số lượng mẫu dương tính tăng lên khoảng 5% (từ 10.674.553 lên 11.203.511 trên tập COCO2017)[cite: 368, 375].

### 3.2. Mạng xương sống Deformable YOLO v5 Backbone (DYB)
* [cite_start]**Khái niệm:** Thay thế phép tích chập thông thường trong cấu trúc nút thắt cổ chai (bottleneck) của mô-đun C3 ở mạng backbone YOLOv5s bằng toán tử tích chập biến dạng[cite: 174].
* **Cấu trúc Mô-đun:**
    * [cite_start]**D2C3:** Sử dụng toán tử tích chập biến dạng **DCNv2**, có trọng số chiếu riêng cho từng vector đặc điểm của điểm lấy mẫu (số lượng tham số lớn hơn)[cite: 179, 180].
    * [cite_start]**D3C3:** Sử dụng toán tử tích chập biến dạng **DCNv3**, nhẹ hơn và kết hợp tích chập nhóm, phù hợp với các tầng mạng có số lượng kênh tensor lớn[cite: 183, 184].
* [cite_start]**Cấu hình tối ưu:** Qua thử nghiệm trên mạng backbone, kết quả tốt nhất đạt được khi sử dụng mô-đun **D3C3 ở giai đoạn downsampling cuối cùng** và mô-đun **D2C3 ở tất cả các giai đoạn trước đó** (tương đương chiến lược $3+1$)[cite: 388, 389, 391].

---

## 4. Thiết Lập Thực Nghiệm Và Công Thức Đánh Giá
### 4.1. Môi trường và Tham số huấn luyện
* [cite_start]**Phần cứng:** Máy chủ DELL PowerEdge 640 trang bị 4 GPU GeForce RTX 3090 (24GB memory)[cite: 292].
* [cite_start]**Phần mềm:** Ubuntu 20.04, CUDA 11.2, Python 3.9, và PyTorch 1.12.1[cite: 293].
* [cite_start]**Tham số:** Optimizer SGD, hàm mất mát CIoU Loss cho hồi quy bounding box, batch size 64 mỗi GPU, learning rate ban đầu 0.01, decay 0.0005, huấn luyện 300 epochs, kích thước ảnh đầu vào $640\times 640$[cite: 294, 295].

### 4.2. Tập dữ liệu (Datasets)
* [cite_start]Tập dữ liệu chung: **COCO2017** và **VOC07 + 12**[cite: 295].
* Tập dữ liệu đặc thù: **Urised11** (Cặn nước tiểu). [cite_start]Tập này gồm 7364 hình ảnh (kích thước $720\times 576$), chứa 11 danh mục với tổng số 58.196 nhãn (chia tỷ lệ train:val là 4:1)[cite: 258, 260].

### 4.3. Các công thức đánh giá (Evaluation Metrics)
[cite_start]Bài báo sử dụng các chỉ số chuẩn trong bài toán phát hiện đối tượng[cite: 299]. Các công thức được định nghĩa chính xác như sau:

* **Độ chính xác (Precision):**
  [cite_start]$P = \frac{TP}{TP + FP}$ [cite: 301, 302]

* **Độ phủ (Recall):**
  [cite_start]$Recall(R) = \frac{TP}{TP + FN}$ [cite: 303]

* **Độ chính xác trung bình (Average Precision - AP):**
  [cite_start]$AP = \frac{1}{n}\sum_{i=1}^{n}\int_{0}^{1}P(R)d\overline{R}$ [cite: 303]

[cite_start]*(Trong đó: TP là True Positive, FP là False Positive, FN là False Negative, n là số lượng danh mục [cite: 304, 305, 306]).*

---

## 5. Kết Quả Và Đánh Giá

### 5.1. Hiệu suất trên tập COCO2017
* [cite_start]**Baseline (YOLOv5s):** Đạt 38.0 AP, 7.2M tham số, 16.4 GFLOPS, tốc độ 75 FPS[cite: 326].
* [cite_start]**DP-YOLO (tích hợp cả PSA và DYB):** Đạt **41.2 AP** (tăng **3.2 AP** so với baseline)[cite: 326].
* [cite_start]**Chi phí tính toán:** Chỉ tăng một lượng nhỏ, với 7.6M tham số (tăng 5.5%) và 17.7 GFLOPS (tăng 4.3%), duy trì tốc độ rất ấn tượng là **69 FPS** trên RTX 3090[cite: 321, 326].
* [cite_start]**Đặc biệt:** Độ chính xác cho mục tiêu nhỏ ($AP_{s}$) tăng vọt **2.2 AP** (từ 22.6 lên 24.8)[cite: 322, 326].

### 5.2. Hiệu suất trên tập Urised11 (Tập dữ liệu Cặn nước tiểu)
* [cite_start]DP-YOLO xử lý kích thước ảnh $[640, 640]$ đạt tổng **49.2 AP**, vượt trội hơn tất cả các phương pháp SOTA khác có kích cỡ mạng tương đương như YOLOV3 (43.0 AP), Faster RCNN (45.6 AP), Retinanet (47.1 AP), FCOS (47.2 AP), SSD (48.4 AP), YOLOX (48.7 AP) và YOLOv5 nguyên bản (48.2 AP)[cite: 401].
* **Phân tích từng danh mục có khác biệt lớn (Intra-class variance):**
  [cite_start]DP-YOLO cho thấy sự cải thiện đáng kể ở 7 danh mục khó nhất so với YOLOv5[cite: 403]. Cụ thể (so sánh mức tăng $AP_{50}$):
    * [cite_start]Tinh trùng (Sperm): Tăng mạnh **+8.4** (từ 58.9 lên 67.3)[cite: 411].
    * [cite_start]Tế bào biểu mô không vảy (Epithn): Tăng **+6.0** (từ 49.8 lên 55.8)[cite: 411].
    * [cite_start]Nấm (Mycete): Tăng **+4.3** (từ 56.8 lên 61.1)[cite: 411].

---

## 6. Kết Luận
* [cite_start]Bằng cách điều chỉnh giới hạn vùng mở rộng mẫu theo dạng cánh hoa (PSA) và khéo léo kết hợp các toán tử DCNv2/DCNv3 ở các tầng khác nhau của mạng xương sống (DYB), DP-YOLO đã khắc phục triệt để hạn chế của YOLOv5 trong việc nhận diện mục tiêu biến dạng, thay đổi tư thế[cite: 440, 441].
* [cite_start]Thành tựu quan trọng nhất là thuật toán tăng cường năng lực phát hiện một cách vượt trội mà không đánh đổi bằng chi phí tính toán khổng lồ, chứng minh tính khả thi ứng dụng thực tiễn cao, đặc biệt trong các lĩnh vực phân tích vi thể y khoa phức tạp[cite: 442, 444].