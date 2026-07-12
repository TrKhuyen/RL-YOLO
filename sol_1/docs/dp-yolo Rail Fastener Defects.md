# BÁO CÁO PHÂN TÍCH CHI TIẾT BÀI BÁO: DP-YOLO

**Thông tin chung về bài báo:**
* [cite_start]**Tên bài báo:** DP-YOLO: A Lightweight Real-Time Detection Algorithm for Rail Fastener Defects[cite: 4, 5].
* [cite_start]**Tác giả:** Lihua Chen, Qi Sun, Ziyang Han và Fengwen Zhai[cite: 6].
* [cite_start]**Nguồn xuất bản:** Tạp chí Sensors (MDPI) [cite: 1, 2][cite_start], xuất bản ngày 28 tháng 3 năm 2025[cite: 9].
* [cite_start]**Mục tiêu nghiên cứu:** Đề xuất DP-YOLO, một thuật toán phát hiện lỗi chốt kẹp ray xe lửa theo thời gian thực, có dung lượng nhẹ dựa trên YOLOv5s, nhằm hoạt động hiệu quả trong các môi trường hạn chế về tài nguyên[cite: 21].

---

## 1. Bốn cải tiến cốt lõi của DP-YOLO

[cite_start]Thuật toán DP-YOLO cải tiến mạng YOLOv5s gốc thông qua bốn chiến lược tối ưu hóa chính[cite: 21].

### 1.1. Mô-đun DSP (Depthwise Separable Convolution Stage Partial)
* [cite_start]**Mục đích:** Giảm số lượng tham số của mô hình trong khi vẫn tăng cường độ chính xác nhận dạng[cite: 22].
* [cite_start]**Cấu trúc:** Tác giả thiết kế mô-đun W3_D để thay thế cho cấu trúc Bottleneck trong mô-đun C3 của YOLOv5s gốc[cite: 74].
* **Thành phần chi tiết của W3_D:**
    * Tích chập tách chiều sâu (Depthwise Separable Convolution - DSC) $1\times1$: Dùng để giảm tham số và thực hiện dung hợp kênh[cite: 182].
    * [cite_start]Tích chập DSC $3\times3$: Duy trì khả năng trích xuất đặc trưng trong khi giảm đáng kể chi phí tính toán[cite: 183].
    * [cite_start]Mô-đun tích chập thông thường: Tối ưu hóa thêm việc trích xuất đặc trưng[cite: 184].

### 1.2. Cơ chế chú ý PSCA (Position-Sensitive Channel Attention)
* **Mục đích:** Cải thiện khả năng nhận thức các đặc trưng quan trọng của mô hình bằng cách kết hợp thông tin không gian và kênh[cite: 208].
* [cite_start]**Nguyên lý hoạt động:** PSCA tính toán các thống kê không gian (giá trị trung bình và độ lệch chuẩn) trên cả hai chiều chiều cao và chiều rộng cho mỗi bản đồ đặc trưng kênh[cite: 23]. [cite_start]Các thống kê này được nhân với nhau trên các chiều tương ứng để tạo ra trọng số riêng cho từng kênh[cite: 24].
* **Các công thức tính toán cốt lõi:**
    * Công thức độ lệch chuẩn tổng quát:
      [cite_start]$$std=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_{i}-\mu)^{2}}$$ [cite: 214]
    * Độ lệch chuẩn của hai chiều cho từng kênh:
      [cite_start]$$\sigma_{c}^{H}=\sqrt{\frac{1}{H}\sum_{j=1}^{H}(x_{c}(h,j)-\mu_{h})^{2}}$$ [cite: 226]
      [cite_start]$$\sigma_{c}^{w}=\sqrt{\frac{1}{W}\sum_{i=1}^{W}(x_{c}(i,w)-\mu_{w})^{2}}$$ [cite: 226]

### 1.3. Mô-đun C3Ghost trong phần Neck
* [cite_start]**Mục đích:** Giảm thiểu sự dư thừa thông qua các phép toán tuyến tính, từ đó giảm chi phí tính toán[cite: 25].
* **Nguyên lý hoạt động:** Mô-đun Ghost chia bản đồ đặc trưng đầu vào thành hai phần: "Đường dẫn chính" trích xuất đặc trưng chính và "Đường dẫn Ghost" trích xuất đặc trưng phụ[cite: 308]. Hai bản đồ đặc trưng này sau đó được dung hợp bằng phép cộng nối[cite: 309].

### 1.4. Hàm mất mát Alpha-IoU
* [cite_start]**Mục đích:** Cải thiện khả năng thích ứng đa tỷ lệ và tăng cường độ mạnh mẽ (robustness) của mô hình bằng cách thay thế hàm mất mát tiêu chuẩn[cite: 26].
* **Công thức Alpha-IoU:**
    * Hàm cơ bản:
      [cite_start]$$L_{\alpha-IoU}=\frac{1-IoU^{\alpha}}{\alpha},\alpha>0$$ [cite: 339]
    * Công thức hàm mất mát hồi quy tối ưu hóa:
      [cite_start]$$L_{\alpha-CIoU}=1-IoU^{\alpha}+\frac{\rho^{2\alpha}(b,b^{\rho^{t}})}{c^{2\alpha}}+(\beta v)^{\alpha}$$ [cite: 344]

---

## 2. Dữ liệu thí nghiệm và Tiêu chí đánh giá

### 2.1. Tập dữ liệu (Dataset)
* [cite_start]**Nguồn gốc:** Tập dữ liệu Fastener-defect-detection trên Roboflow Universe[cite: 84].
* **Quy mô ban đầu:** 2234 hình ảnh (2061 ảnh huấn luyện, 173 ảnh kiểm tra)[cite: 87].
* [cite_start]**Phân loại:** Gồm 6 danh mục: normal fasteners (fastener, fastener_2), defective fasteners (fastener_broken, fastener2_broken), foreign objects (tracked_stuff), và missing fasteners (missing)[cite: 367].
* [cite_start]**Tăng cường dữ liệu:** Mỗi hình ảnh được tăng cường bằng ba phương pháp chọn ngẫu nhiên (thêm nhiễu, đổi độ sáng, cắt, dịch chuyển, xoay, lật gương, cutout)[cite: 88, 89].
* **Quy mô sau tăng cường:** 8936 hình ảnh (6520 ảnh huấn luyện, 2416 ảnh kiểm tra)[cite: 86].

### 2.2. Tiêu chí đánh giá
* **Công thức tính Precision (Độ chính xác):**
  [cite_start]$$Precision = \frac{TP}{TP + FP}$$ [cite: 376]
* **Công thức tính Recall (Độ phủ):**
  [cite_start]$$Recall = \frac{TP}{TP + FN}$$ [cite: 377]
* **Công thức tính Average Precision (AP):**
  $$AP=\int_{0}^{1}P(R)dR$$ [cite: 383]

---

## 3. Kết quả Thí nghiệm và Phân tích

### 3.1. Thí nghiệm Cắt bỏ (Ablation Experiment)
* [cite_start]**Kết quả tổng thể:** DP-YOLO đạt độ chính xác phát hiện là 87.1% [cite: 27] [cite_start]với tốc độ 92 FPS[cite: 410].
* **So sánh với Baseline (YOLOv5s gốc):**
    * [cite_start]mAP0.5 tăng 1.3%[cite: 27, 411].
    * [cite_start]mAP0.5:0.95 tăng 2.1% [cite: 27] [cite_start](hoặc 2.8% theo bảng phân tích cấu phần [cite: 411]).
    * [cite_start]Số lượng tham số giảm 1.3%[cite: 28, 411].
    * Tải lượng tính toán giảm 15.19%[cite: 28, 411].
    * [cite_start]Kích thước mô hình giảm 6.25%[cite: 411].

### 3.2. So sánh với các thuật toán khác
* [cite_start]**Hiệu suất chung:** Thuật toán đạt độ chính xác cao hơn các thuật toán so sánh như Faster R-CNN, Cascade R-CNN, SSD, YOLOX, CenterNet, và YOLOv7[cite: 457, 458]. [cite_start]Nó cũng vượt trội so với các phiên bản YOLOv5 nhẹ khác[cite: 464].
* [cite_start]**Trường hợp ngoại lệ ("missing"):** Độ chính xác mAP0.5 trên lớp "missing" thấp hơn do thuật toán và baseline đều gặp khó khăn khi tập dữ liệu không phân biệt rạch ròi sự vắng mặt của loại chốt kẹp "fastener" hay "fastener_2"[cite: 465, 468]. [cite_start]Việc cải thiện nhận dạng đa tỷ lệ ở DP-YOLO dẫn đến suy giảm khả năng khái quát hóa ở riêng lớp dữ liệu này[cite: 469].

---

## 4. Kết luận và Hướng phát triển
* **Thành tựu:** DP-YOLO có giá trị thực tiễn cao cho việc phát hiện lỗi chính xác, tiết kiệm tài nguyên trong hệ thống bảo trì đường sắt[cite: 29]. Mô hình xử lý tốt hơn các cảnh phức tạp, mục tiêu đa tỷ lệ và bị che khuất[cite: 473].
* [cite_start]**Hạn chế:** Chỉ số $mAP_{0.5:0.95} (\le 72.4\%)$ vẫn chưa tối ưu do sự mất cân bằng lớp dữ liệu, nhiễu nền và khả năng khái quát hóa hạn chế trên các lỗi hiếm gặp[cite: 494].
* **Hướng nghiên cứu tương lai:** Ưu tiên học bán giám sát (SSL) và học ít mẫu (FSL) để giải quyết mất cân bằng dữ liệu, đồng thời kết hợp thông tin ngữ cảnh đa tỷ lệ để triệt tiêu nhiễu nền[cite: 495, 496].