# Phân Tích Chi Tiết Mô Hình DP-YOLO: Phát Hiện Sâu Bệnh

## I. Thông Tin Tổng Quan
* [cite_start]**Mục tiêu chính:** Cải thiện độ chính xác phát hiện sâu bệnh kích thước nhỏ trong khi vẫn giữ mô hình ở dạng hạng nhẹ (lightweight) để triển khai trên các thiết bị nhúng[cite: 18, 23, 35].

---

## II. Bối Cảnh và Thách Thức Nghiên Cứu

## III. Các Cải Tiến Cốt Lõi (Dẫn chứng chi tiết)

### 1. Thay đổi cấu trúc lớp phát hiện (Detection Layers)
* [cite_start]**Giải pháp:** Loại bỏ lớp phát hiện vật thể lớn (P5) và thêm lớp phát hiện vật thể cực nhỏ (P2)[cite: 18, 225].
* [cite_start]**Dẫn chứng:** * Kích thước bản đồ đặc trưng cho các đầu phát hiện mới là $4 \times 4$, $8 \times 8$, và $16 \times 16$ pixel[cite: 184, 229].
    * [cite_start]Việc này giúp mạng tập trung vào các đặc trưng mịn (fine-grained) của biển báo nhỏ thay vì các vật thể lớn không cần thiết trong bối cảnh này[cite: 230, 234].

### 2. Mô-đun DBBNCSPELAN4 (Backbone)
* [cite_start]**Giải pháp:** Kết hợp cấu trúc CSPNet và ELAN với kỹ thuật tích chập tái tham số hóa đa nhánh (Diverse Branch Block - DBB)[cite: 57, 58, 240].
* [cite_start]**Dẫn chứng:** * Sử dụng 4 nhánh tích chập (bao gồm $1 \times 1$, $3 \times 3$, và average pooling) trong quá trình huấn luyện[cite: 247, 248].
    * [cite_start]Khi suy luận (inference), các nhánh này được gộp lại thành một phép toán tích chập $3 \times 3$ duy nhất thông qua 6 phương pháp chuyển đổi[cite: 249]. 
    * [cite_start]Kết quả: Tăng khả năng trích xuất đặc trưng mà không làm tăng tải trọng tính toán khi chạy thực tế[cite: 244, 249].

### 3. Mô-đun PTCSP (Neck)
* [cite_start]**Giải pháp:** Đề xuất mô-đun Partially Transformer - Cross Stage Partial (PTCSP)[cite: 20, 186, 297].
* [cite_start]**Dẫn chứng:** * Chia kênh đầu vào thành 2 phần: một phần xử lý bằng khối CNN truyền thống, phần còn lại xử lý qua cấu trúc Transformer (MHSA_CGLU)[cite: 298, 303, 305].
    * [cite_start]Tích hợp cơ chế Multi-Head Self-Attention (MHSA) để nắm bắt ngữ cảnh toàn cục và Convolutional GLU (CGLU) để tăng cường khả năng mô hình hóa cục bộ[cite: 305, 308, 310].

### 4. Hàm mất mát W3F_MPDIOU (Loss Function)
* [cite_start]**Giải pháp:** Kết hợp ưu điểm của MPDIoU, Focaler-IoU và WIoU v3[cite: 21, 62, 187].
* [cite_start]**Dẫn chứng:** Giúp hội tụ nhanh hơn và giảm tác động của các mẫu dữ liệu chất lượng thấp (low-quality samples)[cite: 62, 187, 466].

---

## IV. Các Công Thức Toán Học Chính

### 1. Hồi quy hộp giới hạn (Bounding Box Regression)
* **Khoảng cách điểm tối thiểu (MPDIoU):**
    * [cite_start]Khoảng cách góc trên bên trái: $d_{1}^{2}=(x_{1}-x_{1}^{gt})^{2}+(y_{1}-y_{1}^{gt})^{2}$ [cite: 393]
    * [cite_start]Khoảng cách góc dưới bên phải: $d_{2}^{2}=(x_{2}-x_{2}^{gt})^{2}+(y_{2}-y_{2}^{gt})^{2}$ [cite: 395]
    * [cite_start]Giá trị MPDIoU: $MPDIoU=IoU-\frac{d_{1}^{2}}{w^{2}+h^{2}}-\frac{d_{2}^{2}}{w^{2}+h^{2}}$ [cite: 405]
    * [cite_start]Loss MPDIoU: $L_{MPDIoU}=1-MPDIoU$ [cite: 405]

* **Cơ chế hội tụ trọng tâm (Focaler-IoU):**
    * [cite_start]$IoU^{focaler}=\begin{cases}\frac{IoU-d}{u-d}, \\ 1, \end{cases}$ [cite: 462]
    * [cite_start]$L_{F\_MPDIoU}=L_{MPDIoU}+IoU-IoU^{focaler}$ [cite: 462]

* **Tối ưu trọng số mẫu (W3F_MPDIOU):**
    * $L_{W3F\_MPDIoU}=rR_{WIoU}L_{F\_MPDIoU}$ [cite: 470]
    * [cite_start]Hệ số lấy nét: $r=\frac{\beta}{\delta\alpha\beta-\delta}$ [cite: 470]
    * [cite_start]Khoảng cách tâm chuẩn hóa: $R_{WIoU}=exp\left(\frac{(x-x_{gl})^{2}+(y-y_{gl})^{2}}{(W_{g}^{2}+H_{gl}^{2})^{*}}\right)$ [cite: 471]

### 2. Chỉ số đánh giá (Evaluation Metrics)
* [cite_start]**Precision (Độ chính xác):** $Precision=\frac{TP}{TP+FP}$ [cite: 591]
* **Recall (Độ thu hồi):** $Recall=\frac{TP}{TP+FN}$ [cite: 591]
* [cite_start]**mAP (Độ chính xác trung bình toàn cục):** $mAP=\frac{1}{n}\sum_{i=1}^{n}AP_{i}$ [cite: 595]