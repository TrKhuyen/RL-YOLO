# Kịch bản chi tiết slide báo cáo tiến độ RL-YOLO

**Đề tài:** Tối ưu mô hình YOLO phát hiện lá sâu bệnh bằng học tăng cường và xây dựng website nhận diện.  
**Ngày soạn:** 03/11/2026. **Quy mô đề xuất:** 32 slide, khoảng 30–35 phút; thời lượng tùy phần trực quan hóa do nhóm bổ sung.  
**Sản phẩm của tài liệu:** nội dung có thể đưa lên slide, vị trí từng vùng, hình/bảng cần dựng và lời trình bày. Đây là kịch bản, chưa phải file PowerPoint.

## Quy ước thiết kế và sử dụng

- Khung 16:9. Tọa độ bên dưới tính theo phần trăm khung slide: `(x, y, rộng, cao)`, gốc ở góc trên trái. Các vùng đã chừa khoảng cách; không cần hiển thị tọa độ trên slide.
- Nền trắng ngà; chữ xanh đen. Dữ liệu dùng xanh lá; KB1 xanh dương; KB2 tím; KB3 cam. Màu nhạt để phân biệt khối, không dùng nền đậm cho bảng nhiều số.
- Tiêu đề thường ở `(5, 4, 90, 10)`, 30–34 pt. Nội dung 20–24 pt, bảng 18–20 pt; chú thích 14–16 pt. Khi bảng chật, giảm số cột, không thu chữ dưới 18 pt.
- Dải thông điệp ở `(5, 84, 90, 8)`: một câu kết luận. Chân trang ở `(5, 94, 90, 4)`: nguồn rút gọn bên trái, số slide bên phải. Nội dung trong mục **Lời trình bày** chỉ để speaker notes.
- Các nhãn `[Tên nhóm]`, `[Thành viên]`, `[GVHD]` là trường cần điền. Hai slide 04 và 07 dành riêng cho người trình bày tự thêm dữ liệu/hình.
- Số AP/P/R/F1 dùng %. Chênh lệch dùng **điểm phần trăm**, ký hiệu “đpt”; dấu chấm là dấu thập phân trong các bảng để khớp report. Mean ± SD là trung bình ± độ lệch chuẩn mẫu giữa seed, không phải khoảng tin cậy.
- Không tự đồng nhất kết quả giữa ba KB: KB1 có test; KB2 hiện là validation; KB3 là validation tại snapshot 15/10/2026. Không dựng biểu đồ xếp hạng ba phương pháp bằng các số thuộc các split/protocol khác nhau.

## Nguồn và mốc bằng chứng

| Mã nguồn trên slide | Tài liệu | Phạm vi dùng |
|---|---|---|
| R1 | [Report_KB1_15_10_2026.md](Report_KB1_15_10_2026.md) | Bản cập nhật trình bày 03/11; kết quả KB1 tổng hợp đến 26/10 |
| R2 | [Report_KB2_DPO_20261103.md](Report_KB2_DPO_20261103.md) | Conditional keep/drop DPO; xác nhận ba seed ngày 03/11 |
| R3 | [Report_KB3_15_10_2026.md](Report_KB3_15_10_2026.md) | Snapshot 15/10/2026 lúc 02:31:17 +07:00; không đại diện tiến độ chạy trực tiếp ngày 03/11 |

Chỉ dùng ba report được yêu cầu làm nguồn chính. README tổng quan có các nhận xét lịch sử, không dùng thay trạng thái mới trong R1/R2/R3. Các đề xuất website cuối bài được ghi rõ là kế hoạch vì ba report chưa xác nhận tiến độ triển khai website.

---

## Slide 01 — Tên đề tài

**Mục đích:** mở đầu bằng đúng phạm vi nghiên cứu và sản phẩm ứng dụng. **Thời lượng:** 30 giây.

**Thiết kế và nội dung hiển thị**

- Góc trên trái `(6, 6, 65, 6)`: “BÁO CÁO TIẾN ĐỘ ĐỀ TÀI”.
- Vùng chính `(6, 23, 76, 30)`, 34–38 pt, chia ba dòng:
  > TỐI ƯU MÔ HÌNH YOLO  
  > PHÁT HIỆN LÁ SÂU BỆNH BẰNG HỌC TĂNG CƯỜNG  
  > VÀ XÂY DỰNG WEBSITE NHẬN DIỆN
- Dưới tiêu đề `(6, 59, 72, 12)`: “Dữ liệu PlantDOC · Ba kịch bản tối ưu · Định hướng tích hợp website”.
- Góc dưới trái `(6, 77, 80, 14)`: `[Tên nhóm] — [Thành viên]` / `Giảng viên hướng dẫn: [GVHD]` / `Ngày báo cáo: [điền ngày trình bày]`.
- Dải trang trí phải `(87, 20, 7, 60)`: biểu tượng lá, khung bounding box, trình duyệt xếp dọc; chỉ là đồ họa khái niệm, không dùng như ảnh kết quả.

**Lời trình bày:** “Nhóm nghiên cứu tối ưu detector YOLO cho bài toán phát hiện lá sâu bệnh, đồng thời hướng tới website để người dùng gửi ảnh và xem kết quả. Báo cáo hôm nay đi từ chuẩn bị dữ liệu đến ba kịch bản tối ưu và công việc tiếp theo.”

**Nguồn:** tên đề tài theo yêu cầu; R1–R3 cho ba hướng nghiên cứu.

---

## Slide 02 — Chuẩn bị dữ liệu: lựa chọn PlantDOC

**Mục đích:** giới thiệu dữ liệu ngay sau tên đề tài. **Thời lượng:** 60 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “Chuẩn bị dữ liệu: nhóm lựa chọn PlantDOC”.
- Cột trái `(5, 19, 42, 56)`: ba khối:
  1. **Dữ liệu được chọn:** “PlantDOC; sử dụng bản dữ liệu đã làm sạch trong dự án”.
  2. **Đơn vị phát hiện:** “Bounding box của đối tượng lá; nhãn kết hợp loại cây và tình trạng khỏe/bệnh”.
  3. **Định dạng học:** “YOLO: class, cx, cy, width, height; 28 class”.
- Cột phải `(52, 19, 43, 56)`: luồng năm hộp dọc, nối mũi tên: “Dữ liệu PlantDOC” → “Kiểm tra ảnh–nhãn” → “Làm sạch và kiểm tra nguồn ảnh” → “Cố định train/validation/test” → “Huấn luyện và đánh giá”.
- Dải thông điệp: “Số liệu sau đây mô tả bản dữ liệu của dự án, không phải quy mô nguyên bản của PlantDOC.”

**Lời trình bày:** “Nhóm đã chọn PlantDOC và sử dụng bản làm sạch của dự án. Bài toán là detection: mô hình cần trả vị trí lá và class tương ứng, thay vì chỉ dự đoán một nhãn cho toàn ảnh. Các thống kê tiếp theo đều thuộc bản dữ liệu đang dùng.”

**Nguồn:** R1 §2; R2 §9.1; R3 §4.1. Không thêm khẳng định về số ảnh nguyên bản hoặc số loại cây khi chưa có nguồn trong ba report.

---

## Slide 03 — Chia dữ liệu và kiểm soát rò rỉ

**Thời lượng:** 65 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “Bộ dữ liệu thực nghiệm: 3.129 ảnh, 28 class”.
- Bảng trái `(5, 21, 55, 48)`:

| Split | Ảnh | Bounding box | Vai trò |
|---|---:|---:|---|
| Train | 2.411 | 10.333 | Học trọng số; sinh preference KB2 |
| Validation | 479 | 1.887 | Chọn cấu hình và checkpoint |
| Test | 239 | 1.063 | Đánh giá sau khi khóa lựa chọn |
| Tổng | 3.129 | 13.283 | 28 class |

- Phải `(65, 21, 30, 49)`: ba thẻ:
  - “Train: 667 ảnh có hậu tố augmentation offline được nhận diện”.
  - “Validation/test: không có các dạng augmentation offline đã nhận diện”.
  - “Audit không phát hiện source ID chung giữa ba split theo quy tắc bỏ hậu tố augment”.
- Ghi chú `(5, 73, 90, 8)`: “Kiểm tra tên/source ID chưa loại trừ mọi near-duplicate khác ID; số ảnh không đồng nghĩa số nguồn độc lập.”
- Dải thông điệp: “Validation để lựa chọn; test để đánh giá lựa chọn đã khóa.”

**Lời trình bày:** “Bản dữ liệu có hơn mười ba nghìn box. Nhóm tách vai trò của từng split và kiểm tra nguồn ảnh nhằm hạn chế rò rỉ. Một phần train là biến thể augment nên không thể xem tất cả ảnh hoặc box là các quan sát độc lập.”

**Nguồn:** R1 §2.1–2.2; R2 §9.1.

---

## Slide 04 — Trực quan hóa dữ liệu trước augmentation [NHÓM TỰ BỔ SUNG]

**Thời lượng dự kiến:** 60–90 giây sau khi bổ sung.

**Thiết kế giữ chỗ; chưa điền số liệu hoặc nhận xét thay nhóm**

- Tiêu đề: “Trực quan hóa dữ liệu trước augmentation”.
- Trái `(5, 20, 43, 51)`: `[CHÈN BIỂU ĐỒ PHÂN BỐ CLASS — NHÓM TỰ THÊM]`.
- Phải `(52, 20, 43, 51)`: `[CHÈN ẢNH DỮ LIỆU CÓ BOUNDING BOX — NHÓM TỰ THÊM]`.
- Dải dưới `(5, 74, 90, 15)`: `[NHẬN XÉT CỦA NHÓM: class ít mẫu / ảnh nhiều lá / nền / kích thước đối tượng]`.
- Chân trang: `[Nguồn ảnh và cách tính biểu đồ — nhóm điền]`.

**Hướng dẫn dựng:** biểu đồ phải ghi rõ đếm ảnh hay box và thuộc split nào. Ảnh nên có chú giải màu/class. Không chèn giá trị giả vào biểu đồ.

**Lời trình bày để điền:** “Biểu đồ cho thấy [nhận xét đã kiểm chứng]. Các ảnh bên phải minh họa [đặc điểm nhìn thấy]. Từ đó nhóm lựa chọn augmentation nhằm [mục tiêu phù hợp].”

---

## Slide 05 — Augmentation: mục tiêu và hai hình thức

**Thời lượng:** 60 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “Augmentation để tăng biến thiên dữ liệu huấn luyện”.
- Trái `(5, 21, 43, 51)`, nhãn **Offline — biến thể được lưu**:
  - “Trong train có 667 ảnh nhận diện bằng hậu tố flip/rotation”.
  - “1.744 file train còn lại không mang các hậu tố augment đã biết”.
  - “Ảnh và nhãn biến đổi phải cùng thuộc split train”.
- Phải `(52, 21, 43, 51)`, nhãn **Online — biến đổi khi đọc ảnh**:
  - “Tạo input khác nhau qua các lượt huấn luyện”.
  - “Kết hợp thay đổi hình học và quang học”.
  - “Recipe phụ thuộc kịch bản; không áp một pipeline ngẫu nhiên cho mọi KB”.
- Ghi chú `(5, 74, 90, 7)`: “667 là số ảnh có hậu tố augment được audit, không phải số ảnh mới sinh trong một lượt online.”
- Dải thông điệp: “Tăng biến thiên của input; chưa có ablation để quy gain AP cho riêng từng phép augment.”

**Lời trình bày:** “Nhóm phân biệt ảnh augment đã lưu trên đĩa với augmentation online. Online giúp cùng một ảnh nguồn xuất hiện dưới nhiều điều kiện, nhưng không làm tăng số nguồn chụp độc lập. Các kịch bản có yêu cầu riêng nên recipe cũng khác nhau.”

**Nguồn:** R1 §2.1, §2.3; R2 §8.1.

---

## Slide 06 — Phương pháp augmentation và bảo toàn bounding box

**Thời lượng:** 90 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “Phương pháp augmentation: biến đổi ảnh đi cùng nhãn”.
- Trái `(5, 19, 61, 54)`: bảng recipe **online stage hai KB1**, ghi rõ ngay trên bảng:

| Nhóm | Phép biến đổi | Xác suất / phạm vi |
|---|---|---|
| Chuẩn hóa kích thước | Resize cạnh dài, pad 114 | 640 × 640; luôn áp dụng |
| Ánh sáng | Brightness/contrast | p=0.5; giới hạn ±0.3 |
| Màu sắc | Hue/Saturation/Value | p=0.4; ±15 / ±30 / ±20 |
| Nhiễu | Gaussian noise | p=0.3; std 0.02–0.12 |
| Mờ | Motion blur | p=0.2; kernel 3 hoặc 5 |
| Hình học | Lật ngang / lật dọc | p=0.5 / p=0.3 |
| Hình học | Xoay | p=0.4; ±15° |
| Tương phản cục bộ | CLAHE | p=0.2; clip limit 1–2 |

- Phải `(70, 19, 25, 54)`: ba hộp dọc:
  - “BBox YOLO được biến đổi đồng bộ, clipping theo biên”.
  - “KB1 train: loại box có phần diện tích còn nhìn thấy < 0.3”.
  - “Normalize [0,1] → tensor”.
- Ghi chú `(5, 75, 90, 7)`: “Val/test: resize/pad xác định. KB2 hiện tại: không random augment để giữ đúng slot/box của preference bank. KB3 dùng recipe native của trial.”
- Dải thông điệp: “Các xác suất xét theo từng phép; một ảnh có thể nhận nhiều phép.”

**Lời trình bày:** “Đây là recipe cụ thể của stage hai KB1, gồm màu sắc, ánh sáng, nhiễu, mờ và hình học. Với detection, lật hoặc xoay ảnh phải biến đổi box tương ứng. KB2 không áp trực tiếp pipeline này vì cặp preference được gắn với slot và box đã lưu.”

**Nguồn:** R1 §2.3, Bảng 3; R2 §8.1; R3 §8.7. GaussNoise dùng độ lệch chuẩn tương đối, không ghi nhầm thành variance.

---

## Slide 07 — Trực quan hóa sau augmentation [NHÓM TỰ BỔ SUNG]

**Thời lượng dự kiến:** 60 giây.

**Thiết kế giữ chỗ**

- Tiêu đề: “Trực quan hóa ảnh và nhãn sau augmentation”.
- Một hàng bốn ô `(5, 22, 90, 45)`, mỗi ô rộng 21%, khe 2%:
  1. `[ẢNH GỐC + BOX]`.
  2. `[BIẾN THỂ HÌNH HỌC + BOX]`.
  3. `[BIẾN THỂ QUANG HỌC + BOX]`.
  4. `[BIẾN THỂ KẾT HỢP + BOX]`.
- Nhãn dưới mỗi ô, y=69: `[Tên phép và tham số thực tế — nhóm điền]`.
- Dải dưới `(5, 77, 90, 12)`: `[NHẬN XÉT CỦA NHÓM VỀ ẢNH VÀ ĐỘ KHỚP BOUNDING BOX]`.

**Hướng dẫn dựng:** cùng một ảnh train để dễ so; giữ tỉ lệ ảnh; ghi recipe và seed nếu cần tái tạo. Đây là vị trí nhóm tự thêm, không yêu cầu tạo hình AI.

**Lời trình bày để điền:** “So với ảnh gốc, các biến thể thay đổi [đặc điểm]. Bounding box sau biến đổi [nhận xét nhóm đã kiểm tra].”

---

## Slide 08 — Ba kịch bản tối ưu: khác nhau ở đối tượng ra quyết định

**Thời lượng:** 70 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “Ba hướng tối ưu được nghiên cứu độc lập”.
- Ba cột `(5, 22, 28, 48)`, `(36, 22, 28, 48)`, `(67, 22, 28, 48)`:
  - **KB1 — Reward-guided**: “Fine-tune trọng số Detect head bằng native loss, proxy và task reward”. Dòng cuối: “RL-inspired; có kết quả test”.
  - **KB2 — Preference/DPO**: “Học ưu tiên giữ/bỏ candidate dựa trên GT và reference cố định”. Dòng cuối: “DPO; hiện có kết quả validation”.
  - **KB3 — Hyperparameter**: “A: chọn cấu hình bằng TPE. B: điều chỉnh trong train bằng PPO”. Dòng cuối: “A là HPO; B là hướng RL”.
- Dải ngang `(5, 74, 90, 8)`: “KB1B và KB2 bắt đầu từ S0 tương ứng; KB3-A khởi tạo từ pretrained gốc, dùng best supervised KB1 làm đối chứng.”
- Dải thông điệp: “Thứ tự trình bày KB1 → KB2 → KB3 không phải chuỗi fine-tune nối tiếp.”

**Lời trình bày:** “Ba KB giải quyết ba cách tác động lên detector. Đặc biệt, KB2 không nối tiếp guided checkpoint của KB1. KB3-A huấn luyện trial mới từ nguồn pretrained gốc và sử dụng supervised KB1 làm mốc so sánh.”

**Nguồn:** R1 §1, §3; R2 §1.1; R3 §3.

---

## Slide 09 — KB1: mục tiêu và câu hỏi nghiên cứu

**Thời lượng:** 65 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB1 — Tối ưu thêm YOLO sau supervised training”.
- Trái `(5, 22, 36, 51)`: **Mục tiêu**: “Kiểm chứng quy trình hai giai đoạn để cải thiện detector so với checkpoint supervised của chính kiến trúc đó”. Dưới là nhãn “Metric chính: canonical mAP50–95”.
- Phải `(46, 22, 49, 51)`: hai thẻ lớn:
  - **RQ1:** “Thêm reward-guided fine-tuning cải thiện bao nhiêu so với S0 tương ứng?”
  - **RQ2:** “Trong các guided checkpoint chọn bằng validation, kiến trúc nào đạt chất lượng tốt nhất?”
- Dải thông điệp: “Câu hỏi bổ sung: phần tăng thêm đến từ continuation, proxy hay reward?”

**Lời trình bày:** “KB1 đo hiệu quả của cả quy trình hai giai đoạn, đồng thời dùng đối chứng để phân tích đóng góp thành phần. Nhóm chọn checkpoint bằng validation rồi mới báo kết quả test.”

**Nguồn:** R1 §1.

---

## Slide 10 — KB1: ý tưởng và objective

**Thời lượng:** 90 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB1 — Dùng task reward để điều chỉnh fine-tuning”.
- Trên `(5, 20, 90, 18)`: bốn hộp ngang: “Prediction trên train” → “Matching với GT” → “Reward P/R/IoU” → “Advantage và loss”.
- Trái `(5, 44, 44, 32)`:
  - “Reward tổng hợp precision, recall, IoU tại các ngưỡng 0.5/0.6/0.7/0.8”.
  - “Trọng số P/R/IoU: 0.25 / 0.45 / 0.30”.
  - “Advantage chuẩn hóa theo EMA reward baseline”.
- Phải `(53, 44, 42, 32)`: công thức lớn:
  > L = 0.25 L_native + 0.10 L_reward  
  > + 0.25 L_proxy + 0.001 L_SP

  Chú giải: “Proxy: confidence TP/FP/FN; L_SP: hạn chế lệch trọng số khỏi S0”.
- Dải thông điệp: “Tên chính xác: reward-guided / RL-inspired; chưa phải REINFORCE chuẩn.”

**Lời trình bày:** “Reward phản ánh chất lượng detection theo GT của train. Nó được dùng để điều chỉnh đóng góp của confidence surrogate, kết hợp native loss và regularization. Implementation không lấy mẫu một detection policy có log-likelihood đầy đủ, vì vậy nhóm dùng thuật ngữ RL-inspired.”

**Nguồn:** R1 §5.1–5.5. Không dùng reward và mAP như hai đại lượng tương đương.

---

## Slide 11 — KB1: kiến trúc tổng thể

**Thời lượng:** 80 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB1 — Pipeline hai giai đoạn cho từng kiến trúc”.
- Sơ đồ giữa `(5, 21, 90, 40)`, các nút trái sang phải:
  > Pretrained → Supervised trên train → S0 của model → Fine-tune Detect head → Chọn best bằng validation → Test đã khóa
- Từ S0 rẽ xuống ba hộp ở `(31, 65, 18, 12)`, `(53, 65, 18, 12)`, `(75, 65, 18, 12)`:
  - “S0: không train thêm”.
  - “S1: native + L2-SP”.
  - “G: native + proxy + reward + L2-SP”.
- Nhãn trên nút fine-tune: “Backbone/neck và BN statistics đóng băng”.
- Dải thông điệp: “Bảy model, mỗi model dùng S0 riêng; so sánh các nhánh trong cùng model.”

**Lời trình bày:** “Sau supervised, nhóm lưu S0 và tách các nhánh đối chứng. Nhánh S1 cho biết tiếp tục fine-tune có ích đến đâu; nhánh G thêm proxy và reward. Việc giữ nguyên S0 trong từng model giúp phép so có cùng điểm khởi đầu.”

**Nguồn:** R1 §1, §3, §6.

---

## Slide 12 — KB1: thiết kế thực nghiệm đã hoàn tất

**Thời lượng:** 70 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB1 — Bảy kiến trúc và hai lớp đối chứng”.
- Trái `(5, 20, 43, 53)`:
  - **7 model:** YOLOv5s, YOLOv8n, YOLOv8s, YOLOv11n, YOLOv11s, YOLO26n, DP-YOLO.
  - **Continuation:** seed 42/43/44; chọn best bằng validation.
  - **Thống kê chính trình bày:** hai seed Ubuntu mới 43/44.
- Phải `(52, 20, 43, 53)`:
  - **Ablation:** YOLOv11n và YOLOv8s; seed 42/43/44.
  - **Các nhánh:** S1, proxy-only, reward-only, G, G bỏ L2-SP.
  - **Cùng ngân sách:** 10.000 micro-step = 5.000 optimizer update; dùng last checkpoint.
- Dải thông điệp: “Đã hoàn tất kế hoạch bổ sung: 58 lượt train dài; continuation và ablation được tổng hợp riêng.”

**Lời trình bày:** “Cohort chính trả lời hiệu quả trước và sau stage hai. Bộ ablation tập trung ở hai model để kiểm ảnh hưởng từng term tại cùng số update. Các seed continuation đều bắt đầu từ một S0 cố định của từng model, chưa phải nhiều supervised initialization.”

**Nguồn:** R1 §6, §8, Bảng 10–11.

---

## Slide 13 — KB1: kết quả đạt được trên test

**Thời lượng:** 90 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB1 — Guided cải thiện so S0 ở cả bảy model”.
- Bảng trái `(5, 20, 63, 57)`; mean test seed 43/44:

| Model | S0 mAP (%) | G mAP (%) | G − S0 (đpt) |
|---|---:|---:|---:|
| YOLOv5s | 30.398 | 31.261 | +0.863 |
| YOLOv8n | 33.201 | 33.561 | +0.359 |
| YOLOv8s | 33.976 | 35.387 | +1.412 |
| YOLOv11n | 37.534 | 37.768 | +0.234 |
| YOLOv11s | 36.271 | 36.442 | +0.172 |
| YOLO26n | 33.357 | 33.831 | +0.474 |
| DP-YOLO | 28.913 | 29.366 | +0.453 |

- Phải `(72, 20, 23, 57)`: hai thẻ số:
  - **YOLOv11n:** “G validation: 40.648%; G test: 37.768%; dẫn đầu cohort guided theo validation”.
  - **YOLOv8s:** “Gain test lớn nhất: +1.412 đpt”.
- Dải thông điệp: “G > S0 trong 21/21 cặp model–continuation seed đã quan sát, gồm seed 42/43/44.”

**Lời trình bày:** “Bảng dùng riêng hai seed Ubuntu mới. Tất cả model đều tăng so với S0. Model có AP cao nhất và model có gain lớn nhất là hai kết luận khác nhau: YOLOv11n dẫn đầu về chất lượng guided, còn YOLOv8s tăng nhiều nhất so điểm xuất phát.”

**Nguồn:** R1 Bảng 13–14, §16. Số làm tròn ba chữ số; delta lấy từ số gốc. n=2 không đủ khẳng định ý nghĩa thống kê.

---

## Slide 14 — KB1: đóng góp riêng của reward

**Thời lượng:** 75 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB1 — Gain toàn bộ stage hai không chỉ đến từ reward”.
- Trái `(5, 22, 43, 47)`:
  - “G−S0: tác dụng gộp của toàn bộ stage hai”.
  - “G−S1: phần thêm của proxy + reward”.
  - “G−P: phần thêm của reward khi đã có proxy”.
- Phải `(52, 22, 43, 47)`: bảng ablation test, cùng 10.000 micro-step, ba seed:

| Model | G − proxy-only (đpt) |
|---|---:|
| YOLOv11n | +0.070727 |
| YOLOv8s | −0.049986 |

- Ghi chú `(5, 73, 90, 8)`: “Trong cohort continuation, G hơn S1 ở 16/21 cặp; YOLOv11s thấp hơn S1 ở cả ba seed.”
- Dải thông điệp: “Lợi ích riêng của reward phụ thuộc kiến trúc và protocol.”

**Lời trình bày:** “Kết quả ủng hộ quy trình hai giai đoạn nhưng chưa cho phép gán toàn bộ gain cho reward. Khi giữ proxy và cùng ngân sách, reward tăng AP nhỏ ở YOLOv11n nhưng chưa tốt hơn proxy-only ở YOLOv8s.”

**Nguồn:** R1 §10, §16.3.

---

## Slide 15 — KB1: định hướng tương lai

**Thời lượng:** 60 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB1 — Mở rộng bằng chứng và cải thiện objective”.
- Ba hàng ngang từ y=22, mỗi hàng cao 16%, có cột “Việc làm” rộng 49% và “Câu hỏi cần trả lời” rộng 37%:
  1. “Nhiều S0 supervised độc lập; dữ liệu ngoài nguồn hiện có” → “Gain có bền vững theo khởi tạo và nguồn ảnh?”
  2. “Phân tích lỗi theo ảnh/source; paired bootstrap AP” → “Độ bất định của gain nhỏ đến đâu?”
  3. “Chẩn đoán gradient/advantage; cải thiện FN surrogate” → “Reward có tác động đúng candidate bị bỏ sót?”
- Dải thông điệp: “KB1 đã hoàn tất phạm vi hiện tại; các mục này là nghiên cứu mở rộng.”

**Lời trình bày:** “Bước tiếp theo nên tập trung tăng độ chắc chắn của kết luận và hiểu gradient thay vì chỉ tăng số lượt chạy. Hạn chế đáng chú ý là FN surrogate hiện dùng max score toàn ảnh, chưa gắn riêng với từng GT bị bỏ sót.”

**Nguồn:** R1 §14, §16.

---

## Slide 16 — KB2: mục tiêu và câu hỏi nghiên cứu

**Thời lượng:** 65 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB2 — Tối ưu YOLO bằng Direct Preference Optimization”.
- Trái `(5, 22, 37, 51)`: **Mục tiêu:** “Dùng GT tạo phản hồi ưu tiên tự động để học quyết định giữ/bỏ candidate tốt hơn”. Dưới: “Model xác nhận: YOLOv8n; bắt đầu từ S0 supervised KB1”.
- Phải `(47, 22, 48, 51)`, ba thẻ:
  - **RQ1:** “Native + DPO cải thiện gì so S0?”
  - **RQ2:** “Thêm DPO có tốt hơn native-only cùng điều kiện và ngân sách?”
  - **RQ3:** “Lợi ích AP đi kèm đánh đổi nào về FP/FN, recall và class?”
- Dải thông điệp: “Preference do GT chấm tự động; không phải nhãn sở thích của con người.”

**Lời trình bày:** “KB2 chuyển sang học từ cặp ưu tiên. Nhóm cần tách cải thiện do tiếp tục native training khỏi phần tăng thêm do DPO, đồng thời kiểm tra ordinary inference thay vì chỉ nhìn preference loss.”

**Nguồn:** R2 §1–3. Cách đánh số RQ ở slide là cách tổng hợp để trình bày mục tiêu trong report.

---

## Slide 17 — KB2: ý tưởng tạo cặp giữ/bỏ candidate

**Thời lượng:** 85 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB2 — Hai phương án chỉ khác một quyết định”.
- Trái `(5, 21, 58, 50)`: sơ đồ chia đôi từ một nút “Ảnh + context prediction S0”:
  - Nhánh A: “Giữ candidate j” → “NMS toàn ảnh”.
  - Nhánh B: “Bỏ candidate j” → “NMS toàn ảnh”.
  - Hai nhánh hội tụ: “GT gate → chosen / rejected”.
- Phải `(68, 21, 27, 50)`:
  - “Keep: cứu/thực sự giữ GT mục tiêu”.
  - “Drop: thực sự giảm FP”.
  - “Cả hai: không mất GT đang được phát hiện; không tăng FP”.
- Dải thông điệp: “Học quyết định cục bộ; kiểm hậu quả trên toàn ảnh sau NMS.”

**Lời trình bày:** “A/B không phải gọi predict hai lần. Nhóm chủ động can thiệp active-bit của một candidate, giữ phần context còn lại giống nhau rồi chấm toàn ảnh. GT chỉ dùng để tạo và chấm cặp khi train; inference thực tế không cần GT.”

**Nguồn:** R2 §5.1–5.3. Có thể vẽ sơ đồ bằng các ô box đơn giản, ghi “minh họa cơ chế”, không trình bày như kết quả ảnh thật.

---

## Slide 18 — KB2: kiến trúc tổng thể

**Thời lượng:** 90 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB2 — Policy học, reference cố định, native giữ supervision”.
- Sơ đồ `(5, 19, 90, 57)`:
  - Hàng trên: “S0 supervised YOLOv8n” rẽ thành “Policy: học cv3” và “Reference: đóng băng”.
  - Trái giữa: “Ảnh + GT train → tạo A/B → GT gate → preference bank”.
  - Giữa: “Batch: 2 ảnh preference + 2 ảnh uniform train” nối tới policy và reference.
  - Dưới policy: “Native loss: foreground 1; background 0.5”.
  - Giữa policy/reference: “DPO từ logit keep/drop”.
  - Hội tụ: “L = L_native(bg=0.5) + 0.1 L_DPO → AdamW cập nhật cv3”.
- Nhãn `(5, 77, 90, 6)`: “Backbone/neck, regression và BN statistics đóng băng; validation bằng YOLO inference thông thường”.
- Dải thông điệp: “Phiên bản hiện tại học score/giữ-bỏ; chưa trực tiếp học sửa tọa độ box.”

**Lời trình bày:** “Policy và reference cùng bắt đầu từ S0. Chỉ nhánh classification được cập nhật; regression giữ cố định để tập trung đánh giá quyết định giữ hoặc bỏ. Native loss vẫn học GT trên cả bốn ảnh, còn DPO bổ sung tín hiệu trên hai quyết định.”

**Nguồn:** R2 §4, §6, §8.1.

---

## Slide 19 — KB2: policy và thiết kế đối chứng

**Thời lượng:** 80 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB2 — Định nghĩa xác suất rõ ràng và so cùng điều kiện”.
- Trái `(5, 21, 43, 52)`:
  > q(keep)=sigmoid(z); q(drop)=1−sigmoid(z)  
  > m=s(z_policy−z_reference)  
  > L_DPO=softplus(−βm)

  Chú giải: “s=+1: chosen keep; s=−1: chosen drop; β=0.1; hệ số loss λ=0.1”.
- Phải `(52, 21, 43, 52)`:
  - “S0: checkpoint ban đầu cố định”.
  - “Native-only: DPO λ=0; cùng background 0.5”.
  - “Native+DPO: thêm DPO λ=0.1”.
  - “3 seed × 2 nhánh × 15.000 update; trong seed dùng cùng stream ảnh/pair”.
  - “Chọn best bằng validation mAP50–95; so thêm last tại 15k”.
- Dải thông điệp: “Probability chuẩn hóa cho keep/drop, chưa phải likelihood toàn bộ detection set sau NMS.”

**Lời trình bày:** “Policy phụ là Bernoulli của một quyết định. Reference tạo điểm so sánh cố định. Đối chứng native-only dùng cùng pipeline và background weighting, nên chênh lệch giữa hai nhánh phản ánh rõ hơn phần thêm của DPO.”

**Nguồn:** R2 §3.2, §8, §10.1.

---

## Slide 20 — KB2: kết quả đạt được qua ba seed

**Thời lượng:** 85 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB2 — AP cải thiện lặp lại trên validation”.
- Nhãn ngay dưới tiêu đề: “YOLOv8n · validation · seed 42/43/44 · mean ± SD”.
- Trái `(5, 25, 54, 45)`:

| So sánh | Native (%) | DPO (%) | DPO − native (đpt) |
|---|---:|---:|---:|
| Best từng run | 38.1196 ± 0.0608 | 38.1707 ± 0.0429 | +0.0511 ± 0.0202 |
| Cùng last 15k | 38.0595 ± 0.0430 | 38.1670 ± 0.0475 | +0.1075 ± 0.0337 |

  Dưới bảng: “S0 mAP50–95 = 37.8218%; best DPO − S0 = +0.3489 đpt”.
- Phải `(64, 25, 31, 45)`: ba cặp điểm nối native→DPO tại 15k:
  - “Seed 42: 38.0099 → 38.1254”.
  - “Seed 43: 38.0823 → 38.2187”.
  - “Seed 44: 38.0864 → 38.1570”.
- Ghi chú `(5, 74, 90, 7)`: “DPO hơn native về mAP/AP50/AP75 ở cả ba seed; mức tăng thêm còn nhỏ.”
- Dải thông điệp: “Đây là kết quả validation; chưa có đánh giá hiệu năng test của chuỗi KB2 này.”

**Lời trình bày:** “Kết quả tốt xuất hiện ở cả ba seed và vẫn giữ khi so cùng 15.000 update. Tuy nhiên, phần lớn gain so S0 cũng xuất hiện ở native-only. Vì thế đóng góp riêng DPO phải báo bằng delta với native, không phải toàn bộ 0.3489 điểm.”

**Nguồn:** R2 §10.2–10.4. Asset gợi ý: [paired_ap.svg](assets/kb2_strength_multiseed_20261103/paired_ap.svg); có thể dùng thay biểu đồ phải nếu vẫn đủ cỡ chữ.

---

## Slide 21 — KB2: đánh đổi và giới hạn kết quả

**Thời lượng:** 85 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB2 — AP tăng, recall tại ngưỡng cố định giảm”.
- Bảng trái `(5, 22, 54, 48)`, best DPO mean ba seed:

| Metric | S0 | DPO | Thay đổi |
|---|---:|---:|---:|
| Precision (%) | 50.9227 | 53.1581 | +2.2354 đpt |
| Recall (%) | 54.1070 | 53.0825 | −1.0246 đpt |
| F1 (%) | 52.4666 | 53.1200 | +0.6534 đpt |
| FP | 984 | 882.67 | −101.33 |
| FN | 866 | 885.33 | +19.33 |

- Phải `(64, 22, 31, 48)`:
  - “P/R/F1, FP/FN: confidence 0.25; matching IoU 0.5”.
  - “Mean AP theo class: 12 tăng, 16 giảm so S0”.
  - “AP small: 4.3965% → 4.0525%”.
  - “Geometry đóng băng: AP75 tăng chưa chứng minh model học sửa box”.
- Dải thông điệp: “Chất lượng xếp hạng tốt hơn chưa đồng nghĩa ít bỏ sót hơn ở mọi threshold.”

**Lời trình bày:** “Ở ngưỡng 0.25, model giảm FP nhưng tăng FN. AP đo cả quá trình xếp hạng prediction nên có thể tăng dù recall tại một ngưỡng giảm. Cải thiện cũng chưa đồng đều giữa class và kích thước đối tượng.”

**Nguồn:** R2 §10.3, §10.6, §12.

---

## Slide 22 — KB2: định hướng tương lai

**Thời lượng:** 65 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB2 — Ưu tiên kiểm chứng trước khi mở rộng”.
- Ba khối ngang `(5, 22, 28, 51)`, `(36, 22, 28, 51)`, `(67, 22, 28, 51)`:
  1. **Xác nhận:** “Khóa recipe/selector; đánh giá holdout/test cùng S0 và native; thêm S0 độc lập khi cần”.
  2. **Đối chứng objective:** “DPO so BCE có/không reference; kiểm gradient strength; sweep β/λ/background có ngân sách”.
  3. **Mở rộng:** “Bank/class coverage; adapter model khác; policy geometry và augment ánh xạ candidate đúng”.
- Dải thông điệp: “Mục tiêu tiếp theo: gain AP bền vững hơn, giảm đánh đổi recall và cải thiện độ phủ.”

**Lời trình bày:** “Nhóm cần kiểm chứng recipe hiện tại trên dữ liệu chưa dùng phát triển trước. Vì Bernoulli DPO có thể viết thành BCE trên contrast trừ reference, cần đối chứng objective phù hợp để xác định lợi ích riêng. Học geometry là một thí nghiệm mới, chưa phải khả năng đã đạt.”

**Nguồn:** R2 §12–13.

---

## Slide 23 — KB3: mục tiêu và câu hỏi nghiên cứu

**Thời lượng:** 70 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB3 — Tối ưu hyperparameter của quá trình huấn luyện”.
- Trái `(5, 22, 36, 50)`: **Mục tiêu:** “Tìm cấu hình hoặc chính sách điều chỉnh hyperparameter giúp detector đạt chất lượng tốt hơn”.
- Phải `(46, 22, 49, 50)`:
  - **RQ-A:** “Local HPO quanh recipe KB1 có tìm được cấu hình tốt hơn anchor và supervised KB1?”
  - **RQ-B:** “Controller RL có điều chỉnh trong train hiệu quả hơn cấu hình/lịch cố định?”
  - **Điều kiện:** “Cùng init, recipe, evaluator, seed và cách hạch toán ngân sách”.
- Dải thông điệp: “KB3-A tạo mốc HPO; KB3-B nghiên cứu điều khiển thích nghi bằng RL.”

**Lời trình bày:** “KB3 tác động lên quá trình huấn luyện thông qua hyperparameter. Nhóm chia thành chọn cấu hình trước run và điều chỉnh trong run. Hai hướng cần đối chứng công bằng trước khi so chất lượng.”

**Nguồn:** R3 §3, §6, §16, §21. RQ được tổng hợp lại cho slide.

---

## Slide 24 — KB3: ý tưởng A và B

**Thời lượng:** 75 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB3 — Chọn cấu hình giữa các run hay điều chỉnh trong run?”.
- Trái `(5, 22, 43, 51)`, nhãn **A — TPE/HPO**:
  > History trial → Đề xuất LR/WD → Train detector mới → Validation → Cập nhật history

  “Base parameters được chọn trước mỗi trial; warmup/decay vẫn theo lịch đã định.”
- Phải `(52, 22, 43, 51)`, nhãn **B — PPO/RL**:
  > Quan sát training → Action đổi HP → Train segment tiếp → Reward → Cập nhật controller

  “Một episode là một trainer liên tục; thay HP tại decision boundary.”
- Dải thông điệp: “TPE là HPO truyền thống; PPO là hướng RL cần kiểm chứng policy.”

**Lời trình bày:** “A thử cấu hình mới sau khi trial kết thúc. B quan sát diễn biến rồi thay đổi HP ngay trong một lần train. B phức tạp hơn vì còn phải học controller và bảo toàn training state.”

**Nguồn:** R3 §6.2–6.3, §16–18.

---

## Slide 25 — KB3: kiến trúc tổng thể và nguồn khởi tạo

**Thời lượng:** 80 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB3 — Tách recipe, pretrained và baseline”.
- Trên `(5, 20, 90, 18)`: ba hộp:
  - “YAML supervised KB1: recipe”.
  - “COCO pretrained gốc: khởi tạo mỗi trial A”.
  - “Best supervised KB1: đối chứng chỉ đọc”.
- Dưới `(5, 44, 90, 33)`:
  - Luồng A chiếm 58% trái: “TPE → cấu hình → detector train mới → native chọn checkpoint → canonical validation chấm trial → TPE”.
  - Luồng B chiếm 32% phải, nét đứt: “Controller ↔ live trainer ↔ metrics/reward”; nhãn “hướng nghiên cứu tiếp”.
- Dải thông điệp: “Trial A không tiếp tục từ best KB1 và không nối trọng số từ trial trước.”

**Lời trình bày:** “Ba artifact có vai trò khác nhau. Recipe của KB1 giúp giữ điều kiện nền; pretrained gốc khởi tạo detector mới; best KB1 chỉ là mốc so. Trong A, native chọn checkpoint nội bộ và canonical evaluator chấm checkpoint đó để search.”

**Nguồn:** R3 §3.2–3.4, §9.

---

## Slide 26 — KB3-A: phạm vi tìm kiếm hiện tại

**Thời lượng:** 70 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB3-A — Local HPO hai chiều quanh recipe KB1”.
- Bảng trái `(5, 22, 52, 43)`:

| Tham số | Anchor | Search space |
|---|---:|---|
| Learning rate lr0 | 0.01 | 0.005–0.015, log scale |
| Weight decay | 0.0005 | 0.00025–0.001, log scale |
| Momentum | 0.937 | Giữ nguyên |
| Augmentation strength | 0.5 | Giữ nguyên |

- Phải `(62, 22, 33, 43)`:
  - “Mục tiêu: 8 trial”.
  - “Trial 0: anchor”.
  - “Trial 1: startup random”.
  - “Trial 2–7: TPE proposals”.
  - “Detector seed 0; sampler seed riêng từng proposal”.
- Ghi chú `(5, 71, 90, 9)`: “Objective: canonical validation mAP50–95 tại checkpoint do native chọn; không dùng test cho search.”
- Dải thông điệp: “Tám trial cùng detector seed không phải tám seed xác nhận độc lập.”

**Lời trình bày:** “A hiện chỉ tìm LR và weight decay trong vùng gần baseline. Đây là pilot local HPO, chưa tối ưu toàn bộ optimizer hay augmentation. Số trial là số cấu hình đã thử, không thay cho lặp nhiều seed.”

**Nguồn:** R3 §6.4–6.5, §9–10.

---

## Slide 27 — KB3-A: kết quả đạt được tại snapshot

**Thời lượng:** 90 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB3-A — Pretrained pilot chưa vượt baseline ở objective chính”.
- Nhãn `(5, 15, 90, 5)`: “Snapshot report: 15/10/2026, 02:31 +07:00 · validation · 2/8 trial hoàn tất”.
- Bảng `(5, 24, 90, 32)`:

| Mốc | mAP50–95 (%) | So anchor mới (đpt) | So KB1 (đpt) |
|---|---:|---:|---:|
| Supervised KB1 reference | 37.8207 | +1.7530 | 0 |
| Trial 0 — pretrained anchor | 36.0677 | 0 | −1.7530 |
| Trial 1 — startup random | 37.7225 | +1.6548 | −0.0982 |

- Dưới trái `(5, 61, 43, 18)`: “Trial 2 đang chạy ở snapshot; chưa có completed canonical result”.
- Dưới phải `(52, 61, 43, 18)`: “Chi phí 2 trial hoàn tất: 300 epoch; 4.6795 giờ; chưa là tổng cost 8 trial”.
- Dải thông điệp: “Trial tốt nhất đã hoàn tất là startup random; chưa chứng minh TPE pretrained tìm tốt hơn.”

**Lời trình bày:** “Hai trial hoàn tất cho thấy startup random tốt hơn anchor mới nhưng còn thấp hơn supervised KB1 về mAP50–95. Chênh anchor mới với baseline cũng cần audit. Đây là trạng thái được đóng băng trong report, không phải tiến độ chạy trực tiếp hôm nay.”

**Nguồn:** R3 §13. Không đưa metric native của trial dở vào bảng canonical. Kết quả scratch lịch sử để phụ lục, không gộp với pretrained.

---

## Slide 28 — KB3-B: cơ chế controller RL

**Thời lượng:** 90 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB3-B — Quan sát, hành động và reward trong training”.
- Ba cột `(5, 22, 28, 48)`, `(36, 22, 28, 48)`, `(67, 22, 28, 48)`:
  - **Observation:** “20 chiều trong quality B: tiến độ, train/val loss, xu hướng, metric, best AP, độ trì trệ, lịch và HP hiện tại”.
  - **Action:** “Điều chỉnh LR, weight decay, momentum, augmentation strength; mask lựa chọn vượt bounds”.
  - **Reward:** “Quality best-gain: r_t = 100(b_{t+1}−b_t), γ=1; b_t là best canonical validation AP của epoch đã train”.
- Dưới `(5, 74, 90, 8)`: “Giữ liên tục weights, optimizer, scheduler, EMA, RNG và dataloader trong một episode”.
- Dải thông điệp: “Cơ chế quality B đã có trong code; chưa có bằng chứng PPO học policy hữu ích trên dữ liệu thật.”

**Lời trình bày:** “Controller quan sát trạng thái tóm tắt, chọn thay đổi HP và nhận reward từ cải thiện best AP. Reward này khác reward hỗn hợp legacy. Các cấu hình của B cũ chưa được coi là protocol pretrained mới tương đương với A.”

**Nguồn:** R3 §17–18. Ghi “quality B” trên slide để không đồng nhất với luồng A pretrained mặc định.

---

## Slide 29 — KB3: kết quả kỹ thuật và định hướng tương lai

**Thời lượng:** 85 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “KB3 — Hoàn thiện mốc A trước khi mở rộng controller B”.
- Trái `(5, 21, 43, 54)`, nhãn **Đã đạt**:
  - “Pipeline A pretrained: recipe/init/reference tách rõ; có trial thực và metric canonical”.
  - “Ghi manifest/hash, checkpoint selection, completed-trial resume và chi phí”.
  - “B có integration gate cho continuity và áp action; chưa chứng minh chất lượng policy”.
- Phải `(52, 21, 43, 54)`, nhãn **Tiếp theo — theo thứ tự**:
  1. “Audit chênh anchor với baseline KB1; hoàn tất search A đã chốt”.
  2. “Khóa ứng viên, xác nhận seed mới; so random search cùng ngân sách”.
  3. “Ưu tiên B LR-only residual quanh strong pretrained A”.
  4. “Kiểm hold-only/fixed, random actions, progress-only; frozen-policy evaluation”.
- Dải thông điệp: “Chưa kết luận A vượt KB1, TPE vượt random hoặc B vượt A.”

**Lời trình bày:** “Tiến bộ hiện tại của KB3 chủ yếu là pipeline có thể kiểm tra và pilot A. Định hướng thực dụng là tạo baseline A mạnh, sau đó giới hạn B trước vào LR để giảm độ khó, kiểm tra controller có dùng feedback thật hay chỉ học một lịch theo epoch.”

**Nguồn:** R3 §11, §14–15, §19–21.

---

## Slide 30 — Tổng hợp tiến độ ba kịch bản

**Thời lượng:** 70 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “Tiến độ nghiên cứu và bằng chứng hiện có”.
- Bảng `(5, 23, 90, 52)`:

| Hạng mục | Kết quả chính | Bằng chứng | Việc tiếp theo |
|---|---|---|---|
| Dữ liệu | 3.129 ảnh, 28 class; split sạch theo audit source ID | Manifest, ảnh/nhãn và audit | Bổ sung trực quan hóa; kiểm near-duplicate sâu hơn |
| KB1 | G hơn S0 ở 21/21 cặp quan sát | Validation và test | Nhiều S0; phân tích bất định/objective |
| KB2 | DPO hơn native về AP ở 3/3 seed | Validation | Holdout/test; BCE ablation; recall/class coverage |
| KB3-A | 2/8 pretrained trial hoàn tất ở snapshot | Validation snapshot 15/10 | Audit anchor; hoàn tất và xác nhận |
| KB3-B | Cơ chế/controller và kiểm chứng kỹ thuật | Integration gates | Learning pilot và frozen-policy evaluation |

- Dải thông điệp: “Mỗi KB có mức hoàn thành khác nhau; bảng này tổng hợp tiến độ, không xếp hạng hiệu năng.”

**Lời trình bày:** “KB1 có bằng chứng hoàn chỉnh nhất trong phạm vi hiện tại. KB2 cho thấy lợi ích AP lặp lại nhưng còn cần holdout. KB3 đã có nền kỹ thuật, còn phần khẳng định chất lượng HPO/RL phải tiếp tục kiểm chứng.”

**Nguồn:** R1 §16; R2 §12–13; R3 §1, §13.

---

## Slide 31 — Website nhận diện: kế hoạch tích hợp

**Thời lượng:** 70 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “Website nhận diện — kế hoạch chuyển mô hình thành ứng dụng”.
- Nhãn nổi `(5, 16, 90, 6)`: “ĐỀ XUẤT TIẾP THEO — ba report chưa xác nhận tiến độ website”.
- Luồng giữa `(5, 30, 90, 24)`: “Người dùng tải ảnh” → “API kiểm tra ảnh” → “YOLO inference” → “Trả box/class/confidence” → “Hiển thị kết quả”.
- Dưới trái `(5, 62, 43, 17)`: **Giao diện:** “Upload ảnh; xem box và class; xem confidence; xử lý trường hợp không có detection”.
- Dưới phải `(52, 62, 43, 17)`: **Tích hợp model:** “Khóa checkpoint/version; thống nhất preprocessing; đo latency và bộ nhớ; đối chiếu output với inference offline”.
- Dải thông điệp: “Chọn model triển khai dựa trên chất lượng đã xác nhận và chi phí phục vụ thực tế.”

**Lời trình bày:** “Đây là luồng website đề xuất, chưa phải báo cáo sản phẩm đã hoàn thành. Nhóm cần đóng gói checkpoint, giữ preprocessing đúng và kiểm tra output API khớp offline trước khi đánh giá trải nghiệm và thời gian đáp ứng.”

**Nguồn:** yêu cầu đề tài và đề xuất của kịch bản. Không thêm ảnh demo, framework, URL triển khai hoặc tỷ lệ hoàn thành khi chưa có thông tin xác nhận.

---

## Slide 32 — Kế hoạch giai đoạn tiếp theo và trao đổi

**Thời lượng:** 60 giây.

**Thiết kế và nội dung hiển thị**

- Tiêu đề: “Kế hoạch tiếp theo”.
- Bốn hàng `(5, 22, 90, 13)`, cách nhau 2%:
  1. **Dữ liệu và báo cáo:** “Hoàn thiện visualization; thống nhất bảng số và phạm vi kết luận”.
  2. **Xác nhận mô hình:** “KB2 holdout/test sau khóa phương pháp; KB3 audit và xác nhận A; phân tích bất định KB1”.
  3. **Nghiên cứu RL tiếp:** “Cải thiện reward/FN surrogate và pilot controller B có đối chứng”.
  4. **Ứng dụng:** “Đóng gói inference, dựng website và đo chất lượng phục vụ”.
- Góc dưới phải `(70, 85, 25, 7)`: “Cảm ơn — Trao đổi”.
- Dải dưới trái `(5, 85, 61, 7)`: “Đầu ra kỳ vọng: bằng chứng thực nghiệm rõ ràng và website nhận diện có thể kiểm thử”.

**Lời trình bày:** “Nhóm ưu tiên củng cố bằng chứng trước khi mở rộng các biến thể, đồng thời chuẩn bị luồng inference cho website. Phần cần tiếp tục đánh giá là lợi ích riêng của reward/preference và khả năng controller tạo cải thiện ngoài các lịch cố định.”

**Nguồn:** tổng hợp hướng tương lai R1–R3; website là kế hoạch đề xuất. Chưa gán ngày hoàn thành khi chưa biết lịch và nhân lực của nhóm.

---

## Phụ lục tùy chọn — dùng khi hội đồng hỏi sâu

Không tính vào 32 slide chính. Các slide này dùng cùng quy ước tọa độ/kiểu chữ.

### A1 — Diễn biến huấn luyện và chọn checkpoint KB1

- Tiêu đề: “KB1 — Validation trajectory và lựa chọn checkpoint”.
- Hình `(5, 20, 63, 58)`: [continuation_validation.svg](assets/kb1_20261103/continuation_validation.svg).
- Phải `(72, 20, 23, 58)`: “Best theo canonical validation”; “Continuation có early stopping”; “Ablation dùng last cùng budget”; “Không gộp best-validation và fixed-final thành một thống kê”.
- Lời trình bày: “Hình mô tả diễn biến validation, không phải đường test. Quy tắc selection của continuation khác phép so fixed-budget.”
- Nguồn: R1 §6.4, §9.

### A2 — Vì sao KB2 chuyển từ whole-action sang conditional?

- Bốn hộp ngang `(5, 24, 90, 35)`: “Whole-action: khác nhiều slot” → “Margin tốt chưa chắc cứu FN thật” → “Audit score/GT” → “Một candidate, gate toàn ảnh”.
- Dưới `(5, 65, 90, 14)`: “Các recipe lịch sử khác action space/mask/numerics; không coi là ablation chỉ thay một biến”.
- Lời trình bày: “Thiết kế conditional xuất phát từ việc cần gắn tín hiệu preference với quyết định có thể trace, thay vì chỉ tối ưu margin toàn action.”
- Nguồn: R2 §7.

### A3 — KB2 tại cùng ngân sách FP

- Bảng `(5, 24, 56, 43)`:

| Tại FP=984 trên validation | S0 FN | Native best FN mean | DPO best FN mean |
|---|---:|---:|---:|
| Cutoff quét riêng từng model | 866 | 861.00 | 857.33 |

- Phải `(66, 24, 29, 43)`: “DPO giảm 3.67 FN so native best”; “Cutoff được tìm trên validation”; “Không thay bảng chính ở confidence 0.25”.
- Lời trình bày: “Khi so cùng số FP, DPO có lợi ích FN nhỏ. Đây là diagnostic tại một điểm, chưa chứng minh toàn PR curve vượt trội hoặc cutoff tổng quát tốt trên test.”
- Nguồn: R2 §10.5.

### A4 — KB3 scratch lịch sử và lý do không gộp pretrained

- Bảng `(5, 23, 90, 37)`:

| Protocol | Mốc | Canonical validation mAP50–95 (%) |
|---|---|---:|
| Scratch v1 | Anchor | 34.8556 |
| Scratch v1 | Winner trial 5 | 36.1185 |
| Historical pretrained KB1 | Reference | 37.8207 |

- Dưới `(5, 67, 90, 13)`: “Scratch winner: +1.2629 đpt so scratch anchor; −1.7022 đpt so KB1. Tám selected weights scratch đã xóa; logs/metric còn lưu”.
- Lời trình bày: “Kết quả scratch chứng minh có cấu hình tốt hơn scratch anchor trong pilot một seed. Nó không phải kết quả pretrained hoặc thành tích RL, và không thay kết quả A v2.”
- Nguồn: R3 §12.

## Ghi chú bàn giao cho người dựng slide

1. Dựng đúng thứ tự 01–32; hai slide 04 và 07 để placeholder, nhóm tự bổ sung. Không điền số hoặc ảnh minh họa giả vào các vị trí này.
2. Sơ đồ 11, 18 và 25 là ba hình kiến trúc trọng tâm; dùng shape/vector và mũi tên để có thể sửa. Không cần tạo bitmap bằng AI.
3. Hình có sẵn nên ưu tiên SVG để chữ rõ; nếu quá nhiều ô nhỏ thì đưa vào phụ lục, dùng bảng đã rút gọn trong phần chính.
4. Giữ nhãn split/seed/selection sát bảng, không chỉ nằm trong speaker notes. KB3 phải có nhãn snapshot 15/10, KB2 phải ghi validation.
5. Không so trực tiếp S0 số lẻ khác nhau giữa report: YOLOv8n validation khoảng 37.8190% (R1), 37.8218% (R2), 37.8207% (R3) do các lượt đánh giá được báo riêng. Giữ baseline cùng report khi tính delta; không lấy một số “thống nhất” rồi tính lại mọi gain.
6. Website chỉ ghi kế hoạch đề xuất cho đến khi nhóm bổ sung bằng chứng triển khai. Tên thành viên/GVHD/ngày trình bày cần điền trước xuất bản.
7. Nếu chỉ có 20–25 phút: vẫn giữ đủ năm phần mỗi KB; gộp 09+10, 12+14, 16+17, 19+20, 23+24, 25+26; chuyển chi tiết phương pháp và các bảng dài vào notes/phụ lục. Không bỏ các slide định hướng tương lai 15/22/29.
