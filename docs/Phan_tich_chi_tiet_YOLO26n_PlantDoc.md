# Phân tích chi tiết bài báo Plant Disease Object Detection on PlantDoc Using YOLO26n (2026)

**Bài báo:** Ovide Decroly Wisnu Ardhi, Rudi Hartono, Nanang Maulana Yoeseph, *Plant Disease Object Detection on PlantDoc Using YOLO26n*, SISFO, 2026.

**Ngày tổng hợp:** 06/10/2026.

**Nguồn chính:** [PDF bài báo](https://ojs.unimal.ac.id/sisfo/article/view/27063/11079) · [Trang bài báo](https://ojs.unimal.ac.id/sisfo/article/view/27063).

> Báo cáo phân biệt mô tả của tác giả, kết quả tính lại từ bảng số liệu, và ví dụ giải thích. Các ví dụ không phải thí nghiệm đã thực hiện trong bài.

## Nhận định chung

**Bài này là một thí nghiệm fine-tune YOLO26n trên một bản PlantDoc đã được chuẩn bị cho phát hiện đối tượng.** Giá trị chính của nó nằm ở việc lập baseline và phân tích lỗi. Tuy nhiên, **quy trình tạo bản dữ liệu 3.940 ảnh chưa được mô tả đủ**, nên chưa thể tái lập chính xác hoặc coi kết quả là phép so sánh trực tiếp với PlantDoc gốc.

Cần sửa một điểm trong nhận xét trước: nói “họ chuyển nhãn sang YOLO” là quá chắc chắn. **Hình 1 bắt đầu bằng dữ liệu có ảnh và tệp nhãn YOLO**, rồi chuẩn bị và kiểm tra dữ liệu; bài không trình bày mã chuyển đổi hay chứng minh tác giả tự chuyển toàn bộ XML gốc. [Nguồn: PDF bài báo](https://ojs.unimal.ac.id/sisfo/article/view/27063/11079).

## 1. Bài toán họ giải quyết chính xác là gì?

Đầu vào là một ảnh có thể chứa nhiều lá, nền cây cỏ, lá chồng lên nhau và triệu chứng khác nhau. Đầu ra của detector gồm:

- **Hộp bao:** đối tượng nằm ở đâu.
- **Nhãn lớp:** lá thuộc loại cây/tình trạng nào.
- **Confidence:** mức tin cậy của dự đoán.

Ví dụ, một ảnh có thể cho kết quả:

| Vùng được phát hiện | Nhãn dự đoán | Confidence minh họa |
|---|---|---:|
| Hộp thứ nhất | Lá ngô bị gỉ sắt | 0,89 |
| Hộp thứ hai | Lá cà chua bị đốm vi khuẩn | 0,72 |
| Hộp thứ ba | Lá cà chua khỏe | 0,65 |

Những số confidence trên **chỉ là ví dụ**, không phải kết quả của bài.

Đặc biệt, cần phân biệt **phát hiện lá bị bệnh** với **phát hiện riêng từng vết bệnh**. Một hộp bao quanh cả lá vẫn có thể mang nhãn “lá bị đốm vi khuẩn”. Khi đó, mô hình không được chấm theo việc khoanh từng đốm nhỏ.

Vì thế, việc bài sử dụng các từ như “disease region”, “small spots” hay “lesion” **không đủ để kết luận toàn bộ dữ liệu được gán hộp quanh từng tổn thương**. Muốn biết phải mở ảnh nhãn thực tế và kiểm tra đối tượng mà hộp đang bao quanh.

Đây là điểm ảnh hưởng trực tiếp đến luận văn: nếu nhãn bao cả lá, reward dựa trên IoU cũng phải đánh giá **hộp của lá**, không thể yêu cầu mô hình tự khoanh chính xác từng đốm bệnh chưa được gán nhãn.

## 2. Họ chỉnh sửa dữ liệu như thế nào — và phần nào thực sự chưa biết?

Từ nội dung bài, có thể xác nhận **chuẩn bị thư mục, kiểm tra cặp ảnh–nhãn, định dạng YOLO và cấu hình dữ liệu**. Nhưng chưa có mô tả đủ để khẳng định họ thực hiện những bước sau:

| Thao tác | Có thể kết luận từ bài không? | Vì sao cần biết? |
|---|---|---|
| Chuẩn bị train/validation/test | Có | Quyết định ảnh nào được học, ảnh nào được đánh giá |
| Kiểm tra ảnh có nhãn tương ứng | Có | Tránh lỗi thiếu nhãn hoặc nhãn gắn nhầm ảnh |
| Dùng tọa độ hộp chuẩn hóa kiểu YOLO | Có | Để framework đọc nhãn đúng |
| Tự chuyển XML gốc sang TXT | Chưa rõ | Chưa có script hoặc quy trình chuyển đổi |
| Sửa các nhãn bệnh sai | Chưa rõ | Nhãn sai ảnh hưởng cả train và đánh giá |
| Gộp/tách/đổi tên các lớp | Chưa rõ | Có thể làm thay đổi bài toán |
| Xóa ảnh trùng hoặc gần trùng | Chưa rõ | Cần để kiểm soát rò rỉ giữa các tập |
| Tạo ảnh mới bằng augmentation trước khi train | Chưa rõ | Có thể giải thích số ảnh tăng, nhưng bài không chứng minh |
| Oversampling lớp ít mẫu | Chưa rõ | Ảnh hưởng mức mất cân bằng |
| Dùng Mosaic, lật, xoay… trong lúc train | Chưa công bố đầy đủ | Không thể suy ra cấu hình chỉ từ tên Ultralytics |

**Chuẩn hóa định dạng không đồng nghĩa với sửa chất lượng dữ liệu.** Một tệp có tọa độ hợp lệ vẫn có thể khoanh sai lá hoặc gán sai bệnh. Kiểm tra “ảnh có nhãn” là kiểm tra kỹ thuật; kiểm tra “nhãn đúng bệnh” là kiểm tra nội dung.

Vì vậy, cách mô tả chính xác nhất là:

> Tác giả sử dụng và kiểm tra một bản PlantDoc ở định dạng YOLO; nguồn gốc và các bước biến đổi tạo ra phiên bản đó chưa được giải thích đầy đủ.

## 3. Định dạng YOLO mà họ sử dụng hoạt động thế nào?

Một ảnh có một tệp nhãn tương ứng. Mỗi dòng biểu diễn một đối tượng:

```text
class_id x_center y_center width height
```

Các tọa độ được chuẩn hóa theo kích thước ảnh; mã lớp bắt đầu từ 0. `data.yaml` xác định đường dẫn các tập và ánh xạ mã lớp sang tên lớp. Đây là quy ước chính thức của Ultralytics. [Nguồn: Object Detection Datasets](https://docs.ultralytics.com/datasets/detect/).

**Ví dụ chuyển đổi do người phân tích đưa ra:**

Ảnh rộng $W=1000$, cao $H=800$. Hộp bao có hai góc:

$$
(x_{\min},y_{\min})=(200,100),\qquad
(x_{\max},y_{\max})=(600,500).
$$

Tọa độ YOLO:

$$
x_c=\frac{x_{\min}+x_{\max}}{2W}=0,4
$$

$$
y_c=\frac{y_{\min}+y_{\max}}{2H}=0,375
$$

$$
w=\frac{x_{\max}-x_{\min}}{W}=0,4,\qquad
h=\frac{y_{\max}-y_{\min}}{H}=0,5.
$$

Nếu lớp có mã 3, dòng nhãn là:

```text
3 0.4 0.375 0.4 0.5
```

Ba lỗi đặc biệt nguy hiểm khi chuẩn bị dữ liệu là:

- Chuẩn hóa $x,w$ theo **chiều cao** thay vì chiều rộng.
- Thay đổi ảnh bằng crop/resize nhưng không biến đổi hộp tương ứng.
- Giữ tọa độ đúng nhưng thay đổi thứ tự lớp trong `data.yaml`.

Ví dụ mã 3 trước đây là “lá khỏe”, sau đó bị ánh xạ thành “lá bệnh”: framework vẫn chạy bình thường nhưng ý nghĩa dự đoán sai hoàn toàn.

## 4. Bản dữ liệu 3.940 ảnh có vấn đề gì cần kiểm tra?

Từ các số đã trích, tính lại được:

| Tập | Ảnh | Tỉ lệ ảnh |
|---|---:|---:|
| Train | 3.244 | **82,34%** |
| Validation | 463 | **11,75%** |
| Test | 233 | **5,91%** |
| Tổng | 3.940 | 100% |

Đây **không phải phép chia 80:10:10 chính xác**. Tuy nhiên, bản thân tỉ lệ này không làm thí nghiệm sai. Điều quan trọng hơn là cách chọn ảnh và việc các tập có độc lập hay không.

Hình 2 còn cho số hộp train/validation/test lần lượt **27.127 / 1.661 / 817**. [Nguồn: PDF bài báo](https://ojs.unimal.ac.id/sisfo/article/view/27063/11079). Từ đó, tính được:

| Tập | Hộp trung bình trên mỗi ảnh |
|---|---:|
| Train | **8,36** |
| Validation | **3,59** |
| Test | **3,51** |

**Tập train có mật độ hộp trung bình cao hơn đáng kể.** Có thể do nhiều ảnh đông lá hơn, do cách chọn ảnh, hoặc do dữ liệu đã được biến đổi. Các số này **chưa chứng minh được nguyên nhân nào**.

So với PlantDoc gốc đã được đọc:

- Số ảnh tăng từ 2.598 lên 3.940, khoảng **51,7%**.
- Số hộp tăng từ 9.216 lên 29.605, khoảng **3,21 lần**.

Chỉ chuyển định dạng nhãn từ XML sang TXT **không tự làm tăng số ảnh hoặc số hộp**. Vì vậy phải có sự khác biệt về phiên bản, số liệu thống kê, phép biến đổi hoặc phạm vi gán nhãn. Bài chưa giải thích đủ để xác định trường hợp nào.

Muốn làm rõ, cần có:

1. Link và version chính xác của bản dữ liệu.
2. Danh sách ảnh trong từng tập.
3. Số **ảnh gốc độc lập** trước augmentation.
4. Quy tắc tạo ảnh/hộp mới.
5. Kết quả kiểm tra ảnh trùng giữa các tập.

**Không thể kết luận có data leakage chỉ từ số ảnh tăng.** Nhưng chưa có thông tin đủ để loại trừ nó. Nếu một ảnh gốc ở train và bản xoay/crop gần giống của ảnh ấy ở test, bài đánh giá sẽ dễ hơn so với kiểm tra trên ảnh hoàn toàn mới.

## 5. Mất cân bằng và “đối tượng nhỏ” được hiểu như thế nào?

Tỉ lệ **2.407:5 ≈ 481:1** cho thấy số đối tượng giữa các lớp rất lệch. Lưu ý đây là **số instance/hộp**, không nhất thiết là số ảnh độc lập.

Một ảnh có 20 hộp cùng lớp cung cấp nhiều ví dụ, nhưng chúng thường chung nền, ánh sáng và nguồn chụp. Vì thế “20 hộp” không tương đương hoàn toàn với “20 ảnh chụp độc lập”.

Với lớp chỉ có 5 mẫu, khó trả lời được:

- Mô hình học dấu hiệu bệnh hay nhớ một vài ảnh?
- Test có đại diện cho biến thiên thực tế không?
- Nếu chỉ có một mẫu test, bỏ sót một đối tượng sẽ làm recall thay đổi bao nhiêu?

Thông tin **48,9% hộp có diện tích chuẩn hóa dưới 0,05** cũng cần đọc cẩn thận:

$$
a_{\text{norm}}=\frac{w_{\text{box}}h_{\text{box}}}{WH}.
$$

Ngưỡng 0,05 nghĩa là hộp chiếm dưới **5% diện tích ảnh**. Nếu giả sử ảnh được đưa về hình vuông 416 × 416 và hộp cũng vuông, ngưỡng tương ứng cạnh khoảng:

$$
416\sqrt{0,05}\approx93\text{ pixel}.
$$

Do đó, nhóm này có thể gồm cả hộp 10 × 10 lẫn hộp gần 90 × 90 pixel. **Không thể diễn giải rằng gần một nửa dữ liệu đều là tổn thương cực nhỏ.**

Đây là một cách chia kích thước tương đối do bài sử dụng. Khi đánh giá đối tượng nhỏ, nên công bố rõ đang dùng **diện tích tương đối** hay **diện tích pixel**, và đo AP của từng nhóm thay vì chỉ đếm số hộp nhỏ.

## 6. Quá trình fine-tuning diễn ra thế nào?

Quy trình được nêu trong bài là:

1. Nạp mô hình với trọng số pretrained.
2. Điều chỉnh đầu dự đoán cho các lớp PlantDoc.
3. Cho ảnh train đi qua mạng.
4. So dự đoán với ground truth.
5. Tính loss và cập nhật toàn bộ mạng.
6. Đánh giá trên validation để lựa chọn checkpoint.
7. Đánh giá checkpoint được chọn trên test.

Ở đây **“end-to-end fine-tuning”** nghĩa là cập nhật cả phần trích xuất đặc trưng, hợp nhất đặc trưng và đầu phát hiện. Nó khác với việc đóng băng backbone rồi chỉ train head.

Có thể hiểu loss ở mức khái niệm gồm:

- Thành phần giúp **hộp dự đoán tiến gần hộp nhãn**.
- Thành phần giúp **dự đoán đúng lớp**.

Không nên tự gán cho bài một công thức loss, hệ số hay optimizer cụ thể khi tác giả không công bố. Tài liệu YOLO26 hiện tại mô tả regression không dùng DFL và có các lựa chọn head khác nhau, nhưng tài liệu hiện tại **không thay thế được cấu hình của lần chạy trong bài**. [Nguồn: Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/).

Với 3.244 ảnh và batch 16, một epoch có khoảng:

$$
\left\lceil\frac{3244}{16}\right\rceil=203\text{ batch}.
$$

Nếu chạy đủ 50 epoch, đó là khoảng **10.150 lượt xử lý batch**. Số lần optimizer thực sự cập nhật có thể khác nếu dùng gradient accumulation hoặc cơ chế khác.

**50 epoch không tự chứng minh mô hình đã hội tụ.** Cần xem đường cong train/validation và checkpoint tốt nhất ở epoch nào. Các khả năng đều tồn tại:

- Validation vẫn tăng ở epoch 50: có thể chưa học đủ.
- Train tiếp tục tốt lên nhưng validation giảm: có thể overfit.
- Cả hai ổn định: có thể đã hội tụ trong cấu hình đó.

`best.pt` chỉ có nghĩa là tốt nhất **theo tiêu chí lựa chọn của lần chạy**. Không nên mặc định nó là checkpoint có precision cao nhất. Tiêu chí fitness, early stopping, seed và phiên bản framework cần được ghi rõ để tái lập. Ultralytics cho phép cấu hình nhiều yếu tố huấn luyện; các giá trị mặc định cũng phải gắn với phiên bản sử dụng. [Nguồn: Model Training](https://docs.ultralytics.com/modes/train/).

## 7. TP, FP, FN và IoU được tính như thế nào?

Trước tiên cần phân biệt hai ngưỡng:

| Ngưỡng | Câu hỏi nó trả lời |
|---|---|
| **Confidence threshold** | Có giữ dự đoán này để đánh giá/hiển thị không? |
| **IoU threshold** | Hộp dự đoán có đủ khớp với hộp nhãn không? |

IoU là:

$$
IoU=\frac{\text{diện tích giao}}{\text{diện tích hợp}}.
$$

**Ví dụ:** hai hộp cùng có diện tích 10.000 pixel², giao nhau 8.000 pixel²:

$$
IoU=\frac{8000}{10000+10000-8000}
=\frac{8000}{12000}\approx0,667.
$$

Hộp này có thể đạt yêu cầu ở IoU 0,5, nhưng không đạt ở IoU 0,75.

Để tính một dự đoán là **TP**, cần đúng lớp, đủ IoU và được ghép với một ground truth chưa bị ghép trước đó. Ví dụ:

| Trường hợp | Cách tính thông thường |
|---|---|
| Đúng lớp, IoU đủ | TP |
| Hộp đặt trên nền, không ghép được nhãn | FP |
| Có lá được gán nhãn nhưng không phát hiện được | FN |
| Đúng vị trí nhưng sai lớp | FP cho lớp dự đoán; FN cho lớp thật |
| Hai dự đoán cùng ghép vào một lá | Tối đa một TP; hộp dư có thể là FP |

**Ví dụ minh họa độc lập:** ảnh có 3 lá được gán nhãn. Detector đưa 4 hộp, trong đó 2 đúng, 2 sai; một lá bị bỏ sót:

$$
TP=2,\quad FP=2,\quad FN=1.
$$

$$
P=\frac{2}{2+2}=0,5,\qquad
R=\frac{2}{2+1}\approx0,667.
$$

$$
F1=\frac{2PR}{P+R}\approx0,571.
$$

Trong object detection, việc xác định **TN** không rõ ràng như phân loại ảnh: nền chứa vô số vùng có thể không phải đối tượng. Vì vậy accuracy thông thường không phải chỉ số trung tâm cho bài toán này.

## 8. Precision, recall và F1 của bài: một phát hiện quan trọng khi tính lại

Người phân tích nhập lại **27 dòng kết quả theo lớp** của Bảng 5 và tính trung bình. Các trung bình khớp với kết quả tổng thể:

| Phép tính lại | Kết quả |
|---|---:|
| Trung bình precision của 27 lớp | **0,53426** |
| Trung bình recall của 27 lớp | **0,56052** |
| Trung bình AP50 của 27 lớp | **0,57289** |
| Trung bình AP50–95 của 27 lớp | **0,41659** |

Đây là phép tính từ bảng, không phải lần chạy lại mô hình. [Nguồn số liệu: PDF bài báo, Bảng 5](https://ojs.unimal.ac.id/sisfo/article/view/27063/11079).

Điều này cho thấy P và R được báo cáo **phù hợp với trung bình theo lớp — macro average**. Nó dẫn tới một sửa đổi quan trọng trong cách diễn giải:

> Recall 0,560 ở đây không nên được đọc là “detector tìm được đúng 56% tổng 817 hộp”.

Để có tỷ lệ trên tổng đối tượng, phải biết tổng TP và FN, hoặc số mẫu của từng lớp để tính trung bình có trọng số. Macro average cho mỗi lớp trọng số bằng nhau, dù lớp có 2 hay 200 đối tượng.

F1 cũng có hai cách tính khác nhau:

$$
F1_{\text{từ P,R trung bình}}
=\frac{2\bar P\bar R}{\bar P+\bar R}
\approx\mathbf{0,547}.
$$

Trong khi:

$$
\text{Macro-F1}
=\frac1C\sum_{c=1}^{C}
\frac{2P_cR_c}{P_c+R_c}
\approx\mathbf{0,510}.
$$

**Hai cách không bằng nhau**, vì F1 là phép biến đổi phi tuyến. Con số 0,547 của bài khớp với cách thứ nhất. Khi viết luận văn, nên ghi đúng cách tính thay vì gọi cả hai là macro-F1.

Một điểm nữa: P/R phụ thuộc confidence threshold. Trong mã Ultralytics hiện tại, các giá trị P/R theo lớp được lấy ở ngưỡng tương ứng cực đại của đường F1 trung bình đã làm trơn. Bài không ghi phiên bản và ngưỡng cụ thể, nên chưa thể xác nhận hoàn toàn cách tính lần chạy đó. [Nguồn: utils.metrics](https://docs.ultralytics.com/reference/utils/metrics/).

## 9. AP, mAP50 và mAP50–95: chúng khác gì nhau?

P/R đo ở một điểm vận hành. **AP** đánh giá chất lượng xếp hạng dự đoán theo confidence.

Với một lớp, quy trình khái niệm là:

1. Gom các dự đoán trong tập test.
2. Sắp xếp theo confidence giảm dần.
3. Xác định TP/FP theo lớp và IoU.
4. Tính precision–recall tích lũy khi thêm dần dự đoán.
5. Tính diện tích dưới đường precision–recall bằng quy tắc nội suy của evaluator.

Mô hình có AP tốt khi đưa các dự đoán đúng lên trước và vẫn tìm được nhiều đối tượng. **Confidence cao nhưng sai có thể làm AP xấu đi.** AP không phải trung bình confidence, cũng không phải “tỷ lệ ảnh đúng”. [Nguồn: YOLO Performance Metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/).

$$
mAP50=\frac1C\sum_{c=1}^{C}AP_c(IoU=0,5).
$$

$$
mAP50\text{–}95=
\frac1{10C}\sum_c
\sum_{t\in\{0,50,0,55,\ldots,0,95\}}AP_c(t).
$$

Với bài này:

- **0,573 mAP50:** kết quả trung bình theo lớp khi hộp cần đạt IoU 0,5.
- **0,417 mAP50–95:** kết quả khi lấy trung bình qua 10 mức IoU.

Khoảng cách **15,6 điểm phần trăm** gợi ý chất lượng định vị giảm khi yêu cầu hộp sát hơn. Tuy nhiên, nó **không chứng minh nguyên nhân chỉ là “vẽ hộp chưa chính xác”**. Những yếu tố khác có thể góp phần:

- Hộp ground truth chưa nhất quán.
- Đối tượng nhỏ bị mất chi tiết.
- Hộp chứa nhiều lá chồng nhau.
- Mô hình chỉ khoanh phần nổi bật của một lá.
- Sai lớp hoặc thiếu phát hiện tại các ngưỡng đánh giá.

Nên kiểm tra thêm AP75, IoU của các dự đoán đúng lớp và lỗi theo kích thước để tách các nguyên nhân.

**Ba lớp không có ground truth test cũng là vấn đề về phạm vi đánh giá.** Kết quả tổng thể khớp với trung bình 27 lớp có trong bảng. Vì vậy, số đó không chứng minh hiệu quả trên đầy đủ 30 lớp được cấu hình. Lớp không có mẫu test là **chưa đánh giá được**, không phải đã đạt AP tốt hoặc xấu.

## 10. Confusion matrix và phân tích theo lớp cần đọc ra sao?

Với ma trận trong Hình 3, hãy nhìn đúng nhãn trục:

- **Trục dọc:** lớp thật.
- **Trục ngang:** lớp dự đoán.
- **Đường chéo:** đúng lớp.
- **Ngoài đường chéo:** nhầm giữa các lớp.
- **Cột background:** ground truth không ghép được với dự đoán.
- **Hàng background:** dự đoán không ghép được với ground truth.

Điểm đáng chú ý là **background trong confusion matrix detection mang nghĩa ghép hộp**, không nhất thiết mô hình có một lớp “background” đầu ra.

Ma trận chuẩn hóa giúp nhìn tỷ lệ, nhưng dễ làm người đọc quên số mẫu. Một ô 1,00 trên lớp có 1 đối tượng không có mức độ chắc chắn tương đương 1,00 trên lớp có 100 đối tượng. Khi báo cáo, nên đặt **ma trận số lượng** cạnh **ma trận chuẩn hóa**.

Từ các lớp đã trích:

| Lớp | Kết quả | Điều có thể diễn giải |
|---|---|---|
| Corn rust leaf | AP50 = 0,959 | Khả năng xếp hạng/phát hiện tốt trên mẫu test của lớp này |
| Cherry leaf | AP50 = 0,247 | Một lớp khó trong thiết lập hiện tại |
| Tomato leaf | AP50–95 = 0,143 | Kết quả kém khi tăng yêu cầu định vị |

Nhưng không thể kết luận riêng từ bảng rằng “ngô dễ hơn cà chua trong mọi hoàn cảnh”. Phải kiểm tra số mẫu, kích thước hộp, độ đa dạng ảnh và nhãn.

Một ví dụ minh họa cách đọc P/R: nếu một lớp có **recall cao nhưng precision thấp**, mô hình có thể đang phát hiện rất nhiều vùng là lớp đó để ít bỏ sót, đổi lại tạo nhiều báo động giả. Nếu **precision cao nhưng recall thấp**, nó có thể chỉ giữ những trường hợp rất rõ, bỏ qua nhiều trường hợp khó. Đây là hai kiểu lỗi khác nhau và cần hai cách cải tiến khác nhau.

Người phân tích cũng thấy bài có chỗ nói một biến thể tên lớp đậu tương không có mẫu test, trong khi bảng có lớp tên gần giống với kết quả. **Cần kiểm tra danh sách class ID trước khi kết luận đây là lỗi:** có thể tồn tại hai nhãn `Soyabean`/`Soybean` khác nhau, hoặc lỗi diễn đạt. Sự không rõ này càng cho thấy cần công bố `data.yaml` và số mẫu test từng lớp.

## 11. Tốc độ và phép so sánh với PlantDoc gốc mạnh đến đâu?

Từ **2,8 ms/ảnh**, nghịch đảo đơn giản là:

$$
\frac{1000}{2,8}\approx357\text{ ảnh/giây}.
$$

Đây chỉ là phép đổi đơn vị lý thuyết. Nó không có nghĩa web sẽ xử lý 357 lượt tải ảnh mỗi giây. Luồng web còn có tải ảnh, giải mã, tiền xử lý, truyền dữ liệu, hậu xử lý, vẽ hộp và tải kết quả xuống. Độ trễ batch 1 cũng khác throughput khi chạy batch lớn.

Muốn tái lập tốc độ cần biết batch, precision FP32/FP16, warm-up, GPU synchronization, số lần đo và phần nào được tính thời gian. Với YOLO26 còn cần biết head/path inference: tài liệu hiện tại có lựa chọn one-to-many dùng NMS và one-to-one không dùng NMS. Không nên mặc định bài đã đo đúng một lựa chọn cụ thể. [Nguồn: Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/).

Về chất lượng:

$$
0,573-0,389=0,184
$$

tức **18,4 điểm phần trăm**. Nếu biểu diễn tăng tương đối:

$$
\frac{0,573-0,389}{0,389}\times100\%
\approx47,3\%.
$$

Hai cách biểu diễn đều là số học, nhưng **không giải quyết được khác biệt điều kiện thí nghiệm**. Dùng cùng tên dataset và cùng mAP50 chưa đủ để so sánh công bằng. Cần cùng tập test, nhãn, lớp, quy tắc evaluator và chính sách tiền xử lý.

Do đó, bằng chứng bài cung cấp hỗ trợ phát biểu:

> Một cấu hình YOLO26n fine-tuned đạt kết quả này trên bản dữ liệu mà tác giả dùng.

Nó chưa tách được phần cải thiện do **kiến trúc**, **pretraining**, **dữ liệu**, **augmentation** hoặc **cấu hình huấn luyện**.

## 12. Bạn nên học gì từ bài để thiết kế luận văn tốt hơn?

Có thể giữ ý tưởng baseline và phân tích lỗi, đồng thời làm chặt hơn ở những điểm sau:

| Phần cần làm | Cách thực hiện rõ ràng |
|---|---|
| Phiên bản dữ liệu | Ghi link/version, số ảnh gốc, số hộp và danh sách lớp |
| Chia tập | Lưu danh sách ảnh; chia theo ảnh/nguồn gốc trước augmentation |
| Chất lượng nhãn | Kiểm tra tọa độ và một mẫu ảnh để xác nhận hộp bao đúng đối tượng |
| Baseline | Fine-tune YOLO pretrained với cấu hình đầy đủ |
| So sánh cải tiến | Cùng split, đầu vào, trọng số khởi tạo và ngân sách huấn luyện |
| Lựa chọn mô hình | Chỉ dùng validation; test giữ cho đánh giá cuối |
| Metric | mAP50, mAP50–95, P/R, cách tính F1, AP từng lớp và số mẫu |
| Độ ổn định | Nhiều seed, báo cáo trung bình và độ lệch chuẩn |
| Tốc độ | Đo trên phần cứng triển khai, phân biệt inference và toàn luồng |

Với cơ chế phản hồi tự động, một đối chứng đặc biệt cần thiết là:

**YOLO baseline → tiếp tục train thông thường thêm một khoảng thời gian tương đương.**

Sau đó mới so với:

**Cùng baseline → tiếp tục train bằng cơ chế phản hồi đề xuất.**

Nếu phương pháp phản hồi được học thêm mà baseline không được học thêm, mức tăng có thể chỉ phản ánh **ngân sách huấn luyện lớn hơn**. Và nhãn dùng để tính phản hồi phải thuộc phần dữ liệu dành cho học; dùng test để tạo reward rồi lại chấm trên test sẽ khiến kết quả mất ý nghĩa kiểm tra độc lập.

**Đánh giá:** bài hữu ích để tham khảo cách fine-tune một detector nhỏ và nhìn lỗi theo lớp. Phần cần thận trọng nhất là **nguồn gốc bản dữ liệu, phạm vi 27/30 lớp, cách lấy trung bình metric và phép so sánh với benchmark cũ**. Đây cũng là những điểm có thể làm rõ hơn để luận văn có bằng chứng thuyết phục hơn.

## Tài liệu nguồn

1. [Plant Disease Object Detection on PlantDoc Using YOLO26n — PDF](https://ojs.unimal.ac.id/sisfo/article/view/27063/11079).
2. [Trang công bố bài báo](https://ojs.unimal.ac.id/sisfo/article/view/27063).
3. [PlantDoc: A Dataset for Visual Plant Disease Detection — bài gốc](https://arxiv.org/abs/1911.10317).
4. [Kho PlantDoc Object Detection chính thức](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset).
5. [Ultralytics — Object Detection Datasets](https://docs.ultralytics.com/datasets/detect/).
6. [Ultralytics — YOLO26](https://docs.ultralytics.com/models/yolo26/).
7. [Ultralytics — Model Training](https://docs.ultralytics.com/modes/train/).
8. [Ultralytics — YOLO Performance Metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/).
9. [Ultralytics — utils.metrics API và mã tính metric](https://docs.ultralytics.com/reference/utils/metrics/).

**Ghi chú nguồn:** tài liệu Ultralytics được đối chiếu tại thời điểm phân tích; các giá trị mặc định và triển khai hiện tại không chứng minh cấu hình chính xác của lần chạy trong bài báo. Các tỷ lệ, trung bình theo lớp và hai cách tính F1 trong báo cáo là phép tính lại từ số liệu công bố, không phải kết quả tái huấn luyện.
