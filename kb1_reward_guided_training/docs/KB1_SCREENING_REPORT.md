# BÁO CÁO KB1 — KIỂM CHỨNG VÀ SCREENING SEED 42

Cập nhật: 2026-09-16T06:24:20.
**Trạng thái: 0/12 lượt train hoàn tất; 0/6 cặp đủ kết quả so sánh.**

## Kết luận ưu tiên: tạm dừng screening dài vì rò rỉ dữ liệu

**149/592 ảnh validation (25,17%) và 67/296 ảnh test (22,64%) có cùng ID nguồn với train.** Đây là ảnh gốc và các bản flip nằm ở các tập khác nhau. Hai cặp được kiểm chứng bằng pixel: sau lật đúng hướng, tương quan lần lượt 0,999855 và 0,999483; sai khác trung bình 0,578 và 1,548 trên thang 0–255, phù hợp với lưu lại JPEG.

Train–validation trùng 115 nhóm nguồn; train–test trùng 60 nhóm; validation–test trùng 38 nhóm. Kiểm tra dựa trên ID nguồn đầy đủ, không chỉ trùng tên ngắn. Có thể còn dạng trùng khác mà phép kiểm tra này chưa bao phủ.

Code pre-data/split_dataset.py shuffle từng ảnh rồi cắt tập, không group theo nguồn. Code augment_class.py tạo hậu tố _augN_hflip/vflip/hvflip. Nếu chia sau khi gộp ảnh augmentation, thuật toán này cho phép rò rỉ đã quan sát; chưa có log để xác nhận chính xác lệnh lịch sử tạo dataset hiện tại.

### Dấu hiệu dùng để kết luận có leak

1. **Cùng ID nguồn xuất hiện ở nhiều split.** Sau khi chỉ bỏ hậu tố augmentation có quy tắc `_augN_hflip`, `_augN_vflip` hoặc `_augN_hvflip`, phần tên còn lại giống hoàn toàn, bao gồm cả mã Roboflow dài. Ví dụ ảnh train `000gb_jpg.rf.eeb741072e585dc54b4332616793266f.jpg` đi cùng ảnh validation `000gb_jpg.rf.eeb741072e585dc54b4332616793266f_aug2_hvflip.jpg`.

2. **Hậu tố mô tả đúng phép biến đổi trong code.** `augment_class.py` tạo ảnh bằng `flip_image_and_boxes`, rồi đặt tên `_aug{n}_{flip_str}`. Đây là quan hệ sinh ảnh có chủ đích, không phải hai ảnh độc lập tình cờ có tên gần giống.

3. **Nội dung pixel xác nhận quan hệ đó.** Với hai cặp mẫu, lật ảnh train theo đúng hậu tố làm tương quan pixel với ảnh validation đạt 0,999855 và 0,999483. MAE chỉ 0,578 và 1,548 trên thang 0–255; phần sai khác nhỏ phù hợp với nén JPEG lại.

4. **Cách chia tập tạo điều kiện cho hiện tượng này.** `split_dataset.py` shuffle từng file ảnh riêng lẻ rồi cắt danh sách thành train/valid/test. Script không nhóm ảnh gốc với các biến thể augmentation theo source ID trước khi chia.

### Leak ảnh hưởng như thế nào?

| Thành phần | Ảnh hưởng |
|---|---|
| Validation | Model được chọn checkpoint trên các mẫu có nội dung đã xuất hiện trong train; mAP, AR, precision, recall và F1 có nguy cơ lạc quan hơn dữ liệu hoàn toàn mới. |
| Early stopping và best checkpoint | Validation dễ hơn có thể làm thay đổi best step và quyết định dừng, nên ảnh hưởng trực tiếp tới checkpoint cuối của cả KB1-A và KB1-B. |
| Test | 67/296 ảnh test cùng nguồn với train, nên test hiện tại không còn là phép đánh giá độc lập đáng tin cậy về khả năng tổng quát hóa. |
| So sánh model | Mọi model dùng cùng split nên bảng vẫn có giá trị chẩn đoán nội bộ, nhưng mức thiên lệch có thể khác theo kiến trúc và khả năng bất biến với flip; thứ hạng có thể đổi trên split sạch. |
| Delta KB1-B | B−A và B−native-only vẫn hữu ích như tín hiệu kỹ thuật trong cùng protocol, nhưng KB1-B chọn checkpoint trên validation bị leak nên chưa thể coi delta là bằng chứng tổng quát hóa. |
| Độ chắc chắn thống kê | Các bản flip cùng nguồn là các mẫu tương quan mạnh, làm số quan sát độc lập thực tế nhỏ hơn số file; khoảng tin cậy nếu coi mọi file độc lập sẽ quá lạc quan. |

Audit chứng minh sự giao nhau theo source ID và xác minh pixel trên hai cặp đại diện. Nó không định lượng chính xác số điểm mAP bị tăng do leak; muốn biết mức thiên lệch phải đánh giá lại trên split sạch. Phép kiểm tra hiện tại cũng có thể bỏ sót ảnh trùng nội dung nhưng mang ID khác, vì vậy các con số trên là bằng chứng tối thiểu đã xác nhận.

Vì vậy, các metric hiện có vẫn mô tả tập đánh giá này nhưng không đủ để chứng minh khả năng tổng quát hóa. Smoke test đạt không có nghĩa thiết kế thực nghiệm đã hợp lệ. Việc train dài được chặn trước khi khởi động; chưa hoàn tất screening hoặc tối ưu hiệu năng.

**Phương án đề xuất:** tạo dataset mới ở thư mục riêng, chia nhóm theo ảnh nguồn trước augmentation; chỉ augment train, giữ validation/test là ảnh gốc; audit không giao nguồn và kiểm tra phân bố đủ lớp. Sau đó chạy lại KB1-A cho sáu model và hai nhánh KB1-B seed 42. Audit tìm thấy 2.463 ID nguồn đầy đủ và cả 2.463 nhóm đều còn ảnh gốc, nên có thể chuẩn bị split mới từ ảnh gốc; vẫn cần kiểm tra các nguồn trùng nội dung nhưng khác ID. Checkpoint hiện tại đã học trên split cũ, nên chia lại tập rồi tiếp tục dùng chúng không tự loại bỏ rò rỉ. KB1-A mới cần khởi tạo lại từ pretrained chung ban đầu. Cần thống nhất việc mở rộng sang chạy lại KB1-A trước khi thực hiện.

Bằng chứng: ../results/dataset_audit/summary.json và overlap_pairs.csv. Dataset và checkpoint gốc chưa bị thay đổi. Kiểm tra test ở đây chỉ audit nguồn ảnh, không đánh giá model hoặc dùng metric test để tuning.


## 1. Phạm vi và tên gọi

KB1 là Huấn luyện YOLO hai giai đoạn: baseline có giám sát và tinh chỉnh có hướng dẫn bởi reward. KB1-A tạo best checkpoint riêng cho sáu model. KB1-B tiếp tục tối ưu từ chính checkpoint tương ứng, rồi so sánh trước/sau trong từng model. Nhánh native-only là đối chứng tiếp tục train, dùng để đánh giá phần bổ sung reward và proxy trong KB1-B.

Tên phương pháp chính xác: reward-guided fine-tuning (RL-inspired). Prediction xác định sau NMS và log-confidence là surrogate; không phải DPO hay REINFORCE chính xác.

## 2. Các vấn đề đã phát hiện và sửa

- Dataloader cũ âm thầm resize trực tiếp khi augmentation lỗi. Nhãn vượt biên rất nhỏ do làm tròn tọa độ đã thực sự kích hoạt lỗi này. Bản mới clip góc box trước transform, kiểm tra ảnh/nhãn và dừng rõ ràng nếu còn lỗi.
- Cố định seed augmentation theo seed/epoch/chỉ số ảnh, cùng thứ tự batch giữa hai nhánh. Sampler có thể tiếp tục đúng vị trí sau khi ngắt.
- Giữ cố định trọng số tích phân DFL của YOLOv8/11; đóng băng backbone/neck và thống kê BatchNorm.
- Native-only trực tiếp tính native loss + L2-SP. Không tạo prediction giả hoặc ghi reward giả.
- Best checkpoint được cập nhật theo score cao nhất. Min-delta chỉ điều khiển early stopping; checkpoint tăng nhẹ vẫn được giữ.
- Lưu optimizer, EMA, RNG và bước train để resume tại ranh giới accumulation; hash code/checkpoint ngăn resume khác cấu hình. Ghi file qua file tạm rồi thay thế.
- Tính matching box trên CPU rồi lấy score có gradient trên GPU để giảm đồng bộ trong vòng lặp.

Audit một epoch augmentation đã đọc đủ **2.072 ảnh train**, thu được **7.929 box** sau augmentation. Đây là kiểm tra kỹ thuật dữ liệu, không phải chỉ số chất lượng model.

## 3. Evaluator chung

Protocol: **kb1_canonical_v2_ar300**. Chọn checkpoint và screening trên **592 ảnh validation**. Tập test chưa dùng để tuning.

| Thành phần | Quy tắc |
|---|---|
| Input | RGB float32 [0,1]; resize cạnh dài 640; pad giữa bằng 114 |
| Nhãn | Clip góc box trước transform; xyxy trong ảnh 640×640 |
| AP | Confidence ≥0,001; IoU 0,50:0,05:0,95 |
| NMS | Theo lớp; IoU 0,60; tối đa 300 box/ảnh |
| Chọn lớp | Một lớp có score cao nhất trên mỗi candidate |
| Metric | torchmetrics với faster-coco-eval; cap300 |
| AR300 | Average Recall qua các ngưỡng IoU, tối đa 300 detection |
| P/R/F1 | Confidence 0,25; matching cùng lớp; IoU 0,50 |
| APs/APm/APl | Diện tích box trong ảnh đã resize, không phải ảnh gốc |

AR300 khác operating_recall. COCO cap100 cũng là protocol hợp lệ; ở đây cố định cap300 và ghi tên riêng. Không trộn điểm evaluator cũ với điểm mới. Ví dụ YOLOv8n baseline mAP50–95 đổi từ khoảng 0,4603 sang 0,4739 sau sửa tiền xử lý; đây không phải tăng do train.

## 4. Phương pháp train và đối chứng

Hai nhánh cùng seed 42, cùng checkpoint KB1-A, cùng augmentation, chỉ train Detect head. Adam, batch 8, accumulation 2 = 16 ảnh/update. LR 5e-7; warmup 500 micro-step từ 1e-7, sau đó cosine về 1e-7. Ngân sách tối đa 10.000 micro-step = 5.000 optimizer update, eval mỗi 500 bước. Early stopping sau 4 lần eval không tăng quá 0,0005 so với mốc patience. Cùng ngân sách tối đa nhưng số bước thực tế có thể khác nhau do early stopping.

**Native-only:** L = 0,25 L_native + 0,001 L2-SP.

**KB1-B:** L = 0,25 L_native + 0,001 L2-SP + 0,10 L_reward + 0,25 L_proxy.

YOLOv5/DP-YOLO dùng ComputeLoss và hyp gắn trong checkpoint (box/objectness/classification). YOLOv8/11 dùng native criterion Ultralytics (box/classification/DFL). Loss chia batch size để quy về mỗi ảnh. Thang native loss khác nhau theo họ model; cùng hệ số không đồng nghĩa cùng độ lớn gradient. Đối chứng chính là ghép cặp trong từng model.

YOLOv5/DP-YOLO load qua backend fuse Conv/BN với feature extractor đóng băng. YOLOv8/11 giữ model chưa fuse, BN eval và DFL cố định. Đây là continued fine-tuning bằng native criterion trong cấu hình chung, không tái hiện đầy đủ native trainer.

Reward gộp precision/recall/IoU theo trọng số 0,25/0,45/0,30 trên IoU 0,5/0,6/0,7/0,8. EMA chuẩn hóa advantage. Proxy tăng confidence TP, giảm FP và dùng max candidate score cho FN. FN proxy chưa định vị riêng từng GT bị bỏ sót; reward không truyền gradient qua box/NMS. B−native đo đóng góp kết hợp reward và proxy, chưa tách riêng reward.

DP-YOLO screening cố định W3F=OFF, PSA=OFF theo mặc định code hiện tại; native loss dùng CIoU và assignment chuẩn. Chưa có metadata môi trường KB1-A chứng minh baseline đã bật W3F/PSA. Tên model không đủ để khẳng định hai thành phần này đã tham gia huấn luyện.

Score chọn best = 0,4 mAP50 + 0,4 mAP50–95 + 0,2 AR300. Step 0 cũng có thể là best sau một lượt đã hoàn tất. Chỉ completed.json xác nhận lượt hoàn tất.

## 5. Kết quả validation

Chỉ đưa vào bảng các lượt hoàn tất và đã kiểm tra best checkpoint nạp lại. Chỉ số hiển thị theo %, delta là điểm phần trăm.

| Model | Giai đoạn | Best step | mAP50 | mAP50–95 | AR300 |
|---|---|---:|---:|---:|---:|

| Model | ΔmAP50–95 B−A | ΔmAP50–95 B−native |
|---|---:|---:|

## 6. Đánh giá và hướng tiếp theo

Đang chạy screening; chưa đủ sáu cặp để kết luận tổng quát hoặc chọn tối ưu tiếp.

## 7. Trạng thái và bằng chứng

| Model | Nhánh | Trạng thái | Bước đã ghi nhận |
|---|---|---|---:|
| yolov5s | native_only | pending | 0 |
| yolov5s | kb1b | pending | 0 |
| yolov8n | native_only | pending | 0 |
| yolov8n | kb1b | pending | 0 |
| yolov8s | native_only | pending | 0 |
| yolov8s | kb1b | pending | 0 |
| yolov11n | native_only | pending | 0 |
| yolov11n | kb1b | pending | 0 |
| yolov11s | native_only | pending | 0 |
| yolov11s | kb1b | pending | 0 |
| dp_yolo | native_only | pending | 0 |
| dp_yolo | kb1b | pending | 0 |

- results_full.csv: điểm từng model/giai đoạn; results_delta.csv: B−A và B−native.
- Mỗi nhánh: manifest.json, baseline.json, train.jsonl, validation.jsonl, best.pt, last.pt và completed.json.
- Smoke test hai bước chỉ kiểm chứng kỹ thuật, không dùng làm kết quả nghiên cứu.
- Kiểm tra tự động: nạp checkpoint vào model dựng mới; gradient hữu hạn/nonzero; head thay đổi; frozen parameters và BN không đổi; best checkpoint khớp metric sau reload.
- Test hành vi: NMS/gradient, matching duplicate/sai lớp, hướng gradient proxy, letterbox nhãn sát biên, sampler/augmentation sau resume và evaluator prediction hoàn hảo.
- Kết quả: ../results/screening_seed42_v2/. Lõi train: run_verified.py.

## 8. Kết quả smoke test của runner hiện tại

| Model | native-only | KB1-B |
|---|---|---|
| yolov5s | PASS (2 bước) | PASS (2 bước) |
| yolov8n | PASS (2 bước) | PASS (2 bước) |
| yolov8s | PASS (2 bước) | PASS (2 bước) |
| yolov11n | PASS (2 bước) | PASS (2 bước) |
| yolov11s | PASS (2 bước) | PASS (2 bước) |
| dp_yolo | PASS (2 bước) | PASS (2 bước) |

Bộ test hành vi: 6/6 đạt; bộ kiểm tra KB1 có sẵn: 19/19 đạt. Các test kiểm tra code, không xác nhận tập validation/test độc lập với train.
