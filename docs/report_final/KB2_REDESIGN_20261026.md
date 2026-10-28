# KB2 — Thiết kế DPO mới và kiểm chứng kỹ thuật ngày 26/10/2026

> Bổ sung cùng ngày: reference đã chuyển từ cache batch1 sang forward cùng
> batch với policy. Xem [bản sửa và kiểm chứng GPU](KB2_REFERENCE_BATCH_FIX_20261026.md).
> Các số pilot CPU dưới đây ghi giai đoạn kiểm chứng trước bản sửa đó.

## Quyết định phạm vi

KB1, KB2, KB3 là ba hướng độc lập. Mỗi hướng lấy best checkpoint supervised
của KB1 tương ứng từng kiến trúc làm điểm bắt đầu và mốc so sánh. KB2 chỉ
tập trung DPO cho detection; không nối tiếp reward-guided KB1 hoặc HPO KB3.

Đơn vị preference là **hai bộ detection trên cùng ảnh**, không phải hai box
quanh một GT. Implementation v1 dùng action có candidate indices và các
quyết định background/class/tọa độ, được chấm chất lượng toàn bộ output.
Không tuyên bố exact likelihood của unordered post-NMS detection set.

## Đã thay đổi

- Policy chuẩn hóa C+1 class/background ở mỗi grid slot, lấy mẫu bốn phía hộp
  từ raw DFL bin distributions. Tính full action log-prob gồm background.
- Sinh nhiều output chỉ trên train, không đưa GT vào sampler. Renderer/NMS
  dùng priorities từ S0 đóng băng; lưu action và rendered A/B để replay.
- GT chấm class-aware one-to-one; Q=0.8*mean F1@IoU0.50:0.05:0.95 +
  0.2*GT-normalized matched IoU. Giữ best/worst đủ gap, bỏ gần hòa/trùng output.
- `L_total = L_native + 0.01*L_DPO`, β=0.1 là cấu hình thử. Chỉ cập nhật head;
  reference/backbone/neck, DFL integral và BN running stats được giữ cố định.
- Native và DPO replay cùng ảnh resize/pad xác định; chưa random augmentation
  với offline actions. Native loss trong v1 dùng các ảnh có preference được giữ;
  coverage của ảnh/class bị loại cần audit trước khi xác nhận nghiên cứu.
- Step0 là ứng viên best; selection bằng canonical validation mAP50–95.
- S0/DPO đều được đo lại bằng ordinary deterministic inference, conf0.001,
  NMS0.60/cap300, cùng transform/evaluator. Không tái dùng AP cũ khác contract.
- Thêm mAP75, AP từng lớp, APs/m/l với support, AR300, P/R/F1@conf0.25/IoU0.5,
  FP/ảnh, FN rate, matched IoU+coverage, fixed train audit và gradient diagnostics.
- Chặn sai source/data/code/runtime, hash mismatch, historical checkpoint,
  output overwrite và final-test evaluation của pilot. Chưa hỗ trợ exact resume.

## Dọn dẹp

Đã xóa **251 file, 6.630.304.190 byte (~6.63 GB)** thuộc checkpoint/feedback
và output KB2 cũ, bao gồm candidate-DPO lịch sử. Code/tài liệu cũ được chuyển
vào `legacy/`, giữ result/metrics nhỏ và [inventory SHA256](../../kb2_preference_optimization/legacy/cleanup_manifest.json).
Không dọn checkpoint KB1/KB3 hoặc dataset/pretrained chung.

## Kiểm chứng đã thực hiện

| Kiểm tra | Kết quả | Giới hạn |
|---|---|---|
| Unit + real-YOLO CPU gradient tests | 18/18 pass | Không thay training dài |
| Full action probability trên support nhỏ | Tổng bằng1 | Policy factorized, chưa phải set marginal |
| Gradient DPO | Class và regression head có signal | Không bảo đảm mAP tăng |
| Fixed reference + BN/backbone | Bất biến qua kiểm tra | Trong phạm vi test/pilot |
| Adapter bốn model | YOLOv8n/s, YOLO11n/s: nc28, DFL16 | Train pilot chỉ YOLOv8n |
| S0/data provenance | PASS | Dataset manifest dự án |
| Sinh cặp input640 | 8 ảnh, 32 output, 8 cặp giữ | Đầu split train, không đại diện toàn dataset |
| Độ đa dạng output | 47/48 cặp so sánh khác nhau | Một so sánh trùng được nhận diện |
| Wrapper + train/save/evaluate CPU | 2 optimizer updates, 2 ảnh validation | Đánh dấu PILOT, không full validation/test |
| Reference khi kết thúc pilot | Không đổi | Được kiểm bằng digest parameters/buffers |
| ΔmAP50–95 trên 2 ảnh pilot | 0 | Không phải bằng chứng gain hoặc không hiệu quả |
| Dependency lock | Offline check pass, 137 packages | Không tải/cài lại môi trường |

Một pilot trước ở input640 với λ=0.1 có weighted DPO/native gradient norm
khoảng1.87 ở batch đầu. Giảm λ mặc định thử xuống0.01; pilot cuối đo khoảng
0.187. Đây chỉ là diagnostic một batch, không phải sweep tham số tối ưu.

Bằng chứng nhỏ: [verification.json](assets/kb2_20261026/verification.json).
Các checkpoint pilot kỹ thuật nằm trong `/tmp/kb2-structured-v1-verified`,
không phải kết quả nghiên cứu dài hạn hoặc artifact cần phát hành.

![Ví dụ thật: GT, chosen và rejected của cùng ảnh train](assets/kb2_20261026/preference_example.png)

Ảnh minh họa lấy từ pilot, không đại diện chất lượng toàn tập. Chosen có Q
cao hơn vì cách gán nhãn preference; điều đó tự nó không chứng minh policy học.

## Phần cần xác nhận tiếp

1. Audit nhiều ảnh/classes/sizes ở input640: pair coverage, chênh Q, mức biến
   dạng do DFL sampling, FP/FN và preference noise.
2. Pilot nhiều batch để chốt temperature, background logit, β/λ và quality gap.
3. Full canonical validation, nhiều continuation seed; khóa method trước test.
4. Xác nhận lợi ích chuyển được sang ordinary deterministic YOLO inference.
5. YOLO26 DFL-free cần policy tọa độ riêng; hiện CLI từ chối, chưa giả lập hỗ trợ.

So với S0 đo hiệu quả quy trình native+DPO tổng thể; muốn quy gain riêng cho
DPO cần ablation phù hợp. Không diễn giải test kỹ thuật này thành kết luận đó.

Hướng dẫn chạy: [KB2 README](../../kb2_preference_optimization/README.md).
Định nghĩa/công thức: [DPO.md](../../kb2_preference_optimization/docs/DPO.md).
Báo cáo [15/10](Report_KB2_15_10_2026.md) là lịch sử objective cũ.
