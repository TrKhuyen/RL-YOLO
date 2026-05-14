# Đề Xuất Nghiên Cứu Mới Cho Bài Toán UAV Pest Detection

## 1) Mục tiêu

Xây dựng một hướng **có novelty học thuật rõ ràng** thay vì chỉ triển khai lại pipeline CE -> RL đã có trong tài liệu.

Bối cảnh hiện tại:
- Bạn đã có pipeline chạy được cho YOLOv5/v8/v11/DP-YOLO.
- Reward hiện tại chủ yếu là recall + small-object recall.
- Điểm cần nâng cấp: tạo đóng góp mới ở mức phương pháp.

---

## 2) Đề xuất 3 hướng novelty chính

## Hướng A: Risk-Aware Reward (ưu tiên không bỏ sót lớp nguy hiểm)

Ý tưởng:
- Không coi mọi lỗi miss là như nhau.
- Thiết kế reward theo mức rủi ro thực địa: bỏ sót sâu hại nặng bị phạt cao hơn bỏ sót lớp nhẹ.

Công thức:

$$
R = \alpha R_{recall} + \beta R_{small} + \gamma R_{risk} - \lambda R_{fp}
$$

Trong đó:
- $R_{risk}$: recall có trọng số theo lớp, với trọng số $w_c$ lấy từ mức độ thiệt hại nông học.
- $R_{fp}$: phạt false positive để tránh model tăng recall bằng cách đoán tràn lan.

Điểm mới:
- Reward gắn trực tiếp với giá trị ứng dụng nông nghiệp, không chỉ metric thị giác máy tính thuần.
- Phù hợp bài toán UAV ngoài đồng, nơi chi phí bỏ sót rất không đồng đều giữa các lớp bệnh/sâu.

---

## Hướng B: Uncertainty-Guided RL (điều chỉnh theo độ bất định)

Ý tưởng:
- Dùng độ bất định dự đoán (uncertainty) để điều tiết mức cập nhật policy.
- Những mẫu model đang rất không chắc chắn sẽ được ưu tiên học nhiều hơn.

Cách làm gợi ý:
- Ước lượng uncertainty từ entropy của phân phối lớp hoặc variance từ test-time augmentation.
- Tạo hệ số $u_i$ cho mỗi ảnh/bbox, rồi scale advantage:

$$
A_i' = u_i \cdot (R_i - b)
$$

Điểm mới:
- REINFORCE trong object detection thường chưa tận dụng uncertainty có hệ thống.
- Có thể giúp ổn định training và tập trung vào case khó (nhòe, occlusion, backlight).

---

## Hướng C: Scale-Adaptive Policy Update (RL theo tầng P2/P3/P4)

Ý tưởng:
- Không dùng một loss RL chung cho mọi scale.
- Tách reward theo scale head (P2/P3/P4), với trọng số động theo mật độ object nhỏ trong batch.

Công thức gợi ý:

$$
L_{RL} = -\sum_{s \in \{P2,P3,P4\}} \eta_s \cdot \mathbb{E}[\log \pi_s(a|s) \cdot A_s]
$$

Trong đó:
- $\eta_s$: trọng số scale, tự điều chỉnh theo tỉ lệ object tương ứng trong batch.
- $A_s$: advantage tính riêng cho scale.

Điểm mới:
- Bám rất sát đặc trưng DP-YOLO (nhấn mạnh P2 cho vật thể nhỏ).
- Tạo novelty ở mức coupling giữa kiến trúc multi-scale và policy gradient.

---

## 3) Gói đề tài khuyến nghị (để dễ publish)

Đề xuất chọn 1 gói chính:

**Gói khuyến nghị mạnh nhất:** A + C
- A (risk-aware) cho giá trị ứng dụng thực tế.
- C (scale-adaptive) cho giá trị thuật toán gắn với DP-YOLO.

Tên đề tài gợi ý:
- **Risk- and Scale-Aware Reinforcement Fine-tuning for DP-YOLO in UAV Pest Detection**

Giả thuyết nghiên cứu:
1. Risk-aware reward tăng recall ở lớp nguy hiểm mà không làm mAP giảm đáng kể.
2. Scale-adaptive RL cải thiện AP_small tốt hơn RL chuẩn một-loss.
3. Kết hợp A+C cho trade-off tốt nhất giữa AP_small, recall lớp nguy hiểm và FPS.

---

## 4) Thiết kế thí nghiệm đề xuất

Baseline:
- B0: Supervised only (không RL)
- B1: RL chuẩn hiện tại (composite reward hiện có)

Ablation:
- E1: B1 + Risk-aware reward
- E2: B1 + Scale-adaptive update
- E3: B1 + Risk-aware + Scale-adaptive
- E4: E3 + freeze backbone
- E5: E3 + no freeze backbone

Metric báo cáo:
- mAP50, mAP50-95
- AP_small
- Recall theo lớp
- Risk-weighted Recall (metric mới)
- FPS và latency trên thiết bị mục tiêu

Tiêu chí thành công tối thiểu:
- AP_small tăng >= 2.0 điểm so với B1
- Risk-weighted Recall tăng >= 3.0 điểm so với B1
- FPS giảm không quá 10%

---

## 5) Kế hoạch triển khai 6 tuần

Tuần 1:
- Chuẩn hóa trọng số rủi ro lớp $w_c$ từ chuyên gia/tri thức nông học.
- Định nghĩa metric Risk-weighted Recall.

Tuần 2:
- Implement Risk-aware reward vào reward.py.
- Chạy pilot 10k steps để kiểm tra ổn định.

Tuần 3:
- Implement scale-adaptive RL loss trong train_rl.py.
- Log riêng theo P2/P3/P4.

Tuần 4:
- Chạy full train cho E1, E2.
- So sánh với B1.

Tuần 5:
- Chạy E3, E4, E5.
- Tổng hợp bảng kết quả và kiểm định thống kê cơ bản.

Tuần 6:
- Viết phần kết quả, ablation, error analysis và đóng góp.

---

## 6) Đóng góp kỳ vọng để viết luận văn/bài báo

Đóng góp 1:
- Một reward mới có ý nghĩa miền ứng dụng (risk-aware) cho UAV pest detection.

Đóng góp 2:
- Một cơ chế RL theo scale cho detector multi-head, đặc biệt phù hợp kiến trúc DP-YOLO.

Đóng góp 3:
- Bộ benchmark so sánh công bằng giữa YOLOv5/v8/v11/DP-YOLO trước và sau RL.

---

## 7) Hành động tiếp theo ngay trong codebase

1. Thêm `risk_aware_reward()` vào `reward.py`.
2. Mở rộng `compute_reward()` với `reward_type: risk | risk_composite`.
3. Sửa `train_rl.py` để hỗ trợ scale-adaptive coefficients cho P2/P3/P4.
4. Cập nhật `configs/hyp.rl.yaml` với `class_risk_weights`, `eta_p2`, `eta_p3`, `eta_p4`.
5. Cập nhật `evaluate.py` để tính thêm `risk_weighted_recall`.

Nếu bạn muốn, bước tiếp theo mình có thể code luôn toàn bộ phần A (Risk-aware) trước, vì đây là phần dễ triển khai và tạo novelty rõ nhất.