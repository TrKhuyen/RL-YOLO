# KB1 — Định nghĩa và phạm vi nghiên cứu hiện hành

**Cập nhật:** 26/10/2026.

**Mục tiêu:** tối ưu thêm chất lượng detector YOLO bằng reward-guided/RL-inspired fine-tuning sau supervised training.

**Báo cáo tổng hợp:** [KB1 — phương pháp, kết quả và kết luận](Report_KB1_15_10_2026.md).

## 1. Hai giai đoạn của KB1

Mỗi kiến trúc YOLO được huấn luyện supervised riêng từ pretrained tương ứng và tạo checkpoint S0. KB1B là stage hai: bắt đầu đúng S0 của model đó, đóng băng feature extractor/BN và tối ưu Detect head bằng native loss, task reward, confidence proxy và L2-SP.

```text
Pretrained → supervised → S0 của từng model
                         → KB1B reward-guided fine-tuning → guided checkpoint G
```

Hai câu hỏi chính:

1. G cải thiện bao nhiêu so với S0 trong cùng model, dataset và evaluator?
2. Guided model nào tốt nhất theo canonical validation mAP50–95 đã khóa?

Validation chọn checkpoint; test đánh giá sau khi khóa lựa chọn. Metric chính là canonical mAP50–95, không thay bằng reward hoặc metric phụ sau khi xem kết quả.

## 2. Objective và các phép so

Objective chính: `L_G = 0.25 L_native + 0.10 L_reward + 0.25 L_proxy + 0.001 L_SP`.

Reward dùng precision/recall/IoU từ prediction–GT train; normalized advantage dựa trên EMA. Proxy hướng confidence theo TP/FP/FN. Native loss cung cấp gradient detection theo GT, gồm định vị. Score giữ gradient, còn box/NMS/matching detach. Implementation được gọi là reward-guided/RL-inspired vì chưa định nghĩa stochastic detection policy có log-likelihood action tường minh.

| Phép so | Nội dung đo | Phạm vi thí nghiệm |
|---|---|---|
| G−S0 | Lợi ích thêm stage hai so supervised | Bảy model, continuation seeds 42/43/44 |
| G−S1 | Thêm reward và proxy so native+SP continuation | Bảy model; early stopping có thể khác số update |
| G−P | Thêm reward khi giữ proxy | Ablation YOLOv11n/YOLOv8s, ba seed, cùng step 10.000 |
| G−R | Thêm proxy khi giữ reward | Cùng bộ ablation |
| G−G-noSP | Tác dụng của starting-point regularization | Cùng bộ ablation |

**Chú thích bảng 1.** S1 là native-only+L2-SP, P là proxy-only, R là reward-only; các nhánh đều giữ native loss. Delta dùng điểm phần trăm AP. Continuation chọn best validation, còn ablation đánh giá last tại cùng 5.000 optimizer update.

**Nhận xét bảng 1.** G−S0 trả lời ý tưởng chính của KB1. Các phép so khác giải thích gain đến từ thành phần nào; cần giữ riêng để không quy toàn bộ cải thiện cho reward hoặc trộn checkpoint khác protocol.

## 3. Trạng thái hiện tại và kết quả

| Hạng mục | Trạng thái |
|---|---|
| Supervised checkpoint sạch | 7/7 model, mỗi model S0 seed 0 |
| Cohort seed 42 | Đủ S0/S1/G, gồm YOLO26n |
| Continuation bổ sung seed 43/44 | 28/28 lượt train |
| Preflight bổ sung | 20/20 lượt |
| Ablation | 30/30 lượt: 2 model × 3 seed × 5 nhánh |
| Evaluation đợt bổ sung | 16/16 lượt, 156 hàng val/test |
| Pipeline bổ sung | completed, đủ 58 lượt train dài theo kế hoạch |

**Chú thích bảng 2.** Số lượt thuộc từng nhóm artifact, không cộng preflight như benchmark accuracy. Seed 42 của sáu model có nguồn training lịch sử; thống kê Ubuntu mới seed 43/44 được báo riêng. Các nhóm đều dùng S0 tương ứng đã xác minh hash/dataset.

**Nhận xét bảng 2.** KB1 đã hoàn tất phạm vi pipeline và các thí nghiệm đã chốt. Nhiều supervised initialization, bootstrap AP và tập kiểm tra độc lập khác là hướng kiểm chứng mở rộng, không phải job đang treo của script hiện tại.

Guided cải thiện S0 trong các cặp model–continuation seed đã khảo sát; YOLOv11n dẫn đầu cohort guided theo validation và test. Tuy nhiên, guided chưa luôn hơn native-only. Ablation G−P có mean test khoảng +0,070727 điểm ở YOLOv11n và −0,049986 điểm ở YOLOv8s; lợi ích tăng thêm của reward phụ thuộc model.

## 4. Nguồn và tái lập

- [Báo cáo tổng hợp KB1](Report_KB1_15_10_2026.md): giải thích công thức, thiết kế, bảng kết quả và kết luận hiện hành.
- [Review 25/10](KB1_RESULTS_REVIEW_20261025.md), [đánh giá 26/10](KB1_FINAL_ASSESSMENT_20261026.md): truy nguyên kết quả chi tiết.
- [Hướng dẫn Ubuntu](UBUNTU_KB1_20261022.md): môi trường, output và resume.
- [Plan](../../kb1_reward_guided_training/checkpoint_reward_guide_trainning/remaining_seed43_44_ablation_v1/plan.json), [completed](../../kb1_reward_guided_training/checkpoint_reward_guide_trainning/remaining_seed43_44_ablation_v1/completed.json), [mean/std](../../kb1_reward_guided_training/checkpoint_reward_guide_trainning/remaining_seed43_44_ablation_v1/summary_mean_std.csv).

Dữ liệu dùng `pre-data/data/v2i_cleanned`: 2411 train, 479 validation, 239 test; audit không thấy known source overlap theo policy hậu tố augment. Mỗi run phải xác minh checkpoint/dataset provenance. Khi đổi code/config/runtime hoặc thiết kế mới, dùng output mới; không sửa manifest lịch sử để vượt guard.
