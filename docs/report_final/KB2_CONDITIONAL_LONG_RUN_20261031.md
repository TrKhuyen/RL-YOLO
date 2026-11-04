# KB2 Conditional V0 — lượt xác nhận dài, 31/10/2026

**Cập nhật: đã hoàn tất cảhai arm15.000 update và kiểm GPU lại.** Xem [review kết quả](KB2_CONDITIONAL_LONG_REVIEW_20261031.md). Phần kế hoạch dưới được giữ để truy vết. Run kiểm tra khả năng
cải thiện muộn của nguyên recipe V0 sau [pilot2k](KB2_CONDITIONAL_PILOT_REVIEW_20261031.md).
Không áp dụng thay đổi native background/lambda được gợi ý trong audit.

## Cấu hình khóa

- YOLOv8n, `conditional_detection_dpo_v0`, seed42, float32/TF32 off, CUDA RTX4060.
- Mỗi arm15.000 update, chạy tuần tự **DPO trước, native-only sau**; tổng30.000
  update. Cảhai khởi tạo từ S0 và optimizer mới; không resume checkpoint pilot.
- Giữ native batch4 (2 full-train +2 preference), DPO2 quyết định/step; input640.
- LR1e-6, AdamW WD5e-4, clip1, beta0,1, lambda0,01; native coefficient1.
- Chỉ classification cv3 trainable; regression/backbone/neck và BN statistics cố định.
- Dùng bank đã audit:784 cặp,663 cặp thuộc DPO train; không refresh/sửa annotation.
- Sampler/bank/seed/model mask và loss giữ nguyên. Đã kiểm bằng CPU sampling:
  toàn bộ4.000 draws trong2k update đầu giống pilot; phần mới kéo dài tới15k.
- Validation đủ479 ảnh tại0 và mỗi1k đến15k; không dùng test. Ngân sách15k cố định,
  không early stop theo việc recall của pilot đã giảm. Lỗi số học/provenance vẫn
  làm trainer dừng và lưu traceback trong log.
- Best theo macro recall với micro recall>=S0, FP<=1,05*S0, mAP>=S0−0,001;
  S0 vẫn là ứng viên. Lưu `best.pt`/`last.pt` riêng cho mỗi arm.

Trainer giữ nguyên để bảo toàn hash bank/objective; metadata `pilot=True` kế thừa
của trainer không mô tả độ dài đợt này. Ngân sách thực tế là15.000/arm, được ghi
trong request/config/plan; companion plan đánh dấu `conditional_v0_long_confirmation`.

## Theo dõi

- Run: `kb2_preference_optimization/outputs/yolov8n_conditional_long_s42_20261030_200209_427399`.
- [Training log](/home/celine/Celine/graduate/RL-YOLO/kb2_preference_optimization/outputs/_launches/yolov8n_conditional_long_s42_20261030_200209_427399/training.log).
- [Plan và PID](/home/celine/Celine/graduate/RL-YOLO/kb2_preference_optimization/outputs/_launches/yolov8n_conditional_long_s42_20261030_200209_427399/launch.json), PID lúc khởi chạy: `88654`.
- DPO: `/home/celine/Celine/graduate/RL-YOLO/kb2_preference_optimization/outputs/yolov8n_conditional_long_s42_20261030_200209_427399/dpo/metrics.jsonl`; native tạo sau khi DPO hoàn tất.
- Marker hoàn tất toàn bộ: `/home/celine/Celine/graduate/RL-YOLO/kb2_preference_optimization/outputs/yolov8n_conditional_long_s42_20261030_200209_427399/completed.json`.
- Process chạy nền với session riêng, stdin đóng và stdout/stderr vào file log.

Kết quả cũ và S0 được giữ nguyên; đã đối chiếu SHA S0 và DPO pilot-last khi khởi chạy.
Marker `completed.json` đã có. Review cuối xác nhận hash/stream và fresh GPU metrics;
primary FN924 so S0 FN866, matched-FP FN860 ở cả native/DPO. Xem review để phân biệt
protocol chính và threshold diagnostic hậu nghiệm.
