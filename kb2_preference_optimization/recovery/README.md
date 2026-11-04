# Pilot chọn cặp khôi phục GT

Code thử nghiệm tách khỏi F2 gốc. Các file Python ở đây có fingerprint riêng
trong audit/manifests, được kiểm tra khi train và review.

Sinh bốn action trên mỗi ảnh train; xét sáu cặp. Cặp khôi phục phải cứu thêm
GT, không mất GT khác, không tăng FP, Q-F2 gap ≥ .02. Giữ tối đa một cặp/ảnh,
ưu tiên khôi phục rồi duy trì cùng tập GT. Native dùng full train với chuỗi
ảnh/RNG cố định độc lập sampler DPO. Ảnh probe vẫn được native sử dụng.

Runner chạy ba pilot seed42/2.000 update: đối chiếu F2, selector mới,
selector mới cộng cân bằng theo class. Sampler nhắm 50% khôi phục nhưng
giới hạn xác suất mỗi ảnh tối đa 5 lần mức lấy đều; tỷ lệ đạt được được báo cáo.
Đây là pilot cơ chế, không phải kết quả run dài hoặc test.

```bash
YOLO_CONFIG_DIR=/tmp MPLCONFIGDIR=/tmp/kb2-matplotlib NO_ALBUMENTATIONS_UPDATE=1 .venv-ubuntu/bin/python scripts/run_kb2_recovery_pilot.py --root kb2_preference_optimization/outputs/yolov8n_recovery_NEW
```

Luôn dùng thư mục mới. Runner chỉ bỏ qua những pilot đã hoàn tất; không resume
trọng số từ một pilot bị gián đoạn. Dùng entrypoint recovery để giữ native/DPO
streams và quy tắc probe đúng protocol này.

Chạy dài riêng phương án cân bằng, đủ15.000 update từ S0, không resume pilot:

```bash
YOLO_CONFIG_DIR=/tmp MPLCONFIGDIR=/tmp/kb2-matplotlib NO_ALBUMENTATIONS_UPDATE=1 .venv-ubuntu/bin/python scripts/run_kb2_recovery_long.py --device cuda --steps 15000
```

Runner tự tạo thư mục `outputs/yolov8n_recovery_long_<timestamp>/` theo giờ
Asia/Bangkok, sinh lại preference rồi train. `pilot_balanced/best.pt` và
`pilot_balanced/last.pt` thuộc run mới; không ghi vào checkpoint cũ. Có thể dùng
`--output-root` để đặt tên, nhưng thư mục đã tồn tại sẽ bị từ chối.
Trainer vẫn giữ flag experimental/pilot dù ngân sách dài; tăng số update
không tự xác nhận chất lượng phương pháp. Validation mỗi1.000 update, best
theo mAP50–95; đây là ngân sách cố định, không có early stopping.

Kết quả ngày 29/10: tăng lượt DPO khôi phục từ 8,03% lên 46,37%, nhưng cả ba
có FN916/FP785 tại 2k; chưa xác nhận lợi ích detection. Xem
[báo cáo đầy đủ](../../docs/report_final/KB2_RECOVERY_PILOT_20261029.md).
