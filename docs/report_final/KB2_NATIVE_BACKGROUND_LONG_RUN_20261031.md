# KB2 — chạy dài background0,5, 31/10/2026

> Lượt mới nhất01/11: **DPOλ0,1/15k đã xong**, best_AP tại12k có mAP38,1363%, FP891/FN882. Xem [review lượt dài tăng λ](KB2_DPO_STRENGTH_LONG_REVIEW_20261101.md). Các kết quả và selector bên dưới thuộc lượtλ0,01 cũ.

> Tiếp nối01/11: đã hoàn tất [pilot tăng λ DPO](KB2_DPO_STRENGTH_PILOT_20261101.md)
> và [audit candidate swap](KB2_CANDIDATE_RANK_AUDIT_20261101.md). Best của pilot
> mới chọn theo AP, lưu đủ checkpoint eval; các checkpoint run dài dưới giữ nguyên.

**Cập nhật: đã hoàn tất hai lượt15.000 bước và đo GPU lại full validation.**
DPO FN886/FP881, native-only FN886/FP878 tại confidence0,25; S0 FN866/FP984.
Best cảhai vẫn S0. Tại cùng FP984, DPO FN859/native-only FN860.
Xem [review kết quả](KB2_NATIVE_BACKGROUND_LONG_REVIEW_20261031.md).

Phần kế hoạch launch dưới được giữ để truy vết lượt xác nhận sau
[pilot đối chứng](KB2_NATIVE_BACKGROUND_PILOT_20261031.md).

## Ngân sách và cấu hình khóa

- Hai lượt tuần tự: **native+DPO trước, native-only sau**, mỗi lượt15.000 update,
  tổng30.000 update trên CUDA RTX4060 Laptop.
- Mỗi lượt khởi tạo lại từ S0 supervised KB1 và optimizer mới, không resume
  checkpoint pilot. Đây là kéo dài cùng recipe từ điểm xuất phát chung.
- Background BCE weight0,5; toàn bộ foreground slot weight1, gồm cả class đúng
  và target0 tại các class khác; assignment/normalization/box/DFL giữ nguyên.
- Giữ classification cv3 trainable; regression/backbone/neck/BN state cố định.
- Native batch4=2 uniform train+2 preference; DPO2 quyết định/step, input640.
- LR1e-6, AdamW weight decay5e-4, gradient clip1, beta0,1, lambda0,01;
  native-only dùng hệ số DPO0. Float32, TF32 tắt, seed42.
- Full bank784 cặp,663 cặp DPO train, giữ nguyên ảnh/annotation/bank.
- Full validation479 ảnh tại0 và mỗi1.000 bước, không dùng test.
- Primary confidence0,25, IoU0,5; NMS0,6, cap300, AP từ confidence0,001.
- Chọn macro recall cao nhất với micro recall>=S0, FP<=1,05*S0 và
  AP>=S0−0,001; S0 vẫn eligible. Không đổi threshold/selector sau khi xem kết quả.
- Lưu best.pt/last.pt riêng mỗi lượt; nếu không điểm train nào đạt gate,
  best vẫn S0. Không early stop theo recall pilot; lỗi số học/provenance vẫn dừng.

Đã kiểm trước launch: CUDA khả dụng, không có job train khác; hash S0 và8
checkpoint pilot khớp; fingerprint implementation background khớp pilot.
Tái dựng bằng CPU sampling xác nhận đủ2.000 batch đầu (ảnh uniform và pair)
trùng pilot. Runner sẽ kiểm lại metrics/probes0/1k/2k sau mỗi lượt và kiểm
hai stream15k byte-identical, checkpoint cũ không đổi khi hoàn tất.

Trainer cũ giữ nguyên để bảo toàn fingerprint. Cờ metadata `pilot=True` kế thừa;
ngân sách15.000 thực tế nằm trong config và plan companion. Runner mới:
`scripts/run_kb2_native_background_long.py`.

## Theo dõi lượt thực tế

- Run: `kb2_preference_optimization/outputs/yolov8n_native_background_long_s42_20261031_020350_005551/`.
- Khởi chạy09:03:50 ngày31/10/2026, giờ Asia/Bangkok; PID lúc launch **124575**.
- [Training log](/home/celine/Celine/graduate/RL-YOLO/kb2_preference_optimization/outputs/_launches/yolov8n_native_background_long_s42_20261031_020350_005551/training.log).
- [Launch/PID/command](/home/celine/Celine/graduate/RL-YOLO/kb2_preference_optimization/outputs/_launches/yolov8n_native_background_long_s42_20261031_020350_005551/launch.json).
- [Plan khóa](/home/celine/Celine/graduate/RL-YOLO/kb2_preference_optimization/outputs/yolov8n_native_background_long_s42_20261031_020350_005551/plan.json).
- [DPO metrics](/home/celine/Celine/graduate/RL-YOLO/kb2_preference_optimization/outputs/yolov8n_native_background_long_s42_20261031_020350_005551/dpo/metrics.jsonl).
- Native metrics/checkpoint đã được tạo sau khi DPO hoàn tất.
- Marker hoàn tất toàn bộ: `completed.json` tại run root; nếu runner lỗi có
  `failed.json` và traceback trong log. Không suy hoàn tất chỉ từ PID biến mất.

Process chạy nền với session riêng, stdin đóng, stdout/stderr ghi vào training.log.
S0 và mọi checkpoint cũ giữ nguyên; toàn bộ output mới nằm trong run trên.
Marker completed.json đã có; review xác nhận checkpoint/hash/stream và metrics
nạp lại khớp. Tổng thời gian trainer khoảng34,5 phút, không tính setup/initial eval.

Đọc log từ repo root:

```bash
tail -f kb2_preference_optimization/outputs/_launches/yolov8n_native_background_long_s42_20261031_020350_005551/training.log
```
