# KB2 — Sửa reference scoring cùng batch, 26/10/2026

## Vấn đề

Preference generation chấm reference từng ảnh (batch1), trong khi policy train
theo batch4. Trên GPU, cùng checkpoint có sai lệch số học theo batch shape.
Vì trainer cũ dùng cached reference, relative margin có thể khác0 ngay trước
optimizer update đầu tiên, khi policy và reference vẫn có cùng trọng số.

Đây là sai lệch tính toán, không phải model đã học hoặc nhãn preference đổi.
Kiểm tra batch1 tái hiện đúng cache; batch4 có chênh gap tùy ảnh.

## Thay đổi

- Policy/reference cùng forward trên cùng tensor ảnh và batch shape mỗi update.
- Reference `no_grad`, không nằm trong optimizer và được kiểm immutable.
- Cả training và fixed train audit dùng live reference log-prob của cùng action.
- Cached reference chỉ phục vụ replay/provenance và diagnostic, không đi vào loss.
- Log thêm `relative_margin_max_abs`, mean/max-abs chênh reference gap so cache.
- Check update đầu: từ chối chạy nếu max-abs relative margin lớn hơn1e-5.
- Config ghi `reference_scoring=same_batch_reference_v1`; không đổi λ=0.01,
  β=0.1, sampler, quality scorer hoặc optimizer trong thí nghiệm đối chiếu.

## Kiểm chứng

21 CPU tests pass; một GPU integration test opt-in được chạy riêng trên
RTX4060 và pass. Test kiểm cả cache cố ý sai không làm lệch loss, gradient
policy vẫn có, reference không nhận gradient và batch4 trên checkpoint thật.

Chạy lại đúng pilot32 ảnh train, 4 draws/ảnh, 20 update, batch4/input640,
validation16 ảnh ở step0/10/20. Hai run có **cùng SHA256 pairs.jsonl**, do đó
chosen/rejected và cached values giống hệt nhau. Giữ run cũ nguyên trạng.

| Chỉ số | Trước sửa | Sau sửa |
|---|---:|---:|
| Initial train relative margin | -0.01383268 | **0** |
| Initial train DPO loss | 0.69383968 | **0.69314718** |
| Initial cached gap error mean | Chưa log riêng | -0.01383268 |
| Initial cached gap error max abs | Chưa log riêng | 0.04860731 |
| Fixed audit DPO loss step0 | 0.69314718 | 0.69314718 |
| Fixed audit DPO loss step20 | 0.69496292 | 0.69495283 |
| Validation mAP50–95 step0 | 0.58532345 | 0.58532345 |
| Validation mAP50–95 step20 | 0.58531570 | 0.58531570 |
| Best step | 0 | 0 |
| Reference immutable | Có | Có |

Sai lệch cache không còn trở thành preference signal ở update đầu. Audit loss
vẫn tăng nhẹ và mAP gần như không đổi; bản sửa không giải quyết hay chứng minh
nguyên nhân của xu hướng tối ưu này. 20 update/16 ảnh không đủ kết luận DPO
cải thiện hoặc không cải thiện generalization. Không tăng λ hoặc đổi recipe
đồng thời để làm đẹp pilot này.

Run mới: `kb2_preference_optimization/outputs/yolov8n_gpu_pilot_samebatch_20261026_1`.
Bằng chứng nhỏ: [reference_batch_fix.json](assets/kb2_20261026/reference_batch_fix.json).

## Tái chạy

Mỗi lần dùng thư mục mới. Code fingerprints đã thay đổi, vì vậy runner sẽ sinh
preference mới; không sửa manifest cũ để bỏ qua provenance check.

```bash
bash kb2_preference_optimization/run_all_kb2.sh --model yolov8n --device cuda --output-root "kb2_preference_optimization/outputs/yolov8n_gpu_pilot_$(date +%Y%m%d_%H%M%S)" --image-size 640 --pilot-images 32 --draws 4 --steps 20 --batch-size 4 --eval-interval 10 --eval-limit 16
```
