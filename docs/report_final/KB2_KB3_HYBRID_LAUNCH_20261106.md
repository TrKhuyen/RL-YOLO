# Hybrid KB3 → KB2: lệnh chạy và kiểm chứng — 06/11/2026

## Phạm vi

Theo thiết kế đã chốt: pretrained YOLOv8n → KB3/TPE → selected checkpoint tốt nhất
→ KB2/native+DPO. Hai phép so chính là hybrid−TPE và hybrid−S0 supervised.
Chưa chạy lượt hybrid15k trong lần triển khai này; smoke chỉ kiểm kỹ thuật.

**Cập nhật phạm vi sau các lượt dài ngày06/11:** KB2 DPO là toàn bộ pipeline;
launcher hiện tại chỉ chạy DPO, bỏ nhánh native-only. Native detection loss vẫn
giữ trong tổng loss. Kết quả và phạm vi hiện hành ở
[KB2_CURRENT_PIPELINE_20261106.md](KB2_CURRENT_PIPELINE_20261106.md);
các kiểm chứng triển khai bên dưới được giữ như lịch sử.

Runner trước đây cố định nguồn S0. Package [hybrid](../../kb2_preference_optimization/hybrid/)
được tách riêng để giữ nguyên bytes/fingerprint của root Python modules,
`conditional/`, `conditional_bg/` và `conditional_strength/`. Bank/trainer hybrid
là phiên bản riêng, tái sử dụng judge, policy Bernoulli, background loss, sampler
và evaluator hiện có. Vòng update giữ nguyên semantics của strength trainer.

## Nguồn checkpoint

Winner hiện tại là **trial2**, canonical validation mAP50–95 lịch sử **38,5071%**.
Nguồn: [evaluation.json](../../kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/quality/tpe_kb1_pretrained_v2/yolov8n/evaluation.json).

Checkpoint cục bộ:
`kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/quality/tpe_kb1_pretrained_v2/yolov8n/tpe/trial_0002/selected_best.pt`.

SHA256: `eb7072dd087de0154049ca8a5151c6ada6ba1fca2e2666c676a9c3d17c61a84d`.

Resolver dùng trial ID và kiểm SHA với evaluation/result/spec, không sửa các path
Windows trong artifacts KB3. Kiểm completed search, winner đạt score tối đa đã ghi,
parameters/seed và dataset identity với mốc supervised. Đây là kiểm integrity của
selection đã lưu, không tái chạy hay chứng thực lại toàn bộ search lịch sử.

Policy/reference đều nạp từ TPE. Reference, backbone/neck, regression và BN buffers
đóng băng; chỉ classification `cv3` được cập nhật. Bank sinh từ TPE, có replay/hash
guards; bank S0 hoặc bank khác provenance bị từ chối. TPE và S0 luôn chỉ đọc.

## Lệnh chạy dài một dòng

Từ repo root:

```bash
bash kb2_preference_optimization/run_hybrid_kb3_kb2.sh --device cuda --steps 15000 --eval-interval 1000 --seed 42 --dpo-weight 0.1 --background-weight 0.5
```

Đây là **một lượt DPO continuation**, không chạy lại TPE và không tự chạy thêm
native-only. Ngân sách/optimizer giữ recipe đã chọn cho KB2:15k update, AdamW
LR1e-6, weight decay5e-4, clip1, beta0,1, lambda0,1, native background0,5,
input640, native batch4 với2 ảnh uniform +2 ảnh preference; DPO2 quyết định/update.
Hyperparameter full training KB3 không tự thay thế hyperparameter continuation này.
Recipe được chuyển sang TPE để thử nghiệm; chưa có bảo đảm gain như khi khởi tạo S0.

Launcher chọn `.venv-ubuntu` hoặc Linux `.venv`, mặc định CUDA và dừng nếu không có
CUDA; không tự chuyển sang CPU. Chạy preflight mà chưa train:

```bash
bash kb2_preference_optimization/run_hybrid_kb3_kb2.sh --device cuda --check-only
```

## Output và cách đọc

Mỗi lần tạo folder mới:

```text
kb2_preference_optimization/outputs/yolov8n_hybrid_tpe2_s42_<timestamp>/
  source.json             # TPE/S0/code/runtime/data identities
  request.json
  run.log
  preferences/            # bank mới từ TPE, train split
  bank_audit.json
  S0_validation.json      # chấm lại S0 theo protocol KB2
  dpo/
    source_validation.json # chấm lại checkpoint TPE
    best.pt
    last.pt
    step_000000.pt ... step_015000.pt
    metrics.jsonl
    stream.jsonl
    completed.json
  comparison.json
  completed.json
```

Best chọn mAP50–95 validation, min_delta1e-4; TPE step0 là ứng viên. Vì vậy,
`best.pt` có thể vẫn là TPE nếu các update không cải thiện. `last.pt` là weights
sau15k. Các file `.pt` dùng format state_dict của KB2; công cụ nạp lại là
`python -m kb2_preference_optimization.hybrid.evaluate`, không mặc nhiên nạp trực
tiếp bằng `YOLO(best.pt)` như một checkpoint Ultralytics.

`comparison.json` ghi metrics best/last, delta so TPE và S0. AP/P/R/F1 dùng fraction,
nhân100 để ra điểm phần trăm; FP/FN dùng số đếm. Chấm lại ba mốc cùng evaluator,
không dùng số TPE lịch sử thay baseline runtime mới. Test không được inference
hoặc dùng chọn checkpoint. Trong diagnostics kế thừa, các trường legacy có chữ
`S0` của từng arm thực chất chỉ **source/reference TPE**; mốc S0 chung được ghi riêng
ở root. GT recovered/lost trong arm vì vậy là so với TPE.

Launcher hiện tại chỉ nhận DPO; không chạy native-only. Hybrid−TPE đo tác dụng
toàn bộ pipeline KB2 sau TPE, còn hybrid−S0 đo tác dụng kết hợp KB3→KB2.

## Kiểm chứng trước khi giao lệnh

- Unit checks: resolver nhận đúng checkpoint sau di chuyển Windows→Linux; từ chối
  checkpoint bị đổi bytes, search chưa đủ, winner sai, bank S0 và metadata source
  khác; kiểm vòng update giữ nguyên và so baseline bỏ qua wall time.
- GPU smoke:32 ảnh tạo bank,2 update mỗi arm native/DPO,8 ảnh validation; xác nhận
  margin ban đầu0, DPO loss log2, reference/geometry/frozen state không đổi và stream khớp.
- Nạp lại checkpoint smoke và đối chiếu metric; kiểm SHA nguồn TPE/S0 và source
  Python của các recipe cũ không thay đổi.

Kết quả kiểm cụ thể được lưu ở
[verification.json](assets/kb2_hybrid_launch_20261106/verification.json).
Các số smoke không đánh giá khả năng cải thiện model và không thay kết quả lượt15k.
