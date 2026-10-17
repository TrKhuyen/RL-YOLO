# Review KB3-A pretrained sau khi sửa — 14/10/2026

Luồng đã chốt: YAML supervised KB1 là cấu hình bắt đầu **gồm nguồn pretrained**;
TPE thay LR/WD; best checkpoint KB1 chỉ là đối chứng. Code cũ ép scratch nên tám trial cũ
không trả lời câu hỏi tối ưu pretrained. Protocol mới `kb3_tpe_v2_kb1_pretrained` tách hẳn
khỏi v1; [thiết kế và lệnh chạy](KB3_A_TPE_PROTOCOL.md).

| Điều cần đúng | Cách thực hiện và kiểm chứng |
|---|---|
| Kế thừa YAML KB1 | Epoch/patience/seed/batch/imgsz/warmup/lrf và recipe native; yolov8n resolves 200/30/0/16/640. Assert actual trainer args không bị ghi đè. |
| Dùng pretrained thật | Đọc source từ trường model của args.yaml, verify nc=80/tensor layout/SHA và mọi tensor backbone được transfer trước train. Không chỉ đặt pretrained=True trên model YAML random. |
| Best KB1 chỉ để so | Kiểm provenance/checkpoint SHA; chặn path/hash trùng source; integration dùng reference giả không thể train để chứng minh trainer không nạp nó. |
| Trial độc lập | Tải lại source mỗi trial; hash khởi tạo model 28 lớp phải giống giữa các trial cùng seed. Không nối tiếp trọng số trial trước. |
| Chỉ LR/WD thay | Momentum/augmentation giữ giá trị KB1; reject RL/action schedules trong trainer A; native optimizer/scheduler/EMA/dataloader liên tục. |
| Cùng quy tắc chọn với KB1 | Native best EMA theo validation fitness và native early stopping/patience; không dùng canonical stopper/segment guards của scratch B. |
| Metrics/checkpoint khớp | Hash native best EMA đối chiếu checkpoint thực, bỏ callback final-eval khỏi epoch log; canonical best/last chấm artifact đã lưu. |
| Đánh giá rõ ràng | Native curves mỗi epoch; bảng checkpoint dùng canonical evaluator chung với KB1. Không gán native AP hoặc loss sang canonical curve, không bịa AP small từng epoch. |
| Resume không trộn protocol | Identity khóa source/recipe/code/data/deps; reject v1 scratch. Completed trials verified/skip; partial trial archive và train lại cùng proposal. |
| Không dùng test để chọn | Pilot chỉ validation; hashes test phục vụ toàn vẹn dữ liệu, không dùng test scores. |

Đã xóa đúng **8 selected_best.pt của trial_0000..0007** trong run scratch v1.
Không còn .pt trong cây tpe cũ; logs, metrics và reports giữ làm lịch sử.
KB1 yolov8n best SHA vẫn `9a7ce8d2710b94453da2add079d3b3635fb2a9f9ae1078fbe260428a67c969f9`;
pretrained nguồn SHA vẫn `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.

## Validation thực hiện

- Regression KB3: **82 tests, 81 pass, 1 legacy integration tùy chọn skip**.
- Native pretrained CPU gates: transfer/identical initialization/read-only source-reference; early stop theo patience và không thêm epoch giả.
- Native CUDA gates: hai integration tests pass trên RTX 4060 laptop; final transfer/actual-args guard được chạy lại trên CUDA sau thay đổi cuối.
- End-to-end CPU smoke: **4 pretrained trial thật/12 epoch trên ảnh tổng hợp**, resume 2→4 giữ kết quả, evaluate/resume không train thêm, snapshot pretrained=False bị reject.
- Preflight cả 5 model pass với pretrained và args.yaml thật; không tạo nghiên cứu mới.
- Lệnh dự kiến `--model yolov8n --trials 8` pass: pretrained source yolov8n.pt, seed 0, 200 epoch/trial, patience 30, tối đa 1.600 epoch cho tám trial.

Logs và summary nằm trong `checkpoint_hyperparameter_optimization/`:
`pretrained_a_regression_tests.log`, `pretrained_a_cpu_smoke.log`,
`pretrained_a_cuda_tests.log`, `pretrained_a_final_cuda_gate.log`,
`pretrained_a_preflight_all.log`, `pretrained_a_final_preflight.log`,
`pretrained_a_validation_summary.json`. Chưa chạy training nghiên cứu trên dữ liệu thật.

## Giới hạn còn lại

Same recipe/source/seed không bảo đảm tái lập chính xác baseline lịch sử do runtime/version/RNG;
trial 0 đo mức lệch này. Một seed và chọn winner trên validation chưa chứng minh ổn định/optimal.
Native best có thể khác canonical best mọi epoch; cố ý giữ cùng loại selection với KB1 để so công bằng.
Resume hiện ở mức trial, không mid-trial. Tiny CPU/GPU gates kiểm tính đúng cơ chế, không chứng minh chất lượng pest dataset.
Mã B scratch cũ được giữ riêng cho nghiên cứu tương lai và không được chạy bởi entry point A.
