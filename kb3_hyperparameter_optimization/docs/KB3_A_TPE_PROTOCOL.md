# KB3 hiện tại: A–TPE, bốn lượt thử quanh recipe KB1

Protocol `kb3_tpe_v1_kb1_start`; cấu hình [`kb3_tpe.yaml`](../configs/kb3_tpe.yaml); entry point `quality.tpe_pipeline`. `run_all_kb3.sh` và `run_quality_kb3.sh` chỉ chạy luồng này. B được giữ làm nghiên cứu tương lai trong [báo cáo chi tiết](KB3_B_FUTURE_RESEARCH_REPORT.md).

## Những gì được train và so sánh

Đọc `kb1_reward_guided_training/checkpoint_based/<model>/args.yaml` để lấy tham số và recipe supervised thật của KB1. Kiểm tra recipe với metadata checkpoint và dataset provenance trước khi train. `best.pt` KB1 chỉ dùng inference làm mốc **pretrained**, không nạp làm khởi tạo KB3. Mỗi trial KB3 khởi tạo từ `<model>.yaml`, scratch, SGD, cùng seed 42 và cùng initial-weight hash.

| Trial (đánh số từ 0) | Cách chọn | Vai trò |
|---:|---|---|
| 0 | Enqueue cấu hình KB1 chính xác | Matched scratch anchor để đo lợi ích đổi hyperparameter |
| 1 | Startup random trong vùng hẹp | Thêm observation trước khi fit TPE |
| 2 | TPE dựa trên 0/1 | Cấu hình mới |
| 3 | TPE dựa trên 0/1/2 | Cấu hình mới |

**Bốn lượt là tổng cộng, gồm anchor; không phải bốn lượt cộng baseline train nữa.** Baseline pretrained KB1 chỉ được đánh giá lại. Không train B, không tự train thêm các seed tuning/final. Một seed chung giảm khác biệt khởi tạo nhưng chưa đo độ ổn định qua seed. Checkpoint thắng validation là kết quả exploratory, không đủ chứng minh tối ưu hoặc thắng ổn định.

Chỉ tìm hai chiều log: `lr0 = [0.5,1.5] × lr0_KB1` và `weight_decay = [0.5,2] × WD_KB1`, cắt trong biên toàn cục. Với recipe hiện tại: LR 0.005–0.015, WD 0.00025–0.001. Giữ momentum 0.937 và augmentation_strength 0.5 tái tạo KB1; giữ warmup, linear decay/lrf, nbs, loss weights, close-mosaic và AMP theo recipe. Các cấu hình cố định base hyperparameter suốt trial, LR thực vẫn chạy scheduler gốc.

Optuna pin 4.5.0. `n_startup_trials=2`: nếu để mặc định 10 thì bốn trial đều chưa dùng TPE. Chọn hai startup để thực sự thử cơ chế TPE trong ngân sách nhỏ; chỉ hai proposal dựa trên mô hình mật độ nên độ tin cậy còn hạn chế. Xem [API TPESampler 4.5.0](https://optuna.readthedocs.io/en/v4.5.0/reference/samplers/generated/optuna.samplers.TPESampler.html).

Sampler dùng `sampler_seed + logical_trial_number`, dựng lại study từ completed observations mỗi proposal. Proposal và objective history lưu trước khi detector train. Đây là biến thể seed theo trial được khai báo, không giữ nguyên chuỗi RNG của một study sống từ đầu đến cuối; kiểm thử bảo đảm resume/direct tạo cùng proposal và TPE thực sự gọi density sampler từ trial 2.

## Epoch, early stopping và evaluator

Tối đa 300 epoch/trial, tổng tối đa 1.200 cho bốn lượt. Canonical stopper không dừng trước 100 epoch; patience 50 tính từ cải thiện đáng kể >0.0005 mAP50–95; kiểm tra stop tại ranh 5 epoch, bảo vệ pha close-mosaic cuối. Native patience vô hiệu hóa để không có hai stopper cạnh tranh. LR decay vẫn theo horizon 300, không co lại khi stop sớm. Giữ horizon này vì scratch pilot cũ đạt peak muộn; chưa có đảm bảo 300 tối ưu.

Lưu best EMA checkpoint theo canonical validation mỗi epoch đã train. Metric FP32/TF32 tắt, conf 0.001, NMS IoU 0.60, max_det 300, operating confidence 0.25; cùng evaluator cho KB1 reference và trial. Native CSV KB1 có thể khác metric nên không lấy trực tiếp để so. Evaluator giữ RNG của train; mỗi trial vẫn là một trainer liên tục, không reset qua segment.

`all` = `baseline` inference → `search` bốn detector → `evaluate` tổng hợp kết quả/log đã có. Không đánh giá test trong pilot; hash test chỉ dùng kiểm tra toàn vẹn. Search/epoch selection dùng validation nên điểm winner có selection bias. Phải chốt thiết kế và checkpoint trước khi một đợt đánh giá test cuối độc lập; entry point pilot không cung cấp `--stage test`.

## Chạy trên PowerShell

Mỗi lệnh dưới đây nằm trên **một dòng**. Dấu `\` nối dòng của Bash không dùng được khi gọi Bash từ PowerShell.

```powershell
bash kb3_hyperparameter_optimization/run_all_kb3.sh --model yolov8n --check-only
bash kb3_hyperparameter_optimization/run_all_kb3.sh --model yolov8n --trials 4 --output-dir kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/quality/tpe_kb1_start_v1/yolov8n
```

Tiếp tục cùng protocol, không chạy lại completed trials:

```powershell
bash kb3_hyperparameter_optimization/run_all_kb3.sh --model yolov8n --trials 4 --resume --output-dir kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/quality/tpe_kb1_start_v1/yolov8n
```

Đọc/tạo lại báo cáo khi đã đủ bốn kết quả, không train thêm:

```powershell
bash kb3_hyperparameter_optimization/run_quality_kb3.sh --model yolov8n --stage evaluate --trials 4 --resume --output-dir kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/quality/tpe_kb1_start_v1/yolov8n
```

Mặc định chỉ `yolov8n`. `--model all` chạy tuần tự bốn trial **mỗi model**, tổng 20 detector, tối đa 6.000 epoch; preflight cả năm model trước khi chạy model đầu. Không tự chọn all cho pilot hiện tại. `--trials N` là tổng tích lũy, không phải thêm N. Không tự tăng từ 4 lên 8 khi resume. Chỉ mở rộng khi chủ động yêu cầu và giữ source/config/data/dependencies/recipe. Mỗi lần source/contract đổi phải dùng output mới; không resume thư mục A/B v3/v4.

Script có khóa OS theo output tránh hai tiến trình cùng train một trial. Completed result được kiểm SHA checkpoint, seed, hyperparameter và không có quyết định RL. Saved proposal được kiểm với completed history. Partial episode/trial chưa commit được archive trong cùng run và train lại cùng proposal; **chưa hỗ trợ mid-trial full-state resume**. Không sửa code trong khi một long job đang chạy vì manifest ghi nguồn code để kiểm tra lần resume sau.

## Kết quả tự động

- `status.json`: stage, trạng thái, trial hiện chạy và đường epoch log.
- `manifest.json`, `train_contract.json`, `kb1_recipe.yaml`, `starting_train.yaml`, `dataset.yaml`: provenance/contract đã khóa.
- `kb1_reference.json`: metrics canonical của pretrained KB1, inference-only.
- `tpe/trial_0000..0003/trial_spec.json`: proposal cố định; `epochs.json`, `result.json`, `selected_best.pt`: curves, kết quả, checkpoint.
- `trial_history.json`: số lượt, tổng epoch và detector giờ thực.
- Sau bốn lượt: `evaluation.json`, `comparison.csv`, `report.md`; bản versioned trong `reports/trials_0004/` và biểu đồ `trajectories.png`.

Báo cáo so từng trial với scratch anchor và pretrained KB1. Chênh lệch với pretrained KB1 có cả ảnh hưởng init/ngân sách lịch sử, không quy riêng cho TPE. Ghi best epoch, minimum val loss, final-minus-best và dấu hiệu mAP giảm ít nhất 0,5 điểm phần trăm trong năm epoch liên tiếp sau best; đây là chẩn đoán đường cong, không khẳng định chính xác epoch bắt đầu overfit. Metric chưa có sẽ để trống, không tự bịa.

Thử nghiệm A này không tạo kết quả RL. Nếu cả ba cấu hình mới không hơn anchor thì chỉ kết luận trong vùng tìm/seed/ngân sách đã thử, không kết luận HPO không hữu ích. Nếu winner hơn anchor, cần seed xác nhận trước tuyên bố ổn định; nếu mở rộng search, giữ report bốn lượt và khai báo tổng ngân sách đã sử dụng.
