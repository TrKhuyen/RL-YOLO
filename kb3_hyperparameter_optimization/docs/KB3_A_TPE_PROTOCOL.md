# KB3-A: YAML supervised KB1 → pretrained → TPE → so checkpoint

Protocol hiện tại: `kb3_tpe_v2_kb1_pretrained`, cập nhật 14/10/2026. Entry point `quality.tpe_pipeline`; cấu hình `configs/kb3_tpe.yaml`. Hai script run_all/run_quality chỉ chạy A–TPE; [B giữ làm nghiên cứu tương lai](KB3_B_FUTURE_RESEARCH_REPORT.md).

## Luồng đã chốt

1. Đọc `checkpoint_based/<model>/args.yaml` của supervised KB1 làm cấu hình xuất phát.
2. Tải lại pretrained nguồn được YAML chỉ định (yolov8n hiện là `yolov8n.pt`, `pretrained: true`) cho mỗi trial.
3. TPE chỉ đổi `lr0` và `weight_decay` giữa trial; mỗi detector dùng base tham số cố định và scheduler gốc.
4. Chọn best checkpoint bằng native validation fitness/early stopping như KB1.
5. Chấm checkpoint KB3-A và best supervised KB1 bằng cùng canonical evaluator rồi so kết quả.

`args.yaml` là cấu hình; pretrained `.pt` được YAML chỉ định là khởi tạo; `checkpoint_based/<model>/weights/best.pt` là đối chứng chỉ đọc. Không dùng best KB1 hoặc trọng số trial trước để train trial sau. Mỗi trial là phiên fine-tune mới từ pretrained nguồn, không phải scratch.

## Recipe và nguồn trọng số

Đối chiếu YAML với train_args của checkpoint supervised KB1 và dataset provenance. Source phải tồn tại, đúng tensor layout kiến trúc, có 80 lớp COCO và khác best KB1. Lưu SHA source trong manifest/result; kiểm tra mọi tensor backbone đã chuyển vào detector 28 lớp và hash khởi tạo giống nhau giữa các trial cùng seed. YOLOv8 weights cũ thiếu yaml_file được kiểm qua tensor layout.

Kế thừa epochs, patience, seed, batch, imgsz, workers, device, warmup và lrf. Giữ SGD, nbs, AMP, augmentation, loss weights, LR schedule, close-mosaic và native validation conf/IoU/max_det. Momentum và augmentation không tìm kiếm. Output/plot/save_period thay để quản lý artifact. Reseed trước native model setup để import framework lần đầu không làm lệch khởi tạo head giữa trial; không bảo đảm tái lập tuyệt đối checkpoint lịch sử.

Yolov8n hiện dùng **yolov8n.pt, seed 0, tối đa 200 epoch, native patience 30, batch 16, imgsz 640**; LR 0.01, WD 0.0005, momentum 0.937, warmup 3, lrf 0.01, nbs 64, close-mosaic 10. Không còn ép 300 epoch/seed 42/canonical patience 50 như protocol scratch cũ. Preflight chỉ đọc/xác minh, không train/tạo output. Smoke dùng ảnh tổng hợp và 3 epoch, không chấm đối chứng KB1; smoke không phải kết quả nghiên cứu.

## TPE và ngân sách

Mặc định bốn trial tổng cộng; `--trials N` là tổng tích lũy. Đợt chạy lại tám lượt dùng `--trials 8`.

| Trial | Cách chọn |
|---:|---|
| 0 | Enqueue tham số KB1 chính xác, cùng pretrained nguồn: anchor kiểm tra mức tái lập |
| 1 | Startup random trong vùng hẹp |
| 2 trở đi | TPE dựa trên completed observations |

Hai chiều log: LR `[0.5,1.5] × lr0_KB1`, WD `[0.5,2] × WD_KB1`, cắt theo global bounds. Yolov8n: LR 0.005–0.015, WD 0.00025–0.001. Optuna 4.5.0, startup=2. Sampler seed 42 là seed chọn tham số, khác detector seed 0. Proposal dùng `sampler_seed + logical_trial_number`, dựng lại study từ history để resume tạo cùng proposal. Không có RL/PPO trong luồng này.

TPE objective là canonical validation mAP50–95 của native best checkpoint. Tổng trial bao gồm trial 0, không cộng thêm baseline detector training. KB1 best chỉ chấm inference. Winner một detector seed chưa chứng minh tối ưu hoặc ổn định; chưa đánh giá test.

## Checkpoint, early stop và metric

Giữ **native Ultralytics validation fitness** để chọn best EMA và native patience như supervised KB1. Không có canonical stopper, ranh segment hay bảo vệ pha cuối như protocol scratch. Detector train liên tục bằng một trainer; stop theo quy tắc native mỗi epoch.

Sau train, chấm native best và last bằng canonical FP32 evaluator chung với KB1 reference: TF32 tắt, conf 0.001, NMS IoU 0.60, max_det 300, operating confidence 0.25. TPE xếp các native best bằng canonical mAP50–95. Kiểm hash EMA native với artifact để epoch/metrics/checkpoint khớp nhau. Bỏ callback final validation khỏi epoch log để không có epoch giả sau early stop. Native best có thể khác epoch có canonical mAP cao nhất; đây là cách chọn cùng loại với KB1.

**Không trộn native/canonical:** curves mỗi epoch là native mAP/loss; bảng checkpoint và final-minus-selected là canonical. AP small chỉ tính tại checkpoint được chấm, không bịa mỗi epoch. Mốc suy giảm dùng native curve và chỉ là chẩn đoán, không chứng minh epoch bắt đầu overfit.

`Δ pretrained anchor = 100 × (canonical mAP_trial − canonical mAP_trial0)`. CSV cũng so với best supervised KB1 có sẵn. Trial 0 kiểm tra mức tái lập; cùng recipe/source được YAML chỉ định chưa bảo đảm tái lập checkpoint lịch sử do runtime/version/RNG. Hash test chỉ kiểm tra toàn vẹn, không dùng test score.

## Lệnh PowerShell: mỗi lệnh một dòng

Kiểm tra trước train:

```powershell
bash kb3_hyperparameter_optimization/run_all_kb3.sh --model yolov8n --trials 8 --check-only
```

Chạy tám lượt pretrained trong thư mục mới, hiển thị trực tiếp trong terminal:

```powershell
bash kb3_hyperparameter_optimization/run_all_kb3.sh --model yolov8n --trials 8 --output-dir kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/quality/tpe_kb1_pretrained_v2/yolov8n
```

Tiếp tục sau gián đoạn:

```powershell
bash kb3_hyperparameter_optimization/run_all_kb3.sh --model yolov8n --trials 8 --resume --output-dir kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/quality/tpe_kb1_pretrained_v2/yolov8n
```

Tạo lại report khi đủ kết quả, không train thêm:

```powershell
bash kb3_hyperparameter_optimization/run_quality_kb3.sh --model yolov8n --stage evaluate --trials 8 --resume --output-dir kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/quality/tpe_kb1_pretrained_v2/yolov8n
```

Không dùng dấu nối dòng Bash trong PowerShell. Hoàn tất thành công khi có `VALIDATION REPORT` và `A-TPE stage complete`, rồi trả lại dấu nhắc; Traceback là chưa hoàn tất thành công. `all` = KB1 inference → search → report, không tự tuning/final seed/test. `--model all --trials 8` là tám lượt mỗi model, tổng 40 detector; preflight cả năm trước model đầu.

## Artifact, resume và lịch sử

Manifest/contract lưu source/reference SHA, code/data/dependency hashes; `kb1_recipe.yaml` là bản đầy đủ args.yaml và `starting_train.yaml` là cấu hình đã resolve. Trial có initial/epochs/result/selected_best JSON, selected_best.pt và native best/last. Report versioned theo tổng trial có evaluation.json, comparison.csv, report.md và curves. Status chỉ trial đang chạy; history tổng epoch/giờ thực.

Khóa OS ngăn hai tiến trình cùng output. Completed trial phải khớp SHA/source/transfer proof/seed/parameters/selection. Partial trial chưa commit được archive và train lại cùng proposal. **Chưa hỗ trợ resume giữa trial**; script resume ở mức trial, dù native checkpoint có optimizer trước strip. Source/config/data/dependencies đổi phải dùng run mới.

Tám checkpoint scratch ở `tpe_kb1_start_v1/yolov8n` đã xóa ngày 14/10/2026 theo yêu cầu; logs/reports giữ làm lịch sử. Không resume thư mục đó hoặc dùng observations/cấu hình thắng scratch như kết quả pretrained mới.