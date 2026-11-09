# KB2 hybrid: giảm LR sau1k và đánh giá dày giai đoạn đầu — 06/11/2026

## Kế hoạch thí nghiệm

Kiểm tra liệu tiếp tục cập nhật nhẹ hơn sau giai đoạn đầu có giúp nâng mAP,
đồng thời tìm checkpoint tốt có thể bị lịch đánh giá1k bỏ qua. Chỉ chạy pipeline
KB2 DPO; các mốc so chính là S0, TPE và hai recipe DPO đã hoàn tất.

Kế hoạch được ghi trước khi chạy trong `plan.json` tại campaign:
`kb2_preference_optimization/outputs/yolov8n_hybrid_schedule_s42_20261106_trial3/`.

| Thành phần | Cấu hình |
|---|---|
| Model/initialization/reference | YOLOv8n, best TPE trial2 |
| Ngân sách/seed | 15.000 update, seed42 |
| DPO λ/beta | 0,03 /0,1 |
| Native background/foreground | 0,5 /1 |
| LR update1–1.000 | 1e-6 |
| LR update1.001–15.000 | 1e-7 |
| Eval giai đoạn đầu | Mỗi250 update tới2.000 |
| Eval còn lại | Mỗi1.000 update; luôn lưu mốc cuối |
| Best | Validation mAP50–95, min_delta1e-4 fraction; TPE step0 hợp lệ |
| Bank/batch/optimizer/mask | Giữ recipe trước, cùng stream15k |
| Test | Không dùng |

22 mốc được lưu:0,250,500,750,1k,1.250,1.500,1.750,2k,3k,…,15k.
Phải phân biệt selected best trên toàn bộ22 mốc với selected best chỉ trên
lịch1k chung của các run cũ. Đánh giá dày không tự làm model học tốt hơn.
Nếu best xuất hiện trước hoặc tại1k, nó chưa chịu tác động giảm LR.

Implementation riêng ở `hybrid_schedule/`; bank, judge, loss, policy, evaluator
và các trainer lịch sử không đổi. Định dạng checkpoint/objective tương thích
hybrid cũ, có thêm training variant, fingerprint, LR schedule và eval grid.
CLI hiện tại hỗ trợ `--lr-drop-step`, `--lr-drop-factor`,
`--early-eval-interval`, `--early-eval-until`; mặc định không đổi recipe cũ.

17 unit checks đạt. GPU smoke2 update xác nhận LR thực trong optimizer đổi đúng
ranh giới, checkpoint/eval hoạt động và reference/frozen state không đổi.

## Thực thi và khả năng tái hiện

Lượt `trial3` hoàn tất15k, train wall32,84 phút gồm periodic eval, chưa tính đầy
đủ setup và chấm lại hậu kiểm. Runtime CPU/thread và số lần eval khác run cũ,
nên không dùng thời gian này để suy ra overhead riêng của DPO hoặc LR schedule.
Lượt `trial1` gặp segmentation fault khoảng update175,
lượt `trial2` gặp segmentation fault khoảng update1.650, trong đường tính
detection loss; log/checkpoint lỗi được giữ riêng. Lượt `trial3` bắt đầu lại từ
cùng TPE/seed/recipe, dùng `CUDA_LAUNCH_BLOCKING=1`, đặt `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` và giới hạn riêng tiến trình vào
CPU16–23. Model vẫn tính toán/huấn luyện trên CUDA; không đổi loss, data hay
trọng số khởi tạo, không chỉnh cấu hình hệ thống.

Kernel có bản ghi `mce: [Hardware Error]: Machine check events logged` lúc13:38:49
ngày06/11 theo log hệ thống. Bản ghi này xảy ra trước các lần crash hiện tại,
chưa xác định được nguyên nhân hoặc liên hệ nhân quả. Không kết luận lỗi thuộc
CPU, driver, PyTorch hay lịch LR từ các stack trace này.

Metric dưới đây dùng full validation479 ảnh/1.887GT và đã nạp lại checkpoint
để hậu kiểm, cùng với kiểm hash, stream và tính tương đương trước mốc đổi LR.

## Kết quả selected best của toàn pipeline

Tỷ lệ tính theo%; FP/FN tại confidence0,25. Tất cả continuation có ngân sách15k;
best500 là thời điểm chọn model, không phải lượt train chỉ chạy500 update.

| Model | Best step | mAP50–95 | AP50 | AP75 | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 supervised | — | 37,8218 | 49,1225 | 42,9680 | 50,9227 | 54,1070 | 52,4666 | 984 | 866 |
| TPE trước KB2 | 0 | 38,5097 | 50,4306 | 43,4945 | 50,8763 | 55,3789 | 53,0322 | 1.009 | 842 |
| KB2 λ0,1, LR cố định, eval1k | 1.000 | 38,5962 | 50,5191 | 43,6132 | 52,8037 | 53,8951 | 53,3438 | 909 | 870 |
| KB2 λ0,03, LR cố định, eval1k | 1.000 | 38,6314 | 50,5556 | 43,6522 | 52,9657 | 53,9481 | 53,4523 | 904 | 869 |
| **KB2 λ0,03, LR giảm, eval dày** | **500** | **38,7133** | **50,6287** | **43,7080** | 52,6208 | 54,2660 | 53,4307 | 922 | 863 |

So recipeλ0,03 cũ, best mới tăng**0,0818 điểm phần trăm mAP**, AP50+0,0730,
AP75+0,0558 và recall+0,3180 điểm; FN giảm6. Precision giảm0,3449 điểm,
F1 giảm0,0216 điểm và FP tăng18. Không phải mọi chỉ số đều tăng so best cũ.

So **S0**, best mới tăng:

- mAP50–95 **+0,8915 điểm**, AP50 **+1,5061**, AP75 **+0,7399**.
- Precision **+1,6981 điểm**, recall **+0,1590**, F1 **+0,9641**.
- TP tăng3, FP giảm62 và FN giảm3.

Như vậy các chỉ số chính AP/P/R/F1 đồng thời tăng so S0 trong seed42 này.
Đây là gain kết hợp KB3→KB2, không quy toàn bộ cho giai đoạn KB2.

So **TPE**, best tăng mAP**0,2036 điểm**, AP50+0,1981, AP75+0,2135,
precision+1,7444 và F1+0,3985 điểm; FP giảm87. Recall giảm1,1129 điểm,
TP giảm21/FN tăng21. Cần giữ cả mốc TPE để thấy đánh đổi của giai đoạn KB2.

Các metric bổ sung của best500:

| Metric (%) | S0 | TPE | KB2 best500 |
|---|---:|---:|---:|
| Macro F1 | 51,3707 | 51,2818 | 51,6693 |
| Macro recall | 54,5463 | 55,9285 | 55,0529 |
| AR300 | 56,2044 | 58,2612 | 58,0731 |
| APsmall | 4,3965 | 4,0997 | 4,1209 |
| APmedium | 17,9704 | 18,4362 | 18,3008 |
| APlarge | 42,5676 | 43,8593 | 44,0333 |

APsmall vẫn thấp hơn S0**0,2756 điểm**; cải thiện các metric chính không đồng
nghĩa cải thiện mọi kích thước. Regression/backbone/neck vẫn đóng băng, nên
không diễn giải AP75 tăng thành bằng chứng KB2 đã học sửa tọa độ box.

## Tách ảnh hưởng đánh giá dày và giảm LR

**Best500 xuất hiện trước khi giảm LR.** Phần tăng selected best từ38,6314%
lên38,7133% đến từ lịch đánh giá tìm được checkpoint sớm hơn. Weights/optimizer
tại1k khớp chính xác run cũ; các mốc250–1.250 cũng tái hiện đúng giữa hai runtime.
Không quy phần tăng best500 cho tác dụng của giảm LR sau1k.

Để kiểm LR, chỉ lấy các mốc0,1k,2k,…,15k, giữ rule min_delta cũ:

| Lịch so | Selected step | mAP50–95 (%) |
|---|---:|---:|
| LR cố định, lịch1k | 1.000 | 38,6314 |
| LR giảm, chỉ xét lịch1k chung | 5.000 | 38,6609 |
| LR giảm, toàn lịch đánh giá dày | 500 | 38,7133 |

Trên cùng lịch1k, LR giảm tăng selected best**0,0295 điểm phần trăm mAP**.
Đây là cải thiện nhỏ, tách khỏi lợi ích chọn mốc sớm. Raw mAP cao nhất trên
lịch1k mới là38,6644% ở6k, nhưng chỉ hơn5k0,0035 điểm, thấp hơn min_delta0,01
điểm nên selector vẫn giữ5k. Min_delta không phải ngưỡng kiểm định thống kê.

## So cùng15.000 update

| Metric (%) trừ số đếm | LR cố định, λ0,03 | LR giảm, λ0,03 |
|---|---:|---:|
| mAP50–95 | 38,3601 | **38,5167** |
| AP50 | **50,4056** | 50,3684 |
| AP75 | 43,1646 | **43,5123** |
| Precision | **55,7119** | 53,8916 |
| Recall | 52,4642 | **53,2061** |
| F1 | **54,0393** | 53,5467 |
| Macro F1 | **52,4106** | 51,8866 |
| Macro recall | 52,7465 | **53,9998** |
| AR300 | 56,4631 | **57,2631** |
| APsmall | **7,2645** | 4,3578 |
| APmedium | **18,5405** | 18,2398 |
| APlarge | 43,4157 | **43,7640** |
| TP | 990 | **1.004** |
| FP ↓ | **787** | 859 |
| FN ↓ | 897 | **883** |

LR giảm giữ mAP cuối cao hơn**0,1566 điểm** và recall cao hơn0,7419 điểm,
giảm14FN nhưng tăng72FP; precision/F1 thấp hơn. Đây là đánh đổi, không phải
cải thiện đồng đều. Last mới chỉ hơn TPE0,0070 điểm mAP, thấp hơn best500,
nên vẫn dùng best500 làm checkpoint cuối được chọn.

GT identity so ordinary TPE:

| Model | GT được cứu | GT bị mất | TP thay đổi ròng |
|---|---:|---:|---:|
| LR cố định, best1k | 6 | 33 | −27 |
| LR giảm, best500 | 3 | 24 | −21 |
| LR cố định, last15k | 22 | 77 | −55 |
| LR giảm, last15k | 8 | 49 | −41 |

Best500 giữ được nhiều GT nguồn hơn best1k cũ, dù cứu ít GT bị bỏ sót hơn.
Không diễn giải net TP tăng thành chỉ tăng số GT được cứu.

![Quỹ đạo mAP, precision và recall](assets/kb2_hybrid_schedule_20261106/learning_curves.png)

## Hậu kiểm và artifact

- Hash54 checkpoint eval hợp lệ:16 mốcλ0,1 cũ,16 mốcλ0,03 cũ và22 mốc mới.
  Weights/optimizer của best/last khớp file step được chọn.
- Toàn bộ stream15k giống nhau. Manifest/code/bank/source/mask/optimizer/batch
  khớp; chỉ thêm metadata lịch LR/eval và sửa nhãn pilot thành long run.
- Weights/optimizer tại1k khớp chính xác recipe LR cố định; checkpoint250–1.250
  khớp giữa runtime trước/sau giới hạn CPU. LR thực trong optimizer đúng1e-6
  tới hết1k,1e-7 từ các mốc sau1k.
- Đã nạp lại6 best/last và checkpoint selected trên lịch chung, chấm full
  validation trên GPU; toàn bộ quality metric, per-class và GT transitions
  khớp log. TPE và S0 cũng được chấm lại.
- Sau khi các lần chấm GPU hoàn tất, bước xuất CSV gặp trường Counter bằng0
  không được ghi thành key. Đã sửa `.get(..., 0)` và hoàn tất từ9 JSON metric
  đã chấm, kiểm lại hash/state/metric trước khi tái sử dụng. Không train lại,
  không thay metric và không bỏ qua kiểm tra chất lượng để hoàn tất báo cáo.
- Reference, frozen state, bank và checkpoint nguồn/đối chứng cũ không đổi.

Bằng chứng: [review.json](assets/kb2_hybrid_schedule_20261106/review.json),
[summary.csv](assets/kb2_hybrid_schedule_20261106/summary.csv),
[deltas.csv](assets/kb2_hybrid_schedule_20261106/deltas.csv),
[classes.csv](assets/kb2_hybrid_schedule_20261106/classes.csv),
[gt_transitions.csv](assets/kb2_hybrid_schedule_20261106/gt_transitions.csv),
[validation.csv](assets/kb2_hybrid_schedule_20261106/validation.csv).

Launcher thí nghiệm:
[run_kb2_hybrid_schedule_trial.py](../../scripts/run_kb2_hybrid_schedule_trial.py).
Script hậu kiểm:
[review_kb2_hybrid_schedule_trial.py](../../scripts/review_kb2_hybrid_schedule_trial.py).
Checkpoint tốt nhất mới:
`kb2_preference_optimization/outputs/yolov8n_hybrid_schedule_s42_20261106_trial3/run/dpo/best.pt`.
Định dạng state_dict KB2; dùng evaluator hybrid để nạp.

## Nhận xét và phạm vi kết luận

**Đã tìm được best mAP cao hơn:38,7133%, tăng0,0818 điểm so recipeλ0,03 cũ.**
Phần tăng selected best này đến từ tìm checkpoint500 nhờ đánh giá dày. LR giảm
giúp tăng nhẹ selected best trên lịch1k chung và giảm suy giảm mAP về cuối,
nhưng chưa vượt đỉnh500. Tiếp tục dùng checkpoint500 làm ứng viên tốt nhất
hiện tại theo selector mAP50–95; không đổi sang checkpoint có precision/F1 cao hơn.

Các chỉ số chính đều tăng so S0 trong lượt này, nhưng còn đánh đổi so TPE,
so recipe KB2 trước và ở object nhỏ. Đây là một continuation seed trên validation
đã dùng phát triển; cần xác nhận seed khác/dữ liệu xác nhận trước kết luận về
độ ổn định hoặc tổng quát hóa. Chưa chạy thêm seed, chưa dùng test, chưa sửa data.
