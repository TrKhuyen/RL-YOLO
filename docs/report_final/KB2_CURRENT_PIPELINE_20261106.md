# KB2 — phạm vi pipeline DPO và bảng kết quả chính

**Chốt ngày 08/11/2026:** dùng hybrid TPE → KB2 DPO λ0,03/background0,5,
best500/mAP50–95 38,7133% của run `yolov8n_hybrid_schedule_s42_20261106_trial3`.
Giữ sampler cũ, không chọn last hoặc sampler preservation35%. Đây là kết quả
validation phát triển seed42 đã được người thực hiện chọn; chưa là xác nhận
độc lập. [Kết quả chốt, cơ sở và minh chứng](KB2_HYBRID_SELECTED_EXPLANATION_20261108.md).

Theo quyết định ngày06/11/2026, **DPO là tên của toàn bộ phương pháp KB2**:
tạo hai action A/B, dùng GT chọn preference, xây/chọn/cân bằng candidate bank,
và tối ưu policy bằng `L_detection + λ L_DPO` với frozen reference.

Bỏ nhánh native-only khỏi luồng chạy và bảng kết quả chính. Native detection
loss vẫn nằm trong phương pháp KB2; native background weight0,5 vẫn giữ nguyên.
Các kết quả và mã đối chứng đã thực hiện được bảo toàn để truy vết, không xóa
checkpoint hoặc sửa số liệu lịch sử. Không yêu cầu chạy thêm native-only.

## Thiết lập đánh giá hiện tại

- KB2 độc lập: bắt đầu từ S0; so checkpoint sau KB2 với S0.
- Hybrid: bắt đầu/reference từ best TPE của KB3; so checkpoint sau KB2 với TPE,
  đồng thời so kết quả cuối với S0.
- Hybrid15.000 update, eval mỗi1.000, chọn best bằng validation mAP50–95;
  TPE step0 vẫn là ứng viên. LR/batch/beta/mask và tổng loss không thay đổi
  khi loại nhánh đối chứng.
- Phép so sau KB2 là đánh giá phương pháp tổng thể. Trong hybrid, gain so S0
  là gain kết hợp KB3→KB2; gain so TPE là gain giai đoạn KB2.

## Kết quả đã xác minh, seed42

Best của hai cấu hình LR cố định ở1.000 update; thử nghiệm LR giảm/eval dày
chọn best500. Cả ba đều có ngân sách15k.
Tỷ lệ tính theo%; FP/FN tại confidence0,25.

| Model | mAP50–95 | AP50 | AP75 | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 supervised | 37,8218 | 49,1225 | 42,9680 | 50,9227 | 54,1070 | 52,4666 | 984 | 866 |
| Best TPE của KB3 | 38,5097 | 50,4306 | 43,4945 | 50,8763 | 55,3789 | 53,0322 | 1.009 | 842 |
| TPE → KB2 DPO, λ0,1 | 38,5962 | 50,5191 | 43,6132 | 52,8037 | 53,8951 | 53,3438 | 909 | 870 |
| TPE → KB2 DPO, λ0,03, LR cố định | 38,6314 | 50,5556 | 43,6522 | **52,9657** | 53,9481 | **53,4523** | **904** | 869 |
| TPE → KB2 DPO, λ0,03, LR giảm/eval dày | **38,7133** | **50,6287** | **43,7080** | 52,6208 | **54,2660** | 53,4307 | 922 | **863** |

In đậm chỉ so ba cấu hình KB2. Best mới tăng0,0818 điểm phần trăm mAP soλ0,03
LR cố định,0,2036 điểm so TPE và0,8915 điểm so S0. AP/P/R/F1 đều tăng so S0,
FP giảm62/FN giảm3; nhưng recall vẫn thấp hơn TPE và APsmall thấp hơn S0.
Best500 xuất hiện trước mốc giảm LR: phần tăng best này đến từ tìm được mốc
sớm nhờ đánh giá dày. Trên cùng lịch1k, LR giảm tăng selected best0,0295 điểm.

Đây là một continuation seed trên validation phát triển479 ảnh/1.887GT,
chưa đánh giá xác nhận độc lập. Bỏ nhánh đối chứng không thay đổi số liệu,
checkpoint hoặc mức độ xác nhận của kết quả đã chạy.

Nguồn số liệu đã nạp lại checkpoint:
[summary.csv](assets/kb2_hybrid_schedule_20261106/summary.csv),
[review.json](assets/kb2_hybrid_schedule_20261106/review.json).
Báo cáo thực nghiệm chi tiết và các đối chứng lịch sử:
[KB2_HYBRID_LAMBDA003_REVIEW_20261106.md](KB2_HYBRID_LAMBDA003_REVIEW_20261106.md).
Thử nghiệm mới nhất:
[KB2_HYBRID_SCHEDULE_REVIEW_20261106.md](KB2_HYBRID_SCHEDULE_REVIEW_20261106.md).

Thử nghiệm tiếp theo giảm LR sau500 thay vì1.000 đã hoàn tất15k: selected best
vẫn500/mAP38,7133%, state dict giống hệt best đang có. Last tăng từ38,5167%
lên38,6300% nhưng không vượt best; giữ nguyên model ứng viên. Xem
[KB2_HYBRID_DROP500_REVIEW_20261106.md](KB2_HYBRID_DROP500_REVIEW_20261106.md).

08/11 đã thử tăng preservation lên 35% bằng sampler riêng, pilot2k/seed42,
giữ lịch LR drop1k. Best1k mAP38,7046%, chưa vượt best500 cũ38,7133%; precision,
F1/AP50/APsmall/APmedium tăng, FP giảm9 nhưng FN tăng1. Ở cùng2k, số GT bị mất
vẫn33 ở cả hai run. Chưa đổi recipe/best hoặc chạy dài sampler mới. Xem
[KB2_PRESERVATION_SAMPLING_PILOT_20261108.md](KB2_PRESERVATION_SAMPLING_PILOT_20261108.md).

## Launcher hiện tại

`run_hybrid_kb3_kb2.sh` và `run_all_kb2.sh` đi qua entrypoint DPO-only
[run_kb2_dpo.py](../../scripts/run_kb2_dpo.py). Các lệnh cũ có `--arms dpo` vẫn
chạy được; native/BCE hoặc nhiều nhánh bị từ chối trước khi nạp model/train.
Engine lịch sử không đổi bytes/fingerprint để giữ khả năng kiểm tra bank và
checkpoint đã lưu. Output mới vẫn giữ `dpo/best.pt`, `dpo/last.pt` và cùng
schema metrics; hybrid thêm metadata mô tả phạm vi phương pháp tổng thể.

Lệnh hybrid một dòng, giữ recipeλ0,1 hiện có; đổi `--dpo-weight 0.03` nếu chủ động
chọn cấu hìnhλ0,03 vừa thử:

```bash
bash kb2_preference_optimization/run_hybrid_kb3_kb2.sh --device cuda --steps 15000 --eval-interval 1000 --seed 42 --dpo-weight 0.1 --background-weight 0.5
```

Lần thay đổi phạm vi launcher không khởi chạy training. Thí nghiệm LR/eval
được yêu cầu sau đó đã hoàn tất, xem báo cáo mới nhất ở trên.

## Kiểm chứng thay đổi launcher

13 unit checks đạt: mặc định chỉ một DPO, từ chối native/BCE/nhánh lặp vàλ0,
giữ recipe từng nguồn, chuyển tiếp S0 đúng, ghi metadata phạm vi pipeline và
giữ các kiểm chứng source/update parity trước đó. Hai launcher được kiểm trực
tiếp đều từ chối `--arms native` trước khi nạp model. Bash syntax và các link
tài liệu hợp lệ.

Đã kiểm GPU preflight qua launcher mới: CUDA/RTX4060, đúng TPE trial2 và mốc S0.
Bank đầy đủ vẫn được chấp nhận; fingerprint source và9 file checkpoint/bank
được bảo vệ của kế hoạch thửλ0,03 không đổi. Không thay đổi số liệu hoặc trọng số
để loại nhánh đối chứng khỏi cách trình bày hiện tại.
