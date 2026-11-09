# Hybrid KB3 → KB2: đối chứng native và giảm background — 06/11/2026

## Kế hoạch đã khóa trước khi chạy

Mục tiêu: kiểm tra ảnh hưởng của native continuation và giảm sức ép background
trong hybrid TPE→KB2. Giữ selector **mAP50–95**,15.000 update, seed42,
eval mỗi1.000 update. Best TPE step0 vẫn là ứng viên; không thay selector theo F1.

| Cấu hình | Native background | Auxiliary DPO λ | Trạng thái đầu thí nghiệm |
|---|---:|---:|---|
| Hybrid hiện tại | 0,5 | 0,1 | Đã có, best1k mAP38,5962% |
| Native-only mới | 0,5 | 0 | Chạy15k |
| Native+DPO mới | 0,25 | 0,1 | Chạy15k |

Cả hai lượt mới bắt đầu từ best TPE trial2, không tiếp tục từ hybrid best/last.
Dùng bank đầy đủ đã xác minh của lượt15k, cùng LR1e-6, AdamW WD5e-4,
beta0,1, clip1, input640, native batch4/preference2 và trainable `cv3`.
Native foreground giữ hệ số1; regression/backbone/neck/BN/reference đóng băng.
Giữ source, bank và checkpoint cũ chỉ đọc; mỗi cấu hình có output riêng.

Campaign:
`kb2_preference_optimization/outputs/yolov8n_hybrid_bg_compare_s42_20261106_051730_406262/`.
Launcher: [run_kb2_hybrid_bg_comparison.py](../../scripts/run_kb2_hybrid_bg_comparison.py).

## Cách diễn giải

- DPO/background0,5 so native/background0,5: kiểm phần đóng góp auxiliary DPO
  trong điều kiện background0,5, cùng bank/stream/budget.
- DPO/background0,25 so DPO/background0,5: kiểm việc thay hệ số background,
  giữ các thành phần khác cố định.
- DPO/background0,25 so native/background0,5 thay hai yếu tố; không dùng phép so
  đó để quy toàn bộ gain cho DPO tại background0,25. Chưa có native/background0,25.

Báo best và last, AP50/AP75, P/R/F1, macro metrics, FP/FN, size/per-class,
GT recovered/lost và probe train. Tiêu chí chọn model vẫn là AP; các metric khác
làm rõ đánh đổi. Không dùng test. Đây là một continuation seed trên validation
đã dùng phát triển, chưa là xác nhận độc lập.

## Kết quả đã hoàn tất và kiểm chứng

Hai lượt mới đều hoàn tất15k. Native/background0,5 dùng14,48 phút train wall gồm
periodic eval; lượt DPO/background0,25 hoàn tất dùng17,95 phút ở chế độ CUDA đồng
bộ. Các số này chưa tính đầy đủ initial eval và các lần thử bị ngắt, không dùng
để kết luận overhead riêng của DPO. Trainer native vẫn chạy reference/auxiliary
để chẩn đoán dù trọng số auxiliary bằng0.

### So selected best theo mAP50–95

Metric tỷ lệ tính theo%; FP/FN là số đếm tại confidence0,25.

| Cấu hình | Best step | mAP50–95 | AP50 | AP75 | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TPE trước continuation | 0 | 38,5097 | 50,4306 | 43,4945 | 50,8763 | 55,3789 | 53,0322 | 1.009 | 842 |
| DPO λ0,1 /background0,5 — mốc cũ | 1.000 | 38,5962 | 50,5191 | 43,6132 | 52,8037 | 53,8951 | 53,3438 | 909 | 870 |
| Native-only /background0,5 | 1.000 | **38,6304** | **50,5586** | **43,6553** | **53,0208** | 53,9481 | **53,4804** | **902** | 869 |
| DPO λ0,1 /background0,25 | **0** | 38,5097 | 50,4306 | 43,4945 | 50,8763 | 55,3789 | 53,0322 | 1.009 | 842 |

**Best của DPO/background0,25 là TPE step0**, không phải checkpoint đã được DPO
cập nhật. Trong15 mốc đã train, mAP cao nhất quan sát là38,4842% ở1k, vẫn thấp
hơn TPE38,5097%. Không có checkpoint mới nào được chọn theo rule AP đã khóa.

Native-only best cao hơn DPO/background0,5 best **0,0342 điểm phần trăm** mAP,
0,0395 điểm AP50,0,0421 điểm AP75,0,2171 điểm precision,0,0530 điểm recall và
0,1366 điểm F1; giảm7FP và1FN. Đây là chênh nhỏ trong **một seed**, chưa phải
kiểm định thống kê hoặc kết luận DPO nói chung không có ích.

So TPE, native-only best tăng0,1207 điểm mAP, còn recipe native+DPO/background0,5
tăng0,0865 điểm. Đối chứng mới chưa hỗ trợ phát biểu rằng phần tăng best AP của
hybrid seed42 đến riêng từ auxiliary DPO λ0,1.

### So cùng15.000 update

| Metric (%) | Native/bg0,5 | DPO/bg0,5 | DPO/bg0,25 |
|---|---:|---:|---:|
| mAP50–95 | 38,3144 | 38,3230 | **38,4427** |
| AP50 | 50,4219 | 50,3502 | **50,5424** |
| AP75 | 43,0837 | **43,1832** | 43,1745 |
| Precision | **55,6429** | **55,6429** | 52,7328 |
| Recall | 52,5172 | 52,5172 | **55,2199** |
| F1 | **54,0349** | **54,0349** | 53,9477 |
| Macro F1 | 52,3103 | **52,4395** | 52,3792 |
| Macro recall | 52,8127 | 52,8784 | **55,3183** |
| AR300 | 56,3786 | 56,4617 | **57,8557** |
| APsmall | 7,2584 | **7,2676** | 5,8781 |
| APmedium | **18,5985** | 18,5040 | 18,4339 |
| APlarge | 43,3519 | 43,3372 | **43,5710** |
| TP | 991 | 991 | **1.042** |
| FP ↓ | **790** | **790** | 934 |
| FN ↓ | 896 | 896 | **845** |

Ở cùng15k, DPO/background0,5 hơn native0,0086 điểm mAP, nhưng TP/FP/FN và micro
P/R/F1 giống nhau. Lợi ích AP này nhỏ và không đảo ngược so sánh selected best.
Không chọn lại tiêu chí sau khi xem kết quả để chỉ giữ phép so có lợi cho DPO.

Giảm background0,5→0,25 trong các lượt DPO tạo thay đổi:

- Recall **+2,7027 điểm**, FN giảm51; TP tăng51.
- mAP **+0,1197 điểm**, AP50 +0,1922 điểm ở cùng15k.
- Precision **−2,9101 điểm**, FP tăng144 và F1 giảm0,0872 điểm.
- APsmall giảm1,3895 điểm dù recall tổng thể tăng; không diễn giải recall tăng
  thành cải thiện đồng đều mọi kích thước hoặc IoU.

Như vậy giảm background có tác dụng giữ recall và làm mAP suy giảm ít hơn ở
cuối quỹ đạo. Tuy nhiên mAP last0,25 vẫn **thấp hơn TPE0,0670 điểm** và thấp hơn
best recipe cũ0,1535 điểm, nên chưa đạt mục tiêu nâng selected best AP.

### Tác động lên GT và các mốc so sánh

Ở15k, so TPE theo GT identity:

| Cấu hình | GT được cứu | GT bị mất | TP thay đổi ròng |
|---|---:|---:|---:|
| Native/bg0,5 | 22 | 76 | −54 |
| DPO/bg0,5 | 22 | 76 | −54 |
| DPO/bg0,25 | **38** | **41** | **−3** |

Probe train tách khỏi DPO cũng hỗ trợ hướng tác động: trong9 candidate
preservation ở15k, DPO/bg0,25 giữ detected9/9, so5/9 ở DPO/bg0,5; mean score
delta so TPE đổi từ−0,02981 sang+0,04778. Trong9 candidate recovery, số detected
đổi0→1 và mean score delta từ−0,04085 sang+0,00874. Đây là mẫu nhỏ thuộc **train**,
vẫn có thể vào native và đã được TPE học; không phải holdout xác nhận độc lập.

Nếu so **last DPO/bg0,25 với S0 supervised**, nhiều chỉ số cùng tăng: mAP+0,6209
điểm, AP50+1,4199, AP75+0,2065, precision+1,8101, recall+1,1129 và F1+1,4811
điểm; FP giảm50/FN giảm21. Đây là kết quả toàn pipeline TPE→native+DPO so S0,
không phải phần gain riêng của DPO. So TPE, last0,25 tăng precision/F1 và giảm75FP
nhưng vẫn tăng3FN và giảm mAP. Cần công bố cả hai mốc, không chỉ chọn mốc so có lợi.

![Learning curves của ba cấu hình](assets/kb2_hybrid_bg_compare_20261106/learning_curves.png)

## Sự cố thực thi và kiểm tính tương đương

Một lượt DPO/bg0,25 dừng giữa chừng; lượt sau gặp SIGSEGV sau mốc9k. Stack trace
dừng ở `torch.cat` trong forward native; chưa đủ xác định nguyên nhân gốc là
driver, PyTorch hay yếu tố môi trường khác. Các thư mục bị ngắt được giữ trong
campaign dưới tên `dpo_bg025_interrupted_*`; không xóa checkpoint/log hoặc sửa
chúng thành completed.

Lượt hoàn tất được khởi tạo mới từ cùng TPE/seed, bật `CUDA_LAUNCH_BLOCKING=1`
để chạy CUDA đồng bộ phục vụ chẩn đoán, theo [tài liệu PyTorch](https://docs.pytorch.org/docs/2.15/cuda_environment_variables.html).
Không thay loss, optimizer, dataset hoặc code trainer. Đã đối chiếu weights và
optimizer ở5k/9k: giống hệt lượt trước khi lỗi; full metric ở9k cũng khớp.
Đây là bằng chứng tính tương đương phần đã tái hiện, không phải chứng minh nguyên
nhân crash hoặc bảo đảm cờ này chữa mọi segmentation fault.

Sau khi hai lượt hoàn tất:

- Hash48 checkpoint eval của ba cấu hình và best/last hợp lệ; best/last khớp
  weights/optimizer của step được chọn.
- Toàn bộ15k stream ảnh/cặp giữa ba cấu hình giống nhau; nguồn TPE, bank,
  native foreground và các hyperparameter còn lại khớp.
- Đã nạp lại6 file best/last trên GPU, chấm full validation479 ảnh/1.887GT;
  mọi quality metric, per-class và GT transitions khớp log. Chấm lại cả TPE và S0.
- Reference/frozen state được giữ; source, bank và các checkpoint cũ không đổi.

Bằng chứng: [review.json](assets/kb2_hybrid_bg_compare_20261106/review.json),
[summary.csv](assets/kb2_hybrid_bg_compare_20261106/summary.csv),
[deltas.csv](assets/kb2_hybrid_bg_compare_20261106/deltas.csv),
[classes.csv](assets/kb2_hybrid_bg_compare_20261106/classes.csv),
[probes.csv](assets/kb2_hybrid_bg_compare_20261106/probes.csv).
Script kiểm: [review_kb2_hybrid_bg_comparison.py](../../scripts/review_kb2_hybrid_bg_comparison.py).

## Kết luận và bước tiếp theo

**Giả thuyết giảm background0,25 giúp nâng best mAP chưa được dữ liệu này hỗ trợ.**
Nó cải thiện recall/GT retention ở cuối quỹ đạo nhưng có đánh đổi FP và không
tạo selected checkpoint vượt TPE. Giữ recipe DPO/background0,5 làm mốc hiện tại;
giữ native-only làm đối chứng, không gọi checkpoint native là kết quả DPO.

Nếu tiếp tục tối ưu DPO sau TPE, một phép thử tiếp theo có kiểm soát là giữ
background0,5 và thử λ nhỏ hơn, thí dụ0,03. Đây là đề xuất chưa chạy: đối chứng
native cho thấy không nên mặc định λ0,1 đã chọn từ S0 cũng tối ưu sau TPE.
Trước kết luận về độ ổn định cần xác nhận theo cặp ở seed43/44; nghiên cứu hiện
vẫn chỉ một continuation seed và validation đã dùng phát triển. Chưa đánh giá test.
