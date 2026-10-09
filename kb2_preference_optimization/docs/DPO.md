# DPO online cho chọn candidate của YOLO

Nhánh `dpo` được thêm trên branch `DPO` ngày 2026-10-07 vào trainer
`train_feedback.py` và runner `run_ablation.py`. Đây là một nhánh ablation riêng,
bên cạnh baseline, sampling, hybrid và pairwise. Mã Level 2 trong `train_rl.py`
là thử nghiệm cũ, có quy tắc chọn cặp khác và không được script này gọi.

Phiên bản hiện tại là `candidate_selection_online_dpo_v2`, nguồn cặp
`online_policy_gt_iou_v2`: GT và hộp của model **đang train** xác định ưu tiên.
Phiên bản v1 chọn cặp bằng hộp reference; các số thực nghiệm v1 ở cuối tài liệu
chỉ mô tả phương pháp cũ. Không resume checkpoint v1 thành v2 hoặc dùng số v1
để kết luận về v2.

## Policy và objective

Với ảnh train sau augmentation `x`, lớp GT `c` và chỉ số candidate `a`, gọi
`q_theta(a,c|x)` là xác suất lớp do YOLO trả về. Định nghĩa policy categorical:

```text
z_theta(a,c|x) = logit(clamp(q_theta(a,c|x), 1e-6, 1-1e-6))
pi_theta(a|x,c) = softmax_a(z_theta(a,c|x))
```

Softmax chạy trên toàn bộ dense candidate của cùng lớp. Đây là phân phối
chuẩn hóa cho hành động **chọn chỉ số candidate**, không phải likelihood của
toàn bộ detection set hoặc likelihood của tọa độ hộp. Không lấy riêng sigmoid
confidence rồi mặc định coi đó là xác suất sinh detection set.

Reference `pi_ref` là bản sao đóng băng của checkpoint KB1 khởi đầu, được tạo
trước khi load policy resume. Reference không nằm trong optimizer và forward
trong `no_grad`/eval. Ảnh và GT sau augmentation được dùng giống hệt cho hai
mô hình; grid candidate phải cùng shape. **Prediction hiện tại của policy**
được detach để chọn cặp bằng GT:

- Mỗi candidate thuộc GT có IoU cao nhất.
- A là candidate có IoU cao nhất trong các candidate thuộc GT đó, với IoU >= 0.5.
- B có điểm lớp GT cao nhất trong các candidate cùng GT với IoU >= 0.1 và
  IoU thấp hơn A ít nhất sai số số học `1e-6`. B có thể có IoU >= 0.5;
  ví dụ A=0.9 và B=0.6 vẫn là cặp được phép.
- Không có A hoặc B hợp lệ thì bỏ GT đó. Candidate thuộc GT khác không làm B;
  IoU hòa nhau không tạo preference. Confidence chỉ chọn hard negative giữa
  các B có chất lượng thấp hơn; không được đảo chosen/rejected của GT.

Các chỉ số A/B **đã chọn từ policy** được dùng nguyên vẹn cho cả policy và
reference khi tính loss của bước đó. Reference chỉ cung cấp log-prob trên
cùng action indices, không chọn lại cặp, không gán nhãn ưu tiên và không được
cập nhật. Bước tiếp theo tạo cặp mới theo hộp policy mới so GT. Nhãn lỗi về
geometry của reference không được dùng để tiếp tục ưu tiên một hộp policy đã
trở nên kém hơn. Phần chọn cặp rời rạc không có gradient; class scores policy
vẫn giữ gradient để tăng ưu tiên A và giảm ưu tiên B.

```text
gap_policy = log pi_theta(A|x,c) - log pi_theta(B|x,c)
gap_ref    = log pi_ref(A|x,c)   - log pi_ref(B|x,c)
L_DPO      = mean(-log sigmoid(beta * (gap_policy - gap_ref)))
L_total    = L_native_YOLO + alpha * L_DPO
```

Đây là objective reference-relative theo công thức trong
[bài DPO gốc, Eq. 7](https://arxiv.org/html/2305.18290v3).
Pipeline offline của bài gốc dùng preference dataset đã có nhãn; ở đây cặp
được tạo lại online từ IoU policy/GT. Vì vậy dùng công thức DPO không tự động
chứng minh mọi bảo đảm lý thuyết của pipeline offline cho bài detection này.
Khi policy giống reference, `L_DPO = log(2)` trên batch có cặp hợp lệ, nhưng
gradient policy vẫn khác 0. Không thêm margin của nhánh pairwise vào công thức.
`beta` điều khiển thang log-ratio trong objective; không phải phép đo KL và
không bảo đảm KL toàn mô hình luôn dưới một ngưỡng cố định.

`alpha=0.1`, `beta=0.1` là cấu hình thử ban đầu, chưa được chứng minh tối ưu.
Nhánh này dùng shuffle giống baseline và không cộng thêm sampling/hybrid/
pairwise/object loss để giữ đối chứng dễ diễn giải. Cả pairwise và DPO v2
đều chọn cặp từ policy hiện tại. Pairwise giữ quy tắc B có IoU < 0.5,
còn DPO v2 cho phép mọi B có IoU thấp hơn A và >= 0.1. So hai nhánh này
thay đổi cả objective lẫn miền chọn B; chưa cô lập riêng tác dụng reference.

## Phạm vi và giới hạn

Gradient DPO đi qua điểm lớp của A/B và các tham số mạng tạo điểm đó; không
đi trực tiếp qua tọa độ hộp. Native loss vẫn học regression và classification.
Nhánh này không trực tiếp xử lý GT không có A >= 0.5, false positive xa GT,
số lượng detection hoặc ranking sau NMS. Chất lượng A/B được tính lại từ
hộp policy so với GT ở từng bước. Cải thiện gap so reference chưa bảo đảm
điểm A cao hơn B tuyệt đối; không suy ra mAP tăng chỉ vì loss DPO giảm.

Checkpoint ghi `candidate_selection_dpo_native_loss`, hash checkpoint nền và
reference, policy/pair schema, `dpo_objective_version`, alpha/beta, số cặp
và số bước có cặp. Checkpoint v1 không được resume vào v2. Resume
từ chối thay checkpoint nền, objective, feedback hoặc các cấu hình train
chính; reference được tái tạo từ nền ban đầu. Resume hiện vẫn chưa phục hồi
đầy đủ RNG/sampler/vị trí batch, nên không dùng resume để tuyên bố tái lập
chính xác run liên tục. Các đối chứng thực nghiệm dùng run mới từ đầu.

## Chạy

Từ root repo, chạy riêng DPO cho YOLOv8n:

```bash
bash kb2_preference_optimization/run_all_kb2.sh --model yolov8n --variants dpo --steps 15000 --patience 5 --dpo-alpha 0.1 --dpo-beta 0.1
```

Chạy baseline/pairwise/DPO cùng protocol, dùng feedback hiện có trên Windows:

```powershell
$env:NO_ALBUMENTATIONS_UPDATE='1'
.venv/Scripts/python.exe kb2_preference_optimization/run_ablation.py --model yolov8n --variants baseline pairwise dpo --feedback kb2_preference_optimization/feedback_data_clean/yolov8n_train_20261007T075619Z_354.jsonl --steps 15000 --eval-interval 1000 --patience 5 --save-interval 2000 --seed 42 --dpo-alpha 0.1 --dpo-beta 0.1 --output-dir kb2_preference_optimization/checkpoint_preference_optimization/dpo_online_v2_comparison_s42
```

Chọn thư mục output mới cho mỗi lần thử. `--variants dpo` chỉ train DPO.
Ngân sách mặc định hiện là tối đa **15.000 update**: với 2.411 ảnh train,
batch 4 và shuffle, loader train bỏ batch cuối không đủ 4 ảnh nên có 602 batch,
tương đương khoảng
24,9 lượt nếu chạy đủ ngân sách. Validation mỗi 1.000 update; early stopping
sau 5 lần liên tiếp không cải thiện best mAP50–95 quá `1e-4` (khoảng
5.000 update), thay cho cấu hình cũ 3 lần mỗi 250 update. Checkpoint trung
gian lưu mỗi 2.000 update. Đây là ngân sách để theo dõi plateau, không phải
bằng chứng đã hội tụ. Giữ cùng ngân sách và quy tắc dừng cho các nhánh;
cần xem cả số bước thực chạy vì early stopping có thể dừng chúng khác nhau.
Các pilot và run v1 bên dưới vẫn giữ số bước lịch sử, không được đổi thành
kết quả 15.000 bước. Learning rate và alpha/beta giữ nguyên.

Native checkpoint đầu vào phải đúng model và hash của feedback. Pilot dùng
2.411 ảnh train, 479 ảnh validation, ảnh 640, batch 4, AdamW lr `1e-6`,
weight decay `5e-4`. AP dùng confidence `0.001`, NMS IoU `0.45`, max_det 100.
Checkpoint tốt nhất được chọn theo validation mAP50-95, gồm cả step 0.
Không dùng tập test để chọn alpha/beta.

Mỗi nhánh ghi `*_last_metrics.jsonl` chứa loss, gradient norm, AP ở các mốc
validation; nhánh DPO còn ghi số cặp, chênh gap so reference, tỷ lệ A>B của
policy/reference và tỷ lệ gap policy cải thiện so reference. V2 ghi thêm
gap policy tuyệt đối, IoU chosen/rejected và chênh lệch chất lượng theo GT.
Các tỷ lệ này
đo trên batch train đang xét trước optimizer update, không phải AP validation.

## Kiểm chứng

```powershell
$env:NO_ALBUMENTATIONS_UPDATE='1'
.venv/Scripts/python.exe -m unittest discover -s kb2_preference_optimization/tests -p 'test_*.py'
$env:KB2_RUN_REAL_DPO='1'
.venv/Scripts/python.exe kb2_preference_optimization/tests/test_dpo_integration.py
```

Test loss kiểm tra policy chuẩn hóa, log(2) khi policy=reference, đúng dấu
gradient A/B, reference không nhận gradient, đảo preference khi chất lượng
hộp policy đảo, geometry/confidence reference không gán nhãn ưu tiên,
cặp A/B cùng trên IoU 0.5, loại positive của GT khác, và batch không có cặp.
Test checkpoint kiểm tra reference không chia sẻ storage với policy,
hash/config resume và từ chối resume v1 thành v2.

Lượt discover sau sửa v2 chạy 53 test: 52 qua, một test GPU được skip và
được chạy riêng thành công.

Test thật YOLOv8n trên batch train augmentation seed 42 có 24 cặp;
loss ban đầu `0.693147`, gradient DPO sau nhân alpha bằng `3.5336%` gradient
native, loss sau một update kết hợp là `0.692895`. Reference không thay đổi;
chênh IoU A/B trung bình là `0.049637`. Thay geometry reference không đổi loss.

Pilot v2 chạy mới baseline/pairwise/DPO 100 bước, seed 42, ghi trong
[report KB2, mục 10](../../docs/report_final/KB2.md) và artifact
`checkpoint_preference_optimization/dpo_online_v2_20261007_s42_100`.

## Kết quả lịch sử v1

Pilot v1 dùng reference chọn cặp, ghi tại
`checkpoint_preference_optimization/dpo_pilot_20261007_s42_100`.

Run v1 YOLOv8n tiếp theo đủ 1.000 update, seed 42, alpha/beta đều 0.1, ghi tại
`checkpoint_preference_optimization/ablation/20261007T075619Z_354/yolov8n`.
Best mAP50–95 là `0.385223` tại step 1.000, thấp hơn historical baseline
cùng protocol `0.385598` khoảng 0.0375 điểm phần trăm. DPO xử lý 17.306 cặp,
active trên toàn bộ 1.000 bước. AP_small tăng nhẹ; chưa thấy lợi ích tổng thể
ở mAP50–95 trên một model và một seed. Chi tiết ở mục 9 của report KB2.

Trong log run DPO, `pairwise=0 pairs=0` là hai trường của nhánh ranking
pairwise riêng đang tắt; `dpo` và `dpo_pairs` mới là loss/count của DPO.
`feedback=native` là diagnostic alias khi feedback-alpha=0, không cộng native
hai lần. Tổng loss đúng là `native + dpo_alpha * dpo`.
