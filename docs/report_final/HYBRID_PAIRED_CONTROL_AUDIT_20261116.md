# Kiểm tra đối chứng hybrid λ=0 và λ=0,03 — 16/11/2026

**Cập nhật sau audit:** khoảng trống dưới đây đã được bổ sung bằng sáu lượt
training mới. [Báo cáo đối chứng hoàn tất](HYBRID_DPO_CAUSAL_CONTROL_20261116.md)
ghi kết quả và giới hạn kết luận. Phần dưới giữ snapshot kiểm tra trước khi chạy.

## Kết luận

**Chưa có đối chứng ghép cặp nhiều seed, cùng lịch LR và cùng lịch đánh giá
dày cho hybrid best500 trong các artifact hiện có tại workspace.** Có đối
chứng hợp lệ ở seed42 với LR cố định, đánh giá mỗi1.000 update. Không được
đánh đồng hai phép so này.

Phạm vi lần kiểm tra: đọc manifest/log, kiểm hash checkpoint và đối chiếu
stream của các run đã có; không chạy training hoặc inference mới.

## Các run đã xác minh

| Run | λ hiệu lực | Seed | LR | Mốc validation | Best |
|---|---:|---:|---|---|---|
| Native-only cũ | 0 | 42 | 1e−6 cố định | 0,1000,...,15000 | 1000 / 38,6304% |
| DPO λ0,03 cũ | 0,03 | 42 | 1e−6 cố định | 0,1000,...,15000 | 1000 / 38,6314% |
| Hybrid chốt | 0,03 | 42 | 1e−6 tới1000; 1e−7 từ1001 | 0,250,500,...,2000; sau đó mỗi1000 tới15000 | 500 / 38,7133% |

Đường dẫn đầy đủ và lịch validation thực đọc từ log nằm trong
[audit.json](assets/hybrid_paired_audit_20261116/audit.json).

Kiểm tra trực tiếp đã đạt:

- Cùng checkpoint TPE trial2, SHA256
  `eb7072dd087de0154049ca8a5151c6ada6ba1fca2e2666c676a9c3d17c61a84d`.
- Cùng provenance, bank manifest, pairs, trainable names, reference-state
  và frozen-state digest giữa cả ba run.
- Ba file stream giống nhau từng byte, đủ15.000 update liên tục. Đã tính lại
  stream digest theo cách trainer ghi và đối chiếu completed.json.
- Hash54 checkpoint tại các mốc eval, cùng best/last của ba run, khớp log.
- Cặp LR cố định chỉ khác các trường arm/dpo_weight/effective_dpo_weight;
  các config còn lại giống nhau. Native có dpo_weight khai báo0,1 nhưng
  effective_dpo_weight bằng0 do arm=native: phải đọc trọng số hiệu lực.
- Các run hoàn tất từ TPE trial2 với ít nhất15k update được tìm thấy chỉ có
  seed42; không thấy cặp seed43/44. Run bị ngắt và smoke không tính là lặp seed.

Native-only cũ không có checkpoint250/500/750. Do đó không thể chấm lại một
checkpoint500 đã lưu để hoàn thiện phép so. Từ sau update1000, LR của native
cũ khác hybrid chốt; lấy giao các mốc eval sau đó vẫn không tạo đối chứng
chỉ khác DPO.

## Phép so công bằng hiện đã có

Đơn vị delta tỷ lệ là điểm phần trăm, tính từ số chưa làm tròn. Đây là cặp
LR cố định, cùng seed42 và stream; không phải cặp scheduled best500.

| Chỉ số DPO λ0,03 − native λ0 | Cùng update1000 | Cùng update15000 |
|---|---:|---:|
| mAP50–95 | +0,001079 | +0,045732 |
| AP50 | −0,002968 | −0,016379 |
| AP75 | −0,003132 | +0,080958 |
| Precision | −0,055173 | +0,068977 |
| Recall | 0 | −0,052994 |
| F1 | −0,028081 | +0,004405 |
| TP | 0 | −1 |
| FP | +2 | −3 |
| FN | 0 | +1 |

Cả hai chọn best1000. Chênh mAP best rất nhỏ, các metric còn lại không cho
thấy DPO vượt đồng đều. Tại15000 DPO cao hơn về mAP nhưng cả hai thấp hơn TPE
ban đầu. Một seed và validation đã dùng phát triển chưa xác nhận lợi ích
ổn định hay ý nghĩa thống kê; min_delta của selector không phải ngưỡng kiểm
định thống kê.

## Phần còn thiếu để thực hiện đúng đề xuất

Khóa protocol trước khi chạy: từ cùng TPE trial2 và bank, background0,5,
β0,1, AdamW LR1e−6, WD5e−4, clip1, mask cv3,15k update; LR giảm hệ số0,1
sau1000; đánh giá mỗi250 tới2000 rồi mỗi1000. Giữ selector mAP50–95,
min_delta1e−4 fraction, cho phép chọn TPE step0.

| Seed continuation | Native λ0, scheduled/dense | DPO λ0,03, scheduled/dense |
|---|---|---|
| 42 | Thiếu | Có run chốt; cần kiểm tính tương thích runtime nếu tái sử dụng |
| 43 | Thiếu | Thiếu |
| 44 | Thiếu | Thiếu |

Tối thiểu thiếu5 run nếu tái sử dụng hợp lệ DPO seed42; chạy lại cả6 run
trên cùng runtime là phương án kiểm soát chặt hơn. Trong mỗi seed, hai arm
phải cùng stream; giữa các seed, stream được thay đổi để kiểm độ ổn định.
Báo cả cùng step500, selected-best theo cùng grid, và last15000; không chọn
lại tiêu chí thuận lợi sau khi thấy kết quả. Ba continuation seed từ cùng
TPE vẫn chưa kiểm độ ổn định qua các TPE initialization khác nhau.

Trainer scheduled hiện chưa hỗ trợ λ0: `hybrid_schedule/train.py` yêu cầu
`arm='dpo'` và `dpo_weight>0`; entrypoint `scripts/run_kb2_dpo.py` cũng chặn
λ0. Cần bổ sung đường chạy đối chứng có ghi nhãn rõ trước khi launch,
giữ nguyên pair/image sampling, native loss và các bước optimizer.

GPU RTX4060 khả dụng khi kiểm tra ngoài sandbox; kiểm tra trong sandbox
không truy cập được driver. Đây là trạng thái hạ tầng, không phải kết quả
thí nghiệm mới.

## Đính chính cách diễn giải

Không viết “chưa có native-only từ TPE”: đối chứng đó đã tồn tại trong
[review λ0,03](KB2_HYBRID_LAMBDA003_REVIEW_20261106.md). Cách diễn đạt đúng là:
“Chưa có đối chứng native-only ghép cặp nhiều seed với cùng scheduled LR và
lịch đánh giá dày của hybrid chốt.” Kiểm tra hiện tại xác nhận khoảng trống
thí nghiệm này; chưa xác nhận DPO thắng hoặc thua trong thí nghiệm còn thiếu.
