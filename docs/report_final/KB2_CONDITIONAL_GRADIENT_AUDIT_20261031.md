# KB2 V0 — audit native/DPO tại GT mục tiêu, 31/10/2026

**DPO đang kéo đúng hướng tại các quyết định được chọn, nhưng phần đóng góp thêm
vào update tổng còn nhỏ. Native loss và cập nhật tham số dùng chung có thể kéo
confidence của GT khác xuống, kể cả khi GT được gán foreground đúng.**

Đã kiểm bằng GPU trên checkpoint của [pilot2.000 update](KB2_CONDITIONAL_PILOT_REVIEW_20261031.md),
không chỉ suy từ norm loss/gradient. Không sửa training recipe, ảnh/annotation hay
checkpoint; các update thử chạy trong RAM rồi khôi phục. Không chạy train dài/test.

## 1. Phạm vi và kiểm chứng

- Run: `outputs/yolov8n_conditional_s42_20261030_180851_313339`, objective V0, seed42.
- Đo đủ784 cặp trong bank tại S0/native-last/DPO-last với cùng input batch và
  ordinary inference. Tách663 cặp DPO train và121 cặp probe thuộc train.
- Lấy đúng32 batch đầu của stream pilot:64 lượt quyết định, gồm15 recovery,
  17 preservation,24 background,8 duplicate. Recovery gồm14 target khác nhau;
  preservation16; các lượt lặp không được tính là mẫu độc lập.
- Tại S0 và DPO-last2k, từng batch được thử độc lập với native-only, weighted
  DPO-only, native+DPO và zero-current-gradient; mỗi nhánh bắt đầu cùng weights/
  optimizer history. Có thêm DPO với optimizer mới chỉ để kiểm cơ chế AdamW.
- Native/DPO/clip norm tại batch đầu S0 khớp log update thật của pilot. Đã đọc
  trực tiếp target scores/fg-mask/GT-index của native assigner và autograd tại
  slot; không tự thay nhãn native bằng binary GT.
- Kiểm16 fixed GT (8 recovery/8 preservation, trải theo class) ngoài batch cập
  nhật, tạo1.024 target–batch comparisons cho hai mốc weights. Đây là phép đo
  lặp trên16 GT train, không phải1.024 mẫu độc lập hoặc đánh giá holdout.
- Phân rã native gradient trên5 quyết định ở4 batch/tại mỗi mốc;20 phép central
  finite-difference đều qua, sai số tương đối lớn nhất0,0179% với bước L2=0,001.
- Hash checkpoint/S0/code giữ nguyên; reference và frozen state không đổi.
  Raw boxes ở toàn bộ bank và các bước thử có max-abs delta0.

Main data: [audit.json](assets/kb2_conditional_gradient_20261031/audit.json),
[pool targets](assets/kb2_conditional_gradient_20261031/pool_targets.csv),
[directions](assets/kb2_conditional_gradient_20261031/directions.csv),
[Adam decisions](assets/kb2_conditional_gradient_20261031/adam_decisions.csv).

## 2. Đo toàn bộ GT mục tiêu sau pilot

| Nhóm | Số GT | Score S0 | Score native-last | Score DPO-last | Ordinary detected S0/native/DPO |
|---|---:|---:|---:|---:|---:|
| Recovery, DPO train | 263 | 0,151178 | 0,114794 | 0,114886 | 0 /4 /4 |
| Preservation, DPO train | 77 | 0,314365 | 0,265637 | 0,266102 | 77 /39 /39 |
| Recovery, bỏ khỏi DPO | 33 | 0,158150 | 0,106642 | 0,106697 | 0 /0 /0 |
| Preservation, bỏ khỏi DPO | 14 | 0,306304 | 0,229609 | 0,229657 | 14 /5 /5 |

Score là true-class sigmoid tại **slot nguồn cố định**, không phải mean score
của những box sống sót. Ordinary detection được kiểm trên toàn bộ candidates/
NMS của ảnh, không bắt GT phải được phát hiện bằng slot nguồn.

Trên263 recovery GT DPO train, DPO cao hơn native ở213 target và thấp hơn ở50;
mean difference chỉ+0,0000917. Trên77 preservation GT,73 target cao hơn native,
4 thấp hơn; mean difference+0,0004654. Ordinary target-detected IDs vẫn giống
native-only ở cảhai nhóm: DPO chưa cứu thêm GT nào so đối chứng trong bank này.

Vì vậy đánh giá từ8 probes trước đó đã được bổ sung bằng toàn bộ bank: score
recovery DPO train giảm khoảng24% so S0;38/77 preservation target bị mất. Các
probe bỏ khỏi DPO vẫn có thể vào native và đã được S0 học, không là holdout độc lập.
Các bảng này là train-target audit, không thay thế full validation FN866→911.

## 3. Có phải DPO sai dấu hoặc native coi GT cần cứu là background?

**Không thấy hai lỗi đó ở các recovery decisions đã audit.**

Tại cả S0 và last,15/15 lượt recovery đều được native assigner gán foreground
và đúng GT ID. Tại S0, native soft target trung bình0,638348 cao hơn score0,147091;
đạo hàm native tại chính output đó âm ở15/15 lượt, tức riêng term đó muốn tăng
logit. DPO keep cũng có đạo hàm âm đúng dấu ở15/15.

Ví dụ với native classification BCE:

```text
d L_native / d z_target  ∝  sigmoid(z_target) - soft_target
```

Điều này chỉ nói hướng tại một output. Bước trên tham số là tổng tác động từ
nhiều output/ảnh khác, rồi còn chịu AdamW và clipping. Một output được gán positive
đúng không bảo đảm update tổng làm score tại đó tăng.

Preservation17/17 lượt cũng gán foreground đúng GT;16/17 term native muốn tăng
score. Một trường hợp có soft target thấp hơn score nên term native muốn giảm
score. Không suy mọi candidate đúng trong bank đều có native gradient giống nhau.

## 4. Native kéo xuống qua phần nào?

Phân rã **hướng thay đổi logit vô cùng nhỏ theo SGD chưa clip**, `-grad(z)·grad(L)`;
số dương là hướng tăng logit. Các con số dưới không phải delta của một bước AdamW.

Ví dụ recovery pair379, batch thật số4, tại S0:

| Thành phần native | Hướng tác động lên logit target |
|---|---:|
| Term của chính output target | +2,4501 |
| Các positive target khác | -3,0799 |
| Class có target0 tại foreground slots | -0,2972 |
| Background slots | -71,7231 |
| Tổng | -72,6501 |

GT này được native nhận đúng và term của chính nó muốn tăng score. Nhưng native
BCE trên các background slots khác, qua classification parameters dùng chung,
tạo tác động giảm lớn hơn lên logit đó. Không có annotation nào bị đổi.

Hai ví dụ khác tại S0: preservation pair258 có own term+3,7221, background
-239,2009, tổng native-162,1809; recovery pair165 có own term+12,4603,
background-442,5660, tổng-312,0853. Cũng có batch native tăng target: pair625 có
tổng+45,7100. Không kết luận mọi native update đều làm hại target.

[Phân rã và finite differences](assets/kb2_conditional_gradient_20261031/supplementary_checks.json).
Các ví dụ minh họa được chọn để giải thích cơ chế, không dùng làm ước lượng tỷ lệ
cho toàn bộ GT. Trong32 batch, hướng SGD native giảm ở6/15 recovery lượt tại S0.

## 5. Actual AdamW có xác nhận không?

Tại S0, optimizer mới, trên32 batch độc lập:

| Nhóm được chọn | Native: số lượt score tăng | DPO-only: số lượt score tăng | Native+DPO: số lượt score tăng |
|---|---:|---:|---:|
| Recovery | 9/15 | 15/15 | 9/15 |
| Preservation | 12/17 | 17/17 | 12/17 |

Ở các quyết định drop, DPO-only giảm score đúng hướng ở24/24 background và8/8
duplicate lượt. Như vậy công thức DPO không chỉ tăng margin nhờ slot không liên
quan như thiết kế whole-action trước; nó thực sự có thể tăng đúng score nguồn.

Ví dụ pair379 từ score0,08420895, sau **một bước**:

| Counterfactual | Score sau update |
|---|---:|
| Native-only | 0,08380181 |
| Weighted DPO-only | 0,08480847 |
| Native+weighted DPO | 0,08380471 |

DPO-only tăng đúng target; thêm DPO vào native chỉ bù được một phần rất nhỏ,
update tổng vẫn giảm score. Đây là kiểm một bước, không phải kết quả train2k
DPO-only hoặc bằng chứng DPO-only tổng quát hóa tốt.

Trên32 batch từ S0, DPO-only có một ordinary GT recovery, không mất GT trong
các ảnh batch đã đo; native+DPO không có recovery/lost ở các bước thử đó. Không
suy train DPO-only lâu sẽ không tăng FP hoặc mất GT ở ảnh khác.

**Lưu ý AdamW:** bước DPO-only đầu tiên với optimizer mới có thể lớn hơn kỳ vọng
vì adaptive normalization. Không dùng độ lớn này để so strength với optimizer
đã có2.000 bước history hoặc suy tăng lambda sẽ tăng update tuyến tính.

## 6. Momentum và targets không nằm trong batch

Tại last2k, load nguyên optimizer history của pilot rồi so từ cùng trạng thái:

- Incremental score `native+DPO - native-only` dương ở15/15 recovery và17/17
  preservation lượt được chọn; mean lần lượt+1,275e-6 và+3,190e-6.
- DPO-only với history vẫn có4/15 recovery lượt giảm score tuyệt đối. Nhưng so
  với bước zero-current-gradient (vẫn có momentum/weight decay), phần đóng góp
  thêm của DPO dương ở15/15. Đây không phải lỗi đảo dấu loss.
- DPO-only với optimizer mới tăng score ở15/15 recovery và17/17 preservation;
  reset optimizer chỉ là diagnostic, chưa là cách sửa được kiểm chứng.

Một GT chỉ nằm trong ít batch; các update từ ảnh khác vẫn thay classifier chung.
Kiểm thêm8 recovery và8 preservation fixed targets, chọn trải theo class. Tại S0,
chỉ xét những comparison mà **ảnh target hoàn toàn không có trong update batch**:

| Nhóm | Comparisons trên8 GT | Native mean delta score | Native+DPO mean delta score |
|---|---:|---:|---:|
| Recovery | 254 | -0,00007703 | -0,00007695 |
| Preservation | 251 | -0,00014475 | -0,00014469 |

Native làm score giảm ở199/254 recovery comparisons và200/251 preservation
comparisons; thêm DPO chưa đảo chiều xu hướng trung bình. DPO-only cũng không
tự bảo vệ targets ngoài batch: mean delta-0,00000476 cho recovery và-0,00001865
cho preservation trong các comparison này.

Đây là counterfactual độc lập từ S0 trên8 target/nhóm, không phải cộng dồn các
delta để dự đoán quỹ đạo2k hoặc coi505 comparisons là505 GT khác nhau. Tại last2k,
mean update từ ảnh khác vẫn âm, với độ lớn khác theo history. Kết quả hỗ trợ cơ
chế interference nhưng chưa phân rã toàn bộ nguyên nhân của45FN tăng validation.

[Fixed-probe interference audit](assets/kb2_conditional_gradient_20261031/interference/audit.json),
[chi tiết1.024 comparisons](assets/kb2_conditional_gradient_20261031/interference/fixed_probe_updates.csv).

## 7. Kết luận và thứ tự sửa nên thử

1. Giữ công thức conditional DPO làm điểm xuất phát: đã có bằng chứng nó tăng
   đúng score được chọn. Chưa có bằng chứng recipe hiện tại giúp ordinary recall.
2. Ưu tiên kiểm soát native classification background pressure và bảo vệ GT S0
   đã đúng; cân bằng số cặp50/50 không cân bằng được toàn bộ gradient native.
   Một ablation hợp lý là giảm **hệ số riêng native background**, giữ phần native
   positive có ích, rồi so native-only và native+DPO cùng cấu hình đó. Chưa mặc
   định bỏ toàn bộ native hoặc background, vì FP có thể tăng.
3. Tune lambda/beta là biến khác, thử riêng với ngân sách nhỏ và cùng bank/mask/
   stream. Không tăng loss rồi chạy dài chỉ vì DPO gradient norm nhỏ; cần ordinary
   recovery/lost/FP/AP và matched actual AdamW. Anchor bằng S0 là một hướng bảo vệ
   score có thể nghiên cứu riêng, chưa được triển khai hoặc kiểm chứng ở đây.
4. Không chuyển sang Gaussian, unfreeze regression hoặc NMS groups cùng lúc:
   audit hiện chỉ ra vấn đề confidence/update chung trong V0 có box đóng băng.

Chưa có thay đổi loss/trainer hay checkpoint mới trong lượt audit này. Các kết
luận cơ chế áp dụng cho batch/target/mốc weights đã đo; không thay bằng chứng
nhiều seed, full validation cho cấu hình mới hay test ngoài nguồn chụp.

## 8. Lệnh tái kiểm

Từ repo root; dùng output mới vì audit từ chối ghi đè:

```bash
.venv-ubuntu/bin/python -m scripts.audit_kb2_conditional_gradients --run kb2_preference_optimization/outputs/yolov8n_conditional_s42_20261030_180851_313339 --output /tmp/NEW_kb2_gradient_audit --device cuda --batches 32
```

```bash
.venv-ubuntu/bin/python -m scripts.audit_kb2_conditional_interference --run kb2_preference_optimization/outputs/yolov8n_conditional_s42_20261030_180851_313339 --output /tmp/NEW_kb2_interference_audit --device cuda --batches 32 --per-group 8
```

Dùng cùng env `YOLO_CONFIG_DIR=/tmp`, `MPLCONFIGDIR=/tmp/kb2-matplotlib`,
`NO_ALBUMENTATIONS_UPDATE=1` như các run trước. Không có file trọng số được ghi.
