# Plan PowerPoint bảo vệ đồ án: Hybrid HPO–DPO trên YOLOv8n

Ngày soạn: 11/11/2026. Kết quả mô hình chốt: 08/11/2026. S0 trong bảng được cập nhật ngày 11/11/2026.

**Cập nhật bản trình chiếu:** Theo yêu cầu rà soát tiếp theo, PPTX/PDF/HTML hiện đã rút còn **27 trang**. Các nhãn cũ 25–29,31 và toàn bộ phụ lục được bỏ; nhãn cũ30→25,32→26. Đã đối chiếu và làm rõ nội dung các nhãn04,06,07,09,11,12,14,17–20. Xem [báo cáo đối chiếu và giải thích](SLIDE_HYBRID_CONTENT_AUDIT_20261111.md) và [hướng dẫn bản slide hiện hành](slides/hybrid_hpo_dpo_20261111/README.md). Phần dưới giữ thiết kế đầy đủ ban đầu để tra cứu; bản slide hiện hành và báo cáo đối chiếu là nguồn cho nội dung đã sửa.

## 1. Phạm vi, nguồn và quy ước bắt buộc

Đây là bản thiết kế nội dung và bố cục để dựng PowerPoint, gồm **32 slide nội dung, 1 bìa tùy chọn và 8 slide phụ lục**. Mỗi slide có nội dung hiển thị, vị trí, hình/công thức, lời thuyết trình và nguồn. Không phải file `.pptx` đã dựng.

Không tìm thấy file có đúng tên `report_kb2_20261108` trong workspace. Tài liệu khớp ngày, tab IDE và toàn bộ bảng người dùng cung cấp là [KB2_HYBRID_SELECTED_EXPLANATION_20261108.md](KB2_HYBRID_SELECTED_EXPLANATION_20261108.md); dùng tài liệu này làm nguồn chính. Kiến trúc và quy trình được bổ sung từ code và [Report_Hybrid_KB3_TPE_KB2_DPO_20261109.md](Report_Hybrid_KB3_TPE_KB2_DPO_20261109.md).

**Ưu tiên số liệu:** [metrics.csv](assets/kb2_hybrid_selected_20261108/metrics.csv), [selection.json](assets/kb2_hybrid_selected_20261108/selection.json), [mechanism_evidence.json](assets/kb2_hybrid_selected_20261108/mechanism_evidence.json) và bảng người dùng gửi. Báo cáo Hybrid ngày 09/11 còn dùng S0 lịch sử ở một số đoạn; không chép các con số đó vào phép so hiện tại.

| Tên dùng xuyên suốt slide | Định nghĩa chính xác |
|---|---|
| S0 | KB3 trial 0, anchor theo recipe KB1, train từ pretrained; native-selected epoch 91 |
| Best TPE | KB3 trial 2; native-selected epoch 139, thực chạy 169 epoch |
| Hybrid chốt | Từ Best TPE → KB2 native + DPO; λ=0,03; seed 42; best update 500 |
| S0 lịch sử | Checkpoint KB1 cũ; chỉ xuất hiện khi trình bày đối chứng lịch sử tại slide 31 |
| HPO | Tối ưu siêu tham số; trong kết quả này là KB3-A với TPE |
| DPO/KB2 | Pipeline preference gồm bank, GT judge, sampling, native loss và DPO loss |

S0 trial 0 và trial 2 **train độc lập từ cùng pretrained**, không nối weights trial 0 sang trial 2. Chỉ đoạn Best TPE → Hybrid mới là fine-tune tiếp weights. Không đưa PPO/KB3-B vào sơ đồ kết quả chính.

Luồng trình bày theo yêu cầu: **Hybrid → YOLOv8n → HPO/TPE → DPO/ánh xạ YOLO → kết quả → giải thích và minh chứng**. Bìa là slide 00 tùy chọn để slide nội dung đầu tiên luôn là kiến trúc Hybrid.

## 2. Hệ thiết kế chung

### 2.1. Kích thước, lưới và vị trí

- PowerPoint widescreen **16:9**, kích thước **13,333 × 7,5 inch**.
- Tất cả tọa độ bên dưới là **(x; y; w; h)** theo inch, gốc ở góc trên trái. Dùng dấu `;` để tránh nhầm dấu phẩy thập phân.
- Lề trái/phải: 0,50. Vùng nội dung rộng 12,33; không đặt chữ/hình sát mép.
- Nhãn chương nhỏ: `(0,50; 0,18; 11,50; 0,25)`, 11–12 pt, in hoa vừa phải.
- Tiêu đề: `(0,50; 0,52; 12,33; 0,65)`, 29–32 pt, tối đa hai dòng ngắn.
- Nội dung chính: y=1,35 đến 6,30. Dòng kết luận: `(0,50; 6,40; 12,33; 0,40)`.
- Footer nguồn: `(0,50; 7,02; 11,70; 0,22)`, 10–11 pt; số slide: `(12,20; 7,00; 0,63; 0,25)`.
- Khi một slide có bố cục riêng, tọa độ riêng thay thế vùng nội dung chung. Tiêu đề/footer vẫn giữ nguyên.

### 2.2. Màu, font và biểu diễn

| Thành phần | Quy định |
|---|---|
| Nền | Kem rất nhạt `#F7F2E6`, gợi lại bảng kết quả người dùng gửi |
| Chữ chính | Nâu đậm `#3F3428` |
| HPO/TPE | Xanh lam `#2463A6`, nền vùng `#E8EFF8` |
| DPO/keep | Xanh lục `#237A57`, nền vùng `#E8F3ED` |
| Reference/frozen | Xám `#7B8490`, nền `#EDF0F2`; thêm chữ “đóng băng” |
| Trade-off/drop/chỗ giảm | Cam đất `#B76532`; không dùng đỏ cho mọi giá trị âm |
| Checkpoint được chọn | Vàng nhạt `#F1D994` với viền nâu |
| Font tiếng Việt | Aptos hoặc Arial; nội dung 20–23 pt; nhãn sơ đồ 17–20 pt |
| Công thức | Cambria Math, 24–28 pt; công thức dài tách dòng |
| Bảng dày | 17–18 pt, hàng tối thiểu 0,28 inch; chú thích ngoài bảng |
| Mũi tên | Nét liền: dữ liệu/weights; nét đứt: metric feedback; nét chấm: tham chiếu/kiểm tra |

Không dựa riêng vào màu: luôn có nhãn `trainable`, `frozen`, `keep`, `drop`, `đo thực nghiệm`, `minh họa`. Công thức dựng bằng PowerPoint Equation; bảng là đối tượng bảng có thể sửa. Không dùng ảnh chụp bảng làm nội dung chính.

### 2.3. Mật độ nội dung và hoạt ảnh

Một slide truyền tải một thông điệp. Công thức có bảng ký hiệu hoặc chú giải cạnh bên; không đặt đoạn văn dài lên slide. Lời giải thích dài nằm trong speaker notes của plan. Hoạt ảnh chỉ dùng Appear/Fade 0,2–0,3 giây để hiện từng bước; không tự chạy. Nguồn kỹ thuật ghi ngắn ở footer, đường dẫn đầy đủ nằm trong speaker notes/mục nguồn cuối tài liệu.

## 3. Nhịp trình bày

| Phần | Slide | Thời lượng gợi ý |
|---|---|---:|
| Tổng thể, detector, dữ liệu | 01–03 | 2–3 phút |
| HPO/TPE | 04–10 | 6–7 phút |
| DPO | 11–22 | 10–12 phút |
| Kết quả và giải thích | 23–31 | 8–10 phút |
| Kết luận | 32 | 1 phút |

Bản đầy đủ khoảng **28–33 phút**, tùy mức giải thích công thức. Nếu chỉ có 20 phút: trình bày 01, 02, 03, 04, 05, 07, 08, 10, 12, 13, 14, 15, 18, 20, 21, 22, 24, 25, 26, 27, 29, 32; các slide còn lại chuyển thành dự phòng. Không cố nói hết 32 slide trong 15 phút.

## 4. Plan chi tiết từng slide

### Slide 00 — Bìa, tùy chọn

**Mục đích:** Giới thiệu đề tài trước khi bắt đầu luồng kiến trúc.

**Bố cục và nội dung:**

- `(0,65; 0,50; 12,00; 0,65)`: tên trường/khoa, 16–18 pt; điền thông tin thực của sinh viên.
- `(0,65; 1,70; 8,60; 1,50)`: “TỐI ƯU PHÁT HIỆN BỆNH LÁ CÂY / BẰNG HYBRID HPO–DPO TRÊN YOLOv8n”, 32–36 pt. Điều chỉnh tên để khớp tên đồ án chính thức.
- `(0,70; 3,65; 7,60; 1,10)`: sinh viên, MSSV, giảng viên hướng dẫn; chưa có dữ liệu nên giữ placeholder có nhãn.
- `(9,20; 1,80; 3,30; 3,30)`: sơ đồ ba khối dọc nhỏ “YOLOv8n → TPE → DPO”; không đưa kết quả mới lên bìa.
- `(0,70; 6,55; 10,00; 0,35)`: địa điểm, ngày bảo vệ; điền sau, không lấy ngày soạn làm ngày bảo vệ.

**Lời nói:** “Em trình bày phương án kết hợp tìm siêu tham số bằng TPE và tinh chỉnh preference keep/drop trên detector YOLOv8n.”

### Slide 01 — Kiến trúc Hybrid: HPO tìm cấu hình, DPO tinh chỉnh quyết định

**Thông điệp:** Đây là hai giai đoạn nối tiếp; đầu ra cuối vẫn là một detector YOLOv8n.

**Bố cục:**

- `(0,50; 1,50; 2,00; 3,65)`: cột nguồn: “Train + GT”, “Validation”, “YOLOv8n pretrained”.
- `(2,90; 1,50; 3,45; 3,65)`: khung xanh lam “Giai đoạn 1 — HPO / KB3”; ba ô: “TPE chọn lr0, WD” → “Train YOLOv8n” → “Chọn Best TPE”.
- `(6,80; 1,50; 3,45; 3,65)`: khung xanh lục “Giai đoạn 2 — DPO / KB2”; ba ô: “Sinh preference từ train GT” → “Native + DPO, chỉ cv3 học” → “Chọn Hybrid best”.
- `(10,70; 2,25; 2,13; 2,10)`: khung vàng “YOLOv8n cuối / Ảnh → box, class, score”.
- `(0,65; 5,45; 12,00; 0,65)`: dải phân biệt “HPO đổi cấu hình huấn luyện giữa các trial; DPO cập nhật weights phân loại sau khi đã chọn TPE”.

**Sơ đồ phải vẽ:** pretrained → khối HPO; Best TPE → DPO; DPO → detector cuối. Train+GT nối tới cả hai khối. Validation đi vào các selector bằng nét đứt. Không vẽ test feedback vào optimizer.

**Lời nói:** “HPO chọn cách huấn luyện detector. Sau đó KB2 dùng chính detector tốt nhất làm nguồn preference và làm reference để tinh chỉnh nhánh phân loại. Suy luận chỉ cần model cuối.”

**Nguồn:** `hybrid/source.py`, `hybrid/bank.py`, `hybrid_schedule/train.py`; báo cáo chốt mục 1.

### Slide 02 — YOLOv8n: cấu trúc được dùng chung trong hai giai đoạn

**Thông điệp:** HPO huấn luyện detector theo recipe native; KB2 chỉ mở nhánh classification.

**Bố cục:**

- `(0,50; 1,55; 1,15; 2,95)`: ảnh đầu vào mẫu hoặc ô “RGB / 640×640”. Nếu dùng ảnh thật, lấy một ảnh train và ghi split.
- `(1,95; 1,55; 2,00; 2,95)`: Backbone “Conv / C2f / SPPF”.
- `(4,25; 1,55; 2,00; 2,95)`: Neck “Upsample / Concat / C2f”.
- `(6,55; 1,40; 2,15; 3,30)`: ba feature maps “P3 80×80×64”, “P4 40×40×128”, “P5 20×20×256”.
- `(9,00; 1,40; 3,83; 1,50)`: nhánh `cv2`, “64 regression logits/slot → DFL → box”; nền xám, nhãn “KB2 frozen”.
- `(9,00; 3,20; 3,83; 1,50)`: nhánh `cv3`, “28 class logits/slot → sigmoid”; xanh lục, nhãn “KB2 trainable”.
- `(0,65; 5,15; 12,00; 0,90)`: ba ô số: “8.400 raw slots”, “3.016.308 tham số”, “374.868 tham số cv3 ≈12,43%; 24 tensor”.

**Chi tiết trình bày:** Mũi tên từ cả ba scale tới cả hai nhánh head. Chú thích nhỏ: “YOLOv8 này không có objectness logit riêng; sigmoid theo class, không softmax 28 class”. Không dùng số tham số bản COCO 80 class thay cho checkpoint 28 class.

**Lời nói:** “Đề tài chọn YOLOv8n làm detector nền cho toàn bộ pipeline. Điểm then chốt là KB2 đổi cách chấm class tại các candidate có sẵn; nhánh tọa độ được giữ nguyên. Đề tài chưa có benchmark inference độc lập để khẳng định FPS.”

**Nguồn:** audit kiến trúc trong báo cáo Hybrid mục 3; `conditional/policy.py::ConditionalAdapter`.

### Slide 03 — Dữ liệu và mục tiêu tối ưu

**Thông điệp:** Cùng dữ liệu phát hiện 28 lớp; validation phục vụ phát triển và chọn mô hình.

**Bố cục:**

- `(0,50; 1,50; 5,70; 3,75)`: ba thẻ xếp dọc: “Train: 2.411 ảnh / 10.333 GT”; “Validation: 479 ảnh / 1.887 GT”; “Test có trong dataset: 239 ảnh”.
- `(6,55; 1,50; 6,28; 3,75)`: minh họa một ảnh có nhiều đối tượng; box nét xanh + nhãn class; bên dưới ba yêu cầu “đúng class — box khớp GT — score/xếp hạng hợp lý”. Nếu chưa chọn ảnh thật, ghi “sơ đồ minh họa”, không ghi là prediction.
- `(0,65; 5,55; 12,00; 0,55)`: “Mục tiêu lựa chọn: validation mAP50–95; đồng thời theo dõi precision, recall, F1, FP, FN”.

**Ghi chú nói:** “Train có 667 ảnh augmentation đã biết. Preference chỉ sinh từ train. Các con số kết quả hôm nay là validation phát triển; test chung có lịch sử đã được KB1 xem, nên không gọi đây là kiểm chứng holdout mới.”

**Nguồn:** dataset manifest trong `mechanism_evidence.json`, báo cáo Hybrid mục 2 và 12.

### Slide 04 — Kiến trúc HPO/KB3: hai vòng tối ưu

**Thông điệp:** Vòng ngoài chọn HP; vòng trong train weights detector.

**Bố cục:**

- `(0,50; 1,50; 3,00; 4,30)`: TPE, chứa “History D”, “Ước lượng ℓ/g”, “Đề xuất hᵢ=(lr0, WD)”.
- `(4,00; 1,50; 5,30; 4,30)`: khung vòng trong, chứa “Nạp pretrained mới” → “Train native” ↔ “Native validation từng epoch” → “Native-best EMA”.
- `(9,80; 1,50; 3,03; 4,30)`: “Canonical evaluate selected-best” → “F(hᵢ)=mAP50–95” → “Lưu (hᵢ,Fᵢ)”.
- `(0,65; 6,00; 12,00; 0,30)`: “Metric quay lại TPE sau khi trial hoàn tất”.

**Mũi tên:** TPE → hᵢ → vòng trong → canonical score; score trở về history bằng nét đứt chạy phía dưới. Validation từng epoch nằm bên trong vòng trong, tránh nhầm với feedback TPE.

**Công thức nhỏ cạnh TPE:** `h* = argmax_{h_i đã thử} F(h_i)`.

**Lời nói:** “TPE không backprop qua mAP và không đổi learning rate base sau mỗi epoch. Nó chờ train, chọn checkpoint và evaluate xong một cấu hình rồi mới chọn cấu hình tiếp theo.”

**Nguồn:** `quality/tpe_pipeline.py::propose/search`, `quality/pretrained_detector.py`.

### Slide 05 — Không gian tìm kiếm và recipe giữ cố định

**Thông điệp:** Miền tìm cố định cho hai biến lr0 và weight decay. Lịch sử mAP thay đổi giá trị TPE ưu tiên thử trong miền này.

**Bố cục:**

- `(0,50; 1,50; 7,20; 3,65)`: bảng 4 cột “Tham số / Anchor / Miền thử / Vai trò”.
- `(8,00; 1,50; 4,83; 3,65)`: thẻ recipe cố định.
- `(0,65; 5,50; 12,00; 0,65)`: “8 trial tổng; cùng pretrained và detector seed 0”.

**Bảng phải nhập:**

| Tham số | Anchor | Miền tìm | Thang |
|---|---:|---|---|
| lr0 | 0,01 | [0,005; 0,015] | Log |
| weight_decay | 0,0005 | [0,00025; 0,001] | Log |
| momentum | 0,937 | Giữ nguyên | — |
| augmentation | Recipe gốc | Giữ nguyên | — |

**Thẻ bên phải:** SGD Nesterov; batch 16; input 640; tối đa 200 epoch; patience 30; warmup 3; `lrf=0,01`. Chữ nhỏ “augmentation_strength=0,5 là abstraction của recipe, không nhân 0,5 lên mọi phép augment”.

**Lời nói:** “Chỉ hai biến LR và WD thay đổi giữa các trial. Các thành phần còn lại giữ theo recipe để giảm phạm vi tìm kiếm và làm phép so dễ kiểm tra hơn.”

**Nguồn:** `configs/kb3_tpe.yaml`; `tpe_pipeline.py::local_distributions`; manifest run tám trial.

### Slide 06 — TPE đề xuất trial mới từ các trial đã chạy

**Thông điệp:** Miền ở slide 05 quy định giá trị hợp lệ. Các cặp cấu hình và mAP đã đo giúp TPE quyết định giá trị nên thử tiếp.

**Bố cục:** Bảng hai kết quả đã có trước trial 2, bên dưới là ba bước chia nhóm theo điểm, xét từng tham số trong miền, rồi đưa cấu hình cho trainer.

| Trial đã chạy | lr0 | Weight decay | mAP50–95 validation |
|---|---:|---:|---:|
| 0 | 0,010000 | 0,000500 | 36,0677% |
| 1 | 0,005674 | 0,000582 | 37,7225% |

**Lời nói:** “Trial 1 có mAP cao hơn nên vào nhóm điểm tốt. Với từng tham số, TPE xét những giá trị trong khoảng ở slide 05 và ưu tiên giá trị có mật độ tương đối cao trong nhóm tốt. Cặp được đề xuất cho trial 2 là lr0 khoảng 0,005355 và weight decay khoảng 0,000913. Sau khi train và chấm YOLOv8n, ta mới có mAP 38,5071% để đưa lại vào lịch sử.”

**Ghi chú kỹ thuật:** `n_startup_trials=2`, `n_ei_candidates=24`, `multivariate=False`. Sampler sinh 24 giá trị nội bộ cho từng tham số và dùng tỷ số ℓ/g để xếp hạng. Slide 07 trình bày công thức.

**Nguồn:** `tpe_trials.csv`, `tpe_pipeline.py::propose`, [mã nguồn Optuna 4.5.0 TPESampler](https://optuna.readthedocs.io/en/v4.5.0/_modules/optuna/samplers/_tpe/sampler.html).

### Slide 07 — Công thức TPE: từ Expected Improvement tới ℓ/g

**Thông điệp:** Density ratio là tiêu chí đề xuất HP; không phải gradient cập nhật YOLO.

**Bố cục:**

- `(0,50; 1,50; 8,15; 1,15)`: định nghĩa `F(h)=validation mAP50–95`, đổi sang minimization `y=−F(h)`.
- `(0,50; 2,90; 8,15; 1,15)`: công thức mật độ.
- `(0,50; 4,30; 8,15; 1,55)`: công thức EI và ratio, tách tối đa hai dòng.
- `(9,00; 1,50; 3,83; 4,35)`: bảng ký hiệu h, y, y*, ℓ, g, γ.

**Công thức hiển thị:**

\[
\ell(h)=p(h\mid y<y^*),\qquad g(h)=p(h\mid y\ge y^*)
\]
\[
p(h)=\gamma\ell(h)+(1-\gamma)g(h)
\]
\[
EI(h)=\int_{-\infty}^{y^*}(y^*-y)p(y\mid h)\,dy
\ \propto\ \left[\gamma+(1-\gamma)\frac{g(h)}{\ell(h)}\right]^{-1}
\]

**Kết luận dưới slide:** “Tối đa EI ⇔ ưu tiên ℓ(h)/g(h) cao, theo mô hình TPE”.

**Notes:** γ trong lý thuyết là khối lượng nhóm tốt. Hàm `default_gamma(n)` trong Optuna trả **số** observation tốt: `min(ceil(0,1n),25)`. Với n=2…7 của run này, nhóm tốt chỉ có 1 observation. Không ghi một split 20% tự đặt. Implementation independent sampling xét từng HP; đồ thị chung h là cách viết lý thuyết, không khẳng định đã fit joint density LR–WD.

**Lời nói:** “TPE mô hình hóa HP khi biết objective thuộc nhóm tốt hay còn lại. Công thức EI dẫn tới ưu tiên ratio ℓ/g lớn. Đây là đề xuất có xác suất, nên trial kế tiếp vẫn có thể kém hơn.”

**Nguồn:** báo cáo Hybrid mục 4; [Bergstra et al., 2011](https://papers.nips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html); source Optuna 4.5.0.

### Slide 08 — Một trial: Train → Val → chọn epoch → Eval → cập nhật TPE

**Thông điệp:** Có hai tầng lựa chọn với hai đường đánh giá.

**Bố cục:**

- `(0,50; 1,50; 12,33; 3,00)`: timeline 6 ô ngang, mỗi ô khoảng 1,80 rộng, cách 0,25.
- `(0,50; 4,85; 5,95; 1,20)`: hộp “Tầng 1: native val chọn epoch/EMA trong trial”.
- `(6,80; 4,85; 6,03; 1,20)`: hộp “Tầng 2: canonical score xếp hạng giữa trial”.

**Sáu ô timeline:**

1. TPE chọn `lr0`, `WD`.
2. Nạp pretrained; train nhiều epoch bằng native loss.
3. Native val sau epoch; cập nhật best và patience.
4. Early stop hoặc đủ 200 epoch; lấy `selected_best.pt`.
5. Canonical eval selected-best và last; chỉ selected-best cung cấp objective.
6. Lưu `(hᵢ,Fᵢ)`; TPE đề xuất trial kế tiếp.

**Công thức trong hộp tầng 1:** `e* = NativeSelect{AP_native(EMA_e)}`. Hộp tầng 2: `F(h)=AP_canonical(EMA_e*)`.

**Notes:** Native val NMS IoU 0,7; canonical 0,6; max_det 300. Fitness native của version dùng trong run là AP50–95, không chép công thức cũ `0,1AP50+0,9AP50–95`. LR thực vẫn warmup/decay dù `lr0` của trial cố định. Không dùng canonical để chọn lại mọi epoch.

**Lời nói:** “Validation trong lúc train giúp chọn epoch. Evaluation sau trial chấm lại checkpoint đã chọn theo cùng protocol chuẩn để trả một score cho TPE. Hai bước không nên gộp làm một.”

**Nguồn:** `pretrained_detector.py::epoch_end/train_detector`; báo cáo Hybrid mục 7.1.

### Slide 09 — TPE chọn gì sau mỗi vòng? Ví dụ đúng từ 8 trial

**Thông điệp:** TPE dùng toàn bộ completed history; trial mới không thừa kế weights trial cũ.

**Bố cục:**

- `(0,50; 1,50; 8,00; 3,95)`: 4 dòng tuần tự, mỗi dòng “history → cách chọn → score”.
- `(8,85; 1,50; 3,98; 3,95)`: khung “Từ trial 3 trở đi” và ba lưu ý.
- `(0,65; 5,75; 12,00; 0,45)`: “Score ở slide này là observations lịch sử, không thay bằng số nạp lại của bảng cuối”.

**Bốn dòng:**

- Trial 0: chưa có history → enqueue anchor `(0,01; 0,0005)` → **36,0677%**.
- Trial 1: mới có 1 trial → startup random `(0,005673682; 0,000581614)` → **37,7225%**.
- Trial 2: đủ 2 observations → TPE chọn `(0,005354944; 0,000913034)` → **38,5071%**.
- Trial 3: dùng history trial 0–2 → TPE chọn cấu hình mới → **37,0877%**.

**Khung phải:** “Không tăng đơn điệu”; “Không chỉ nhìn trial vừa xong”; “Sampler seed=42+i, detector seed=0”.

**Lời nói:** “Trial 2 tốt nhất không có nghĩa trial 3 sẽ tốt hơn. Mỗi lần, code dựng study từ toàn bộ history đã hoàn tất, dùng sampler seed theo logical trial, rồi train một detector mới từ pretrained.”

**Nguồn:** `tpe_pipeline.py::propose/search`; `tpe_trials.csv`.

### Slide 10 — Kết quả HPO: chọn trial 2 trong 8 cấu hình đã thử

**Thông điệp:** Winner là tốt nhất trong ngân sách đã thử; chưa chứng minh HP tối ưu toàn cục.

**Bố cục:**

- `(0,50; 1,50; 8,00; 4,40)`: biểu đồ cột 8 trial, trục y “Canonical mAP50–95 lịch sử (%)”; trial 2 xanh đậm, trial khác xanh nhạt/xám. Trục cột bắt đầu 0; nếu dùng bản zoom phải ghi rõ thang cắt.
- `(8,85; 1,50; 3,98; 4,40)`: thẻ winner và chi phí search.

**Nhãn trên cột:** `36,0677 / 37,7225 / 38,5071 / 37,0877 / 38,3459 / 36,2870 / 37,2655 / 38,2293`.

**Thẻ phải:** “Trial 2”; `lr0=0,005354944`; `WD=0,000913034`; “Best epoch 139 / chạy 169”; “Tổng search: 8 trial, 1.387 epoch, 22,7038 giờ theo result”.

**Chú thích:** “Re-evaluate checkpoint TPE cho bảng cuối: 38,5097%; giữ score lịch sử khi mô tả search”.

**Lời nói:** “TPE tìm được trial 2 trong local search. Các trial sau vẫn được thực hiện đủ ngân sách nhưng không vượt winner. Một trial random ban đầu chưa phải đối chứng random search cùng ngân sách.”

**Nguồn:** `assets/hybrid_tpe_dpo_20261109/tpe_trials.csv`; có thể dùng phần biểu đồ phù hợp từ `tpe_search.svg`, tránh chữ quá nhỏ.

### Slide 11 — Chuyển giao Best TPE sang DPO

**Thông điệp:** Policy và reference khởi đầu giống nhau; optimizer KB2 được tạo mới.

**Bố cục:**

- `(0,50; 2,00; 3,10; 2,60)`: checkpoint “Best TPE / trial 2”.
- `(4,25; 1,50; 3,75; 1,75)`: bản sao Policy, xanh lục; “Chỉ cv3 học”.
- `(4,25; 3,75; 3,75; 1,75)`: bản sao Reference, xám; “Toàn bộ frozen”.
- `(8,65; 1,50; 4,18; 4,00)`: checklist chuyển giao: cùng dataset/source; kiểm checkpoint hash; bank sinh từ TPE; AdamW mới; cùng batch policy/reference.
- `(0,65; 5,85; 12,00; 0,45)`: công thức `θ₀=w_TPE; θ_ref=w_TPE; ∇θ_ref L=0`.

**Lời nói:** “Không tiếp tục optimizer SGD của KB3. Cùng một checkpoint TPE được nạp thành policy và reference; reference giữ mốc logit ban đầu, còn policy có phần classification được học.”

**Nguồn:** `hybrid/source.py`; `hybrid_schedule/train.py` phần khởi tạo.

### Slide 12 — Kiến trúc DPO/KB2: dữ liệu preference và hai nhánh loss

**Thông điệp:** Native loss học trên ảnh/GT; DPO loss học quyết định ưu tiên tại candidate.

**Bố cục:**

- `(0,50; 1,50; 3,30; 3,65)`: khối offline “Train + GT → TPE raw output → A/B + GT judge → preference bank”.
- `(4,20; 1,50; 4,30; 3,65)`: khối online “Batch 4 ảnh → Policy / Reference → logits đúng slot/class”.
- `(8,90; 1,50; 3,93; 3,65)`: hai ô “L_native, bg=0,5” và “L_DPO, β=0,1” nối vào “Tổng loss → AdamW → cv3”.
- `(0,65; 5,55; 12,00; 0,65)`: công thức lớn `L_total = L_native,bg0.5 + 0,03 L_DPO`.

**Mũi tên:** Native GT đi tới native loss. Preference record đi tới chọn logit/nhãn keep-drop. Reference chỉ đi tới DPO loss, không có đường gradient quay lại. Validation nằm ngoài khối train và chỉ tới selector.

**Lời nói:** “KB2 không chỉ thêm một DPO loss đơn độc. Kết quả hiện tại đến từ tổ hợp native loss, preference bank, sampler và vùng tham số cho phép học. Vì vậy gain so TPE là gain của pipeline KB2.”

**Nguồn:** `hybrid_schedule/train.py`; `conditional/policy.py`; `conditional_bg/loss.py`.

### Slide 13 — Ánh xạ DPO sang YOLO: mỗi khái niệm tương ứng với gì?

**Thông điệp:** Action dùng trong đề tài là Bernoulli keep/drop tại raw slot, class.

**Bố cục:**

- `(0,50; 1,45; 8,25; 4,75)`: bảng 3 cột “DPO / Trong YOLO của đề tài / Cách hiểu”.
- `(9,10; 1,65; 3,73; 4,20)`: hình một candidate j, class c; hai nhánh KEEP/DROP; ô dưới “Mạng chỉ nhận ảnh”.

**Các dòng bảng:**

| DPO | Ánh xạ |
|---|---|
| Input x | Ảnh train, resize/pad xác định |
| Policy πθ | YOLOv8n từ TPE, cv3 học |
| Reference πref | Bản TPE đóng băng |
| Response/action y | Keep hoặc drop candidate tại `(j,c)` |
| Chosen y_w | Phương án qua GT judge |
| Rejected y_l | Phương án còn lại |
| Probability | `q(keep)=sigmoid(z_jc)`; `q(drop)=1−sigmoid(z_jc)` |
| Preference context | Dùng cho dựng/chấm A/B; không phải tensor input phụ |

**Chú thích:** “Không định nghĩa xác suất của toàn bộ set detection sau NMS bằng confidence trung bình hoặc −detection loss”.

**Lời nói:** “YOLO không có sẵn likelihood của cả tập detection. Đề tài định nghĩa policy phụ nhị phân tại một slot/class rồi dùng cấu trúc DPO trên policy đó. Đây là thiết kế ánh xạ của đề tài.”

**Nguồn:** `conditional/policy.py::pair_loss`; báo cáo Hybrid mục 10.2.

### Slide 14 — Tạo cặp A/B bằng cách bật/tắt một candidate

**Thông điệp:** Hai phương án dùng chung raw predictions; chỉ một active-bit thay đổi trước NMS.

**Bố cục:**

- `(0,50; 1,50; 2,75; 3,85)`: ảnh và raw candidates từ TPE; đánh dấu candidate j.
- `(3,70; 1,50; 3,65; 1,70)`: A: KEEP j → NMS → set A.
- `(3,70; 3,65; 3,65; 1,70)`: B: DROP j → NMS → set B.
- `(7,85; 1,50; 4,98; 3,85)`: GT judge toàn ảnh; “So ID GT được match + FP” → chosen/rejected.
- `(0,65; 5,70; 12,00; 0,55)`: “Box, class, score nguồn giữ nguyên khi dựng A/B”.

**Minh họa trên slide:** sơ đồ 8 GT, B có 5 TP/2 FP; A có 6 TP/2 FP và giữ cả 5 GT cũ → chọn KEEP. Dán nhãn “ví dụ minh họa, không phải ảnh đo thực nghiệm”.

**Notes quan trọng:** Với recovery, KEEP cưỡng bức candidate score thấp tham gia renderer; không tự tăng source score lên 0,25. Judge không loại lại candidate đó bằng operating threshold. Đánh giá model sau train vẫn áp threshold thật. Không kể A/B là hai lần predict ngẫu nhiên hay hai ảnh augment.

**Lời nói:** “NMS có thể làm các box cạnh tranh nhau nên judge phải chấm hậu quả trên toàn ảnh. Cặp hợp lệ offline chỉ cung cấp hướng học, chưa có nghĩa model thực tế đã tăng score qua ngưỡng.”

**Nguồn:** `conditional/pairs.py::render/judge/build_pairs`.

### Slide 15 — Bốn nhóm preference và điều kiện GT judge

**Thông điệp:** Chỉ nhận cặp có lợi và không làm mất GT khác theo renderer.

**Bố cục:**

- `(0,50; 1,50; 8,00; 4,45)`: bảng 4 nhóm, 3 cột “Nhóm / Candidate / Chosen”.
- `(8,85; 1,50; 3,98; 4,45)`: điều kiện tập GT và FP, font công thức 23 pt.

**Nội dung bảng:**

| Nhóm | Candidate | Chosen |
|---|---|---|
| Recovery | GT đang bỏ sót; đúng class; score [0,05;0,25); IoU≥0,60 | Keep nếu cứu GT mục tiêu |
| Preservation | Candidate đang giữ một GT đã được phát hiện | Keep nếu bỏ gây mất GT |
| Background | FP, max IoU với mọi GT≤0,10; tâm ngoài GT | Drop nếu giảm FP |
| Duplicate | Prediction dư cho GT đã match | Drop nếu giảm FP |

**Khung phải:** `M_l ⊆ M_w`; `FP_w ≤ FP_l`; KEEP cần `g_target ∈ M_w \ M_l`; DROP cần `FP_w < FP_l`. Dưới cùng: “Matching đúng class, IoU 0,5; NMS 0,6; cap 300”.

**Notes:** `w/l` lần lượt chosen/rejected; kiểm GT identity thay vì chỉ đếm tổng TP. Recovery thử tối đa 3 candidate/GT và nhận tối đa 1 pair recovery/GT; mỗi group cap 4 accepted pairs/ảnh. Không coi mọi FP là background.

**Lời nói:** “Nếu giữ candidate cứu một lá nhưng làm NMS mất một lá khác, cặp không đạt gate. GT đánh giá proposal đã có từ model, không tạo box mới.”

**Nguồn:** `conditional/pairs.py::PairConfig/judge/build_pairs`.

### Slide 16 — Preference bank và cách lấy mẫu huấn luyện

**Thông điệp:** Số cặp không cân bằng, nhưng xác suất lấy mẫu được điều chỉnh theo group/class/target.

**Bố cục:**

- `(0,50; 1,50; 5,70; 3,40)`: bảng bank.
- `(6,55; 1,50; 6,28; 3,40)`: sơ đồ “keep/drop → group → mixture class/target → cap theo ảnh → 2 pair/update”.
- `(0,50; 5,20; 12,33; 0,85)`: dải minh họa batch 4 ảnh: `[uniform 1] [uniform 2] [pair image 1] [pair image 2]`.

| Group | Tổng | DPO train | Probe |
|---|---:|---:|---:|
| Recovery | 167 | 149 | 18 |
| Preservation | 80 | 68 | 12 |
| Background | 468 | 391 | 77 |
| Duplicate | 58 | 54 | 4 |
| Tổng | 773 | 662 | 111 |

**Nhãn phải:** “392 ảnh có pair DPO train”; “Group xấp xỉ 25% sau cap”; “class_mix=0,5”; “Native dùng toàn bộ GT trong cả 4 ảnh; DPO chỉ lấy 2 quyết định”.

**Notes:** 448 ảnh có pair trong toàn bank. Probe tách khỏi DPO theo ảnh nhưng ảnh vẫn có thể vào native uniform, và model nguồn đã học train; không gọi probe là independent holdout. Sampler chốt là sampler cũ cân bằng gần 25%, không thay bằng preservation 35% của pilot sau đó.

**Lời nói:** “Bank background lớn hơn nhiều recovery/preservation. Nếu lấy đều từng record, background sẽ chiếm đa số. Sampler điều chỉnh exposure; điều này chưa chứng minh các group có gradient bằng nhau.”

**Nguồn:** bank manifest; báo cáo Hybrid mục 12; `conditional/bank.py::sampling_probabilities` và trainer chốt.

### Slide 17 — DPO gốc: so chosen/rejected tương đối với reference

**Thông điệp:** Loss thưởng cho việc tăng ưu tiên tương đối của chosen so với rejected.

**Bố cục:**

- `(0,65; 1,60; 12,00; 1,75)`: công thức DPO lớn, chia hai dòng.
- `(0,50; 3,70; 5,95; 1,90)`: định nghĩa x, y_w, y_l, πθ.
- `(6,80; 3,70; 6,03; 1,90)`: định nghĩa πref, β; câu “không cần huấn luyện reward model riêng trong KB2”.
- `(0,65; 5,90; 12,00; 0,40)`: “Bước kế tiếp: thay response bằng action KEEP/DROP đã định nghĩa”.

\[
L_{DPO}=-\mathbb E\log\sigma\!\left[\beta\left(
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}\right)\right]
\]

**Lời nói:** “DPO học từ cặp ưu tiên bằng chênh log-probability so với reference. Trong đề tài, GT tự tạo nhãn ưu tiên cho action phụ keep/drop. Không đưa PPO hay policy-gradient vào update này.”

**Notes:** Nền tảng paper gốc là preference learning cho mô hình ngôn ngữ. Chỉ trình bày cấu trúc objective cần dùng; không nói paper đã chứng minh tăng mAP cho YOLO.

**Nguồn:** [Rafailov et al., DPO, mục 4](https://arxiv.org/html/2305.18290v3#S4); ánh xạ riêng tại `conditional/policy.py`.

### Slide 18 — Rút gọn DPO sang logit của YOLO

**Thông điệp:** Policy Bernoulli làm log-odds bằng đúng class logit.

**Bố cục:**

- `(0,50; 1,50; 8,15; 4,65)`: bốn hàng công thức, hiện từng hàng theo click.
- `(9,00; 1,50; 3,83; 4,65)`: bảng ký hiệu và code đối chiếu 3 dòng.

**Bốn hàng:**

\[
q_\theta(keep)=\sigma(z_\theta),\qquad q_\theta(drop)=\sigma(-z_\theta)
\]
\[
\log q_\theta(keep)-\log q_\theta(drop)=z_\theta
\]
\[
s=\begin{cases}+1&chosen=keep\\-1&chosen=drop\end{cases},\qquad m=s(z_\theta-z_{ref})
\]
\[
L_{DPO}=\frac{1}{B_p}\sum_{r=1}^{B_p}\operatorname{softplus}(-\beta m_r),\quad B_p=2
\]

**Bảng ký hiệu:** zθ: raw logit đúng `(j,c)`; zref: logit TPE cùng ảnh/slot/class; β=0,1: độ nhạy margin; Bp=2: số pair. Code trong hộp nhỏ: `margin=sign*(z-z0)`; `softplus(-beta*margin).mean()`.

**Lời nói:** “Đẳng thức log-odds giúp tính trực tiếp bằng logit, tránh vòng sigmoid rồi log. Đây là loss thật trong code. β=0,1 khác với λ=0,03 là hệ số cộng DPO vào tổng loss.”

**Nguồn:** `conditional/policy.py::pair_loss`. Công thức tương đương BCE có reference xem phụ lục A04.

### Slide 19 — Gradient DPO tác động lên KEEP và DROP ra sao?

**Thông điệp:** Có thể chứng minh hướng gradient cục bộ; metric toàn model vẫn phải đo.

**Bố cục:**

- `(0,65; 1,50; 12,00; 1,00)`: đạo hàm lớn.
- `(0,50; 2,90; 5,95; 2,55)`: thẻ KEEP, z↑, sigmoid(z)↑; dấu gradient âm.
- `(6,80; 2,90; 6,03; 2,55)`: thẻ DROP, z↓, sigmoid(z)↓; dấu gradient dương.
- `(0,65; 5,75; 12,00; 0,55)`: “Ở khởi tạo: zθ=zref ⇒ margin=0 ⇒ loss=ln2≈0,6931”.

\[
\frac{\partial\ell_{DPO}}{\partial z_\theta}
=-\beta s\,\sigma[-\beta s(z_\theta-z_{ref})]
\]

**Ví dụ nhỏ trong thẻ KEEP:** “Minh họa: score 0,20; giả sử Δz=0,4 → score≈0,2716”; gắn nhãn rõ “giả định, không phải update đã đo”. DROP dùng mũi tên định tính, không bịa số run.

**Notes:** Hai thẻ mô tả gradient descent trực tiếp trên z. Trong mạng thật, weights dùng chung, có native gradient, clipping và AdamW nên không bảo đảm mọi candidate KEEP sẽ tăng score sau từng update. Reference không tạo KL penalty tường minh hoặc hard constraint.

**Lời nói:** “Hướng học có cơ sở toán học: keep có lực tăng logit, drop có lực giảm logit. Nhưng từ đây chưa suy ra AP tăng; kết luận AP phải dựa trên bảng validation.”

**Nguồn:** `conditional/policy.py`; `clarification_examples_20261109.json`.

### Slide 20 — Tổng loss: giữ native learning và thêm preference nhẹ

**Thông điệp:** λ điều chỉnh sức can thiệp DPO; hệ số background chỉ áp trên slot không được assign.

**Bố cục:**

- `(0,65; 1,50; 12,00; 0,85)`: `L_total=L_native,bg0.5+λL_DPO`, λ=0,03.
- `(0,50; 2,65; 7,65; 2,90)`: native loss và công thức classification có trọng số.
- `(8,50; 2,65; 4,33; 2,90)`: bảng “β / λ / α_bg” và ý nghĩa.
- `(0,65; 5,85; 12,00; 0,40)`: “Chỉ cv3 nhận gradient cập nhật; box/DFL giữ cấu trúc loss native”.

\[
L_{native,bg}=7,5L_{box}+0,5L_{cls,bg}+1,5L_{DFL}
\]
\[
L_{cls,bg}=\frac{1}{S}\sum_{j,c}w_j\,BCEWithLogits(z_{jc},t_{jc}),\qquad
w_j=\begin{cases}1&j\in foreground\\0,5&j\in background\end{cases}
\]

**Bảng phải:** β=0,1: slope/margin DPO; λ=0,03: hệ số DPO trong tổng loss; α_bg=0,5: trọng số BCE slot background. Chữ nhỏ: “S=max(Σ target_score,1), giữ mẫu số native”.

**Notes:** Công thức là dạng tóm tắt theo gain; batch reduction/class weighting của thư viện được kế thừa. Background là unassigned slot theo TAL. Các zero class-target nằm tại foreground slot vẫn weight 1; không giảm mọi negative class label xuống 0,5. Do backbone/regression frozen, box/DFL không học tọa độ qua cv3. Tuy vậy assignment native vẫn được tính lại và có thể phụ thuộc class score.

**Minh chứng để nói khi được hỏi:** Audit batch đầu có 100 foreground slots và 33.500 background slots; native background classification loss 0,231566→0,115783 khi nhân 0,5, foreground giữ 0,600880. Điều này xác nhận cách áp trọng số; số background slots nhiều hơn không tự chứng minh background gradient chiếm ưu thế. Không gọi phép weighting này là Focal Loss.

**Lời nói:** “Native loss tiếp tục dùng toàn bộ GT của batch để giữ tín hiệu detection. Chỉ phần classification của slot background được nhân 0,5. DPO là một bổ sung có hệ số nhỏ, không thay toàn bộ loss YOLO.”

**Nguồn:** `conditional_bg/loss.py::BackgroundBCE/BackgroundDetectionLoss`; `hybrid_schedule/train.py`.

### Slide 21 — Một update DPO thực hiện theo thứ tự nào?

**Thông điệp:** Policy/reference cùng batch; chỉ policy cv3 được cập nhật.

**Bố cục:**

- `(0,50; 1,50; 8,10; 4,60)`: sơ đồ dọc 6 bước, mỗi bước cao 0,60–0,65.
- `(8,95; 1,50; 3,88; 4,60)`: thẻ optimizer và kiểm tra tính đúng.

**Sáu bước:**

1. Lấy 2 pair theo plan; lấy thêm 2 ảnh uniform train.
2. Policy native forward với GT trên 4 ảnh → native loss; BN running stats frozen.
3. Policy raw forward có autograd → logits và box.
4. Reference raw forward `no_grad` trên cùng batch 4 ảnh → logit gốc.
5. Lấy 2 logit đúng slot/class → DPO loss → cộng total loss.
6. Backward cv3 → clip norm 1 → AdamW step; tới mốc thì evaluate.

**Thẻ phải:** AdamW; LR=1e−6; WD=5e−4; FP32; TF32 off; input 640. Kiểm `reference_unchanged`, `frozen_state_unchanged`, initial margin≈0 và geometry parity.

**Lời nói:** “Reference không phải một cache score tùy batch. Code forward reference bằng đúng ảnh và batch của policy để tránh sai lệch phép so. Forward ở eval mode vẫn có thể giữ autograd cho nhánh policy.”

**Nguồn:** `hybrid_schedule/train.py` vòng train; `conditional/policy.py::raw`; native adapter.

### Slide 22 — Lịch train/eval và chọn Hybrid best500

**Thông điệp:** Best500 được tìm nhờ đánh giá sớm; LR drop sau 1k chưa tác động tới checkpoint này.

**Bố cục:**

- `(0,50; 1,50; 8,10; 3,75)`: đồ thị mAP theo update 0–15.000; thêm inset 0–2.000 để đọc rõ best500.
- `(8,95; 1,50; 3,88; 3,75)`: thẻ schedule/selector.
- `(0,65; 5,60; 12,00; 0,60)`: “Ngân sách 15.000 update; chọn update 500 theo validation AP”.

**Điểm bắt buộc đánh dấu:** step0=38,5097%; step250=38,6834%; **step500=38,7133%**; step1000=38,6314%; step15000=38,5167%.

**Thẻ phải:** Eval step 0; mỗi 250 tới 2k; sau đó mỗi 1k. “LR 1e−6 ở update 1–1000; 1e−7 từ update 1001”. “Cập nhật best nếu mAP > best+1e−4 fraction (=0,01 điểm %)”. Step0 TPE hợp lệ.

**Notes:** Có 22 mốc eval. Pareto và recall gate cũ chỉ diagnostic; selector của run chốt dùng mAP, không chọn theo F1 hay macro recall. Train không dừng ở 500; 500 là checkpoint được giữ sau run đủ ngân sách. Step250 có F1 53,4715%, cao hơn F1 của best500; điều này bình thường vì selector là mAP.

**Lời nói:** “Train lâu hơn không làm metric tăng đều. Mô hình cuối được chọn theo mAP nên best nằm ở 500. Đánh giá dày phát hiện một checkpoint tốt sớm; nó không tự thay đổi gradient huấn luyện.”

**Nguồn:** `assets/hybrid_tpe_dpo_20261109/dpo_validation.csv`; `hybrid_schedule/schedule.py`; `hybrid_schedule/train.py::validate`.

### Slide 23 — Protocol so sánh: cùng ảnh, cùng evaluator, hai nhóm metric

**Thông điệp:** AP và P/R/F1 không được tính tại cùng confidence threshold.

**Bố cục:**

- `(0,50; 1,50; 5,95; 3,70)`: khung “AP/AR”, xanh lam.
- `(6,80; 1,50; 6,03; 3,70)`: khung “Operating P/R/F1”, xanh lục.
- `(0,65; 5,60; 12,00; 0,60)`: “479 ảnh validation / 1.887 GT / 28 class — ordinary deterministic YOLO inference”.

**Khung AP:** confidence thu candidate 0,001; NMS IoU 0,60; max_det 300; mAP50–95 trên IoU 0,50…0,95; AP50, AP75, APsmall/medium/large; AR300 riêng.

**Khung operating:** confidence 0,25; class-aware one-to-one matching IoU 0,50; micro P/R/F1 từ số đếm tổng; macro F1/recall trung bình theo class. Cùng NMS/cap chuẩn.

**Lời nói:** “AP đo chất lượng xếp hạng trên đường precision–recall; recall trong bảng chính là recall ở confidence 0,25. AR300 là average recall kiểu COCO, không phải tên khác của recall 0,25.”

**Nguồn:** `metrics.py`, `evaluate.py`; báo cáo chốt mục 2.

### Slide 24 — Bảng kết quả tổng hợp: đúng bảng người dùng cung cấp

**Thông điệp:** Hybrid có mAP/P/F1 tốt hơn TPE, đi kèm recall giảm.

**Bố cục dựng chính xác:**

- Tiêu đề rút gọn “Kết quả validation: S0 → TPE → Hybrid”.
- Bảng `(0,50; 1,42; 12,33; 4,72)`, 16 hàng gồm header, chiều cao đều khoảng 0,295; body 17–18 pt; header 17 pt đậm.
- Độ rộng 5 cột lần lượt `2,45 / 2,75 / 2,20 / 2,25 / 2,68` inch.
- Header nền `#E9DECA`; body kem, đường ngang mảnh `#C9C0AE`, không kẻ dọc đậm. Tên metric căn trái, số căn phải; dùng decimal alignment nếu có.
- Highlight nền rất nhẹ cột Hybrid; in đậm giá trị tốt nhất của từng hàng theo đúng bảng dưới. Với TP, recall và AR cao hơn là tốt; FP/FN thấp hơn là tốt.
- `(0,65; 6,32; 12,00; 0,43)`: “So TPE: mAP +0,2036 điểm %; FP −87; FN +21”.
- Footer ghi “Tỷ lệ theo %; Δ tỷ lệ là điểm phần trăm; TP/FP/FN là số đếm”.

| Metric | S0 (KB3 trial 0) | Best TPE | Hybrid chốt | Δ so TPE, điểm % / count |
|---|---:|---:|---:|---:|
| **mAP50–95** | 36,0686 | 38,5097 | **38,7133** | **+0,2036** |
| AP50 | 47,7023 | 50,4306 | **50,6287** | +0,1981 |
| AP75 | 41,2404 | 43,4945 | **43,7080** | +0,2135 |
| Precision | 49,0785 | 50,8763 | **52,6208** | +1,7444 |
| Recall | 49,3906 | **55,3789** | 54,2660 | −1,1129 |
| F1 | 49,2340 | 53,0322 | **53,4307** | +0,3985 |
| Macro F1 | 47,4128 | 51,2818 | **51,6693** | +0,3875 |
| Macro recall | 50,6517 | **55,9285** | 55,0529 | −0,8756 |
| AR300 | 55,6950 | **58,2612** | 58,0731 | −0,1881 |
| APsmall | 1,5128 | 4,0997 | **4,1209** | +0,0212 |
| APmedium | 16,4392 | **18,4362** | 18,3008 | −0,1354 |
| APlarge | 40,7118 | 43,8593 | **44,0333** | +0,1741 |
| TP | 932 | **1.045** | 1.024 | −21 GT |
| FP ↓ | 967 | 1.009 | **922** | −87 detection |
| FN ↓ | 955 | **842** | 863 | +21 GT |

**Lời nói:** “Điểm chính là mAP tăng thêm 0,2036 điểm so TPE. Precision và F1 cũng tăng, nhưng recall giảm. So S0 trial 0, hybrid tăng cả P/R/F1, vì đó là mốc so khác.”

**Nguồn:** `assets/kb2_hybrid_selected_20261108/metrics.csv`; báo cáo chốt mục 2. Không lấy S0 37,8218% ở báo cáo Hybrid cũ thay vào cột này.

### Slide 25 — Mức tăng nằm ở giai đoạn nào?

**Thông điệp:** Tách phần chênh HPO và phần continuation KB2; không gán toàn bộ gain cho DPO.

**Bố cục:**

- `(0,50; 1,50; 8,10; 4,45)`: waterfall mAP gồm S0, cộng chênh KB3, cộng chênh KB2, Hybrid. Trục y dùng 0–45% hoặc ghi rõ nếu dùng zoom.
- `(8,95; 1,50; 3,88; 4,45)`: ba KPI xếp dọc.

**Dữ liệu waterfall:** `S0 36,0686 → +2,4410 → TPE 38,5097 → +0,2036 → Hybrid 38,7133`.

**Ba KPI:** “S0→TPE +2,4410 điểm”; “TPE→Hybrid +0,2036 điểm”; “S0→Hybrid +2,6446 điểm”.

**Dòng kết luận:** “Phân rã chênh lệch checkpoint; chưa phải ablation nhân quả từng thành phần”.

**Lời nói:** “Phần lớn chênh số học so anchor nằm ở giai đoạn HPO. KB2 tạo thêm một mức cải thiện nhỏ từ chính checkpoint TPE. Vì native continuation cũng có mặt, chưa quy riêng toàn bộ 0,2036 điểm cho một hạng loss DPO.”

**Nguồn:** `mechanism_evidence.json::stage_gain`; `metrics.csv`.

### Slide 26 — Vì sao precision và F1 tăng dù recall giảm?

**Thông điệp:** 87 FP giảm đủ bù việc mất ròng 21 TP trong công thức P/F1.

**Bố cục:**

- `(0,50; 1,50; 4,35; 4,55)`: bảng đếm hai model và transition GT.
- `(5,20; 1,50; 7,63; 2,10)`: công thức P/R/F1 thay số.
- `(5,20; 3,95; 7,63; 2,10)`: điều kiện đại số và kết quả.

**Bảng trái:** TPE `(TP,FP,FN)=(1045,1009,842)`; Hybrid `(1024,922,863)`; “Cứu 3 GT / mất 24 GT → ròng −21”.

\[
P:\ \frac{1045}{1045+1009}=50,8763\%
\ \to\ \frac{1024}{1024+922}=52,6208\%
\]
\[
R:\ \frac{1045}{1887}=55,3789\%
\ \to\ \frac{1024}{1887}=54,2660\%
\]
\[
F1=\frac{2TP}{2TP+FP+FN}:\quad53,0322\%\to53,4307\%
\]

**Ô dưới phải:** đặt T=1045, F=1009, G=1887, d=21, f=87. `P tăng ⇔ Tf>Fd`: `90.915>21.189`. `F1 tăng ⇔ Tf>d(F+G)`: `90.915>60.816`.

**Lời nói:** “Đây là giải thích chính xác từ số đếm đo được. Nó không phải câu chuyện model cứu nhiều lá hơn TPE: thực tế mất 24, cứu 3. Lợi ích operating point chủ yếu là giảm false positive.”

**Nguồn:** `mechanism_evidence.json::precision_F1_algebra`; GT transitions của checkpoint chốt.

### Slide 27 — Vì sao AP75 tăng khi box không được học lại?

**Thông điệp:** AP phụ thuộc thứ hạng confidence và tập detection, không chỉ tọa độ box.

**Bố cục:**

- `(0,50; 1,50; 5,95; 3,75)`: “Trước” — 1 GT, box A IoU=0,80 score=0,80; box B IoU=0,05 score=0,90. Thứ tự “FP → TP”.
- `(6,80; 1,50; 6,03; 3,75)`: “Sau” — giữ nguyên box; A score=0,80; B score=0,60. Thứ tự “TP → FP”.
- `(0,65; 5,55; 12,00; 0,65)`: “Ví dụ synthetic đã tính evaluator: AP75 0,50→1,00; P/R tại conf 0,25 không đổi”.

**Thiết kế hình:** GT nét đứt nâu; A xanh lục; B cam; bố cục box giống hệt trước/sau. Thêm dải nhãn đầu hai panel “MINH HỌA, KHÔNG PHẢI PREDICTION THẬT”.

**Lời nói:** “Tọa độ không đổi nhưng đẩy false positive xuống sau true positive vẫn có thể nâng AP. Trong model thật, class score còn có thể đổi class argmax, threshold và NMS selection. Ví dụ này chứng minh cơ chế khả thi, không xác định chính box nào tạo gain của run.”

**Notes:** Chỉ scale toàn bộ score bằng cùng hàm tăng nghiêm ngặt mà giữ rank/tập output/NMS/cap thì AP không đổi. Không viết “AP75 tăng chứng minh box khít hơn”.

**Nguồn:** `clarification_examples_20261109.json::toy_AP_example`; báo cáo chốt mục 5; [COCOeval](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

### Slide 28 — Minh chứng run thật phù hợp với cơ chế score/selection

**Thông điệp:** Có kiểm frozen state và có AP tăng; chưa có phân rã nhân quả từng bước hậu xử lý.

**Bố cục:**

- `(0,50; 1,50; 5,95; 3,90)`: ba hàng kiểm kỹ thuật, mỗi hàng có trạng thái và nguồn.
- `(6,80; 1,50; 6,03; 3,90)`: ba KPI và một chú thích IoU.
- `(0,65; 5,75; 12,00; 0,45)`: “Kết luận phù hợp: classification/selection thay đổi trong khi raw geometry giữ nguyên”.

**Bên trái:** “Reference state không đổi”; “Frozen backbone/regression/BN stats không đổi”; “Raw-box drift=0 trên các kiểm tra train/probe đã báo cáo”. Ghi đúng phạm vi probe, không viết đã exhaustively kiểm mọi raw slot mọi ảnh.

**Bên phải:** AP50 +0,1981 điểm; AP75 +0,2135 điểm; mAP +0,2036 điểm. Chú thích “Matched mean IoU: 0,870426→0,869820; tập match thay đổi, không phải đo box trên cùng GT”.

**Lời nói:** “Kiểm tra state xác nhận model không học lại hình học. AP vẫn tăng là quan sát phù hợp với cơ chế score và selection. Hiện chưa biết riêng đổi thứ hạng, đổi class hay NMS đóng góp bao nhiêu.”

**Nguồn:** báo cáo chốt mục 1 và 5; frozen/probe audits trong review/manifest đã tổng hợp.

### Slide 29 — Cải thiện theo class và hạn chế theo kích thước

**Thông điệp:** Gain AP trải trên nhiều class nhưng vẫn có class và size giảm.

**Bố cục:**

- `(0,50; 1,50; 8,10; 4,55)`: diverging horizontal bar chart, chọn 5 class tăng và 2 class giảm như dưới; trục “Δ AP50–95 so TPE, điểm %”, vạch 0 rõ.
- `(8,95; 1,50; 3,88; 4,55)`: thẻ tổng kết “20/28 tăng; 8/28 giảm” và bảng size nhỏ.

**Các thanh:** Tomato Septoria leaf spot +0,9838; Tomato mold leaf +0,9230; Corn leaf blight +0,7785; grape leaf black rot +0,7654; Squash Powdery mildew leaf +0,6510; Tomato leaf yellow virus −0,3281; Tomato leaf mosaic virus −1,0552.

**Bảng size bên phải:** Small 4,0997→4,1209%; Medium 18,4362→18,3008%; Large 43,8593→44,0333%. Chú thích “small: 94 GT trên ảnh transform; còn khó”.

**Notes:** Macro mAP nhận đóng góp `ΔAP_class/28`, không nhân GT support. So S0 trial 0: 23 class tăng/5 giảm, nhưng không dùng thống kê này thay cho 20/28 của riêng giai đoạn KB2. Tomato mold là ví dụ AP tăng dù TP giảm 5; Tomato mosaic có FP giảm 2 nhưng AP giảm, cho thấy FP tại một threshold không quyết định AP.

**Lời nói:** “20 class tăng AP hỗ trợ nhận xét lợi ích không chỉ nằm ở một class. Tuy nhiên APsmall còn thấp, APmedium giảm và một số bệnh giảm AP; vì vậy không kết luận mọi nhóm đều tốt lên.”

**Nguồn:** `class_contributions.csv` lọc `baseline=TPE`; `metrics.csv`; báo cáo chốt mục 9.

### Slide 30 — Vì sao chọn λ=0,03 và checkpoint500? Minh chứng hiệu chỉnh

**Thông điệp:** Phân biệt tác dụng λ trên gradient, so sánh cùng lịch eval, và việc phát hiện best sớm.

**Bố cục:**

- `(0,50; 1,50; 6,00; 2,10)`: bảng gradient batch đầu.
- `(6,85; 1,50; 5,98; 2,10)`: bảng so λ cùng lịch eval1k.
- `(0,50; 4,00; 12,33; 2,00)`: timeline “best500 → LR drop sau1000”; ghi phần gain từ dense eval.

**Bảng trái:**

| Batch đầu | λ=0,1 | λ=0,03 |
|---|---:|---:|
| Native grad norm | 3,825481 | 3,825481 |
| Weighted DPO grad norm | 0,276375 | 0,082913 |
| DPO/native | 7,2246% | 2,1674% |

**Bảng phải:** eval1k, λ0,1 best 38,5962%; eval1k, λ0,03 best 38,6314%; chênh +0,0352 điểm. Dưới bảng “Cùng bank/image-pair stream trong phép so đã kiểm”.

**Timeline dưới:** Best ở 500 trước LR drop 1k. Dense eval tìm 38,7133% thay vì selected-best 38,6314% trên grid1k. Chú thích “Không có checkpoint500 của λ0,1 trên cùng grid dày để tách trọn hiệu ứng λ”.

**Lời nói:** “Giảm λ từ 0,1 xuống 0,03 làm riêng weighted DPO gradient còn 30% ở cùng batch; không có nghĩa toàn bộ update còn 30%. Phần tăng do chọn best500 không được giải thích bằng LR drop xảy ra sau đó.”

**Nguồn:** `mechanism_evidence.json::same_initial_batch_gradient`; `metrics.csv`; báo cáo chốt mục 6 và 8.

### Slide 31 — Đối chứng bổ trợ: điều gì đã được kiểm, điều gì còn thiếu?

**Thông điệp:** Đối chứng lịch sử hỗ trợ DPO trong recipe cũ; chưa thay thế native-only từ chính TPE.

**Bố cục:**

- `(0,50; 1,50; 7,40; 3,90)`: bảng đối chứng lịch sử 3 seed, nhãn “S0 KB1 lịch sử; recipe khác Hybrid chốt”.
- `(8,25; 1,50; 4,58; 3,90)`: hai ô “Có bằng chứng” và “Cần kiểm tiếp”.
- `(0,65; 5,75; 12,00; 0,45)`: “Không dùng 3 seed cũ để gọi Hybrid hiện tại là đã xác nhận 3 seed”.

**Bảng trái:**

| Phép so lịch sử | Native-only | DPO | Δ DPO−native |
|---|---:|---:|---:|
| Best mAP, mean±SD qua 3 seed | 38,1196±0,0608% | 38,1707±0,0429% | +0,0511±0,0202 điểm |
| Cùng update 15.000 | 38,0595±0,0430% | 38,1670±0,0475% | +0,1075±0,0337 điểm |

**Có bằng chứng:** AP cao hơn native ở 3/3 seed trong recipe S0-start; pair/image stream ghép cặp. **Cần kiểm tiếp:** native-only từ TPE trial2, cùng mask/bank/budget/schedule/selector; nhiều seed hybrid; dữ liệu xác nhận mới.

**Notes:** ± là SD mô tả, không phải confidence interval. Không có kiểm định ý nghĩa thống kê. Recipe lịch sử dùng λ0,1 và khởi tạo từ checkpoint KB1, không phải S0 trial0 của bảng chính. Nếu thời gian ngắn, chuyển slide này xuống phụ lục, vẫn giữ giới hạn ở kết luận.

**Lời nói:** “Thí nghiệm lịch sử cho thấy DPO có lợi thế AP so native-only trong recipe đã thử. Nhưng hybrid hiện tại chưa có arm native-only từ cùng TPE, nên chưa tách được phần đóng góp nhân quả riêng của DPO.”

**Nguồn:** `Report_KB2_DPO_20261103.md` mục 10, 13; `historical_S0_controls.csv`, `historical_paired_deltas.csv`.

### Slide 32 — Kết luận: kết quả đạt được và bước xác nhận tiếp theo

**Thông điệp:** Hybrid cải thiện validation trong run chốt, có đánh đổi và giới hạn xác nhận rõ ràng.

**Bố cục:**

- `(0,50; 1,50; 12,33; 1,15)`: dải lớn “YOLOv8n + KB3-TPE → KB2-DPO / mAP50–95 = 38,7133%”.
- `(0,50; 3,00; 5,95; 2,40)`: ba kết quả đã đạt.
- `(6,80; 3,00; 6,03; 2,40)`: ba bước tiếp theo.
- `(0,65; 5,85; 12,00; 0,55)`: “+2,6446 điểm so S0 trial0; +0,2036 điểm so TPE; precision↑ / recall↓ so TPE”.

**Kết quả:** tích hợp pipeline HPO→DPO có artifact kiểm tra; AP/P/F1 tăng so TPE, FP giảm 87; chỉ fine-tune classification, inference một detector.

**Bước tiếp theo:** native-only continuation từ cùng TPE; khóa recipe và kiểm nhiều seed/dữ liệu mới; xử lý recall ở ảnh đông đối tượng và object nhỏ.

**Lời kết đề xuất:** “Trên validation phát triển, Hybrid đạt mAP 38,7133%, tăng 0,2036 điểm so TPE và 2,6446 điểm so S0 trial 0. Cơ chế keep/drop và score selection phù hợp với kết quả, nhưng chưa thay thế ablation để quy riêng mức tăng cho DPO. Bước tiếp theo là kiểm native-only từ cùng TPE và xác nhận trên nhiều seed, dữ liệu mới.”

**Nguồn:** báo cáo chốt mục 10; bảng chính và phạm vi xác nhận đã nêu.

## 5. Phụ lục để trả lời hội đồng

### A01 — YOLOv8n giải mã tọa độ và score như thế nào?

**Mục đích:** Trả lời câu hỏi về 8.400 slot và vì sao không có objectness riêng.

**Bố cục:** trái `(0,50; 1,50; 7,65; 4,70)` là một feature grid và một slot; phải `(8,50; 1,50; 4,33; 4,70)` là công thức DFL integral và score. Footer theo master.

**Nội dung:** ba scale 80²+40²+20²=8.400 slot; mỗi slot 64 regression logits và 28 class logits. Với một cạnh `p_k=softmax(r)_k`, `d=Σ_{k=0}^{15}k p_k`. Decode bốn khoảng cách tới grid point thành box. Classification `p_jc=σ(z_jc)`, label=argmax class, score=max class. Trong KB2, regression và features trước cv3 frozen.

**Lời nói:** “DFL integral là kỳ vọng phân phối khoảng cách, không lấy mẫu box ngẫu nhiên. Action keep/drop không phải lấy mẫu các bin DFL.”

**Nguồn:** audit model, `adapter.py`; báo cáo Hybrid mục 3.

### A02 — TPE: phân biệt γ lý thuyết và số điểm good trong Optuna

**Bố cục:** trái `(0,50; 1,50; 6,00; 4,70)` bảng history n và số good; phải `(6,85; 1,50; 5,98; 4,70)` công thức EI và chu trình candidate.

**Bảng:** n=2,3,4,5,6,7 → `n_good=min(ceil(0,1n),25)=1`. Lưu ý có prior/kernel heuristic trong estimator; không vẽ một Gaussian đơn “chính xác” từ đúng 1 điểm rồi coi đó là estimator thật. Run dùng independent sampling, không tối ưu thêm bandwidth trong đề tài.

**Công thức:** Bayes `p(y|h)=p(h|y)p(y)/p(h)`; thay mô hình hai vùng vào EI để ra tỷ lệ nghịch với `γ+(1−γ)g/ℓ`. Nếu minh họa bằng curve thì gắn nhãn synthetic.

**Lời nói:** “Search nhỏ có ít thông tin để ước lượng mật độ. Nó vẫn là TPE từ trial 2 nhưng chưa đủ cơ sở nói đã học chắc chắn một vùng tối ưu.”

**Nguồn:** source Optuna 4.5.0; code config/propose trong repo.

### A03 — SGD, LR schedule và native loss trong HPO

**Bố cục:** trái `(0,50; 1,50; 6,00; 4,70)` native loss/assignment; phải `(6,85; 1,50; 5,98; 4,70)` schedule/EMA và bảng recipe.

**Nội dung:** `L_native=7,5L_box+0,5L_cls+1,5L_DFL`; class BCE; box CIoU; DFL nội suy giữa hai bin của target distance. Native TAL dùng class score và IoU để gán soft target; không phải matching metric sau NMS. SGD Nesterov momentum0,937; batch16/nbs64; warmup3; linear LR factor `q(e)=max(1−e/200,0)×0,99+0,01`. Trial early stop ở169 vẫn dùng horizon200. EMA phục vụ native val/checkpoint.

**Lời nói:** “TPE chọn base LR. Learning rate thực vẫn thay đổi theo warmup và schedule; không gọi LR từng optimizer step là HP mới được TPE chọn.”

**Nguồn:** `pretrained_detector.py`, recipe manifest; báo cáo Hybrid mục 6.

### A04 — DPO keep/drop cũng là BCE có tâm theo reference

**Bố cục:** công thức trên `(0,65; 1,50; 12,00; 1,70)`; bảng bên trái `(0,50; 3,60; 5,95; 2,45)`; giải thích bên phải `(6,80; 3,60; 6,03; 2,45)`.

\[
L_{DPO}=BCEWithLogits\big(\beta(z_\theta-z_{ref}),a_w\big),\qquad a_w\in\{0,1\}
\]

**Bảng:** keep → aw=1, s=+1; drop → aw=0, s=−1. Lúc đầu zθ=zref nên probability preference=0,5 và loss=ln2.

**Giải thích:** BCE này nằm trên relative logit, khác native class BCE theo GT soft targets. Không có KL penalty tường minh trong objective code. Chưa có ablation hard-example BCE phù hợp để khẳng định reference-centered objective vượt mọi BCE khác.

**Lời nói:** “DPO nhị phân có dạng BCE tương đương về đại số. Giá trị của thiết kế cần xét cả cách tạo preference, reference và thí nghiệm, không chỉ tên của loss.”

**Nguồn:** `conditional/policy.py`; báo cáo chốt mục 3.

### A05 — Hai ví dụ gate: nhận và loại cặp

**Bố cục:** hai cột `(0,50; 1,50; 5,95; 4,60)` và `(6,80; 1,50; 6,03; 4,60)`; mỗi cột gồm hai set ID, FP và kết luận.

**Ví dụ nhận:** rejected match `{g1,g2,g3}`, FP=2; chosen match `{g1,g2,g3,g4}`, FP=2; target g4 → KEEP hợp lệ.

**Ví dụ loại:** rejected `{g1,g2,g3}`, FP=2; chosen `{g1,g2,g4}`, FP=2 → không hợp lệ dù tổng TP bằng nhau, vì mất g3. Có thể thêm dòng drop: FP 3→2 nhưng mất một GT → vẫn loại.

**Nhãn:** “Ví dụ tập hợp minh họa; không phải thống kê train/validation”.

**Lời nói:** “So GT identity tránh trường hợp cứu một GT nhưng làm mất GT khác mà tổng TP không thay đổi.”

**Nguồn:** `conditional/pairs.py::judge`.

### A06 — Công thức sampler và giới hạn xác suất theo ảnh

**Bố cục:** trên `(0,65; 1,50; 12,00; 1,65)` công thức; dưới trái `(0,50; 3,50; 6,00; 2,55)` bảng ký hiệu; dưới phải `(6,85; 3,50; 5,98; 2,55)` quy tắc cap/probe.

\[
p_r^{raw}=\frac{1}{|S|\,|G_s|}\left[(1-\alpha_c)\frac{1}{n_k}
+\alpha_c\frac{1}{C_kU_{kc}m_u}\right],\qquad\alpha_c=0,5
\]

**Ký hiệu:** S: các sign hiện có; Gs: group trong sign; nk: số pair group k; Ck: số class của group; Ukc: target units class c; mu: số pair cùng unit. Unit là `(image,GT ID)`, hoặc `(image,slot)` khi không có target GT.

Sau đó water-fill/cap tổng xác suất của mỗi ảnh: `p(image i)≤5/392≈0,0127551` trong bank train hiện tại. Probe tách theo ảnh khỏi DPO; native uniform vẫn có thể gặp ảnh đó.

**Lời nói:** “Đây là cân bằng exposure theo xác suất, không phải cân bằng gradient hoặc bảo đảm AP của mọi class tăng.”

**Nguồn:** báo cáo Hybrid mục 12.2; `conditional/bank.py::sampling_probabilities`.

### A07 — Sổ bằng chứng: nhận định nào được hỗ trợ tới đâu?

**Bố cục:** bảng full-width `(0,50; 1,50; 12,33; 4,80)`, 4 cột “Nhận định / Chứng cứ / Mức kết luận / Còn thiếu”. Body 17–18 pt.

| Nhận định | Chứng cứ | Mức kết luận | Còn thiếu |
|---|---|---|---|
| TPE winner tốt hơn anchor | 8 trial, winner canonical AP | Tốt nhất các trial đã thử | HPO khác cùng budget, nhiều seed |
| KEEP/DROP có hướng gradient | Đại số + `pair_loss` | Đúng tại logit cục bộ | Bảo đảm AP toàn model |
| Hybrid tăng mAP so TPE | Cùng evaluator, checkpoint reload | +0,2036 điểm validation | Native-only từ TPE, dữ liệu mới |
| P/F1 tăng, recall giảm | TP/FP/FN và công thức | Giải thích chính xác metric | Ổn định qua run khác |
| Geometry frozen | Digest + raw-box probes | Phù hợp cơ chế score/selection | Counterfactual score/class/NMS |
| λ0,03 hợp lý trong lượt thử | Gradient batch đầu, so grid1k | Hiệu chỉnh thực nghiệm | Sweep/ablation đầy đủ |

**Lời nói:** “Cơ chế, quan sát metric và bằng chứng nhân quả là ba mức khác nhau. Đề tài báo cáo đúng mức mà artifact hiện tại hỗ trợ.”

**Nguồn:** báo cáo chốt và code đã liệt kê.

### A08 — Câu hỏi hội đồng và câu trả lời ngắn

**Bố cục:** 4 thẻ `(0,50; 1,50; 5,95; 2,05)`, `(6,80; 1,50; 6,03; 2,05)`, `(0,50; 3,95; 5,95; 2,05)`, `(6,80; 3,95; 6,03; 2,05)`; câu hỏi 21 pt đậm, câu trả lời 18–19 pt. Các câu còn lại nằm trong speaker notes.

**Bốn thẻ:**

- “TPE đổi HP mỗi epoch?” → “Sau mỗi trial hoàn tất; scheduler native vẫn chạy trong trial.”
- “DPO có cần GT khi dùng model?” → “GT dùng để tạo preference và native training; inference chỉ nhận ảnh.”
- “AP75 tăng nghĩa là box khít hơn?” → “Chưa; raw geometry frozen, score/ranking/selection có thể làm AP đổi.”
- “Gain 0,2036 điểm hoàn toàn do DPO loss?” → “Đây là gain pipeline KB2; cần native-only từ cùng TPE để tách riêng.”

**Speaker notes thêm:**

- “Vì sao không lấy last?” → Selector khóa theo validation mAP, best500 cao hơn last; không chọn theo cảm quan.
- “Vì sao β khác λ?” → β ở margin DPO; λ nhân DPO khi cộng total loss.
- “Vì sao recall giảm vẫn nói tốt hơn?” → Tốt hơn theo mục tiêu mAP/P/F1; báo rõ recall giảm, không gọi tốt hơn trên mọi metric.
- “Có phải hai model khi inference?” → Chỉ model cuối; reference/judge thuộc training.
- “Tại sao chọn YOLOv8n?” → Detector nền kích thước nhỏ, artifact/recipe thống nhất; chưa dùng kết quả này để chứng minh thắng mọi detector hoặc FPS cụ thể.
- “Đã test độc lập chưa?” → Kết quả này dùng validation phát triển; chưa có xác nhận mới độc lập cho hybrid.

## 6. Danh mục hình, bảng cần dựng

Các mục dưới là chỉ dẫn dựng hình trong PowerPoint hoặc bằng công cụ vẽ/plot chuẩn; không phải ảnh thực nghiệm mới đã được sinh.

| Mã | Dùng ở slide | Cách dựng và dữ liệu |
|---|---|---|
| F01 | 01 | Sơ đồ vector Hybrid ba giai đoạn; shapes và connector PowerPoint |
| F02 | 02 | Sơ đồ YOLO có 3 scale và hai nhánh cv2/cv3; lấy số từ audit |
| F03 | 04 | Sơ đồ vòng ngoài TPE/vòng trong detector; feedback metric nét đứt |
| F04 | 06–07 | Density curves minh họa; ghi synthetic, không giả là trace run |
| F05 | 08 | Timeline Train–Val–Select–Eval–History |
| F06 | 10 | Cột trial; `tpe_trials.csv`; giữ score lịch sử |
| F07 | 12 | Sơ đồ DPO offline/online, hai loss và gradient |
| F08 | 14 | Candidate active-bit, hai render A/B và GT judge |
| F09 | 22 | Curve mAP/update; `dpo_validation.csv`; inset early 0–2k |
| T01 | 24 | Bảng kết quả dựng editable, dữ liệu `metrics.csv` |
| F10 | 25 | Waterfall stage gain; `mechanism_evidence.json::stage_gain` |
| F11 | 27 | Box/ranking synthetic; `clarification_examples_20261109.json` |
| F12 | 29 | Diverging bar; `class_contributions.csv`, chỉ `baseline=TPE` |
| T02 | 30 | Batch1 gradient + λ cùng grid1k; evidence JSON/CSV |
| T03 | 31 | Đối chứng lịch sử, tách nhãn source/recipe với bảng chính |

Nếu muốn bổ sung ảnh prediction thật, ưu tiên một ảnh TP giữ được, một FP bị giảm, một trường hợp mất GT; cần đối chiếu output và GT thật trước khi đưa lên. Hiện các CSV tổng hợp và ví dụ synthetic không đủ để tự gán một ảnh bất kỳ là “case thật giúp tăng AP”. Plan không yêu cầu chạy lại model để hoàn thành nội dung hiện có.

## 7. Nguồn đối chiếu khi dựng PowerPoint

| Nội dung | Nguồn local |
|---|---|
| Kết quả chốt, scope, giải thích | [KB2_HYBRID_SELECTED_EXPLANATION_20261108.md](KB2_HYBRID_SELECTED_EXPLANATION_20261108.md) |
| Bảng kết quả đúng S0 trial0 | [metrics.csv](assets/kb2_hybrid_selected_20261108/metrics.csv) |
| Checkpoint/config chốt | [selection.json](assets/kb2_hybrid_selected_20261108/selection.json) |
| Stage gain, gradient, metric algebra | [mechanism_evidence.json](assets/kb2_hybrid_selected_20261108/mechanism_evidence.json) |
| AP theo class, đóng góp mAP | [class_contributions.csv](assets/kb2_hybrid_selected_20261108/class_contributions.csv) |
| Ví dụ score/ranking đã phân nhãn synthetic | [clarification_examples_20261109.json](assets/kb2_hybrid_selected_20261108/clarification_examples_20261109.json) |
| Kiến trúc, TPE, native training, DPO | [Report_Hybrid_KB3_TPE_KB2_DPO_20261109.md](Report_Hybrid_KB3_TPE_KB2_DPO_20261109.md), không chép S0 cũ |
| 8 trial lịch sử | [tpe_trials.csv](assets/hybrid_tpe_dpo_20261109/tpe_trials.csv) |
| Curve DPO chốt | [dpo_validation.csv](assets/hybrid_tpe_dpo_20261109/dpo_validation.csv) |
| TPE proposal/history/selection | [tpe_pipeline.py](../../kb3_hyperparameter_optimization/quality/tpe_pipeline.py) |
| Recipe và native-best EMA | [pretrained_detector.py](../../kb3_hyperparameter_optimization/quality/pretrained_detector.py) |
| TPE config mặc định; run thực là 8 trial | [kb3_tpe.yaml](../../kb3_hyperparameter_optimization/configs/kb3_tpe.yaml) |
| Transfer winner | [hybrid/source.py](../../kb2_preference_optimization/hybrid/source.py) |
| Preference source/bank | [hybrid/bank.py](../../kb2_preference_optimization/hybrid/bank.py) |
| A/B renderer, proposal và GT gate | [conditional/pairs.py](../../kb2_preference_optimization/conditional/pairs.py) |
| Sampler sign/group/class/target và cap ảnh | [conditional/bank.py](../../kb2_preference_optimization/conditional/bank.py) |
| Trainable mask và Bernoulli DPO | [conditional/policy.py](../../kb2_preference_optimization/conditional/policy.py) |
| Native foreground/background BCE | [conditional_bg/loss.py](../../kb2_preference_optimization/conditional_bg/loss.py) |
| Train/eval/selection run chốt | [hybrid_schedule/train.py](../../kb2_preference_optimization/hybrid_schedule/train.py) |
| LR/eval schedule | [hybrid_schedule/schedule.py](../../kb2_preference_optimization/hybrid_schedule/schedule.py) |
| Evaluator và operating metrics | [metrics.py](../../kb2_preference_optimization/metrics.py), [evaluate.py](../../kb2_preference_optimization/evaluate.py) |
| Native-only/DPO lịch sử | [Report_KB2_DPO_20261103.md](Report_KB2_DPO_20261103.md), [paired deltas](assets/hybrid_tpe_dpo_20261109/historical_paired_deltas.csv) |

Nền tảng phương pháp: [TPE — Bergstra et al. 2011](https://papers.nips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html), [Optuna 4.5.0 source](https://optuna.readthedocs.io/en/v4.5.0/_modules/optuna/samplers/_tpe/sampler.html), [DPO — Rafailov et al.](https://arxiv.org/html/2305.18290v3#S4), [COCOeval source](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py). Các nguồn này cung cấp cơ sở thuật toán/metric, không chứng minh trực tiếp mức tăng của run đề tài.

## 8. Kiểm tra trước khi nộp slide

1. Mọi cột S0 của phép so chính đều là trial0; mAP=36,0686%, TP=932, FP=967, FN=955.
2. Chênh toàn pipeline là +2,6446 điểm so S0 trial0; chênh KB2 so TPE là +0,2036 điểm; không dùng gain S0 lịch sử +0,8915 điểm cho bảng mới.
3. TPE tìm LR/WD giữa trial; không mô tả PPO, điều chỉnh HP từng epoch hoặc nối weights giữa trial.
4. Phân biệt native epoch selection, canonical trial ranking và KB2 checkpoint selection.
5. β=0,1; λ=0,03; background=0,5; ba hệ số khác vai trò.
6. Best500 là checkpoint của run ngân sách15k; LR giảm từ update1001 chưa tác động best500.
7. AP confidence0,001 khác operating confidence0,25; AR300 khác recall operating.
8. Ghi cả recall giảm, FN tăng so TPE; không đánh dấu mọi số Hybrid là tốt nhất.
9. Ví dụ AP75/score được ghi synthetic; số đo class/gradient được ghi thực nghiệm.
10. Không dùng đối chứng 3 seed S0-start để gọi hybrid đã có 3 seed hay để quy nhân quả toàn bộ gain cho DPO.
11. Không ghi FPS/latency chưa đo hoặc gọi test hiện tại là holdout mới độc lập.
12. Footer nguồn đọc được; bảng editable; equation không bị cắt; thử trình chiếu 16:9 trên màn hình và xuất PDF để kiểm font tiếng Việt.


## 9. Bổ sung cuối deck — đối chứng DPO ngày16/11/2026

Mục tiêu: trả lời “DPO có tạo lợi ích thêm so với native continuation không?”.
Thêm4 slide sau27 slide hiện có (vị trí28–31, nhãn27–30).

| Slide | Thông điệp | Nội dung / lời nói trọng tâm |
|---|---|---|
| 28 | Thiết kế can thiệp ghép cặp | 6 run mới; seed42/43/44; λ0/0,03; cùng nguồn/stream/LR/eval; chính step500, bổ sung best/last |
| 29 | Lợi ích AP nhỏ ở3 seed | Bảng số liệu gốc; mean+0,021936 ±SD0,026541 điểm; TP/FN và danh tính GT bằng nhau; FP giảm1 ở seed43 |
| 30 | Tách gain native và DPO | TPE→native:+0,158948; native→DPO:+0,021936 điểm tại500 trung bình; best seed42 chỉ hơn0,028133 |
| 31 | Kết luận có điều kiện và kế hoạch | Best mean+0,014128; last mean−0,012596; chưa holdout độc lập. Khóa recipe, holdout mới, thêm seed/TPE init và đối chứng preference |

Nguồn: [báo cáo đối chứng](HYBRID_DPO_CAUSAL_CONTROL_20261116.md),
[paired_results.json](assets/hybrid_paired_control_20261116/paired_results.json).
Bước tiếp là đề xuất nghiên cứu, không trình bày như thí nghiệm đã hoàn tất.
Kiểm tra: số liệu khớp JSON; không gọi SD là CI; không quy toàn bộ hybrid−TPE
cho DPO; giữ kết quả bất lợi tại15000; ghi chú thuyết trình và nguồn trên slide.


## 10. Chỉnh slide DPO và thực nghiệm λ

Các số trong yêu cầu áp dụng theo chân slide của bản 31 trang. Bỏ 09 và 23, sau đó đánh số lại. Thông số evaluator chuyển vào ghi chú của bảng kết quả.

| Số cũ | Số mới | Nội dung |
|---|---|---|
| 11 | 10 | Từ một checkpoint TPE tạo policy được huấn luyện và reference đóng băng |
| 12 | 11 | Bank chuẩn bị trước, native loss trên 4 ảnh, DPO loss trên 2 quyết định KEEP / DROP |
| 15 | 14 | Bốn tình huống KEEP / DROP và ví dụ cùng TP nhưng mất một GT |
| 16 | 15 | Tách 662 cặp train và 111 probe, cân bằng nhóm và tạo batch 4 ảnh |
| 25 | 23 | So λ=0/0,03/0,1 khi LR cố định, và λ=0/0,03/1 khi giảm LR và eval dày. Báo cả checkpoint tốt nhất và điểm ở step 15.000 |

Nguồn thực nghiệm λ gồm [review 06/11](KB2_HYBRID_LAMBDA003_REVIEW_20261106.md), [review 13/11](KB2_HYBRID_LAMBDA1_REVIEW_20261113.md) và [đối chứng 16/11](HYBRID_DPO_CAUSAL_CONTROL_20261116.md).
