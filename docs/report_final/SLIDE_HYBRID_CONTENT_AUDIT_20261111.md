# Đối chiếu nội dung slide Hybrid HPO–DPO và bản rút gọn

Ngày: 11/11/2026. Phạm vi theo yêu cầu: slide nhãn cũ 04, 06, 07, 09, 11, 12, 14, 17, 18, 19, 20; bỏ 25–29, 31 và toàn bộ phụ lục từ sau slide 32.

## 1. Kết quả chính

Các công thức TPE, DPO Bernoulli, đạo hàm một cặp và tổng loss có cơ sở đúng theo implementation. Những phần cần sửa chủ yếu là thời điểm của history, cách gọi vai trò checkpoint TPE và sơ đồ luồng loss. Bản slide đã cập nhật để các điểm này xuất hiện trực tiếp trên trang, không chỉ nằm trong Notes.

Hai câu hỏi trọng tâm:

- **Trial 0 có hoàn tất.** `history=[]` là trạng thái **trước khi đề xuất trial0**. Sau khi train/evaluate trial0, code append kết quả vào history và dùng nó khi đề xuất trial1.
- **Best TPE đúng là đầu vào của KB2.** Cụ thể, KB2 nhận checkpoint rồi nạp thành hai bản YOLO cùng weights ban đầu: policy để học tiếp và reference đóng băng làm mốc tính DPO. Điều đó chính là cách thực thi giai đoạn KB2 trong pipeline Hybrid.

Không phát sinh train hoặc model inference. Có kiểm số nhỏ trên loss thật được trích từ code, đọc trial specs/results và manifest đã lưu.

## 2. Quy ước số slide và các trang đã bỏ

Các câu hỏi về trial0 và checkpoint khớp nhãn `09`/`11` ở chân slide, không khớp thứ tự trang PowerPoint khi đếm bìa là trang1. Vì vậy yêu cầu xóa được thực hiện theo **nhãn bản cũ**:

| Nhãn cũ | Xử lý | Nhãn mới |
|---|---|---|
| 00–24 | Giữ; sửa các trang được yêu cầu đối chiếu | 00–24 |
| 25–29 | Bỏ | — |
| 30 | Giữ phần hiệu chỉnh λ và eval sớm | 25 |
| 31 | Bỏ | — |
| 32 | Giữ kết luận | 26 |
| A01–A08, phần sau 32 | Bỏ toàn bộ | — |

**Bản mới: 27 trang**, gồm bìa00 và nội dung01–26, không còn phụ lục. Bảng kết quả cũ24 vẫn được giữ. PPTX/PDF/HTML, Notes, custom show, ảnh tổng quan và các preview đều đã đồng bộ; preview của trang bị bỏ được dọn khỏi thư mục xuất.

## 3. Đối chiếu từng slide

### Slide 04 — Vòng ngoài HPO và vòng trong train YOLO

**Kết luận:** đúng về kiến trúc hai vòng; sửa chữ để nêu rõ lúc cập nhật history.

Luồng thực tế:

1. `propose(number, history)` nhận những trial trước đã hoàn tất.
2. Trial0 enqueue anchor, trial1 startup random; từ trial2 mới dùng TPE với history đủ2.
3. Cấu hình của trial được giữ, detector bắt đầu từ pretrained nguồn; native scheduler vẫn chạy trong trial.
4. Native validation chọn best EMA epoch.
5. Canonical evaluator chấm selected-best và last; score selected-best là objective trả cho TPE.
6. Sau khi kiểm result hợp lệ mới `history.append(...)`.

Đã sửa khung bên trái và khung canonical trên slide để nêu đúng các bước này. Không có điều chỉnh HP sau từng epoch bằng TPE.

Nguồn: [propose/search](../../kb3_hyperparameter_optimization/quality/tpe_pipeline.py:187), [append history](../../kb3_hyperparameter_optimization/quality/tpe_pipeline.py:244), [native-best và canonical eval](../../kb3_hyperparameter_optimization/quality/pretrained_detector.py:134).

### Slide 06 — Hai density và 24 candidates

**Kết luận:** trực giác ℓ/g đúng; đã làm rõ đó là một chiều minh họa.

Run gọi `TPESampler` với startup2, candidates24 và không bật multivariate. Với default independent sampling, estimator được dùng cho từng tham số; 24 là ứng viên nội bộ cho một tham số, không phải24 detector được train. Hình curve trên slide là hình giải thích theo `log(lr0)`, không phải density joint LR–WD được export từ run.

Đã thay thẻ “đề xuất HP” bằng “fit từng tham số → sinh24 ứng viên cho mỗi tham số → chấm logℓ−logg”.

Nguồn: [TPE sampler trong repo](../../kb3_hyperparameter_optimization/quality/tpe_pipeline.py:187), [Optuna 4.5.0 source](https://optuna.readthedocs.io/en/v4.5.0/_modules/optuna/samplers/_tpe/sampler.html).

### Slide 07 — Công thức TPE và EI

**Kết luận:** công thức đúng theo mô hình hai vùng của TPE. Đã phân biệt bài toán maximize của code với dạng minimization dùng trong phép suy diễn.

Đặt `F(h)` là canonical mAP của native-selected checkpoint, và `y=−F(h)`:

\[
\ell(h)=p(h\mid y<y^*),\quad g(h)=p(h\mid y\ge y^*),
\quad p(h)=\gamma\ell(h)+(1-\gamma)g(h).
\]
\[
EI(h)=\int_{-\infty}^{y^*}(y^*-y)p(y\mid h)dy
\propto\left[\gamma+(1-\gamma)\frac{g(h)}{\ell(h)}\right]^{-1}.
\]

Code thực tế dùng `direction='maximize'` cho F. Việc đổi dấu chỉ để trình bày EI ở dạng minimization, không phải repo có một objective khác.

γ lý thuyết là khối lượng nhóm tốt. `default_gamma(n)` của Optuna là **hàm trả số observations tốt**, `min(ceil(0,1n),25)`. Trong các proposal TPE của run8 trial, n=2…7 nên good count=1. Không ghi γ=0,1 cố định hoặc “20% good”. Đây là acquisition heuristic trên mật độ ước lượng; không bảo đảm trial tiếp theo tăng AP.

Nguồn: [objective/proposal](../../kb3_hyperparameter_optimization/quality/tpe_pipeline.py:189), [Bergstra et al., 2011, TPE](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf), [Optuna 4.5.0](https://optuna.readthedocs.io/en/v4.5.0/_modules/optuna/samplers/_tpe/sampler.html).

### Slide 09 — Vì sao trial0 ghi chưa có history?

**Kết luận:** nội dung cũ thiếu trạng từ chỉ thời điểm, dễ khiến người đọc hiểu nhầm trial0 chưa hoàn tất. Đã sửa bảng thành “History TRƯỚC đề xuất” và “mAP SAU trial”.

| Trial được đề xuất | History trước đề xuất trong `trial_spec.json` | Kết quả sau khi trial hoàn tất |
|---|---|---:|
| 0 | `[]` | 36,0677% |
| 1 | `[0,3606765866…]` | 37,7225% |
| 2 | `[0,3606765866…, 0,3772250712…]` | 38,5071% |
| 3 | `[0,3606765866…, 0,3772250712…, 0,3850706518…]` | 37,0877% |

Diễn tiến: **D₀=∅ → train/evaluate trial0 → D₁={(h₀,F₀)} → đề xuất trial1**. Tất cả8 result đều có `failed=false`; số observations trước trial i bằng i. Checkpoint KB1 lịch sử là reference đối chiếu, không được add vào study như một trial hoàn tất.

Nguồn: [khởi tạo history](../../kb3_hyperparameter_optimization/quality/tpe_pipeline.py:207), [enqueue anchor](../../kb3_hyperparameter_optimization/quality/tpe_pipeline.py:196), [append sau result](../../kb3_hyperparameter_optimization/quality/tpe_pipeline.py:244), thư mục `tpe/trial_0000…0007/trial_spec.json` của run báo cáo.

### Slide 11 — Best TPE là đầu vào KB2 hay policy/reference?

**Kết luận:** cả hai mô tả nói về hai cấp của cùng một pipeline. Tiêu đề cũ “trở thành policy/reference” làm mờ bước chuyển giao, nên đã đổi thành **“Best TPE là checkpoint đầu vào để khởi tạo KB2”**.

Ở cấp Hybrid: HPO trả checkpoint winner → KB2 nhận checkpoint → KB2 fine-tune → detector cuối.

Trong KB2, code thực hiện:

```python
policy = BackgroundAdapter(source_checkpoint('yolov8n'), device, background_weight)
reference = ConditionalAdapter(source_checkpoint('yolov8n'), device)
reference.model.requires_grad_(False)
```

`source_checkpoint()` resolve `trial_0002/selected_best.pt`. Do đó:

\[
\theta_0=w_{TPE},\qquad\theta_{ref}=w_{TPE}.
\]

Ban đầu weights giống nhau; sau update, policy cv3 thay đổi, reference vẫn là mốc TPE. AdamW mới của KB2 không tiếp tục state SGD của HPO. TPE sampler là thuật toán tìm HP, không phải mạng reference hoặc mạng policy. Inference chỉ dùng detector policy cuối.

Nguồn: [resolver](../../kb2_preference_optimization/hybrid/source.py:68), [hai bản model](../../kb2_preference_optimization/hybrid_schedule/train.py:69), [trainable cv3](../../kb2_preference_optimization/conditional/policy.py:20), manifest chốt `source_kind=kb3_tpe_winner`, `source_tpe.trial=2`.

### Slide 12 — Luồng native và DPO loss

**Kết luận:** phép cộng loss đúng; đã vẽ lại để thấy policy/reference có vai trò khác nhau.

- **Sinh bank offline:** ảnh train + GT + raw predictions từ TPE → render A/B → GT judge → bank cố định.
- **Train KB2:** batch4 ảnh gồm2 uniform +2 ảnh của pair.
- **Native:** `policy.native_loss(images, targets)` trên cả4 ảnh, dùng toàn bộ GT trong ảnh.
- **DPO:** policy raw forward có autograd; reference raw forward cùng batch dưới `no_grad`; lấy đúng2 logit theo `(batch_index,slot,class_id)`.
- `total=native+0,03*dpo`; backward chỉ trên trainable cv3.

Đã bỏ nhãn “online” để tránh hiểu nhầm preference được sinh online trong khi train. Các mũi tên native chỉ từ policy/GT; reference chỉ cung cấp zref cho DPO.

Nguồn: [batch và các forward/loss](../../kb2_preference_optimization/hybrid_schedule/train.py:180), [total/backward/step](../../kb2_preference_optimization/hybrid_schedule/train.py:218).

### Slide 14 — Tạo A/B và GT judge

**Kết luận:** đúng về active-bit, nhưng cần nói rõ render cưỡng bức candidate khác ordinary prediction.

`render(context, slot, keep)` chỉ đổi active-bit của đúng1 slot. Box, class, score nguồn giữ nguyên; cả hai set đều qua NMS toàn ảnh và cap300. Với recovery, KEEP cho candidate score thấp tham gia dù chưa qua0,25; **không tăng score gốc lên0,25**. GT judge dùng matching sau render và kiểm tập GT/FP; không tạo tọa độ từ GT.

Ví dụ8GT, KEEP6TP/2FP, DROP5TP/2FP là synthetic minh họa. Gate KEEP chỉ nhận nếu GT mục tiêu có trong chosen nhưng không trong rejected, tất cả GT của rejected được giữ, FP không tăng. Gate DROP còn cần FP giảm nghiêm ngặt. Cặp hợp lệ offline chưa bảo đảm policy đã học đưa candidate qua threshold khi inference.

Đã thêm các phân biệt đó lên slide và Notes.

Nguồn: [render](../../kb2_preference_optimization/conditional/pairs.py:25), [matching](../../kb2_preference_optimization/conditional/pairs.py:47), [judge](../../kb2_preference_optimization/conditional/pairs.py:69).

### Slide 17 — Objective DPO tổng quát

**Kết luận:** công thức policy/reference log-ratio đúng. Đã bổ sung nghĩa expectation và tách công thức tổng quát khỏi ánh xạ riêng của KB2.

`E` trung bình qua preference pairs. Policy/reference trong công thức là xác suất của chosen/rejected action. KB2 tự định nghĩa action Bernoulli KEEP/DROP tại `(j,c)`. Mạng chỉ nhận ảnh; record cung cấp chỉ mục lấy logit. Không lấy confidence trung bình, −native loss hoặc mAP làm log-probability của toàn detection set; cũng không có reward model học riêng trong run này.

Nguồn: [DPO, mục4](https://arxiv.org/html/2305.18290v3#S4), [implementation loss](../../kb2_preference_optimization/conditional/policy.py:46).

### Slide 18 — Bernoulli và công thức trong code

**Kết luận:** đúng; đã làm rõ q là policy phụ nhị phân và s được xác định bởi chosen.

\[
q_\theta(keep)=\sigma(z_\theta),\quad q_\theta(drop)=\sigma(-z_\theta),
\quad \log\frac{q_\theta(keep)}{q_\theta(drop)}=z_\theta.
\]
\[
m_r=s_r(z_{\theta,r}-z_{ref,r}),\quad
L_{DPO}=\frac1{B_p}\sum_r softplus(-\beta m_r).
\]

`s=+1` khi `chosen_keep=True`, `−1` nếu chosen DROP; β0,1, Bp2. λ0,03 nhân **sau** mean DPO trong total loss. Reference được detach. Đây không phải sampling các bin box hoặc likelihood toàn output YOLO.

Nguồn: [pair_loss](../../kb2_preference_optimization/conditional/policy.py:46). Kiểm số16 trường hợp cho công thức DPO tổng quát trên Bernoulli và code softplus khớp tới sai số float64, xem `content_validation.json`.

### Slide 19 — Đạo hàm một cặp so với đạo hàm total loss

**Kết luận:** đạo hàm cũ đúng cho một cặp nhưng chưa nhấn mạnh phạm vi. Đã đổi tiêu đề và thêm hệ số batch/λ.

Với một cặp:

\[
\frac{\partial\ell_r}{\partial z_{\theta,r}}
=-\beta s_r\sigma[-\beta s_r(z_{\theta,r}-z_{ref,r})].
\]

Trong `λ × mean(ℓr)`, đạo hàm đối với mỗi logit riêng được lấy nhân thêm `λ/Bp`. Khi policy=reference:

\[
\left|\frac{\partial(\lambda L_{DPO})}{\partial z_{\theta,r}}\right|
=\frac{\lambda\beta}{2B_p}=0,00075.
\]

Con số này là đạo hàm của hạng DPO tại output logit riêng, không phải độ lớn update weights của model. Nếu lấy mẫu trùng record, gradient tới weights/logit được cộng theo computational graph. Native loss và chia sẻ weights còn tương tác; clipping/AdamW làm bước update khác gradient descent trực tiếp trên z. Đã bỏ ví dụ giả định Δz=0,4 để phần slide này tập trung vào kết luận kiểm được.

Nguồn: [pair_loss mean](../../kb2_preference_optimization/conditional/policy.py:63), [cộng total và optimizer](../../kb2_preference_optimization/hybrid_schedule/train.py:218); kiểm autograd16 trường hợp trong artifact.

### Slide 20 — Native loss, background0,5, β và λ

**Kết luận:** đúng theo recipe đang lưu. Đã bổ sung box/DFL không học lại tọa độ ở KB2.

`args.yaml` trial2 có box7,5, cls0,5, DFL1,5; `cls_pw=0,0`. Do đó native loss được tóm tắt bằng:

\[
L_{native,bg}=7,5L_{box}+0,5L_{cls,bg}+1,5L_{DFL},
\quad L_{total}=L_{native,bg}+0,03L_{DPO}.
\]

Wrapper BCE nhân0,5 **toàn bộ class BCE tại unassigned background slot**; foreground slot vẫn weight1, kể cả những class-target bằng0 tại slot đó. Assigner và mẫu số native `S=max(Σ target_score,1)` được giữ. Native adapter chia batch size sau loss gốc, phù hợp cách library nhân batch size. β0,1 trong DPO, λ0,03 ngoài DPO và background0,5 có ba vai trò khác nhau.

Model vẫn tính native box/DFL loss nhưng regression/backbone frozen và không có đường gradient học tọa độ qua cv3. Assignment được tính lại theo native forward; không suy rằng mọi scalar box loss không đổi qua mọi update.

Nguồn: [BackgroundBCE](../../kb2_preference_optimization/conditional_bg/loss.py:31), [native batch reduction](../../kb2_preference_optimization/adapter.py:43), `trial_0002/native/args.yaml:83–86`, `ultralytics/utils/loss.py` của môi trường. Kiểm trực tiếp wrapper BCE xác nhận zero target foreground giữ1 và slot background nhân0,5.

## 4. Kiểm tra và file đã cập nhật

- [PPTX](slides/hybrid_hpo_dpo_20261111/BAO_CAO_TOT_NGHIEP_HYBRID_HPO_DPO_20261111.pptx), [PDF](slides/hybrid_hpo_dpo_20261111/BAO_CAO_TOT_NGHIEP_HYBRID_HPO_DPO_20261111.pdf), [HTML](slides/hybrid_hpo_dpo_20261111/BAO_CAO_TOT_NGHIEP_HYBRID_HPO_DPO_20261111.html).
- [Kiểm nội dung máy đọc được](slides/hybrid_hpo_dpo_20261111/content_validation.json):8 trial histories, source trial2/config,16 phép so DPO/autograd, native BCE weighting.
- [Kiểm cấu trúc và bảng số](slides/hybrid_hpo_dpo_20261111/validation_report.json): số trang/Notes, XML/relationships/media/bounds và60 ô kết quả gồm delta.
- [Script audit](slides/hybrid_hpo_dpo_20261111/audit_slide_content.py) chạy chỉ loss trên tensor nhỏ; [generator](slides/hybrid_hpo_dpo_20261111/build_hybrid_slides.py) cập nhật nội dung/Notes, bỏ trang và đánh lại số.

PDF/PNG được render từ cùng layout với PPTX. Chưa có Microsoft PowerPoint/LibreOffice để render lại native PPTX trong môi trường này; phạm vi xác nhận hiển thị như README.
