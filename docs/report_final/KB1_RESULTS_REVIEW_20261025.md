# Đánh giá KB1 sau lượt continuation và ablation — 25/10/2026

**Phạm vi hiện hành:** tối ưu thêm detector YOLO bằng reward-guided/RL-inspired fine-tuning. Bản tổng hợp phương pháp và kết luận thống nhất nằm trong [báo cáo KB1](Report_KB1_15_10_2026.md); tài liệu này giữ vai trò review kết quả và truy nguyên artifact.

KB1 mới đã hoàn tất: **20 preflight, 28 continuation, 30 ablation, 16 lượt evaluation**. Có 156 hàng metric (val/test), 116 delta paired và 48 hàng mean/std. Kiểm tra artifact và tính lại các bảng không phát hiện mismatch. Lần review này không train hoặc inference lại detector.

**Kết luận chính:** continued head fine-tuning tăng test mAP so với S0 trong các seed đã thử; guided không luôn hơn native-only. Thí nghiệm mới đã tách được hiệu ứng tăng thêm của reward khi giữ proxy cho hai model: YOLOv11n có mức tăng test nhỏ, còn YOLOv8s giảm test AP ở cả ba seed. Chưa có bằng chứng reward tốt hơn proxy-only trên mọi kiến trúc.

## 1. Tính toàn vẹn và phạm vi kiểm tra

- Pipeline completed/plan/locked checkpoint/summary hashes khớp; 58 checkpoint lựa chọn không thay đổi. Hash best/last/manifest của cả 58 run khớp completed.
- Cả 58 run dùng đúng supervised source đã ghi và cùng runtime trong plan. Dataset fingerprint và bảy S0 vẫn khớp provenance; hash 31 file code/config hiện tại khớp kế hoạch.
- 16 evaluation có đủ hàng, hash manifest/full/delta và nguồn checkpoint khớp. Val có 479 ảnh, test 239 ảnh mỗi hàng; supervised seed 0, continuation seed đúng 42/43/44.
- Cả 30 ablation chạy đủ 10.000 micro-step / 5.000 optimizer update; evaluator thực sự đọc last.pt step 10.000. Continuation dùng best.pt được chọn bằng validation và có early stopping.
- Đối chiếu từng hàng results_all.csv với CSV evaluation gốc, tính lại 116 delta và 48 mean/sample-std: khớp. [Audit JSON](assets/kb1_20261025/audit_summary.json) lưu các nhóm kiểm tra.
- Bảng test 18 hàng Windows lịch sử và 21 hàng Ubuntu trước đó vẫn còn. Hash completed của bảng Ubuntu cũ khớp. Khi đánh giá lại cùng sáu model, chênh mAP50–95 tối đa so với CSV Windows cũ khoảng 0,004497 điểm; các S0 của đợt mới khớp S0 Ubuntu trước đó.
- Không tính lại AP từ prediction thô: kiểm tra này xác minh artifact/evaluator contract và số học, không thay thế lượt inference độc lập hay xác nhận nội dung nhãn.

## 2. Continuation vừa chạy so với seed 42 trước đó

Bảng là **test mAP50–95 (%)**. Hai cột delta là mean ± sample std **điểm phần trăm**, chỉ qua hai seed Ubuntu mới 43/44 (n=2). Seed 42 để cạnh nhau nhằm đối chiếu, không được âm thầm gộp vào thống kê mới: sáu model seed 42 được train trên Windows; YOLO26n seed 42 được train lại trên Ubuntu. S0 giữ nguyên seed 0.

| Model | G seed 42 trước | G seed 43 mới | G seed 44 mới | G−S0, n=2 (điểm) | G−S1, n=2 (điểm) |
|---|---:|---:|---:|---:|---:|
| YOLOv5s | 31.249851 | 31.242937 | 31.278473 | +0.863037 ± 0.025128 | +0.235108 ± 0.062782 |
| YOLOv8n | 33.760384 | 33.388162 | 33.733153 | +0.359160 ± 0.243946 | +0.305267 ± 0.247720 |
| YOLOv8s | 35.241696 | 35.387424 | 35.387179 | +1.411769 ± 0.000173 | +0.377224 ± 0.066607 |
| YOLOv11n | 37.730387 | 37.796175 | 37.738842 | +0.233561 ± 0.040541 | +0.004964 ± 0.011175 |
| YOLOv11s | 36.499283 | 36.495739 | 36.389121 | +0.171609 ± 0.075390 | -0.453047 ± 0.024650 |
| YOLO26n | 33.804214 | 33.836004 | 33.826095 | +0.473715 ± 0.007007 | +0.382070 ± 0.135382 |
| DP-YOLO | 29.469168 | 29.346496 | 29.384807 | +0.452892 ± 0.027090 | +0.054295 ± 0.050629 |

**Chú thích bảng 1.** AP là test mAP50–95 (%); delta là điểm phần trăm. Hai cột mean/std chỉ dùng paired continuation seed 43/44 (n=2); seed 42 được giữ riêng để đối chiếu nguồn lịch sử.

**Nhận xét bảng 1.** Guided cải thiện S0 trên mọi model trong hai seed mới, nhưng YOLOv11s thua native-only và YOLOv11n chỉ nhỉnh hơn gần 0. Không dùng thứ hạng gain thay cho thứ hạng AP tuyệt đối.


- G hơn S0 ở **14/14** cặp model–seed mới. G hơn native-only ở **11/14** cặp; vẫn gồm reward+proxy và khác số update thực tế.
- YOLOv11s thấp hơn native-only ở cả seed 42/43/44: lần lượt −0,462094 / −0,435618 / −0,470477 điểm trên bảng Ubuntu. Không có cơ sở nói guided luôn tốt hơn tiếp tục native loss.
- YOLOv11n continuation G−S1 đổi dấu giữa seed 43/44 (−0,002939 / +0,012866 điểm); trung bình +0,004964 là quá nhỏ để gọi là ưu thế ổn định.
- YOLOv8s tăng G−S0 khoảng +1,412 điểm trên cả hai seed mới. YOLO26n tăng khoảng +0,474 điểm; G−S1 dương ở cả 42/43/44. Đây là hiệu quả stage hai tổng thể, chưa tách riêng reward cho các model không có ablation.
- Trong cohort continuation, YOLOv11n vẫn là guided model test cao nhất. So sánh cả S1/G, native-only YOLOv11n seed 43 đạt 37,799114%, nhỉnh hơn guided cùng seed 37,796175%. Không dùng thứ hạng test để chọn lại hyperparameter.

## 3. Ablation cùng số update: ba seed 42/43/44

Đây là bộ thí nghiệm mới trên cùng Ubuntu, S0, sampler/augmentation seed và lịch LR trong từng seed. Tất cả nhánh đánh giá tại step cuối 10.000; không so best ở step khác nhau rồi gọi là matched-update. Thời gian không bằng nhau. Mean ± sample std mAP50–95 test (%) qua 3 continuation seed, vẫn chỉ một supervised initialization.

| Nhánh | YOLOv11n | YOLOv8s |
|---|---:|---:|
| S0 | 37.533948 ± 0.000000 | 33.975533 ± 0.000000 |
| S1: native+SP | 37.625388 ± 0.007269 | 35.085922 ± 0.039491 |
| P: native+SP+proxy | 37.764661 ± 0.020501 | 35.457566 ± 0.027894 |
| R: native+SP+reward | 37.641009 ± 0.008999 | 34.376187 ± 0.041921 |
| G: native+SP+proxy+reward | 37.835387 ± 0.015414 | 35.407580 ± 0.038423 |
| G không SP | 37.835494 ± 0.015465 | 35.407823 ± 0.038346 |

**Chú thích bảng 2.** AP dùng %, mean ± sample SD qua ba continuation seed 42/43/44 tại cùng step cuối 10.000. S0 là cùng checkpoint được đánh giá lại, không phải ba supervised training độc lập.

**Nhận xét bảng 2.** G dẫn đầu giữa S1/P/R/G trên YOLOv11n, còn P có AP cao hơn G trên YOLOv8s. SP và no-SP gần như trùng nhau ở cả hai model; hiệu quả thành phần phụ thuộc model.


| Delta test (điểm phần trăm) | YOLOv11n, n=3 | YOLOv8s, n=3 |
|---|---:|---:|
| G−S0 | +0.301440 ± 0.015414 | +1.432047 ± 0.038423 |
| G−S1 | +0.209999 ± 0.010706 | +0.321657 ± 0.034339 |
| G−P: thêm reward khi giữ proxy | +0.070727 ± 0.007724 | -0.049986 ± 0.048898 |
| G−R: thêm proxy khi giữ reward | +0.194379 ± 0.024409 | +1.031393 ± 0.047579 |
| G−G-noSP | -0.000106 ± 0.000110 | -0.000243 ± 0.000260 |

**Chú thích bảng 3.** Delta paired được tính trong từng seed trước khi tổng hợp mean và sample SD n=3, đơn vị điểm phần trăm AP. G−P giữ proxy và thêm reward; G−G-noSP thể hiện thay đổi khi có SP.

**Nhận xét bảng 3.** Reward tăng thêm nhỏ ở YOLOv11n nhưng giảm AP ở YOLOv8s khi giữ proxy. Proxy là thành phần đối chứng quan trọng để giải thích gain tổng; bảng chưa chứng minh significance trên dữ liệu hoặc initialization khác.


**Reward có đóng góp riêng hay không?** Trong protocol này, G−P là phép so phù hợp để đánh giá phần tăng thêm của reward khi proxy giữ nguyên. YOLOv11n: +0,063074 / +0,070587 / +0,078520 điểm qua seed 42/43/44, dương cả ba seed nhưng nhỏ. YOLOv8s: −0,005087 / −0,102085 / −0,042787 điểm, âm cả ba seed. Vì vậy hiệu ứng reward phụ thuộc model; không thể suy từ G−S0 hoặc G−S1 dương rằng reward tự nó là nguyên nhân chính.

**Proxy là đối chứng mạnh:** YOLOv8s proxy-only đạt mean 35,457566%, cao hơn G 35,407580%. Reward-only chỉ đạt 34,376187%, thấp hơn native-only 35,085922%. Trên YOLOv11n, proxy-only cũng tăng đáng kể so với native-only; reward-only chỉ nhỉnh hơn native-only khoảng 0,015620 điểm. Hiệu ứng reward khi đi kèm proxy khác hiệu ứng reward khi không có proxy.

**L2-SP:** bỏ SP làm AP trung bình nhỉnh hơn khoảng 0,000106 điểm ở YOLOv11n và 0,000243 điểm ở YOLOv8s; chênh lớn nhất trong các seed chỉ khoảng 0,000540 điểm. Chưa có bằng chứng thực nghiệm hệ số SP=0,001 mang lại lợi ích trong thiết kế head-only/LR rất nhỏ này. Không đồng nhất kết quả đó với việc mọi loại regularization đều vô ích.

## 4. Validation–test không đồng thuận và các metric phụ

G−P trên validation: YOLOv11n **+0,002163 ± 0,011067 điểm**, đổi dấu giữa seed; YOLOv8s **+0,082380 ± 0,005695 điểm**, dương cả ba seed. Nhưng test YOLOv8s G−P âm cả ba seed. Đây là sự không đồng thuận giữa split; chưa đủ dữ liệu quy nguyên nhân riêng cho overfitting, class distribution hoặc một bug. Cần prediction/error analysis và holdout độc lập, không tuning trên test này.

YOLOv11n continuation best-validation và ablation final-step cho kết luận G−S1 khác nhau vì protocol khác: continuation chọn checkpoint sớm; fixed-budget giữ last ở 10.000. Không dùng fixed-budget test tốt hơn để kết luận luôn train lâu hơn, khi validation không ủng hộ chọn checkpoint đó.

AP nhỏ vẫn là hạn chế: YOLOv8s S0 APs 5,244923%, fixed-budget G mean 4,866260%, P 4,891231%. YOLOv11n G mean APs 3,822819% cao hơn S0 3,374792%, nhưng thấp hơn native-only 3,874824% và gần P 3,824248%. Chưa có bằng chứng reward giải quyết vật thể nhỏ; GT vẫn bao lá chứ không annotate lesion riêng.

G−P tăng mAP của YOLOv11n nhưng AR300 giảm nhẹ: G 54,382606% so P 54,507210%. YOLOv8s G macro-F1 46,159887% cao hơn P 45,904268%, dù AP thấp hơn. Cần giữ metric chính đã khóa; không đổi sang metric thắng sau khi xem test.

## 5. Chi phí và giới hạn suy luận

Fixed-budget có cùng 5.000 optimizer update nhưng không cùng thời gian. Trung bình thời gian ghi trong completed (giây): YOLOv11n S1/P/G ≈ 1141/1550/1657; YOLOv8s ≈ 1603/2330/2674. Đây không phải tổng end-to-end, không gồm đầy đủ loading/baseline và phần tính toán bị mất sau crash. Chưa có đối chứng cùng thời gian.

- Các repeated evaluation S0 không là seed độc lập. Nhiều continuation seed vẫn có điều kiện trên cùng S0 seed 0.
- N=2 cho continuation mới và n=3 cho ablation còn nhỏ; mean/std không phải confidence interval hoặc kiểm định significance. Chưa bootstrap paired theo ảnh/source từ prediction thô.
- Test 239 ảnh đã được xem từ trước; kết quả mới là exploratory. Không được gọi đây là xác nhận trên holdout độc lập.
- G−P chỉ được kiểm cho YOLOv11n/YOLOv8s. Không ngoại suy cho YOLO26n, DP hoặc các kiến trúc khác.
- Cách gọi vẫn là reward-guided / RL-inspired head fine-tuning. Thêm seed/ablation không biến surrogate thành REINFORCE chuẩn.

## 6. Việc nên làm tiếp

1. Cập nhật kết luận KB1: stage hai có lợi so S0; giá trị riêng của reward nhỏ và phụ thuộc model. Dùng P làm đối chứng chính bên cạnh S1.
2. Giữ nguyên artifact/test hiện tại. Phân tích lỗi và class trên validation, đặc biệt vì G−P validation/test của YOLOv8s trái dấu; chưa chọn phương pháp cuối chỉ từ kết quả test này.
3. Nếu cần củng cố đóng góp reward, ưu tiên nhiều supervised initialization cho YOLOv11n và đối chứng S1/P/G, khóa thiết kế rồi xác nhận trên nguồn ảnh/holdout mới. Không cần mặc định mở rộng reward-only/no-SP ra tất cả model trước khi có giả thuyết rõ.
4. Xuất prediction/GT để bootstrap paired theo ảnh/source và phân tích AP75, box-size/class errors. Thiết kế FN candidate/advantage mới là thí nghiệm riêng cần validation và output mới.

## 7. Nguồn artifact

- [Results mới, 156 hàng](../../kb1_reward_guided_training/checkpoint_reward_guide_trainning/remaining_seed43_44_ablation_v1/results_all.csv).
- [Delta từng seed](../../kb1_reward_guided_training/checkpoint_reward_guide_trainning/remaining_seed43_44_ablation_v1/paired_deltas.csv), [mean/std](../../kb1_reward_guided_training/checkpoint_reward_guide_trainning/remaining_seed43_44_ablation_v1/summary_mean_std.csv).
- [Plan](../../kb1_reward_guided_training/checkpoint_reward_guide_trainning/remaining_seed43_44_ablation_v1/plan.json), [completed](../../kb1_reward_guided_training/checkpoint_reward_guide_trainning/remaining_seed43_44_ablation_v1/completed.json).
- [21 hàng Ubuntu trước đó](../../kb1_reward_guided_training/results/canonical_clean/screening/ubuntu_all_models_seed42_20261022_v1/results_full.csv).
- [18 hàng Windows lịch sử](../../kb1_reward_guided_training/results/canonical_clean/screening/test/results_full.csv).
- [Báo cáo tổng hợp trước](Report_KB1_15_10_2026.md), [hướng dẫn Ubuntu](UBUNTU_KB1_20261022.md).
