# Đánh giá toàn bộ KB1 theo thiết kế ngày 15/10 — cập nhật 26/10/2026

**Phạm vi hiện hành:** tối ưu thêm detector YOLO bằng reward-guided/RL-inspired fine-tuning. Bản tổng hợp phương pháp và kết luận thống nhất nằm trong [báo cáo KB1](Report_KB1_15_10_2026.md); tài liệu này giữ vai trò review kết quả và truy nguyên artifact.

**Kết luận về hoàn thành:** KB1 đã chạy xong pipeline hai giai đoạn, cohort bảy model, các continuation seed đã chốt, bộ ablation bổ sung và toàn bộ evaluation/tổng hợp trong `run_remaining_kb1.sh`. Có thể chốt kết quả thực nghiệm trong phạm vi data, S0 và protocol hiện có. Các mục mở rộng trong phần 17 của báo cáo 15/10 như supervised initialization mới, holdout độc lập và thiết kế reward mới vẫn chưa chạy; đây không phải các job còn treo của pipeline đã hoàn tất.

Bản đánh giá này đọc lại [báo cáo KB1 15/10](Report_KB1_15_10_2026.md), [định nghĩa logic KB1](KB1.md), đối chiếu implementation/manifest và kết quả hiện tại; không train hoặc inference lại. [Review 25/10](KB1_RESULTS_REVIEW_20261025.md) giữ các bảng chi tiết từng seed và ablation.

## 1. Kiến trúc thực sự của KB1/KB1B

KB1 là quy trình huấn luyện và so sánh nhiều detector YOLO. Mỗi kiến trúc có pretrained, supervised checkpoint S0 và nhánh fine-tuning riêng. KB1B là giai đoạn reward-guided cập nhật trọng số từ chính S0 đó. Ý tưởng nghiên cứu chính là pretrained → supervised → reward-guided fine-tuning, đo mức cải thiện của G so với S0 và tìm model guided tốt nhất theo validation đã khóa.

```text
Pretrained riêng từng kiến trúc
    → Stage 1: native supervised training → S0 (best native checkpoint)
    → Stage 2: fine-tune Detect head
        ├─ S1: native loss + L2-SP
        └─ G/KB1B: native loss + reward term + confidence proxy + L2-SP
    → Chọn best bằng canonical validation mAP50–95
    → Khóa checkpoint và đánh giá test
```

- **Bảy model:** YOLOv5s, YOLOv8n/s, YOLOv11n/s, YOLO26n và DP-YOLO. Mỗi stage hai bắt đầu đúng checkpoint supervised tương ứng, được kiểm SHA/dataset provenance.
- **Stage một:** native trainer, SGD, input 640, tối đa 200 epoch, patience 30, supervised seed 0. Batch/recipe và thời điểm dừng khác nhau giữa họ model. S0 được chọn bằng fitness native; không khẳng định mọi S0 là epoch có canonical AP cao nhất.
- **Stage hai:** đóng băng backbone/neck, BN statistics và DFL projection; chỉ train Detect head bằng Adam, LR 5e-7→1e-7, warmup 500, batch 8, accumulation 2, clip gradient 0,5. Native loss cung cấp gradient định vị; reward/proxy hướng chủ yếu qua confidence, còn box/NMS/matching detach.
- **Reward:** tính từ prediction–GT train bằng greedy matching một-một cùng class; trung bình qua IoU 0,5/0,6/0,7/0,8, trọng số precision/recall/IoU = 0,25/0,45/0,30. EMA mean/second moment decay 0,99 tạo normalized advantage, clip [-3,3].
- **Proxy:** log-confidence TP, log(1-confidence) FP, thêm FN term dựa trên max score toàn ảnh. Gradient không đi qua việc lựa chọn box/class hoặc matching; inference không sample một policy stochastic có log-likelihood action tường minh.
- **Objective G:** `0.25 native + 0.10 reward + 0.25 proxy + 0.001 L2-SP`. G−S0 đo hiệu quả quy trình stage hai tổng thể. G−S1 đo thêm reward+proxy. G−P trong ablation mới đo phần tăng thêm của reward khi giữ proxy.
- **DP-YOLO:** nhánh custom có P2/P3/P4/P5, D2C3/D3C3, PTCSP/C3Ghost, W3F/PSA bật theo manifest. DCNv3 giản lược và W3F là công thức custom; PTCSP attention dùng pooling tối đa 400 token. Chưa tái lập nguyên vẹn một paper DP duy nhất và chưa có ablation từng module kiến trúc.
- **YOLO26n:** canonical inference dùng one-to-many + NMS chung; native loss giữ dual-head, regression không DFL. Criterion hiện dùng gain one-to-many/one-to-one 0,8/0,2 trong stage hai vì runner không gọi lịch update criterion.
- **Evaluator:** TorchMetrics/faster-coco-eval, mAP50–95, conf 0,001, class-aware NMS 0,60, cap 300, cùng resize/pad 640. Đây là protocol canonical của dự án; native validation AP và benchmark ngoài không thay thế nó.

Tên phương pháp chính xác vẫn là **reward-guided / RL-inspired head fine-tuning**, phù hợp với một hybrid objective có task feedback. Bộ kết quả mới không làm confidence surrogate trở thành REINFORCE chuẩn.

## 2. Hoàn tất đến đâu?

| Hạng mục | Trạng thái xác nhận | Phạm vi |
|---|---|---|
| Supervised sạch | Hoàn tất 7/7 | Mỗi model một S0 seed 0, hash/provenance khớp |
| Cohort seed 42 | Hoàn tất S0/S1/G cho cả 7 model | Sáu model lịch sử; YOLO26n train lại S1/G trên Ubuntu; bảng test 21 hàng |
| Continuation mới | Hoàn tất 28/28 | 7 model × seeds 43/44 × native-only/guided |
| Preflight đợt bổ sung | Hoàn tất 20/20 | Tất cả tổ hợp model/mode sẽ dùng trong đợt bổ sung |
| Ablation | Hoàn tất 30/30 | 2 model × 3 seed × 5 nhánh, đủ 10.000 micro-step / 5.000 update |
| Evaluation đợt bổ sung | Hoàn tất 16/16 | 156 hàng val/test, 116 delta và 48 mean/std |
| Pipeline tổng | `status=complete` | Đủ 58 lượt train dài theo kế hoạch script bổ sung |
| Nhiều supervised seed | Chưa có | Continuation 42/43/44 vẫn dùng S0 seed 0 |
| Holdout độc lập mới / bootstrap prediction | Chưa có trong bộ này | Các test hiện tại dùng split đã xem trước |

**Chú thích bảng 1.** Các số lượt thuộc từng nhóm được ghi trong completed/plan. 58 là số training run dài bổ sung; 20 preflight được đếm riêng và cohort seed 42 có nguồn lịch sử riêng.

**Nhận xét bảng 1.** Pipeline đã hoàn tất phạm vi đã lập kế hoạch. Các dòng chưa có supervised seed mới hoặc tập đánh giá độc lập là giới hạn cho kết luận mở rộng, không phải job đang chờ của script.


Các run seed 42 trước đó và 58 lượt bổ sung là các nhóm riêng; không gọi 58 là tổng toàn bộ lịch sử KB1 hoặc trộn các protocol thành một thí nghiệm. Script bổ sung đã kết thúc, không còn nhánh validation/test nào của kế hoạch đó đang chờ.

## 3. Kiểm tra tính đúng và khả năng truy nguyên

Đối chiếu lại ngày 26/10: pipeline/plan/summary hashes, 58 checkpoint lựa chọn, best/last/manifest hashes của 58 training run, nguồn supervised, runtime và toàn bộ 16 evaluation đều khớp. Cả 30 ablation kết thúc tại budget 10.000; evaluator đọc last.pt step 10.000. Dataset vẫn có fingerprint `cbd68ffc1da935ac300fba953ea5f9c97c831f53887e1b9391d500f9e40c0c95`. [Audit kiểm tra ngày 26/10](assets/kb1_20261026/completion_audit.json) không ghi mismatch.

Bộ review 25/10 đã đối chiếu từng hàng CSV tổng hợp với CSV gốc, tính lại delta/mean/sample-std. Review hiện tại xác nhận các artifact vẫn nguyên trạng. Đây là kiểm tra integrity và protocol trên kết quả lưu, chưa phải tái tính numerical AP bằng một lượt inference GPU mới.

Điểm mạnh của implementation là paired source, step-zero candidate, validation selection, frozen-state/gradient checks, strict reload, manifest/runtime, output mới và fail khi thiếu hàng. Các guard này làm phép so sánh có thể kiểm tra được. Chúng không thay thế kiểm định thống kê hoặc xác nhận tính đúng của annotation.

## 4. RQ1: Guided có cải thiện supervised không?

**Có, trong các điều kiện đã thử.** Cohort seed 42 có G>S0 ở 7/7 model; hai seed mới có G>S0 ở 14/14 cặp. Tổng mô tả là 21/21 cặp model–continuation seed dương. Không dùng con số này như 21 quan sát thống kê độc lập vì chung dataset, S0 và khác model/host lịch sử.

Bảng dưới dùng riêng hai seed Ubuntu mới 43/44: AP là mean test mAP50–95 (%); delta là điểm phần trăm. Không gộp seed 42 Windows vào mean/std này.

| Model | S0 test | G test mean 43/44 | Mean G−S0 (điểm) | Mean G−S1 (điểm) |
|---|---:|---:|---:|---:|
| YOLOv5s | 30.397668 | 31.260705 | +0.863037 | +0.235108 |
| YOLOv8n | 33.201498 | 33.560658 | +0.359160 | +0.305267 |
| YOLOv8s | 33.975533 | 35.387301 | +1.411769 | +0.377224 |
| YOLOv11n | 37.533948 | 37.767509 | +0.233561 | +0.004964 |
| YOLOv11s | 36.270821 | 36.442430 | +0.171609 | -0.453047 |
| YOLO26n | 33.357334 | 33.831050 | +0.473715 | +0.382070 |
| DP-YOLO | 28.912759 | 29.365651 | +0.452892 | +0.054295 |

**Chú thích bảng 2.** S0 và G là canonical test mAP50–95 (%); delta dùng điểm phần trăm, mean chỉ trên hai seed Ubuntu mới 43/44. S0 của từng model giữ nguyên seed 0.

**Nhận xét bảng 2.** G−S0 dương ở bảy model; YOLOv8s tăng nhiều nhất, YOLOv11n có AP tuyệt đối cao nhất. G−S1 của YOLOv11s âm, cho thấy gain so supervised chưa chứng minh ưu thế so mọi cách continuation.


YOLOv8s có mức tăng G−S0 lớn nhất, khoảng +1,412 điểm trong hai seed mới. YOLO26n cũng cải thiện rõ so với S0 của chính nó, nhưng chưa đạt AP tổng của YOLOv11n. Hai model này không cần thắng mọi model khác để câu hỏi về lợi ích stage hai được trả lời dương.

**Guided có luôn hơn native-only không?** Không. Trong các seed 42/43/44, G>S1 ở 16/21 cặp mô tả (5/7 ở seed 42, 11/14 ở seed mới). YOLOv11s thua S1 ở cả ba seed; YOLOv11n G−S1 gần 0 và đổi dấu trong continuation. Vì vậy lợi ích G−S0 không thể quy toàn bộ cho reward.

## 5. RQ2: Kiến trúc guided nào tốt nhất?

**YOLOv11n dẫn đầu trong tập ứng viên guided best-validation của cohort chính.** Nó đứng đầu validation và test ở cả seed 43 và 44:

| Seed | Guided validation mAP50–95 (%) | Guided test mAP50–95 (%) |
|---|---:|---:|
| 43 | 40.643573 | 37.796175 |
| 44 | 40.652746 | 37.738842 |

**Chú thích bảng 3.** Các checkpoint guided được chọn bằng validation riêng từng seed. Val và test có lần lượt 479 và 239 ảnh; AP đều dùng cùng canonical contract.

**Nhận xét bảng 3.** YOLOv11n dẫn đầu tập guided trên cả hai split trong từng seed; chênh giữa seed cần được báo cáo mô tả. Không chọn seed 43 chỉ vì test cao hơn seed 44.


Mean hai seed mới là **40,648159% validation / 37,767509% test**; đây là mô tả kết quả của các checkpoint đã khóa, không chọn seed theo test. Seed 42 cũ cũng có YOLOv11n đứng đầu guided.

Nếu mở tập ứng viên sang mọi nhánh hoặc sang checkpoint cuối của ablation, điểm test có thể nhỉnh hơn: S1 YOLOv11n seed 43 đạt 37,799114%; fixed-budget G YOLOv11n seed 42 đạt 37,851104%. Không lấy các số đó thay cho RQ2 đã định nghĩa hoặc chọn checkpoint mới dựa trên test. Cần giữ riêng kết luận “guided model đứng đầu” và “checkpoint nào có test cao nhất trong một tập thí nghiệm cụ thể”.

## 6. Ablation giải thích điều gì?

Trên YOLOv11n/YOLOv8s, S1/P/R/G/G-noSP được so tại cùng step 10.000, cùng LR schedule và seed trong từng cặp. Mean ± sample std qua ba continuation seed (điểm phần trăm test AP):

| So sánh | YOLOv11n | YOLOv8s |
|---|---:|---:|
| G−S0: stage hai tổng thể | +0.301440 ± 0.015414 | +1.432047 ± 0.038423 |
| G−S1: thêm reward+proxy | +0.209999 ± 0.010706 | +0.321657 ± 0.034339 |
| G−P: thêm reward khi giữ proxy | +0.070727 ± 0.007724 | -0.049986 ± 0.048898 |
| G−R: thêm proxy khi giữ reward | +0.194379 ± 0.024409 | +1.031393 ± 0.047579 |
| G−G-noSP: tác dụng SP | -0.000106 ± 0.000110 | -0.000243 ± 0.000260 |

**Chú thích bảng 4.** Mean ± sample SD của paired delta test qua ba continuation seed, đơn vị điểm phần trăm. Tất cả nhánh train có cùng 10.000 micro-step / 5.000 update; thời gian thực khác nhau.

**Nhận xét bảng 4.** G−P mới là phép so phần thêm reward khi có proxy: dương nhỏ với YOLOv11n, âm với YOLOv8s. Mức cải thiện G−S0 lớn hơn không thể quy hoàn toàn cho riêng reward.


- **YOLOv11n:** reward tăng thêm khoảng +0,071 điểm so với proxy-only, dương cả ba seed. Đây là tín hiệu nhỏ có điều kiện trên một S0; chưa có CI theo ảnh/source hoặc supervised seed độc lập.
- **YOLOv8s:** reward giảm AP so với proxy-only ở cả ba seed. P đạt mean test 35,457566%, cao hơn G 35,407580%; proxy là đối chứng cần thiết để đánh giá đúng đóng góp.
- **Reward-only** không thay thế tốt tổ hợp proxy+reward. Ở YOLOv8s nó còn thấp hơn native-only; không có cơ sở gọi riêng reward là nguyên nhân chủ đạo của toàn bộ gain.
- **SP=0,001:** bỏ SP gần như không đổi AP, chênh dưới khoảng 0,000540 điểm trong các seed đã thử. Chưa thấy lợi ích thực nghiệm rõ của thành phần này ở LR/head-only hiện tại.
- Validation G−P ở YOLOv8s dương cả ba seed (mean +0,082380 điểm), nhưng test âm cả ba seed. Đây là bất đồng giữa split cần phân tích; chưa đủ bằng chứng quy riêng nguyên nhân cho overfit, nhãn, hoặc một bug.

Kết luận về reward đã rõ hơn pilot: **có tín hiệu thêm giá trị ở YOLOv11n, chưa tổng quát sang YOLOv8s; chưa được ablate riêng trên năm model còn lại.** Thí nghiệm phủ định lợi ích reward phổ quát vẫn là kết quả nghiên cứu có giá trị, không làm quy trình hai giai đoạn bị coi là chưa chạy xong.

## 7. Đánh giá kỹ thuật và giới hạn thí nghiệm

**Về kiến trúc/quy trình:** pipeline hai giai đoạn đã được triển khai và kiểm chứng input/output rõ; adapter cho nhiều họ YOLO và DP/YOLO26 đặc thù đã chạy thật. Điều này hỗ trợ mô tả một framework reward-guided fine-tuning thực nghiệm. Chưa có chứng minh estimator policy-gradient unbiased hoặc tái lập đầy đủ các module paper gốc.

**Về chất lượng phát hiện:** AP tổng tăng nhỏ đến vừa, phụ thuộc model. APs còn thấp và không cải thiện nhất quán: trong fixed-budget, YOLOv8s G APs 4,866260% thấp hơn S0 5,244923%; YOLOv11n G APs 3,822819% cao hơn S0 nhưng thấp hơn S1. Class hiếm như Potato leaf có rất ít nguồn ảnh test. Các AP này phản ánh kết quả trên box được gán nhãn; chất lượng theo class/kích thước và độ hỗ trợ dữ liệu cần được đọc cùng AP tổng.

**Về DP:** DP-YOLO đứng cuối cohort về AP tổng, dù stage hai tăng so S0. Vì bản DP custom, pretrain transfer, batch và nhiều module/loss/assignment thay đổi cùng lúc, chưa xác định nguyên nhân kém hơn hoặc quy kết kết quả cho một module/paper cụ thể.

**Về chi phí:** guided cần prediction forward, matching và reward ngoài native loss. Fixed-budget có cùng số update, không cùng thời gian. Thời gian G ghi nhận trung bình khoảng 1.657 giây với YOLOv11n và 2.674 giây với YOLOv8s, cao hơn S1 tương ứng khoảng 1.141/1.603 giây. Chưa có bằng chứng ưu thế khi khớp thời gian.

**Về độ chắc chắn:** continuation mới được tổng hợp n=2 riêng; ablation n=3 trên hai model. Tất cả vẫn dựa trên một supervised initialization seed 0. Test 239 ảnh đã được xem nhiều lần; source-ID audit chưa bao phủ near-duplicate khác ID. Mean/std chưa thay thế CI/bootstrap/independent holdout. Không nên tuyên bố significance hoặc tổng quát hóa ngoài dataset hiện tại.

**Về objective:** FN proxy còn tăng max score toàn ảnh thay vì candidate cho từng missed GT; hệ số proxy kết hợp reward có thể âm khi advantage <−2,5. Đây là giới hạn thiết kế đã biết, chưa được log tần suất/gradient để xác nhận là nguyên nhân của kết quả. Không cần sửa chúng để đánh dấu các job đã chạy là hoàn tất; sửa sẽ tạo một biến thể nghiên cứu mới cần output/protocol mới.

## 8. Những việc cũ đã giải quyết và việc còn mở

| Phát hiện/kế hoạch trong report 15/10 | Trạng thái hiện tại |
|---|---|
| YOLO26n thiếu native-only/guided test | Đã hoàn tất cặp Ubuntu và test |
| Chỉ một continuation seed sạch | Đã có 42/43/44; báo tách môi trường lịch sử và hai seed mới |
| Chưa tách reward khỏi proxy | Đã có P/R/G/no-SP trên hai model, ba seed |
| Early stopping làm khác số update | Đã có bộ matched-update ở hai model; matched-time còn thiếu |
| Ubuntu/provenance/partial evaluator | Đã có môi trường, relocation guard, output mới, fail khi thiếu hàng |
| Seed và provenance trong bảng evaluation | Đã ghi supervised/continuation seed, source/runtime/checkpoint hashes |
| Supervised seed độc lập / holdout mới | Chưa thực hiện |
| Near-duplicate theo nội dung / prediction error analysis / paired CI | Chưa có bộ xác nhận đầy đủ trong các artifact được review |
| FN theo từng GT / trọng số reward không âm / mở khóa feature extractor | Chưa thử như biến thể mới |
| Full training interruption–resume parity | Có run đã resume rồi hoàn tất; chưa có test đối chiếu toàn bộ với run liên tục |

**Chú thích bảng 5.** Trạng thái đối chiếu phát hiện ngày 15/10 với artifact đến 26/10. Một số mục đã hoàn tất ở hai model, không phải ở mọi kiến trúc hoặc mọi điều kiện.

**Nhận xét bảng 5.** Nội dung cần phân biệt là hoàn tất pipeline với phạm vi kiểm chứng. Bộ thí nghiệm hiện đủ để báo cáo kết quả hai giai đoạn và ablation có điều kiện; các bước còn mở có thể ưu tiên theo giả thuyết nghiên cứu.


Không yêu cầu thực hiện toàn bộ roadmap nghiên cứu mở rộng mới được chốt KB1 hiện tại. Điều cần thiết là giới hạn kết luận đúng phạm vi dữ liệu và thí nghiệm đã có. Nếu tiếp tục phát triển, ưu tiên P/S1/G trên nhiều S0 và holdout mới, phân tích validation trước khi sửa loss hoặc mở rộng thêm nhiều ablation.

## 9. Kết luận có thể dùng cho báo cáo

> KB1 triển khai quy trình supervised training và reward-guided fine-tuning Detect head cho bảy kiến trúc YOLO trên dataset sạch 28 class. Guided checkpoint cải thiện canonical test mAP50–95 so với supervised checkpoint tương ứng trong các continuation seed đã khảo sát; YOLOv11n đứng đầu cohort guided theo validation và cũng đạt AP test cao nhất trong cohort này. Các đối chứng cho thấy lợi ích vượt native-only phụ thuộc model. Ablation ba continuation seed tại cùng số update xác nhận reward làm tăng AP nhỏ khi giữ proxy ở YOLOv11n, nhưng giảm AP ở YOLOv8s; vì vậy cải thiện toàn bộ stage hai chưa thể quy riêng cho reward trên mọi kiến trúc. Kết quả hỗ trợ một phương pháp hybrid reward-guided fine-tuning có hiệu quả phụ thuộc mô hình, cần xác nhận trên nhiều supervised initialization và holdout độc lập để mở rộng kết luận.

Các số liệu chi tiết: [review kết quả 25/10](KB1_RESULTS_REVIEW_20261025.md), [CSV 156 hàng](../../kb1_reward_guided_training/checkpoint_reward_guide_trainning/remaining_seed43_44_ablation_v1/results_all.csv), [mean/std](../../kb1_reward_guided_training/checkpoint_reward_guide_trainning/remaining_seed43_44_ablation_v1/summary_mean_std.csv), [pipeline completed](../../kb1_reward_guided_training/checkpoint_reward_guide_trainning/remaining_seed43_44_ablation_v1/completed.json).
