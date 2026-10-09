# KB3-B: thiết kế, cơ sở lý thuyết, hạn chế và kế hoạch nghiên cứu tương lai

Ngày review: 09/10/2026. Phạm vi: YOLO phát hiện sâu bệnh, tối ưu quá trình train bằng RL. Tài liệu này ghi lại B để nghiên cứu sau; entry point hiện tại **chỉ chạy A–TPE, không chạy B**. Các nhận xét về code dựa trên mã trong repository; bằng chứng từ nghiên cứu khác được dẫn nguồn riêng. Chưa có thực nghiệm đối chứng đủ để kết luận B cải thiện YOLO trong dự án này.

## 1. Quyết định hiện tại và câu hỏi nghiên cứu

Mục tiêu chung của KB1, KB2, KB3 vẫn là cải thiện YOLO bằng RL theo những cách khác nhau. A–TPE là HPO truyền thống dùng làm thử nghiệm và xây mốc cho KB3; TPE **không phải RL**. B là hướng RL điều khiển hyperparameter trong quá trình SGD train YOLO. Tạm dừng B không đồng nghĩa bác bỏ tính khả thi lý thuyết của hướng này.

Câu hỏi cần kiểm chứng là: sau khi đã có cấu hình và lịch train thông thường tốt, policy sử dụng phản hồi trong train có cải thiện chất lượng trên dữ liệu chưa dùng để học/chọn policy không? Cần phân biệt ba mức kết quả: code đổi tham số đúng; policy học được hành vi hữu ích; hành vi đó làm tốt hơn đối chứng trên seed/model hoặc dữ liệu mới. Chỉ mức đầu đã có các kiểm thử kỹ thuật; hai mức sau vẫn cần thực nghiệm.

Đợt hiện tại dùng bốn lượt A: một mốc scratch có recipe KB1 và ba cấu hình mới quanh mốc, chỉ tìm `lr0` và `weight_decay`. Checkpoint supervised KB1 là mốc pretrained bên ngoài. Báo cáo A sẽ làm cơ sở quyết định có mở lại B hay không. B không cần phải được giữ bằng mọi giá nếu một lịch train cố định đã đáp ứng mục tiêu; nhưng chưa thể kết luận B kém chỉ từ vài episode chưa học đủ.

## 2. Phương pháp B ban đầu và các phiên bản đã có

Ý tưởng gốc:

```text
Khởi tạo YOLO từ kiến trúc YAML + tham số ban đầu
→ train một segment bằng SGD
→ quan sát train/validation và tham số hiện tại
→ agent chọn thay đổi hyperparameter
→ tiếp tục chính detector đó trong segment kế tiếp
→ reward từ validation
→ lặp đến horizon/early stop
→ cập nhật policy qua nhiều episode khởi tạo detector mới.
```

Một **episode** là một lần train detector từ đầu; không phải một epoch, một action, hay train lại từ pretrained. Trọng số detector liên tục trong episode. Giữa episode khởi tạo detector mới, còn policy có thể tiếp tục học. SGD cập nhật trọng số YOLO theo detection loss; PPO cập nhật trọng số controller theo reward. PPO không thay SGD để trực tiếp tối ưu từng trọng số detector.

| Giai đoạn | Thiết kế | Ý nghĩa và giới hạn |
|---|---|---|
| Legacy ban đầu | A cố định so với B đổi bốn tham số theo segment; các worker/adapter và reward tổng hợp | Log 10 epoch, 3 episode trong ảnh người dùng thuộc luồng thăm dò cũ. Điểm thấp ở scratch horizon ngắn không chứng minh B thất bại. Không dùng log đó làm kết quả cuối của protocol mới. |
| Quality v3 | Trainer liên tục, recipe và checkpoint đối chứng KB1; 300 epoch, 128 trial A/128 episode B; tách tuning/final seed | Ngân sách rất lớn nhưng 128 là lựa chọn thiết kế, không có định lý đảm bảo số episode này đủ cho PPO. |
| Quality v4 | Search theo đợt 4, có thể mở rộng; canonical early stop; PPO rollout ngắn hơn; tuning/final đầy đủ khi chốt | Giữ được công việc đã hoàn thành và chi phí rõ hơn; bốn episode vẫn chỉ thích hợp thăm dò cơ chế. |
| Hiện tại | A–TPE riêng, bốn lượt; B và tài liệu cũ giữ để tham khảo | Script mặc định không tạo policy, không chạy episode B, tuning B hay final B. |

Mã B quality còn ở [`quality/pipeline.py`](../quality/pipeline.py), [`quality/detector.py`](../quality/detector.py), [`quality/ppo.py`](../quality/ppo.py), [`quality/state.py`](../quality/state.py); cấu hình cũ [`kb3_quality.yaml`](../configs/kb3_quality.yaml). Không nên gọi trực tiếp pipeline cũ để chạy nghiên cứu mới trước khi chốt lại thiết kế và ngân sách.

### Chi tiết mục tiêu B legacy còn trong repository

Legacy `reward.py` dùng gain giữa **hai metric hiện tại liên tiếp**, không phải quality best-gain:

```text
gain = 0.45 ΔmAP50–95 + 0.25 Δrecall + 0.20 ΔAPsmall + 0.10 ΔmAP50
reward = gain − 0.01 time_ratio − 0.10 overfit_signal
         − 0.02 clipped_actions + 0.20 terminal_mAP50–95
failure reward = −1.
```

Time ratio chỉ có khi khai báo time budget. Overfit signal lấy từ mAP giảm hoặc val loss tăng khi train loss giảm; đây là heuristic, không phải chẩn đoán overfit được chứng minh. Mix nhiều metric và hệ số/phạt tùy chọn tạo mục tiêu khác với tối đa best mAP cuối; gamma=0.99 trong cấu hình legacy còn giảm trọng số thưởng muộn. Cấu hình PPO legacy có hidden size 64, LR 3e−4, GAE 0.95, clip 0.2, value 0.5, entropy 0.01 và 8 update epoch. Các hệ số không được hiểu là thông số tối ưu đã xác nhận cho YOLO.

Legacy observation dùng tiến độ, loss được nén, metric hiện tại, stale/time fraction và hyperparameter. Việc nén loss có thể mất độ nhạy nếu loss nằm ngoài vùng hữu ích; chất lượng representation cần đọc trajectory thực. Legacy có patience theo segment/no improvement và time-budget truncation; điều đó khác canonical stopping v4. Các thay đổi quality nhằm làm mục tiêu/state và liên tục trainer dễ kiểm chứng hơn, nhưng không tự chứng minh chất lượng policy tăng. Chi tiết legacy: [`envs/yolo_hpo_env.py`](../envs/yolo_hpo_env.py), [`reward.py`](../reward.py), [`kb3_default.yaml`](../configs/kb3_default.yaml).

## 3. B quality đang thực hiện cụ thể những gì

### 3.1. Khởi tạo, recipe và tính liên tục

- Đọc `checkpoint_based/<model>/args.yaml` của supervised KB1 làm recipe thật; xác minh checkpoint, kiến trúc, lớp và provenance dataset.
- Khởi tạo bằng `<model>.yaml`, `pretrained=False`, `resume=False`. Không nạp `best.pt` KB1 để train B. `best.pt` KB1 chỉ được inference cùng canonical evaluator.
- Trong một episode giữ model, SGD optimizer và momentum buffer, scheduler, EMA, AMP scaler, dataloader và gradient accumulation. Không gọi nhiều lần train độc lập với các scheduler bị reset.
- Warmup và decay lấy theo recipe KB1. Hành động `lr0` thay base LR, actual LR vẫn nhân hệ số lịch; weight decay theo quy tắc chuẩn hóa batch/nbs của trainer. Phải phân biệt giá trị đầu vào với giá trị optimizer thực dùng.
- `augmentation_strength=0.5` tái tạo augmentation KB1; điều chỉnh quanh mốc theo mapping đã định nghĩa, HSV giữ theo recipe. Close-mosaic cuối train được bảo vệ để agent không bật lại trái với lịch.
- `workers=0` phục vụ kiểm soát augmentation và RNG. OOM làm batch bị thay đổi phải fail thay vì âm thầm thay protocol.

### 3.2. Observation và action

Observation gồm 16 đặc trưng và 4 hyperparameter chuẩn hóa: tiến độ, cờ loss có dữ liệu, log train/val loss, gap, xu hướng loss, precision, AR300, mAP50, mAP50–95, AP small, biến động mAP, best đã train, stale fraction, hệ số scheduler và tham số hiện tại. AR300 là average recall theo evaluator COCO; không đồng nhất với recall tại một confidence vận hành.

| Tham số | Hành động | Biên hiện có |
|---|---|---|
| `lr0` | Nhân 0.5, 0.8, 1, 1.2, 1.5 | 0.0001–0.02 |
| `weight_decay` | Nhân 0.8, 1, 1.2 | 0.0001–0.002 |
| `momentum` | Cộng −0.02, 0, +0.02 | 0.80–0.98 |
| `augmentation_strength` | Cộng −0.1, 0, +0.1 | 0–1 |

Policy có bốn categorical head, khả năng kết hợp tối đa `5 × 3 × 3 × 3 = 135` action; chưa trừ action bị mask tại biên. Các head dùng chung representation nhưng sampling được factorize; cách biểu diễn này chưa mô hình hóa đầy đủ phụ thuộc tức thời giữa các lựa chọn. Luôn có action giữ nguyên. Mask chỉ chặn vượt biên, chưa chứng minh mọi tổ hợp bên trong biên đều an toàn hoặc tốt.

### 3.3. Reward, checkpoint và stopping

Quality reward tại segment: `r_t = 100 × (best_t − best_(t−1))`; `best` là best canonical validation mAP50–95 của **các epoch đã train**. Với gamma=1, return cộng dồn bằng `100 × (best_final − best_prefix)`. Prefix chung hoàn tất warmup và chưa do policy điều khiển; đánh giá detector ngẫu nhiên epoch 0 không được chọn làm best checkpoint.

Reward này ưu tiên checkpoint tốt nhất thay vì final epoch. Nó không trực tiếp thưởng val loss, AP small hoặc tốc độ; thay mục tiêu sang các đại lượng đó phải định nghĩa và kiểm chứng lại. Best chỉ tăng nên nhiều segment có reward=0. Đợt tăng best có thể chịu ảnh hưởng của các action trước, không chỉ action vừa chọn. Nếu horizon thay đổi do early stop, return tương ứng mục tiêu tại horizon thực tế đó.

Mỗi epoch chọn checkpoint bằng cùng canonical evaluator FP32, TF32 tắt, confidence đầu vào 0.001, NMS IoU 0.60, max_det 300, operating confidence 0.25. Dùng EMA copy, giữ nguyên RNG của train trong evaluator. Native best của Ultralytics không được trộn trực tiếp với canonical best. Native patience bị vô hiệu hóa; stopper chung kiểm soát bằng canonical mAP.

V4 search: tối đa 300 epoch, segment 5, không stop trước 100 epoch, patience 50, major improvement >0.0005; stop tại ranh segment và bảo vệ pha close-mosaic cuối. Mốc kiểm soát patience không thay việc lưu best của mọi epoch. Tuning/final v4 chạy đủ horizon để giảm lệch do search bị cắt. Early stop tiết kiệm tài nguyên nhưng có thể bỏ lỡ policy/cấu hình cải thiện muộn.

### 3.4. PPO và cách lưu trạng thái

Controller quality dùng hai lớp ẩn 128 Tanh, actor bốn head và critic; Adam LR 3e−4, gamma=1, GAE lambda=0.95, PPO clip=0.2, value coefficient=0.5, entropy coefficient=0.01, max gradient norm=0.5, 10 update epoch và target KL=0.02. V4 dùng rollout_steps=64, minibatch=32, hoàn tất episode trước khi update và force update tại ranh lưu candidate mỗi bốn episode.

Lưu policy model, optimizer, RNG riêng, rollout chưa update và metadata để tiếp tục search. Seed policy init khác vai trò seed detector mỗi episode. Nếu trainer bị ngắt giữa episode, inference checkpoint hiện tại không chứa toàn bộ optimizer/scaler/dataloader/RNG; episode đó phải chạy lại. Episode đã commit không được train lại. Đây là giới hạn resume detector cần giải quyết riêng nếu muốn chống lãng phí do mất điện/driver reset.

## 4. Cơ sở lý thuyết: B có thể tốt hơn A ở đâu?

Gọi `F(h; ξ)` là chất lượng tốt nhất của detector train với hyperparameter cố định h, recipe lịch cố định và ngẫu nhiên ξ. A tìm `max_h E[F(h; ξ)]`. B tìm policy `π(a_t | o_≤t)` để chọn dãy tham số trong chính trajectory, tối ưu kỳ vọng chất lượng checkpoint trên seed/task mới.

Nếu policy class **thực sự chứa** mọi lịch cố định của A, giữ cùng recipe, init, dữ liệu, horizon và hàm đánh giá, thì nghiệm tối ưu lý tưởng của B không thấp hơn A về mặt không gian lựa chọn. Điều kiện này không tự động đúng với B hiện tại: action thay đổi tương đối, start bị cố định, có prefix, biên và mask nên B có thể không thực hiện ngay cấu hình A bất kỳ từ epoch đầu. Cũng không có đảm bảo PPO tìm được nghiệm lý tưởng với ngân sách hữu hạn. Vì vậy không thể suy ra “B chắc chắn tốt hơn A” từ việc B linh hoạt hơn.

B có lợi tiềm năng khi early/late training cần các mức LR, regularization hoặc augmentation khác nhau; seed tạo tiến độ hội tụ khác nhau; hoặc phản hồi loss/validation chứa thông tin hữu ích để quyết định. Nếu lịch linear/cosine/warmup/close-mosaic đã giải quyết tốt các pha này, hoặc observation quá nhiễu, lợi ích thêm có thể nhỏ. A vốn đã có **lịch LR và close-mosaic theo thời gian** dù base hyperparameter cố định; không nên so B với một baseline bị tắt hết scheduler để làm B có vẻ mạnh.

Một lịch thay đổi theo epoch có thể tốt nhưng chưa chứng minh cần RL hoặc phản hồi trạng thái. Phải so thêm lịch theo epoch và replay một schedule học được trên seed mới. Nếu replay tốt ngang controller, kết luận có thể là học lịch hữu ích, chưa phải lợi ích của điều khiển theo trạng thái.

### Bằng chứng nghiên cứu liên quan và phạm vi áp dụng

| Nghiên cứu | Bằng chứng liên quan | Điều không được suy ra |
|---|---|---|
| [PPO, Schulman và cộng sự, 2017](https://arxiv.org/abs/1707.06347) | Policy gradient với surrogate có clipping và nhiều minibatch update sau tương tác | Không chứng minh số episode tối thiểu hay ưu thế trên YOLO. |
| [Learning an Adaptive Learning Rate Schedule, 2019](https://meta-learn.github.io/2019/papers/metalearn2019-xu.pdf) | Controller RL đổi LR theo lịch sử train; CNN/ResNet trên Fashion-MNIST/CIFAR-10. Dùng 1.000 episode, mỗi trainee 20/25 epoch; kết quả cải thiện tùy cặp model/dataset. | Có cơ sở cho B điều khiển LR, nhưng không phải bằng chứng chỉ bốn episode YOLO đủ, hay đổi đồng thời bốn tham số sẽ tốt hơn. |
| [Hyp-RL, 2019](https://arxiv.org/abs/1906.11527) | RL chọn **cấu hình tiếp theo giữa những lần train**, đánh giá trên 50 dataset | Thuộc hướng A dùng RL ở vòng ngoài; không phải evidence cho B đổi tham số trong một lần train. |
| [Population Based Training, 2017](https://arxiv.org/abs/1711.09846) | Population vừa train, vừa khai thác bằng sao chép trọng số tốt hơn và khám phá hyperparameter | Cho thấy tối ưu lịch trong train có ý nghĩa; PBT không đồng nhất PPO và có chi phí population. |
| [Population Based Augmentation, ICML 2019](https://proceedings.mlr.press/v97/ho19b.html) | Học augmentation thay đổi theo thời gian trên bài toán phân loại | Là hướng population; cần thử transfer sang detection, không phải chứng minh trực tiếp B PPO của dự án. |

Các phân tích áp dụng cho YOLO trong báo cáo này là giả thuyết thiết kế dựa trên những cơ chế trên, không phải kết quả YOLO đã được các nguồn đó xác nhận.

## 5. Những vấn đề B cần giải quyết và lý do

| Vấn đề | Vì sao ảnh hưởng kết quả | Việc cần làm trước/ngay khi thử B |
|---|---|---|
| Observation không Markov | Hai detector có loss/mAP giống nhau nhưng trọng số, momentum buffer, EMA và gradients khác; action có thể có hậu quả khác | Xem bài toán như POMDP; thử lịch sử ngắn/RNN hoặc thêm norm/trend phù hợp, kiểm chứng ablation. Không gọi vector loss là trạng thái đầy đủ. |
| Nhiễu và reward thưa | SGD, augmentation và validation hữu hạn; nhiều best-gain bằng 0, khó gán công cho action | Ghi reward distribution, tỷ lệ zero, advantage/value error; kiểm tra reward có phân biệt schedule đối chứng. Smoothing reward phải giữ rõ mục tiêu. |
| Credit assignment dài | Tăng augmentation có thể giảm mAP ngay nhưng tốt về sau; giảm LR có thể chỉ có tác dụng sau nhiều epoch | Giữ action đủ lâu, ghi hậu quả qua nhiều segment, kiểm tra lambda/critic; không thưởng riêng một bước tăng mAP rồi kết luận action tối ưu. |
| Ít episode | Nhiều segment cùng một detector có tương quan; không thay thế nhiều trajectory độc lập | Định nghĩa ngân sách bằng episode, transition và policy update thực tế; dùng hold-out detector seed và nhiều policy-training seed khi xác nhận. |
| Bốn tham số tương tác | LR và momentum ảnh hưởng ổn định; weight decay và augmentation cùng regularize | Khởi động với LR-only quanh mốc tốt, thêm một tham số khi có evidence; tránh ngay 135 tổ hợp với vài episode. |
| Scheduler/optimizer ghi đè action | Sửa args.yaml chưa chắc optimizer dùng giá trị mới; warmup có thể ghi đè momentum/LR | Assert actual LR, momentum, normalized WD từng epoch; action sau warmup; hold-equivalence regression trên GPU. |
| Phụ thuộc batch/nbs | Batch thay đổi kéo theo WD và accumulation; “cùng WD” có thể là khác regularization thực | Khóa batch/nbs/memory policy, fail OOM, ghi optimizer thực dùng. |
| Augmentation và dataloader | Nhiều worker hoặc pipeline cache có thể vẫn dùng augmentation cũ; rebuild thay RNG | Thiết kế cập nhật tại ranh rõ ràng; kiểm tra ảnh/transform thực; giữ RNG khi hold, không rebuild tùy tiện. |
| Validation bị học quá mức | Agent xem validation lặp lại; chọn policy/epoch nhiều lần gây lạc quan | Tách controller-reward validation và policy-selection validation nếu dữ liệu cho phép; cùng protocol cho đối chứng, test chỉ sau freeze. |
| Search early stop | Truncation có thể ưu ái schedule thắng sớm, bỏ loại cải thiện muộn | Định nghĩa mục tiêu có horizon và stopping rõ ràng, kiểm tra sensitivity, xác nhận đủ horizon; không coi 100 epoch là thay thế chắc chắn 300. |
| Đối chứng thiếu công bằng | KB1 pretrained khác init/seed/horizon; native/canonical khác metric; A chưa tune tốt | Dùng scratch matched anchor và A–TPE, cùng evaluator, matched seed và horizon; pretrained KB1 trình bày riêng. |
| Policy chọn rồi đánh giá vẫn học | Dẫn đến đánh giá một policy tiếp tục thích nghi với reward thay vì policy đã chốt | Freeze trọng số policy và optimizer; vẫn được đọc observation nhưng không cập nhật từ validation/test để học policy. |
| Test leakage | Xem test rồi điều chỉnh reward/actions/budget là dùng test để thiết kế | Chốt policy/cấu hình/evaluator trước, ghi test-access audit. Hash dữ liệu test để kiểm tra toàn vẹn không phải dùng điểm test. |
| Resume không đầy đủ | Khôi phục weights-only mất SGD/EMA/scaler/RNG, không phải cùng trajectory | Chỉ tái sử dụng episode commit; muốn mid-episode resume phải lưu/restore tất cả state và kiểm tra equivalence. |
| Controller không transfer | Lịch học trên model nhỏ/horizon ngắn có thể không phù hợp model lớn/full resolution | Ghi domain, normalize tiến độ/LR, thử transfer có kiểm soát; không gọi thành công proxy là thành công full YOLO. |
| Khó quy lợi ích cho feedback | Một lịch theo epoch hoặc random schedule có thể tốt ngang PPO | Ablation observation, hold, random schedule, schedule replay và strong conventional schedules. |

Dataset hiện có 2.411 ảnh train, 479 validation, 239 test, 28 lớp. Validation cho từng lớp có thể ít mẫu; AP small/metric từng lớp càng nhiễu. Kiểm tra source-id cho thấy chưa thấy chồng lấn ID giữa split nhưng không chứng minh ảnh gần trùng nội dung đã được loại hoàn toàn. Kiểm tra split theo nguồn trước khi tăng mạnh ngân sách; nếu đổi split thì phải làm lại baseline công bằng, không giữ checkpoint cũ như cùng protocol.

## 6. Tại sao bốn episode chưa đủ để đánh giá năng lực B?

Với horizon 300, segment 5, prefix 5, một episode có tối đa `(300−5)/5 = 59` quyết định. Nếu stop ở 100, có 19. Bốn episode đem lại khoảng 76–236 transition, cùng những trajectory tương quan. Với rollout=64, cập nhật sau episode và candidate boundary của v4, đây thường chỉ là khoảng 1–2 đợt update lớn; số chính xác phải đọc log, không suy từ “10 PPO update epoch” rằng đã có thêm dữ liệu môi trường.

Mười lượt tối ưu PPO trên minibatch tái dùng dữ liệu đã có, không tạo mười episode mới. Bốn episode có thể phát hiện bug, reward toàn zero, controller vượt biên hoặc bất ổn; cũng có thể tìm được một schedule tốt tình cờ. Chúng không đủ cơ sở kết luận policy đã hội tụ, B nói chung tốt/kém hơn A, hoặc controller generalize. Không có con số 128 tự động sửa được vấn đề: cần learning curve, nhiều seed và đối chứng.

Một policy chưa train kỹ mà thắng A vẫn phải đánh giá selection bias và randomness. Ngược lại policy chưa học đủ mà thua A là evidence cho thiết kế/ngân sách đã thử, chưa phải phản chứng cho tất cả phương pháp điều khiển RL.

## 7. Chi phí có thể phát sinh

### 7.1. Tính đầy đủ ngân sách

Gọi N là số episode học policy, K số candidate policy đánh giá tuning, V số seed tuning/candidate, S số seed cuối/policy được chọn, E_i là epoch thực của episode i và H là full horizon tuning/final. Khi đánh giá một policy đã chọn:

```text
Detector runs của B = N + K×V + S
Detector epochs của B = Σ_i E_i + (K×V + S)×H
GPU/elapsed time = tổng thời gian detector + evaluator + retry + chi phí hệ thống.
```

Nếu thử R thiết kế reward/action/observation hoặc M policy-training seed, chi phí học/tuning nhân theo R/M tương ứng. Phải tính riêng số lần train đối chứng A/default/random và final/test inference. Một policy candidate không chỉ là file `.pt`: để biết chất lượng khi freeze phải train các detector mới, nên tuning tốn GPU đáng kể.

| Ngân sách minh họa cho B | Detector runs | Epoch tối đa nếu H=300 |
|---|---:|---:|
| Chỉ thăm dò 4 episode | 4 | 1.200 |
| V4 chốt sau 4: 4 học + 1 candidate×2 tuning + 3 final | 9 | 2.700 |
| V4 chốt sau 8: 8 học + 2 candidate×2 tuning + 3 final | 15 | 4.500 |
| V3: 128 học + 8 candidate×3 tuning + 5 final | 157 | 47.100 |

Cấu hình v3 cả A/B và đối chứng có thể tới 324 detector runs, 97.200 epoch/model. V4 chốt cả các nhánh ở đợt bốn có thể tới 24 runs, 7.200 epoch/model. **Các số này là lịch sử thiết kế; đợt A hiện tại chỉ có bốn detector trainings, tối đa 1.200 epoch, không chạy tuning/final bổ sung.**

### 7.2. Ước lượng trên máy hiện có và các khoản ngoài detector

Log scratch pilot yolov8n cũ ghi `22330.8147835` giây/300 epoch, khoảng 6,203 giờ. Đây là run có recipe/batch trước đây; không phải thời gian benchmark cho TPE mới, model s, hay policy B. Nếu tạm dùng 6,203 giờ/run để hình dung quy mô, bốn run khoảng 24,8 giờ; B v3 157 run khoảng 40,6 ngày liên tục; cả 324 run khoảng 83,7 ngày/model. Early stop và batch thật làm thời gian khác; cần đo trial 0 mới để lập dự toán.

Các khoản cần tính thêm:

- Native validation và canonical validation mỗi epoch: inference/NMS/COCO metrics, sao chép EMA FP32, CPU label preparation; policy nhỏ thường không phải phần tốn chính.
- FP32 evaluator, precision/backend cố định để công bằng; AMP train vẫn được dùng. Windows workers=0 có thể làm data loading thành nút thắt.
- Batch khác giữa n/s, model lớn hơn, augmentation động, tích lũy gradient, CUDA VRAM 8 GB; chạy song song detector trên một GPU có thể OOM/chậm cả hai.
- Giờ tuning controller/reward và những run thất bại; mất điện, restart hoặc driver reset có thể buộc chạy lại episode chưa commit.
- Điện và GPU thuê: `chi phí = GPU giờ × giá thuê/giờ`, hoặc `kWh = công suất trung bình đo được(kW) × số giờ`; cộng lưu trữ/truyền dữ liệu. Không đưa giá giả định làm giá thực tế.
- Disk: selected detector weights, full resume states nếu phát triển thêm, policy candidates, trajectory JSON và logs. Full resume state lớn hơn inference-only weights; cần retention sau khi xác nhận nhưng không xóa checkpoint trước khi xác minh.
- Thời gian kỹ thuật để giữ compatibility với Ultralytics callbacks/dataloader/optimizer khi phiên bản đổi; pin dependency và chạy regression trước nâng cấp.
- Test và confidence interval/bootstrap tốn inference/CPU; bootstrap ảnh phản ánh nhiễu dataset, không thay thế nhiều detector seed hay nhiều policy-training seed.

“Không ưu tiên tiết kiệm GPU” vẫn cần thiết kế thí nghiệm có khả năng trả lời câu hỏi. Một trăm episode reward sai hoặc validation leakage có thể tốn hơn mà không tăng độ tin cậy. Nên ưu tiên chất lượng thiết kế và bằng chứng trước khi mở rộng số run.

## 8. Các hướng phát triển khả thi

| Hướng | Đề xuất và lợi ích dự kiến | Rủi ro/chi phí và điều kiện dùng |
|---|---|---|
| **B LR-only có residual** | Bắt đầu từ recipe A tốt; action nhỏ nhân quanh LR của lịch gốc, có hold, giới hạn mức thay đổi và vùng ổn định | Ít chiều và cơ chế rõ hơn; có thể không cải thiện nếu lịch gốc tốt. Cần update optimizer đúng và đối chứng hold tương đương. |
| B theo pha | Warmup cố định; policy quyết định ít thời điểm hơn, giữ action đủ segment; khóa close-mosaic cuối | Giảm số quyết định và nhiễu; có thể bỏ lỡ thích nghi nhanh. Chọn thời điểm theo learning curve, không chỉ giảm horizon. |
| Lịch sử/RNN | Dùng cửa sổ loss/trend và action gần đây; RNN nếu ablation cho thấy cần | Cải thiện thông tin quan sát có thể giúp; thêm tham số và khó học với ít episode, không khắc phục mọi state ẩn. |
| Học khởi tạo controller | Bias ban đầu hướng hold/lịch hợp lý; behavior cloning từ nhiều schedule đã kiểm chứng rồi RL fine-tune | Cần dữ liệu trajectory/action/reward thực; log KB1 một schedule không cho counterfactual của action khác. Phải kiểm tra policy có còn exploration. |
| Proxy hoặc curriculum | Học controller trên model/resolution/data rút gọn rồi xác nhận nguyên dataset/full horizon | Tiết kiệm tương tác học; thay đổi dynamics, AP small và close-mosaic. Phải đưa transfer-validation/full training vào ngân sách. |
| Meta-controller | Học từ nhiều seed/model/task, đánh giá task giữ lại | Có thể amortize chi phí policy; cần nhiều dữ liệu và tách task rõ, chi phí đầu cao. Không gọi đổi vài seed cùng dataset là task generalization. |
| Offline/model-based RL | Replay/log/surrogate hỗ trợ pretraining hoặc lựa chọn thí nghiệm | Dataset bốn trial cố định thiếu coverage action/state; extrapolation và model bias cao. PPO on-policy hiện có không tự hỗ trợ offline replay. |
| A dùng RL ở vòng ngoài | Controller học cấu hình tiếp theo giữa full runs theo hướng Hyp-RL | Vẫn đạt mục tiêu RL-HPO nhưng là A-RL, khác B; chi phí dựng meta-data/model cần tính. TPE hiện tại chỉ là đối chứng. |
| PBT/PBA làm đối chứng động | Population perturb/copy hoặc augmentation schedules để kiểm tra lợi ích lịch thay đổi | Có thể hữu ích cho nghiên cứu; cần nhiều detector/state copy và GPU giờ. PBT không tự trở thành PPO, phải đặt tên đúng. |

Khuyến nghị ưu tiên khi mở lại B: **LR-only residual quanh cấu hình A tốt, observation đơn giản có lịch sử ngắn, full recipe/lịch gốc được bảo toàn**. Chỉ thêm WD/momentum/augmentation sau khi ablation cho thấy có ích và ngân sách đủ. B dùng A tốt làm điểm xuất phát **tham số**, vẫn khởi tạo detector scratch để giữ câu hỏi nghiên cứu hiện tại. Nếu muốn B fine-tune checkpoint KB1 thì đó là protocol pretrained mới, phải đổi cả đối chứng và mô tả mục tiêu.

## 9. Thiết kế đánh giá để kết quả B có ý nghĩa

1. **Khóa dữ liệu/recipe/evaluator.** Giữ train/validation/test rõ ràng; audit nguồn ảnh, classes và hashes. Nếu đủ dữ liệu, tách reward-validation khỏi policy-selection-validation theo source; lưu ý giảm kích thước validation làm nhiễu tăng. Cross-validation có thể thay thế nhưng tốn nhiều run.
2. **Xây baseline mạnh.** Scratch recipe KB1; A–TPE đã tune; lịch linear/cosine hoặc schedule theo pha có ngân sách tuning được ghi. Pretrained KB1 là mốc chất lượng thực tế riêng. Không so native KB1 CSV với canonical B.
3. **Kiểm tra causal mechanics.** Hold-action phải tương đương baseline cùng seed và scheduler; parameters thực dùng phải khớp action; không reset optimizer/EMA; eval không làm thay RNG; terminal transition và GAE đúng khi stop.
4. **Học policy và chọn candidate.** Ghi số transition/update/episode, reward, entropy, KL, gradient norm, value explained variance và boundary-hit rate. Seed học policy và seed detector có vai trò khác nhau; candidate chọn bằng seed/dữ liệu dành tuning.
5. **Freeze trước final.** Chốt checkpoint controller và hành vi stochastic/deterministic, stopping/horizon, evaluator, config/hash. Đánh giá các detector seed chưa dùng cho policy learning/selection; không cập nhật policy trong final.
6. **Đánh giá test sau khi chốt.** Test không dùng làm reward, chọn checkpoint hoặc quyết định ngân sách. Nếu thay thiết kế sau khi thấy test, phải khai báo đó là exploratory và cần test giữ lại khác cho xác nhận.
7. **Phân tích bằng bảng matched seed.** mAP50–95 chính; mAP50, AP small, per-class AP, precision/recall/F1 tại confidence vận hành và AR300 riêng; mean/std và paired delta/interval. Report best epoch, final-minus-best, val loss curve, actual epochs, tổng GPU giờ gồm policy học/tuning/final. Winner một seed không đại diện mean.

Các ablation quan trọng: hold-only; random actions có cùng biên/tần suất; LR-only so với bốn tham số; feedback đầy đủ so với chỉ progress; lịch replay đã học so với feedback controller; reward best-gain so với biến thể giữ cùng mục tiêu; có/không early stop. Không chạy toàn bộ tích Descartes ngay: ưu tiên kiểm tra nào phân biệt rõ nguyên nhân thất bại trước.

Hai kiểu công bằng nên report riêng: (a) chất lượng detector cuối với recipe/horizon/seed tương ứng, trong đó khai báo chi phí học policy; (b) chất lượng đạt được dưới tổng ngân sách GPU bằng nhau giữa các phương pháp. Không so bốn run A với hàng trăm episode B mà giấu chi phí policy training. Đồng thời chi phí học policy có thể được phân bổ nếu thực sự tái sử dụng cho nhiều detector/task, nhưng phải chứng minh transfer trước.

## 10. Lộ trình nghiên cứu và điều kiện chuyển bước

| Bước | Công việc | Evidence cần có để tiếp tục |
|---|---|---|
| 0 — A hiện tại | Bốn trial A–TPE, đọc toàn bộ curves/cost, so anchor và pretrained reference | Recipe/evaluator hoạt động, không lỗi init/metric, xác định plateau và độ nhạy LR/WD. Bốn trial không đủ tuyên bố tối ưu. |
| 1 — thiết kế B nhỏ | LR-only residual, giữ schedule; dự toán học/tuning/final và seed; định nghĩa tiêu chí thực tế | Contract cụ thể, phân biệt proxy/full-horizon, không tự đặt 128 là “chuẩn lý thuyết”. |
| 2 — kiểm tra cơ chế | Tiny synthetic CPU/GPU, hold equivalence, actual optimizer/EMA/RNG, stop và resume | Các gate pass, action ảnh hưởng đúng và giữ nguyên thực sự tương đương. Chưa được tính là kết quả khoa học. |
| 3 — pilot B | Ít episode để đo reward/update, trạng thái và chi phí; checkpoint controller riêng | Reward phân biệt hành vi, policy không collapse/saturate, có dấu hiệu cải thiện trên tuning seed. Không đòi pilot chứng minh superiority. |
| 4 — tăng có căn cứ | Tăng ngân sách theo learning curve; chọn checkpoints trên tuning, lặp seed học policy khi khả thi | Lợi ích đủ lớn và ổn định so strong baseline; có ablation để giải thích, không chỉ training reward tăng. |
| 5 — xác nhận | Policy freeze, held-out detector/task, full horizon phù hợp, test sau chốt | Mean/paired delta và chi phí được report, dấu hiệu overfit/variance minh bạch. |

Trước pilot cần chốt ngưỡng cải thiện có ý nghĩa thực tế, ngân sách tối đa và điều kiện ngừng để sửa thiết kế. Nếu reward gần như toàn zero, actual action bị scheduler ghi đè, feedback không hơn progress-only, hoặc tuning score không cải thiện trong những đợt đã định, phải kiểm tra cơ chế/đối chứng thay vì tự động tăng GPU. Khi tăng ngân sách sau nhiều lần xem validation, ghi số vòng quyết định và không xem đó là đánh giá độc lập.

## 11. Trạng thái bằng chứng và những việc chưa làm

Đã có regression cho tính liên tục, SGD/EMA, scheduler/hold và evaluator; có synthetic CPU/GPU gates trong `tests/test_quality_ultralytics.py`, cùng tests protocol/state/PPO/staged search. Các gate chứng minh hợp đồng kỹ thuật trên các trường hợp đã kiểm tra, không chứng minh B học tốt trên pest dataset hay trên mọi phiên bản Ultralytics.

Scratch pilot cũ 300 epoch đạt best canonical mAP50–95 khoảng 0,37666 tại epoch 255, final khoảng 0,36796; checkpoint pilot đã được xóa theo yêu cầu trước đây, logs giữ lại. Đây là pilot scratch, **không phải kết quả B**; chỉ hỗ trợ nhận xét rằng scratch có thể hội tụ muộn và dùng 10 epoch không đủ đánh giá. Cấu hình khác đợt hiện tại nên không dùng thay matched anchor mới.

Chưa có bằng chứng đủ về learning curve PPO dài, ổn định giữa policy-training seed, transfer giữa n/s/model family, thắng strong A–TPE, hay generalization test của B đã freeze. Chưa có mid-episode full-state resume được xác nhận. Chưa có đủ offline action coverage để train controller chỉ từ KB1/A logs. Những khoảng trống này phải được ghi trong mọi báo cáo kết quả B tương lai.

## 12. Tài liệu, mã và dữ liệu cần giữ để thử lại sau

- Mã quality B và pin môi trường; các tests trong `tests/test_quality_*.py`.
- Cấu hình B v4 và lịch sử review: [quality protocol](KB3_QUALITY_PROTOCOL.md), [review staged search](KB3_STAGED_SEARCH_REVIEW.md), [review KB1-start](KB3_KB1_START_REVIEW.md), [compute/correctness review](KB3_COMPUTE_AND_CORRECTNESS_REVIEW.md). Đây là lịch sử, không phải entry point mặc định hiện tại.
- KB1 supervised `args.yaml`, `best.pt`, dataset provenance. Không xóa checkpoint đối chứng.
- A–TPE `manifest.json`, `starting_train.yaml`, `train_contract.json`, trial specs, `epochs.json`, `result.json`, selected weights, `evaluation.json`, comparison và curves. Logs cố định của A không chứa hậu quả của các action B chưa thực hiện.
- Nếu mở lại B, dùng output/protocol mới; không resume policy legacy/v3/v4 như thể có cùng state/reward/evaluator. Ghi lại seed rules, candidate selection và chi phí thực tế trước khi chạy.

## 13. Nguồn chính

- [PPO — Schulman et al.](https://arxiv.org/abs/1707.06347)
- [RL learning-rate controller — Xu et al., bản paper](https://meta-learn.github.io/2019/papers/metalearn2019-xu.pdf)
- [Hyp-RL — Jomaa et al.](https://arxiv.org/abs/1906.11527)
- [Population Based Training — Jaderberg et al.](https://arxiv.org/abs/1711.09846)
- [Population Based Augmentation — Ho et al., ICML 2019](https://proceedings.mlr.press/v97/ho19b.html)

Kết luận áp dụng cho dự án: B có cơ sở nghiên cứu thực sự, nhất là điều khiển LR; lợi ích của B trên YOLO vẫn là giả thuyết cần kiểm chứng. A–TPE hiện tại xây đối chứng và hiểu dynamics trước. Khi có dữ liệu đó, một B nhỏ, kiểm soát được và đánh giá đúng sẽ có giá trị hơn việc mở ngay không gian bốn tham số với ngân sách episode rất lớn.
