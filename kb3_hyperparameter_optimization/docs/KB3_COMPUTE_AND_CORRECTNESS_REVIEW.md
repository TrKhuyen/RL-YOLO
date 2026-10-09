# Review KB3 trước khi chạy thêm thí nghiệm dài

**Cập nhật sau review:** người dùng ưu tiên chất lượng kết quả. Luồng mới và
các kiểm chứng implementation nằm trong [KB3_QUALITY_PROTOCOL.md](KB3_QUALITY_PROTOCOL.md),
dùng `run_quality_kb3.sh`. Các snapshot/run cũ bên dưới được giữ để audit;
khuyến nghị tiết kiệm compute trong review lịch sử không quyết định protocol mới.

Cập nhật 09/10/2026: quality v2 dùng supervised KB1 best.pt làm đối chứng
pretrained, args.yaml của cùng model làm recipe và bỏ lượt train pilot mặc định.
A/B vẫn scratch; xem protocol hiện tại ở liên kết trên.

Ngày review: 08/10/2026, múi giờ Asia/Bangkok.

Phạm vi: đọc source, phiên bản trainer đang cài, JSON/CSV/checkpoint manifest của
run `20261008T014055204488Z`, đối chiếu tài liệu gốc. Không chạy training,
không sửa trainer, không dừng process đang chạy trong lần review này.
Các số liệu là snapshot khi đọc; run vẫn đang tiến triển.

Kết luận: nền tảng train từ đầu và RL điều khiển hyperparameter hoạt động.
Chưa nên tiếp tục mở rộng ma trận 5 model hoặc tăng tất cả lên 100 epoch.
Cần chốt protocol, sửa các lỗi chọn checkpoint/state/reward, làm recovery và
đo hiệu năng trước; sau đó chỉ dùng một model nhỏ để kiểm chứng giả thuyết.
Không có cơ sở đảm bảo PPO sẽ tốt hơn HPO chỉ bằng việc tăng epoch.

## 1. Vì sao một lệnh chạy hơn nhiều giờ

`run_all.py` mặc định chạy 5 kiến trúc, 5 phương pháp, 3 search trial/episode,
3 seed đánh giá, 10 epoch mỗi detector. Một model thực tế gồm:

| Phương pháp | Search / học policy | Đánh giá lại | Tổng detector | Tổng epoch |
|---|---:|---:|---:|---:|
| Default | 0 | 3 | 3 | 30 |
| Random Search | 3 | 3 | 6 | 60 |
| Random Schedule | 0 | 3 | 3 | 30 |
| Bandit | 3 | 3 | 6 | 60 |
| PPO | 3 | 3 | 6 | 60 |
| **Tổng / model** | **9** | **15** | **24** | **240** |

Cả 5 model là **120 detector / 1.200 epoch**, chưa có Optuna. 10 epoch không
phải ngân sách toàn script. Default này quá lớn cho kiểm tra ban đầu và quá ít
transition cho từng policy PPO: chi phí trải trên nhiều architecture/phương pháp.

YOLOv8n đã hoàn thành 24 detector. Tổng elapsed trong worker:

| Phương pháp | Giờ ghi trong worker, gồm search và đánh giá |
|---|---:|
| Default | 0,538 |
| Random Search | 0,954 |
| Random Schedule | 0,642 |
| Bandit | 1,141 |
| PPO | 1,168 |
| **Tổng** | **4,443** |

Đây không phải toàn wall time: import Python/framework, một phần setup initial
evaluation, khởi tạo YOLO và orchestration không được tính đầy đủ. Nếu mọi model
đều mất như v8n thì riêng elapsed đã khoảng 22,2 giờ; đây là minh họa theo giả
định, không phải dự báo, vì model/hardware load khác nhau. Các run v8s đã lưu có
thời gian khác nhau đáng kể; cần profiling để xác định nguyên nhân.

Với 15 adaptive episode/model, mỗi episode gọi 1 process initial evaluation và
5 process train-segment. Cộng 9 fixed run là **99 worker process/model**.
Ngoài native validation mỗi epoch, worker luôn canonical-evaluate last và best
ở cuối segment/full run. Theo cấu hình mặc định, một model có khoảng 183 lượt
canonical evaluation, cộng 240 lượt native validation trong train và 15 lượt
native validation epoch 0; số canonical giả định best tồn tại, như các run đang đọc.

Snapshot 100 response có **77 response best_epoch = final_epoch**. Hai lần
canonical evaluation của cùng epoch có thể gộp sau khi xác nhận cùng weights.
Cache best theo content/epoch + evaluator/dataset hash; không cache bằng tên
`best.pt`, vì file này bị ghi đè. Cache exact duplicate không cần thay training.

## 2. Điều đã xác nhận và điều chưa được chứng minh

Đã xác nhận:

- Kiến trúc YAML và `pretrained=False`; mỗi episode/trial bắt đầu độc lập theo seed.
- LR trong các response đã đọc khớp các nhóm LR ở CSV: **0 mismatch**.
- Checkpoint có live model, optimizer, EMA đủ precision và scheduler; worker có
  logic restore, ngoài optimizer/scaler/EMA upstream.
- Policy evaluation dùng `deterministic=True, learn=False`; seed 101–103 khác
  seed học policy 42–44 trong run hiện tại.
- Reward chỉ nhận train/validation; run-all chưa đánh giá test để chọn phương pháp.
- Không thấy giao nhau của source ID theo quy tắc suffix augmentation hiện có
  giữa train/valid/test. Đây không phải kiểm tra near-duplicate bằng nội dung ảnh.
- Các split có đủ 28 class; số annotation từng class không cân bằng, đặc biệt
  test có class rất ít. Khi báo cáo cần nêu hạn chế và xem metric từng class.

Chưa chứng minh:

- Resume trajectory tương đương chạy liên tục: model state có restore nhưng
  worker tái tạo RNG/dataloader mỗi segment.
- Initial evaluation và first training có trọng số khởi tạo bằng nhau tuyệt đối.
  Cần test state hash ở thời điểm sau setup, trước update đầu.
- Augmentation đã áp đúng các transform thực tế, chứ không chỉ đúng `trainer.args`.
- PPO học điều khiển có ích trên dữ liệu thật; test train thành công không chứng
  minh thuật toán có lợi hơn lịch cố định.
- Chạy 10 epoch đã hội tụ; loss/mAP hiện tại chưa cho phép khẳng định điều đó.

Integration test hiện tại có 2 ảnh/split, ảnh 64px, 2 epoch; kiểm tra nc, epoch,
optimizer parameter, scheduler base LR và EMA update count. Nó không so từng
tensor optimizer/EMA/RNG trước-sau boundary, không có equivalence control và
không chứng minh throughput/GPU convergence trên ảnh 640px.

## 3. Các việc bắt buộc trước lượt train nghiên cứu kế tiếp

### P0 — Cứu artifact và thống nhất cách chọn checkpoint

Source: `adapters/ultralytics_worker.py` và `envs/yolo_hpo_env.py`.

Worker chọn native best rồi canonical-evaluate. Environment ưu tiên
`best_metrics` và bỏ qua candidate final dù final canonical score cao hơn.
Snapshot có **9/100 response** gặp final mAP canonical cao hơn reported best.
Ví dụ PPO seed 44: reported best epoch 9 = 0,0216558; final epoch 10 = 0,0244691.

Việc cần làm:

1. Chốt trước metric chính và các epoch đủ điều kiện chọn checkpoint.
2. Chọn checkpoint theo chính evaluator đó. Để tiết kiệm, có thể chỉ cho phép
   chọn ở segment boundary cho mọi phương pháp, và canonical-evaluate một lần
   mỗi boundary. Khi đó phải gọi đúng là best trong tập boundary đã đánh giá,
   không gọi là best trên mọi epoch.
3. Với run cũ, chấm các checkpoint **còn tồn tại**: last, native best,
   selected_best; sửa bảng hậu kiểm trong thư mục mới. Không sửa JSON gốc.
4. Lưu immutable checkpoint + metric + epoch/hash để trỏ metric vào đúng weights.

Tại sao: ranking khác evaluator làm lựa chọn checkpoint sai; nhiều lượt chấm
cùng weights chỉ tăng chi phí. Sửa hậu kiểm các file còn lưu chỉ cần inference.

Giới hạn phục hồi: response cũ thường trỏ cùng `last.pt`/`best.pt`, các file này
đã bị ghi đè. Một score epoch 6 còn trong JSON không chứng minh weights epoch 6
còn tồn tại. Không gán score epoch 6 cho last epoch 10. Nếu weights đã mất thì
không thể phục hồi chính xác checkpoint đó chỉ từ log.

### P0 — Resume và không chạy lại phần đã hoàn thành

Source: `run_all.py`, `train_agent.py`, `adapters/command.py`.

Run-all luôn tạo run mới, từ chối output nonempty và không có resume/skip kế hoạch
toàn pipeline. Policy có resume ở ranh giới episode, nhưng orchestration chưa dùng
nó; command adapter reset epoch về 0 và không có recovery một episode dở.

Cần manifest theo stage/seed, trạng thái completed/in-progress/failed, checkpoint
và next epoch, atomic JSON, kiểm tra hash config/data/code/evaluator trước khi
skip/resume. Một worker dừng sau CSV nhưng trước response/state JSON có thể để
epoch checkpoint và manifest lệch nhau; recovery phải đối chiếu checkpoint thật.
Policy cần resume optimizer/RNG/counter và rollout chưa hoàn thành nếu muốn giữ
chính xác episode dở. Nếu chưa hỗ trợ điều đó, chỉ cam kết resume episode boundary.

Không chỉ thêm `if evaluation.json exists: skip`: file có thể incomplete, sai
protocol hoặc chứa detector failed. Không đổi code giữa lúc run cũ tiếp tục sinh
worker mới; run đang chạy dùng source mới ở các child process sau đó.

### P0 — Freeze protocol và báo chi phí trước khi chạy

Cần ba chế độ riêng: check-only, pilot một model, final. Default không tự chạy
all architecture/method. Trước launch in tổng detector, epoch, số decision,
ước lượng wall time từ log thật và giới hạn ngân sách người dùng đặt.

Manifest cần data split/hash, source hash gồm thay đổi chưa commit, phiên bản
Torch/Ultralytics thực cài, model YAML, initialization seed, image size,
batch/nbs, optimizer, warmup/scheduler/precision, observation/reward/evaluator
version. Freeze version giúp tránh silent policy incompatibility và so sánh
các run khác protocol như thể giống nhau.

### P1 — State PPO giữ được thông tin loss

Source: `envs/yolo_hpo_env.py:81`.

`tanh(7)` float32 khoảng 0,9999983; `tanh(10)` và `tanh(70)` đều bằng 1,0.
Trong trajectory thật, loss inputs từng đều thành 1,0. PPO khó phân biệt loss
giảm hay tăng qua các state này.

Đề xuất `log1p(loss)` rồi chuẩn hóa bằng thống kê chỉ lấy từ pilot/training,
cùng trend/relative change và gap đã scale. Không dùng eval/test để fit normalizer.
Lưu normalizer cùng policy, freeze khi đánh giá; xử lý loss epoch 0 chưa có
train loss bằng validity flag hoặc đo một train-eval batch không update, không
để 0 giả tạo như một loss thật. Chỉ thêm feature khi test cho thấy cần.

Tại sao: đây là thông tin đầu vào của controller. Đổi encoder/normalizer làm
policy cũ không còn cùng semantics, dù số chiều observation không đổi. Cần
policy mới hoặc migration có kiểm chứng; không load rồi xem là resume đúng.
Detector weights cũ vẫn giữ giá trị để inference/diagnostic.

### P1 — Reward khớp mục tiêu lựa chọn và không bị penalty lấn át

Source: `reward.py`, `configs/kb3_default.yaml`.

HPO tối ưu mAP50–95; PPO tối ưu hỗn hợp delta mAP/AR300/AP-small/mAP50 và penalties.
Đây có thể là lựa chọn multi-objective hợp lệ, nhưng không phải cùng objective
nếu báo cáo chỉ xếp hạng bằng mAP50–95. `recall` của canonical hiện là AR300,
khác operating recall tại confidence 0,25; tên trong báo cáo phải nói rõ.

Ở PPO eval seeds 101/102/103, tổng metric_gain lần lượt khoảng
0,05294 / 0,02624 / 0,04042; clipping cost đều 0,06. Policy tăng momentum tới
0,98 rồi tiếp tục tăng trong 3 segment. Reward âm ở đây không có nghĩa detector
ngừng cải thiện; penalty có thể lớn hơn phần cải thiện.

Chốt một score J chung cho HPO/RL/checkpoint. Ưu tiên mAP50–95 làm mục tiêu chính
trong phiên bản đơn giản, AR300/AP-small báo cáo bổ sung; nếu dùng composite,
phải giữ trọng số trước search và dùng composite cho cả A/B. Mask action không
còn hợp lệ ở bounds, giữ action hold; PPO phải dùng cùng mask khi sample và khi
tính log probability/ratio lúc update. Không tự đổi mask chỉ ở `act()`.

Với horizon hữu hạn và mục tiêu quality cuối, `r_t = J_{t+1} - J_t`, `gamma=1`
cho tổng reward bằng `J_T - J_0`, khi bỏ penalties/bonus. Nếu chọn best eligible
checkpoint là mục tiêu, J dùng running-best của các boundary đủ điều kiện và
state chứa giá trị đó. Đây là lựa chọn thiết kế cụ thể, không phải khẳng định
reward hiện tại sai thuật toán. Nếu giữ discount khác 1 và muốn shaping bảo toàn
mục tiêu terminal, phải xử lý potential/terminal đúng theo [Ng–Harada–Russell](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).

`max_seconds=None` khiến time_cost luôn 0. Với mục tiêu quality dưới ngân sách
epoch cố định, có thể bỏ time penalty và báo cost riêng; không cần thêm một
truncation thời gian chỉ để làm term có tác dụng. Nếu bật time truncation, phải
chốt đó là terminal của bài toán hay cutoff cần bootstrap value.

### P1 — Đủ dữ liệu on-policy, không lặp update để thay cho dữ liệu mới

Source: `agents/ppo_agent.py`, `runner.py:54`.

Run hiện tại: 3 episode × 5 segment = **15 transition/model**; mỗi lần update
lặp 8 vòng trên 5 transition. Entropy episode cuối 4,8819, gần max log(135) =
4,9053; không đủ để kết luận controller đã học tốt. Negative PPO loss không phải
chỉ số detector tốt hơn.

[PPO gốc](https://arxiv.org/abs/1707.06347) phân biệt lấy dữ liệu tương tác và tối
ưu surrogate nhiều lần. Lặp optimize không sinh thêm transition. Cần rollout
batch lớn hơn từng episode 5 bước, có minibatch, log KL/clip fraction/value fit,
giới hạn update khi policy thay đổi quá mạnh. Không có định lý yêu cầu đúng một
số episode cố định cho dataset này; số lượng phải do pilot và ngân sách quyết định.

Thu gọn action ở pilot: LR trước, augmentation sau; momentum/weight decay giữ
cố định hoặc chọn bằng HPO. 4 chiều hiện có 135 tổ hợp; thêm transition cho một
action space lớn trên cả 5 architecture tốn hơn tập trung kiểm chứng 1–2 chiều.
Bandit là đối chứng ít dữ liệu, không phải chứng minh adaptive state policy tốt.

Không đem logs nhiều policy/phiên bản cũ làm replay cho PPO thông thường rồi coi
là on-policy. Logs có thể dùng debug state/reward, phân tích schedule và thiết kế
test. Normalizer mới không cho phép tái sử dụng rollout cũ một cách tự động.

### P1 — Giữ trainer sống trong một episode

Hiện một segment = một process mới. Hướng khuyến nghị là native in-process
adapter: mỗi episode tạo detector/optimizer một lần, action áp tại boundary,
trainer tiếp tục epoch kế tiếp. Dataloader và RNG không restart mỗi segment.
JSON worker vẫn giữ cho external backend và test contract nếu cần.

Lưu checkpoint recovery ở boundary; chọn best và ghi metadata trong callback
sau validation. [Ultralytics callbacks](https://docs.ultralytics.com/usage/callbacks/)
phân biệt `on_train_epoch_end` trước validation và `on_fit_epoch_end` sau validation.
Đổi hyperparameter cần đúng thời điểm so với scheduler và transform dataset.
Augmentation phải rebuild/cập nhật transform ở boundary theo API đang cài,
không chỉ set args sau dataloader đã xây xong. Tách RNG của policy, training và
evaluation; constructor/model load cho evaluation không được âm thầm tiêu thụ
RNG training hoặc đổi model live sang inference head. Dataloader canonical hiện
có generator riêng, nhưng vẫn cần kiểm chứng toàn evaluation callback.

Nếu giữ subprocess backend, thêm fixed segmented control cùng boundary, restore
RNG/generator và kiểm chứng continuity. Không gọi nó tương đương continuous
training nếu chưa có test. Không cần giả định bitwise equality AMP trên mọi GPU;
so đúng state tại checkpoint và đặt tolerance có giải thích cho trajectory.

### P1 — Baseline train từ đầu phải là một recipe có sức cạnh tranh

Hiện Default = initial values trong search config, SGD, warmup=0, lrf=1,
close_mosaic=0, FP32, nbs=batch. Nó không phải default training recipe Ultralytics.
Tên cần rõ là controlled fixed configuration.

Tắt scheduler/warmup giải quyết overwrite action, nhưng không chứng minh là
recipe tối ưu cho scratch convergence. Chốt một warmup và base LR schedule
chung nếu pilot cho thấy cần; RL điều khiển multiplier của schedule hoặc một
effective LR có semantics rõ ràng. Áp cùng recipe cho A/B, log requested/applied.
Khi A nói hyperparameter cố định, một scheduler đã định trước vẫn có thể làm
effective LR thay đổi; cần phân biệt cấu hình cố định với LR hằng số.

Không đổi optimizer, nbs, loss gains hoặc augmentation cho riêng RL rồi quy toàn
bộ cải thiện cho policy. Nếu đổi effective batch, phải xử lý gradient accumulation
và scaling LR/weight decay công bằng, không đơn giản tăng batch để lấy speed.

### P2 — Đo throughput rồi bật tối ưu, không đoán tốc độ

Worker hiện `amp=False`, `workers=0`, batch=8, imgsz=640, deterministic=True.
GPU đang dùng là RTX 4060 Laptop 8GB; một snapshot utilization=10% không chứng
minh cả run CPU-bound hoặc bị treo. Cần timings train/data/validation/checkpoint/
setup để xác định bottleneck.

Thử AMP và workers nhỏ trên cùng microbenchmark có giới hạn số batch; kiểm tra
loss hữu hạn, GPU memory, throughput và resume scaler. [PyTorch AMP](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
mô tả cả tăng tốc lẫn trường hợp CPU-bound không được lợi; không hứa hệ số speedup
cho dataset này. [Ultralytics train options](https://docs.ultralytics.com/modes/train/)
mô tả workers/cache/AMP; worker Windows có chi phí spawn nên không mặc định tăng
lên 8 mà không đo. Không giảm imgsz hoặc dùng subset cho final chỉ vì chạy nhanh:
đó là đổi protocol và có thể đổi AP-small, ranking và state distribution.

Checkpoint hiện gọi upstream save rồi load và save lại kèm training state mỗi
epoch. Có thể serialize một lần, lưu boundary recovery và best eligible một
cách atomic. Chỉ đánh giá candidate đang đổi; gộp best=last và cache best cũ.
Đây là việc có thể giảm overhead mà không thay objective.

## 4. Protocol khả thi, giữ cả KB3-A và KB3-B

1. **Dừng mở rộng ma trận**: chọn YOLOv8n làm model chính để chứng minh A/B;
   kiến trúc thứ hai chỉ thêm sau khi implementation/protocol đã khóa. Việc chọn
   v8n ở đây theo chi phí, không dùng test để chọn model thắng.
2. **Kiểm chứng tĩnh và toy trước**: state float32 phân biệt loss; reward/return
   algebra; action mask/logprob; selected checkpoint đúng evaluator; pipeline
   resume/skip/budget. Các bước này không cần train dataset thật.
3. **Micro integration đủ cụ thể**: một run nhỏ deterministic trên vài batch,
   so uninterrupted và checkpoint/resume cùng schedule; kiểm tra model/EMA/
   optimizer/scaler/scheduler, actual transform, applied LR/momentum/WD. Thử
   ngắt và recovery. Giới hạn batch/wall time, không chạy cả ma trận để test.
4. **Một baseline pilot có ích**: dùng một seed scratch để quan sát curve tới
   horizon đã khai báo từ đầu, ghi epoch milestones và latency. Chọn horizon
   nghiên cứu theo convergence/chi phí; không mặc định 10 đủ hoặc 100 tốt nhất.
   Nếu cho phép tiếp tục từ milestone, horizon tối đa và scheduler đã định trước.
5. **Search tập trung**: giữ A0 fixed và A1 Random Search/TPE; B0 random schedule
   để phân biệt hiệu quả học, B2 PPO là nhánh chính. Bandit tùy chọn đối chứng;
   thêm nó có lý do dữ liệu ít, không bắt buộc chạy mọi phương pháp trên mọi model.
   Random Search là baseline có cơ sở từ [Bergstra–Bengio](https://jmlr.org/papers/v13/bergstra12a.html).
6. **Ngân sách search tính cả cost**: cùng epoch budget và cùng evaluator cadence
   chưa đủ; báo cả wall/GPU time, số validation, lỗi, epoch thực dùng. Đặt budget
   trước, không cấp thêm cho phương pháp vì nhìn thấy nó đang kém trên eval.
7. **Final evaluation sau khi khóa**: chạy cấu hình/policy đã khóa trên các seed
   mới, cùng horizon/data/recipe/checkpoint selection. Ba seed detector không
   tương đương ba lần train policy độc lập. Báo giới hạn nếu chỉ có một policy
   training seed; nhiều-seed reporting quan trọng theo [Henderson et al.](https://arxiv.org/abs/1709.06560).
8. **Test sau cùng**: chỉ đánh giá phương pháp đã chọn, cùng evaluator và metric
   từng class; không dùng test làm reward hoặc thay reward sau khi xem test.

Multi-fidelity/successive halving là phương án giảm search cost theo
[Hyperband](https://jmlr.org/papers/v18/16-558.html), không phải sửa mặc định hiện tại.
Nó đòi hỏi checkpoint promotion đúng, horizon/schedule tương thích và chấp nhận
rủi ro config học chậm bị loại. Policy học trên ít epoch/ảnh nhỏ/subset không tự
được coi là policy cho horizon/dataset đầy đủ; phải có bước kiểm chứng transfer.
Không chuyển KB3-B thành HPO chọn một config duy nhất nếu đề tài yêu cầu adaptive RL.

## 5. Cái gì giữ lại, cái gì phải làm mới

| Thay đổi / mục đích | Artifact dùng lại | Có cần train detector lại? |
|---|---|---|
| Sửa bảng/cost/metadata | JSON, CSV, timestamps | Không |
| Chấm lại best/last còn lưu | Checkpoint thật còn tồn tại | Không; chỉ inference |
| Phân tích state/reward/action | Raw metrics và trajectory | Không |
| Test normalizer/PPO/mask/recovery | Toy data, CPU tests, vài batch | Không cần full training |
| Resume policy cùng protocol ở episode boundary | Policy + optimizer/RNG + meta | Giữ các episode đã hoàn thành; chỉ học thêm |
| Đổi state/reward/action space | Detector cũ làm pilot/candidate | Policy cần phiên bản mới; không xóa detector cũ |
| Tiếp tục detector với mục đích pilot | last với training state | Có thể giữ prefix, cần adapter extension/test; không gọi là final fresh run |
| So sánh final protocol khác recipe/horizon | Run cũ làm pilot riêng | Chỉ retrain các phương pháp/seed final đã chọn, không toàn ma trận cũ |

**Không hứa resume 10 → 100 epoch chỉ bằng CLI**: trainer đang cài lấy `epochs`
từ checkpoint khi check_resume, không cho overrides epochs qua whitelist, rồi
assert `start_epoch < epochs`. Một last đã xong 10/10 có thể bị từ chối. Cần
thiết kế extension rõ ràng, kiểm thử bằng checkpoint nhỏ và ghi lịch cũ/mới.
Đổi horizon sau khi policy đã học còn đổi observation progress và bài toán RL.

Run hiện tại có giá trị pilot, tài nguyên recovery và đo chi phí. Checkpoint còn
lưu có thể cứu được lựa chọn trong tập còn tồn tại; không đủ khôi phục mọi epoch
đã bị overwrite hoặc biến policy cũ thành policy đã học theo state mới.

## 6. Điều kiện cho phép chạy dài lần kế tiếp

- [ ] Một protocol/manifest có version, hash và một kiến trúc chính.
- [ ] Kế hoạch in chính xác số detector/epoch/decision và ngân sách wall time.
- [ ] Skip/resume không train lại completed stage; test interruption thành công.
- [ ] State loss có range/trend hữu ích ở float32, normalizer frozen khi evaluate.
- [ ] Objective HPO/RL/checkpoint nhất quán; reward scale được kiểm tra bằng log.
- [ ] PPO toy state-dependent task học được; KL/clip/value fit được ghi.
- [ ] Mask action đúng trong sample và update; hold luôn hợp lệ.
- [ ] Actual optimizer/augmentation được kiểm tra tại boundary.
- [ ] Live training state và recovery được kiểm chứng, không chỉ có key trong file.
- [ ] Canonical evaluation không lặp cùng checkpoint; checkpoints có identity rõ.
- [ ] AMP/workers chỉ chọn sau microbenchmark và validation ổn định.
- [ ] Baseline pilot hỗ trợ horizon chọn; chưa dùng eval/test để tune reward.
- [ ] Search cost tách final retrain cost; cùng seeds/protocol khi so final.

Các thay đổi nên triển khai trong một lượt có checklist này, freeze rồi mới
launch. Nếu một gate thất bại, sửa ở toy/micro test trước khi dùng thêm giờ GPU.
Run đang chạy không được âm thầm trộn source/protocol mới. Khuyến nghị chốt
artifact ở boundary trước khi chuyển sang phiên bản kế tiếp; chưa có cơ chế
graceful-stop đã kiểm chứng nên review này không tự dừng run của người dùng.
