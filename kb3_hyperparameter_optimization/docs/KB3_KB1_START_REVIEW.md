# Review và validation: YAML KB1 làm điểm xuất phát, weights KB1 làm đối chứng

Ngày: 09/10/2026. Protocol: `kb3_quality_v3_kb1_start`.

## Kết luận về logic

Luồng hiện tại đáp ứng yêu cầu: từng model lấy recipe thực từ supervised KB1,
HPO thử cấu hình bắt đầu từ recipe đó, PPO bắt đầu mỗi episode bằng recipe đó
và đổi tham số trong train. Best checkpoint KB1 là đối chứng chỉ đọc. Best
checkpoint A/B được chấm cùng evaluator và so với KB1 tương ứng.

KB3 vẫn scratch theo mục tiêu đã thống nhất. Đối chứng KB1 là pretrained, nên
so sánh với KB1 đo chất lượng cuối của các cách làm; không quy toàn bộ chênh lệch
cho HPO/RL. Nhóm default scratch cùng recipe/horizon/seeds phục vụ phép so sánh
riêng tác dụng tối ưu tham số. Không nạp supervised best KB1 để tiếp tục train.

## Các điểm được kiểm tra và sửa

| Điểm | Quy tắc thực tế | Bằng chứng |
|---|---|---|
| Recipe đúng nguồn | Đọc args.yaml của từng model, so với train_args checkpoint | Preflight cả 5 model; fixture LR=.006, momentum=.91, decay=.0009 chứng minh không hardcode mặc định |
| Reference đúng artifact | SHA256 weights/data/class config; kiến trúc đúng model, nc=28 | `prepare_reference`; kiểm tra cả 5 provenance |
| Point xuất phát rõ ràng | starting_train.yaml chứa kwargs thực; augmentation_strength=.5 tái tạo recipe | Snapshot roundtrip; smoke từ chối snapshot đổi pretrained=True |
| Không dùng weights KB1 để train | YOLO(architecture.yaml), pretrained=False, resume=False | CPU integration đặt một reference .pt không hợp lệ; chỉ constructor YAML được gọi, reference giữ nguyên SHA |
| A cố định hyperparameter | Trial 0 là recipe KB1; trial khác chọn bộ tham số trước train | Full pipeline gate; scheduler vẫn giảm actual LR chung |
| B thay tham số trong train | Prefix hoàn thành warmup; action ở boundary, trainer chạy liên tục | Hold giống baseline từng tensor/final live+EMA; EMA/gradient accumulation không reset |
| LR/decay/momentum có hiệu lực | LR0 nhân scheduler factor; decay chuẩn hóa nbs; momentum sau warmup đúng target | CPU và CUDA integration kiểm tra optimizer groups thực |
| Augmentation có hiệu lực | Rebuild transforms khi strength đổi, giữ private RNG; final close-mosaic được giữ | CPU/CUDA gates, action sau close không mở lại mosaic/mixup |
| Reward đúng objective | r=100*(best_after-best_before), gamma=1 | Integration: prefix .1, best .4, final .2 => return 30; initial .8 không được chọn như trained checkpoint |
| Chọn best đúng weights | Canonical EMA mọi epoch, lưu full precision và SHA | Fake canonical curve chọn epoch 2; payload epoch và score khớp |
| Policy không đổi RNG detector | Fork cả CPU/CUDA khi init/load; sampling RNG riêng | CUDA RNG gate, hold equivalence |
| Số học đánh giá thống nhất | FP32, TF32=False, deterministic=True, cùng batch/imgsz/NMS | Backend/RNG restoration gate, kể cả evaluator ném lỗi |
| Test không chọn checkpoint | Khóa config/policy/epoch/seed bằng validation trước test | Ranking test đảo ngược vẫn giữ checkpoint chọn trên validation; test không gọi trainer |
| Best artifact và kết luận thống kê tách rõ | Bảng KB1/A/B ghi best-seed đã chọn bằng validation; kết luận chính dùng mean/paired seeds | Unit report và JSON comparison metadata |
| Resume không trộn experiment | Hash source/config/data/dependencies/reference; kiểm tra cả YAML snapshot | Synthetic end-to-end resume không train/inference lại; snapshot tampered bị chặn |
| Nhiều model không fail muộn | Preflight cả matrix trước execution model đầu | Gate thiếu reference model thứ hai => không model nào train |

## Đối chiếu lý thuyết

HPO chọn một vector hyperparameter cho mỗi detector; scheduler/warmup là recipe
cố định chung. PPO học policy trên quá trình train có state quan sát, action và
reward. Trainer không restart tại decision boundary nên reward phản ánh tác dụng
action trên cùng trajectory. Trọng số, momentum buffers, EMA, scaler và gradient
accumulation tiếp tục qua boundary.

Observation là bản tóm tắt loss/metrics/progress/hyperparameters, không chứa toàn
bộ model/optimizer/data-order state. Vì vậy bài toán thực tế có tính quan sát
một phần (POMDP); không tuyên bố vector này chứng minh tính Markov đầy đủ.

PPO lấy rollout từ policy hiện tại, không update giữa episode, update xong xóa
rollout, tính log-prob joint bằng tổng log-prob của các head với cùng action masks.
Clipped surrogate, GAE và minibatch dùng rollout on-policy. Đây là cấu trúc phù
hợp [PPO của Schulman et al.](https://arxiv.org/abs/1707.06347), không phải bằng
chứng policy đã hội tụ trên dataset này.

Với B, prefix không phụ thuộc policy. Tổng reward telescoping bằng
100*(best_final-best_prefix); tối đa hóa expected return tương ứng tối đa hóa
expected best validation mAP sau train, với phần prefix là hằng số đối với policy.
Lambda=.95 là lựa chọn ước lượng advantage; gamma=1 là mục tiêu hữu hạn không
chiết khấu. Loss/AP-small/AR300 là observation hoặc metric phụ, không penalty
âm thầm đổi objective.

Các field optimizer/warmup/nbs/AMP/close_mosaic dựa trên args.yaml thực và code
Ultralytics 8.4.163 đang cài, phù hợp [ý nghĩa cấu hình Ultralytics](https://docs.ultralytics.com/usage/cfg).
RNG/backend flags được giữ riêng để kiểm soát các nguồn biến thiên theo
[hướng dẫn reproducibility PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Giới hạn của kết luận

- KB1 dùng pretrained, native-best selection, 200 epoch dự kiến/patience 30;
  KB3 scratch, fixed 300 epoch, canonical-best selection. Đây là đối chứng lịch sử,
  không phải một thí nghiệm chỉ đổi duy nhất HPO/RL.
- Giữ 300 epoch vì scratch pilot cũ đạt đỉnh 255; recipe mới không được bảo đảm
  hội tụ ở cùng epoch. Horizon chung không có nghĩa recipe nguồn được copy nguyên
  mọi training-control field; các ngoại lệ ghi ở starting_parameters.json.
- A khám phá absolute parameters, B khám phá adaptive schedules; search spaces
  không đồng nhất hoàn toàn. Không diễn giải kết quả như chỉ đổi thuật toán optimizer.
- Final detector seeds độc lập với search/tuning seeds, nhưng validation images
  vẫn được dùng khi chọn epoch và policy. Generalization phải đọc stage test.
- Test baseline từng được xem trong phân tích trước; không dùng nó để tune protocol
  mới, và không gọi toàn bộ quá trình thiết kế là nghiên cứu chưa từng nhìn test.
- Một experiment có một policy training seed. Năm final detector seeds không
  tương đương năm policy độc lập. Muốn kết luận ổn định của RL cần replicate policy.
- AMP/batch thật được kế thừa từ KB1. GPU gate chỉ dùng 64px/batch 2, không thay
  thế theo dõi OOM/loss/convergence của research run 640px.
- Resume chính xác giữa detector chưa được hỗ trợ; detector dở được archive và
  chạy lại. Chọn weights best không chứa optimizer resume state.

## Validation đã thực hiện

- Suite KB3: **60 test, 59 pass, 1 legacy integration skip**; quality CPU integration bật.
- GPU: **1 integration pass trên RTX 4060**, AMP bật thực; 3 detector x 3 epoch
  trên ảnh giả 64px, baseline/hold bằng nhau và action thực có hiệu lực.
- Full synthetic smoke: **12 detector x 3 epoch**, A/B/tuning/validation/frozen test,
  snapshot integrity và resume pass; không dùng kết quả toy để tuyên bố chất lượng.
- CUDA RNG gate và backend restoration gate pass.
- Preflight recipe/provenance/class-count/architecture/dataset của **5 model** pass:
  YOLOv8n, YOLOv8s, YOLO11n, YOLO11s, YOLO26n.
- Chưa launch research training trên dữ liệu thật. Baseline KB1 chỉ inference;
  không sửa checkpoint, args.yaml hay supervised_manifest.json của KB1.

Log nằm trong checkpoint_hyperparameter_optimization/:
`quality_validation_v3_tests.log`, `quality_validation_v3_cuda.log`,
`quality_validation_v3_smoke.log`, `quality_validation_v3_baselines_all.log`.
Experiment chuẩn bị: `quality/kb1_start_v3/<model>/`, có manifest, recipe snapshot,
starting_train.yaml, starting_parameters.json và kb1_reference.json.
