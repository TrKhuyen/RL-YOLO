# KB3: đối chứng KB1, recipe KB1 và HPO/RL từ đầu

**Tài liệu lịch sử v4.** Script hiện tại chỉ chạy A–TPE; xem
[protocol A–TPE](KB3_A_TPE_PROTOCOL.md) và [báo cáo B](KB3_B_FUTURE_RESEARCH_REPORT.md).
Các lệnh finalize/test A/B dưới đây không thuộc entry point hiện tại.

Protocol tại thời điểm v4: `kb3_quality_v4_staged_search`. `run_quality_kb3.sh` mặc định chỉ
chấm KB1 bằng inference; `run_all_kb3.sh` chạy baseline và một đợt search A/B rồi dừng.
`--finalize` cho phép chạy tiếp tuning, validation retraining và frozen test;
`--stage evaluate` cũng có thể chốt tìm kiếm, sau đó chạy riêng `--stage test`.
Chi tiết thay đổi ngân sách: [KB3_STAGED_SEARCH_REVIEW.md](KB3_STAGED_SEARCH_REVIEW.md).
Luồng 10 epoch cũ nằm ở `run_legacy_kb3.sh`.

## Baseline và khởi tạo là hai việc khác nhau

| Vai trò | Nguồn | Cách dùng |
|---|---|---|
| Đối chứng pretrained | `kb1_reward_guided_training/checkpoint_based/<model>/weights/best.pt` | Đánh giá canonical validation; không train thêm, không sửa weights |
| Recipe xuất phát KB3 | `checkpoint_based/<model>/args.yaml` của supervised KB1 | Lấy hyperparameter, optimizer, scheduler, augmentation và loss gains |
| Trọng số khởi tạo KB3 A/B | YAML kiến trúc, `pretrained=False` | Ngẫu nhiên theo seed; không load checkpoint KB1 |
| Đối chứng đo riêng HPO/RL | Nhóm `default` trong final evaluation | Scratch cùng recipe, horizon và seed với A/B |

Không dùng `configs/hyp.pest.yaml` của KB1 cho YOLOv8: đó là recipe YOLOv5,
khác định nghĩa loss. `pest.yaml` là YAML dữ liệu, không phải hyperparameter.
Recipe được kiểm tra với `train_args` trong checkpoint; weights, dataset và
class configuration được xác minh bằng provenance/SHA256 trước khi train.

KB1 yolov8n là một lượt supervised pretrained, seed 0, 200 epoch dự kiến và
patience 30; dừng ở epoch 166, native best ở 136. Vì khác khởi tạo, thời lượng
và tiêu chí chọn best lịch sử, đây là đối chứng chất lượng thực tế, không phải
phép đo riêng tác dụng HPO/RL. Không nhân cùng checkpoint thành 5 seed độc lập
hoặc gán độ lệch chuẩn cho nó. `comparison.json` ghi reference này riêng.

## Kế thừa và ngoại lệ

SGD, warmup, linear LR (`cos_lr=False`), lrf, AMP, nbs=64, loss gains, HSV,
geometric/mixing augmentation và close_mosaic lấy từ args.yaml tương ứng.
Batch/imgsz cũng mặc định lấy từ KB1 (yolov8n: 16/640); có thể chốt override chung
trong config trước khi tạo experiment. Workers=0 để RL cập nhật transforms trực tiếp.

`augmentation_strength=0.5` tái tạo geometric/mixing recipe KB1: degrees=10,
translate=.1, scale=.5, flipud=.3, fliplr=.5, mosaic=1, mixup=.1 với yolov8n.
Giá trị khác nhân các cường độ theo strength/.5; xác suất chặn ở 1, HSV giữ nguyên.
10 epoch cuối luôn đóng mosaic/mixup/cutmix/copy_paste kể cả sau action RL.
Weight decay dùng đúng hệ số batch/accumulation/nbs của Ultralytics.

Các ngoại lệ có chủ đích:

- KB3 vẫn scratch theo mục tiêu ban đầu; KB1 dùng pretrained.
- Giữ tối đa 300 epoch cho detector KB3: pilot scratch lịch sử đạt đỉnh ở 255.
  Không lấy mốc 136 của pretrained để kết luận scratch hội tụ. Recipe mới
  không được bảo đảm có cùng đường hội tụ với pilot cũ.
- Search A/B có cùng quy tắc dừng: ít nhất 100 epoch, patience 50 epoch,
  min_delta .0005 theo canonical validation mAP50–95, dừng tại boundary 5 epoch.
  Nếu đã vào pha close-mosaic theo horizon 300 thì chạy hết pha này.
  Tuning và final retraining vẫn đủ 300 epoch, không early stop.
  Chọn EMA có canonical mAP50–95 tốt nhất trong mọi epoch đã train.
  Số epoch thực chạy của search A/B có thể khác; báo chi phí thực, không gọi là bằng GPU time.
- Search/tuning/final dùng seed tách biệt, không kế thừa seed 0 KB1.
- Không kế thừa model/pretrained/resume/data/output path của lượt KB1.

## Hai kịch bản và đánh giá

A dùng Random Search: trial đầu là bộ tham số KB1, các trial sau khám phá bounds.
Mỗi trial chọn LR0/decay/momentum/augmentation trước khi train và giữ cố định;
actual LR vẫn thay đổi theo warmup/linear schedule chung.

B dùng PPO: mỗi episode bắt đầu từ bộ tham số KB1; 5 epoch đầu hoàn thành warmup.
Từ epoch 6, policy đọc state, đổi tham số và train tiếp từng đoạn 5 epoch.
Một trainer chạy liên tục, giữ model/optimizer/scaler/EMA/gradient accumulation/RNG
qua boundary. Đổi augmentation chỉ rebuild transforms và mang theo private RNG
Albumentations; hold không rebuild.

State dùng log-loss/trend, flag loss chưa có tại epoch 0, best mAP, tiến độ,
staleness, schedule factor và normalized hyperparameters. Mask vượt bounds dùng
cả sampling lẫn PPO update. Reward `100*delta(best_trained_mAP)`, gamma=1; return
khớp thay đổi objective từ cuối prefix. Policy có RNG riêng; inference cô lập RNG.
PPO gom 64 transition, minibatch 32, KL stop và lưu rollout chưa update khi resume.
Chỉ update tại cuối episode, đồng thời force update tại boundary lưu candidate mỗi 4 episode.
Boundary này không đổi giữa chạy một lần và chạy từng đợt, tránh đổi chính sách do resume.

Canonical evaluator dùng cùng resize/pad, FP32, NMS conf .001/IoU .60/max_det 300.
Mọi lần chấm dùng TF32=False, deterministic=True; khôi phục backend flags/RNG
ngay sau inference để không đổi trạng thái train. PPO init/load giữ RNG cả CPU/CUDA.
Metric chính mAP50–95; recall trong state là AR300, precision tại conf .25.
Reference KB1 được chấm cùng evaluator/batch/imgsz của experiment mới;
không so native CSV KB1 trực tiếp với canonical score KB3.

HPO và PPO chọn ứng viên qua 2 tuning seed rồi khóa config/policy, retrain trên
3 final detector seed chung. Final gồm `default`, `random-search`,
`random-schedule`, `ppo`; báo mean/sample std và paired PPO-minus-HPO.
KB1 nằm riêng với một lượt train lịch sử. A/B không có cùng tập lịch tham số
có thể đạt; đây là so sánh hai chiến lược, không chỉ riêng thuật toán optimizer.

Search/tuning/final retraining dùng validation images; final giữ lại detector seed mới.
Sau đó stage `test` chỉ inference trên các checkpoint đã khóa bằng validation.
`test_comparison.json` chứa score từng seed, mean/std và đối chứng KB1 trên test.
`checkpoint_comparison.csv` và `checkpoint_comparison_test.csv` so KB1/A/B;
cùng best-seed checkpoint được chọn bằng validation trong cả hai bảng, dù ranking
trên test đảo ngược. Kết luận chính dựa trên mean/paired seeds, không chỉ best seed.
Test không dùng để chọn reward/config/policy/epoch/seed.
Test baseline lịch sử đã được xem trong báo cáo scratch/pretrained trước đó;
không dùng điểm test ấy để sửa protocol tiếp theo.

Đợt đầu: **4 HPO + 4 PPO, tối đa 2.400 epoch**, rồi dừng trước tuning/final/test.
Có thể resume lên 8 mỗi nhánh. Chốt ở 4: 8 search + 4 tuning + 12 final =
**24 detector / tối đa 7.200 epoch**. Chốt ở 8: 16 search + 8 tuning + 12 final =
**36 detector / tối đa 10.800 epoch**. Search thực tế có thể ngắn hơn nhờ early stopping.
Mốc 8 là giới hạn mở rộng tự động, không phải giới hạn bắt buộc của nghiên cứu.
`--search-target 12`, `16`... cho phép tăng trước khi tuning, giữ kết quả/policy
và kiểm tra search seeds không trùng tuning/final. Số ứng viên HPO được tune bằng
số policy candidate đã lưu, nên tuning A/B vẫn cùng số lượt khi mở rộng.
Tại mốc 4, tune một HPO config và một policy; tại mốc 8, tune hai ứng viên mỗi bên.
Không dùng tuning/final/test để quyết định kéo dài cùng thí nghiệm: khi `evaluate`
bắt đầu, `selection_lock.json` khóa quần thể tìm kiếm. Số 4/8 là ngân sách thăm dò,
không phải số episode được chứng minh đủ học PPO. Search bị dừng sớm là proxy của
chất lượng 300 epoch; có thể bỏ sót ứng viên cải thiện muộn. Tuning/final đủ horizon
kiểm chứng các ứng viên còn lại, không chứng minh ứng viên bị loại là kém.
Reference chỉ thêm inference, không thêm train. Ngân sách không đảm bảo PPO thắng
hay hội tụ; cần đọc diagnostics/tuning scores. Một experiment có một policy
training seed, không phải 5 policy độc lập.

## Chạy và resume

Từ repository root trong PowerShell (mỗi lệnh trên một dòng):

```bash
# Xác minh recipe/checkpoint/data/CUDA; không train.
bash kb3_hyperparameter_optimization/run_quality_kb3.sh --check-only

# Đợt đầu: 4 A + 4 B, rồi dừng. Không thêm --resume ở lần đầu tạo thư mục này.
bash kb3_hyperparameter_optimization/run_all_kb3.sh --model yolov8n --output-dir kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/quality/kb1_start_v4/yolov8n

# Nếu kết quả search còn cải thiện: lên tổng 8 A + 8 B, giữ lại các lượt đã xong.
bash kb3_hyperparameter_optimization/run_all_kb3.sh --model yolov8n --resume --search-target 8 --output-dir kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/quality/kb1_start_v4/yolov8n

# Chốt tại số lượt đã hoàn thành, chọn ứng viên và train đánh giá; không mở rộng search.
bash kb3_hyperparameter_optimization/run_all_kb3.sh --model yolov8n --resume --stage evaluate --output-dir kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/quality/kb1_start_v4/yolov8n

# Test chỉ inference trên checkpoint đã chọn bằng validation.
bash kb3_hyperparameter_optimization/run_all_kb3.sh --model yolov8n --resume --stage test --output-dir kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/quality/kb1_start_v4/yolov8n
```

`--stage search` / `evaluate` / `test` cho phép tách bước; `pilot` là alias baseline, không train.
`--search-target` là số lượt tích lũy, không phải số lượt thêm. Resume đợt bị gián đoạn
giữ nguyên target đang chạy; resume đợt hoàn thành mặc định tiến thêm một batch.
Không dùng thư mục v3 cho v4: manifest cũ bị từ chối, không sửa hay xóa kết quả cũ.
Đổi model bằng `--model yolov8s`... cần checkpoint/provenance/args.yaml KB1 tương ứng.
`--model all` hỗ trợ YOLOv8n/s, YOLO11n/s, YOLO26n, preflight cả 5 model trước khi
chạy bất kỳ model nào; mỗi model có output/config/policy riêng. Với matrix, resume
dùng thư mục cha `kb1_start_v4`; với một model, dùng thư mục con tương ứng.
`--smoke` là integration nhỏ; không dùng reference KB1 để so chất lượng.

`starting_train.yaml` chứa cấu hình train đầy đủ đã resolve (LR, momentum, decay,
loss, augmentation, warmup, scheduler, batch, initialization...).
`starting_parameters.json` ghi augmentation_strength=.5, vai trò reference,
SHA256 nguồn và các ngoại lệ có chủ đích. Snapshot YAML bị thay đổi sẽ chặn resume,
kể cả `--check-only`. Không copy `best.pt` vào input train.

Manifest hash code/dependencies/config/dataset bytes và reference/recipe SHA256.
Resume từ chối input khác; không resume v1/v2/v3 bằng code v4. Đổi recipe hoặc lịch
train sau khi bắt đầu phải tạo experiment mới. Riêng tăng target bằng `--search-target`
trước selection lock giữ nguyên thuật toán, recipe, lịch update và identity, nên được
tiếp tục cùng experiment. Detector dở được archive `.interrupted_*` và chạy lại;
chưa hỗ trợ exact resume giữa detector. Selected weights là inference
artifact, không chứa optimizer resume state. Auto-OOM thay batch sẽ dừng;
chốt batch chung trong config mới trước khi chạy lại.

## Checkpoint đã xóa và kiểm chứng

Theo yêu cầu ngày 09/10/2026 đã xóa đúng `selected_best.pt` của scratch pilot
`quality/20261008T132415770380Z/pilot`. Giữ logs/curves/evaluation lịch sử và
`checkpoint_deleted.json`; điểm số vẫn là phép đo lịch sử, nhưng không còn weights
để inference lại. Không dùng pilot làm baseline hiện tại. KB1 không bị sửa.

Unit gates: baseline inference-only/cache/checksum, recipe thực KB1,
state/reward/mask, RNG isolation, PPO learning, save/load và skip completed detectors.
Real CPU gate: baseline/hold bằng nhau từng tensor và final live/EMA SHA256,
action thay LR/decay/momentum/augmentation đúng, linear schedule, EMA không reset,
mosaic/mixup không mở lại trong phase cuối. Full synthetic smoke chạy A/B/tuning/
validation/test/resume, không train lại detector hoặc inference lại test đã hoàn thành.
GPU gate bật AMP thật trên RTX 4060 bằng ảnh giả 64px, kiểm tra hold/action tương tự CPU.
Các gate xác minh
implementation, không chứng minh convergence hay PPO thắng HPO.

```powershell
$env:KB3_TEST_QUALITY='1'
.venv/Scripts/python.exe -m unittest discover -s kb3_hyperparameter_optimization/tests -q
.venv/Scripts/python.exe -m kb3_hyperparameter_optimization.tests.quality_smoke
```
