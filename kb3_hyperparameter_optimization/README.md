# KB3 — Hyperparameter optimization for YOLO

**Luồng hiện tại chỉ chạy A–TPE**, protocol `kb3_tpe_v1_kb1_start`.
Đọc [protocol A–TPE](docs/KB3_A_TPE_PROTOCOL.md) và
[báo cáo B để nghiên cứu tương lai](docs/KB3_B_FUTURE_RESEARCH_REPORT.md).
`run_all_kb3.sh` đánh giá checkpoint supervised KB1, chạy **4 detector trainings tổng cộng**,
rồi xuất báo cáo validation. Không chạy B, không train lại để tạo báo cáo.

Trial 0 giữ recipe KB1 làm scratch anchor; trial 1 startup random trong vùng hẹp;
trial 2/3 dùng TPE. Chỉ tìm `lr0`/`weight_decay`, giữ momentum/augmentation theo KB1.
Mọi trial khởi tạo từ YAML kiến trúc, cùng seed 42; best checkpoint KB1 chỉ làm đối chứng pretrained.
Tối đa 300 epoch/trial, canonical early stopping từ epoch 100 với patience 50;
tổng tối đa 1.200 epoch. Resume không tự mở rộng số lượt. Pilot chưa đánh giá test.
Mặc định chỉ `yolov8n`; `--model all` là 4 trial mỗi model, tổng 20 detector.
TPE là HPO truyền thống, chưa phải kết quả tối ưu bằng RL.

```bash
bash kb3_hyperparameter_optimization/run_quality_kb3.sh --check-only
bash kb3_hyperparameter_optimization/run_quality_kb3.sh --model all --check-only
bash kb3_hyperparameter_optimization/run_quality_kb3.sh --stage baseline
# Chạy bốn lượt; dùng thư mục mới, không resume manifest A/B cũ:
bash kb3_hyperparameter_optimization/run_all_kb3.sh --model yolov8n --trials 4 --output-dir kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/quality/tpe_kb1_start_v1/yolov8n
```

## Các hướng A/B và lệnh legacy (lịch sử, không chạy bởi script hiện tại)

Các ví dụ legacy bên dưới dùng `run_legacy_kb3.sh`; không dùng để kết luận chất lượng final.
KB3 được chia thành hai kịch bản con dùng chung trainer, search space và evaluator.

YOLO **train từ đầu** bằng kiến trúc `.yaml`, `pretrained=False`. Mỗi trial/episode
khởi tạo trọng số ngẫu nhiên theo seed; không lấy checkpoint KB1/KB2. Trong một
episode, trọng số đang train, optimizer, scheduler và EMA được giữ giữa các segment.
RL chọn `lr0`, `weight_decay`, `momentum`, `augmentation_strength`; optimizer SGD
cập nhật trọng số YOLO bằng detection loss thông thường.

## Chạy toàn bộ bằng Bash

Từ thư mục repository, dùng Git Bash, WSL hoặc Linux:

```bash
# Kiểm tra dataset, CUDA và cả 5 kiến trúc; chưa train.
bash kb3_hyperparameter_optimization/run_legacy_kb3.sh --check-only

# Chạy thử tất cả phương pháp trên một model: 2 epoch, 1 trial/episode, 1 seed mới.
bash kb3_hyperparameter_optimization/run_legacy_kb3.sh --model yolov8n --smoke

# Mặc định: 5 model, 10 epoch/run, segment 2 epoch, 3 trial/episode, 3 seed mới.
bash kb3_hyperparameter_optimization/run_legacy_kb3.sh

# Ngân sách lớn hơn cho thí nghiệm.
bash kb3_hyperparameter_optimization/run_legacy_kb3.sh \
  --epochs 100 --segment-epochs 5 --trials 20 --episodes 20
```

Script chạy Default, Random Search, Random Schedule, Bandit và PPO. Thêm
`--with-optuna` để chạy cả Optuna/TPE, sau khi cài `uv pip install --python .venv optuna`.
Chọn một tập con bằng `--methods default ppo`; dùng `--device cpu` nếu không có GPU.
Model hỗ trợ: `yolov8n`, `yolov8s`, `yolov11n`, `yolov11s`, `yolo26n`.

Random Search/TPE khóa best config, Bandit/PPO khóa policy sau search, rồi train
lại từ đầu trên cùng `--eval-seeds` (mặc định 101, 102, 103). Policy evaluation
không học thêm. Reward và chọn checkpoint chỉ dùng validation. Script chưa chạy
test set; phần test dành cho đánh giá cuối sau khi đã chọn phương pháp.

Kết quả nằm trong `checkpoint_hyperparameter_optimization/run_all/<timestamp>/`:
`run_manifest.json`, YAML dataset với đường dẫn hiện tại, config và trajectory
của từng phương pháp, policy/checkpoint detector, `evaluation.json` và
`<model>/comparison.csv`. Mỗi lần chạy tạo thư mục mới, dừng nếu detector thất bại.
Warmup, LR decay và close-mosaic được tắt cho mọi phương pháp để không ghi đè
tham số RL; `nbs=batch` tránh mất gradient tích lũy tại ranh giới segment.

`--smoke` chỉ kiểm tra khả năng chạy, chưa đủ để kết luận chất lượng nghiên cứu.
`--backend simulated --smoke --model yolov8n` kiểm tra pipeline không train YOLO thật.

## KB3-A — Traditional HPO

Mỗi trial chọn một bộ hyperparameter trước khi train và giữ cố định đến cuối:

```text
Configuration -> full YOLO training -> validation objective
```

Phương pháp:

- Default configuration.
- Random Search cố định.
- Optuna/TPE cố định.

Entry points:

```bash
python -m kb3_hyperparameter_optimization.traditional_hpo.default_baseline
python -m kb3_hyperparameter_optimization.traditional_hpo.random_search --trials 20
python -m kb3_hyperparameter_optimization.traditional_hpo.optuna_search --trials 20
```

## KB3-B — Adaptive RL

Agent quan sát trạng thái và thay đổi hyperparameter sau mỗi segment:

```text
State -> Action -> train K epoch -> validation -> Reward -> next State
```

Phương pháp:

- Random Schedule.
- Independent Bandit.
- PPO.

Entry points:

```bash
python -m kb3_hyperparameter_optimization.adaptive_rl.random_schedule --episodes 20
python -m kb3_hyperparameter_optimization.adaptive_rl.train_agent --agent bandit --episodes 20
python -m kb3_hyperparameter_optimization.adaptive_rl.train_agent --agent ppo --episodes 20
python -m kb3_hyperparameter_optimization.adaptive_rl.evaluate_policy --agent ppo --checkpoint kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/adaptive_rl/ppo_agent.pt --seeds 101 102 103
```

## So sánh đúng

KB3-A và KB3-B phải dùng cùng model, data split, search bounds, evaluator, seed đánh giá và ngân sách GPU/epoch. Sau search:

1. Khóa cấu hình tốt nhất của KB3-A.
2. Khóa policy tốt nhất của KB3-B.
3. Chạy lại hai phương pháp trên cùng các seed chưa dùng khi search.
4. Đánh giá test chỉ một lần sau khi đã chọn xong.

Thiết kế chi tiết: `docs/KB3_TWO_SCENARIOS.md`.

## Backend mô phỏng

Backend mô phỏng chỉ dùng cho test code:

```bash
uv run python -m kb3_hyperparameter_optimization.adaptive_rl.train_agent \
  --agent ppo --episodes 3 --backend simulated
```

## Backend Ultralytics

Ví dụ PowerShell:

```powershell
$worker = 'python -m kb3_hyperparameter_optimization.adapters.ultralytics_worker --model yolov8n.yaml --data kb1_reward_guided_training/configs/pest.yaml --total-epochs 100 --dataset-root pre-data/data/v2i_cleanned --canonical-eval'

uv run python -m kb3_hyperparameter_optimization.adaptive_rl.train_agent `
  --agent ppo --episodes 20 --backend command --trainer-command $worker
```

`--total-epochs` của worker phải trùng `experiment.total_epochs`. Nếu reward dùng `AP_small`, phải bật `--canonical-eval`.
Worker từ chối `.pt` ở `--model`; `.pt` chỉ dùng nội bộ để tiếp tục segment trong cùng episode.

## Kiểm thử và tổng hợp

```bash
uv run python -m unittest discover -s kb3_hyperparameter_optimization/tests -v
uv run python -m kb3_hyperparameter_optimization.compare_hpo \
  A2=kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/traditional_optuna_final.json \
  B2=kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/adaptive_rl/adaptive_ppo_evaluation.json
```

Kế hoạch tổng thể vẫn nằm trong `docs/KB3_IMPLEMENTATION_PLAN.md`.

Integration test với Ultralytics thật, dữ liệu nhỏ tạm thời và CPU:

```powershell
$env:KB3_TEST_ULTRALYTICS = '1'
.venv/Scripts/python.exe -m unittest kb3_hyperparameter_optimization.tests.test_ultralytics_segments -v
```
