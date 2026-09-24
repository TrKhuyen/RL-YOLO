# KB3 — Hyperparameter optimization for YOLO

KB3 được chia thành hai kịch bản con dùng chung trainer, search space và evaluator.

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
python -m kb3_hyperparameter_optimization.adaptive_rl.evaluate_policy --agent ppo --checkpoint POLICY.pt --seeds 101 102 103
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
$worker = 'python -m kb3_hyperparameter_optimization.adapters.ultralytics_worker --model yolov8n.pt --data kb1_reward_guided_training/configs/pest.yaml --total-epochs 100 --dataset-root pre-data/data/v2i_cleanned --canonical-eval'

uv run python -m kb3_hyperparameter_optimization.adaptive_rl.train_agent `
  --agent ppo --episodes 20 --backend command --trainer-command $worker
```

`--total-epochs` của worker phải trùng `experiment.total_epochs`. Nếu reward dùng `AP_small`, phải bật `--canonical-eval`.

## Kiểm thử và tổng hợp

```bash
uv run python -m unittest discover -s kb3_hyperparameter_optimization/tests -v
uv run python -m kb3_hyperparameter_optimization.compare_hpo \
  A2=runs/kb3/traditional_optuna_final.json \
  B2=runs/kb3/adaptive_ppo_evaluation.json
```

Kế hoạch tổng thể vẫn nằm trong `docs/KB3_IMPLEMENTATION_PLAN.md`.
