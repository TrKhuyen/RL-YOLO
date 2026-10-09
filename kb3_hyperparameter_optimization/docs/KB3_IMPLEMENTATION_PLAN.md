# Kế hoạch triển khai KB3: Tối ưu quá trình huấn luyện YOLO bằng học tăng cường

## 1. Mục tiêu

KB3 xây dựng một tác nhân học tăng cường (RL agent) có nhiệm vụ quan sát diễn biến huấn luyện YOLO và lựa chọn cách điều chỉnh siêu tham số theo từng giai đoạn. Mục tiêu là tìm được mô hình có chất lượng tốt trong cùng một ngân sách tính toán, sau đó so sánh công bằng với baseline, KB1 và KB2 để chọn mô hình triển khai lên website nhận diện sâu bệnh.

KB3 không dùng RL để cập nhật trực tiếp trọng số YOLO. Trọng số mô hình vẫn được cập nhật bằng optimizer thông thường; RL chỉ đóng vai trò bộ điều khiển siêu tham số của quá trình huấn luyện.

### Kết quả cần đạt

- Xây dựng môi trường RL bao quanh vòng lặp huấn luyện YOLO.
- Xây dựng agent có thể chọn hoặc điều chỉnh siêu tham số sau mỗi đoạn huấn luyện.
- Đảm bảo tập test không tham gia chọn action, tính reward hoặc chọn checkpoint.
- So sánh RL-HPO với cấu hình mặc định, random search và Optuna/TPE trong cùng ngân sách.
- Tái lập được thí nghiệm bằng cấu hình, seed, log và checkpoint.
- Xuất bảng kết quả phục vụ so sánh KB1, KB2, KB3 và lựa chọn mô hình web demo.

## 2. Câu hỏi nghiên cứu

1. Điều chỉnh siêu tham số thích nghi bằng RL có cải thiện `mAP50-95`, recall và `AP_small` so với cấu hình cố định không?
2. RL có đạt kết quả tốt hơn random search hoặc Optuna/TPE khi sử dụng cùng ngân sách GPU không?
3. Chính sách học được có ổn định trên nhiều seed và có thể chuyển sang lần huấn luyện mới không?
4. Những siêu tham số và thời điểm điều chỉnh nào ảnh hưởng nhiều nhất đến kết quả?
5. Mức cải thiện độ chính xác có đủ lớn để bù cho chi phí huấn luyện bổ sung không?

## 3. Phạm vi

### Trong phạm vi

- Bài toán object detection trên bộ dữ liệu PlantDoc đã được nhóm vẽ lại bounding box theo định dạng YOLO.
- Sử dụng cùng dataset split với KB1 và KB2.
- Tối ưu quá trình huấn luyện cho một kiến trúc YOLO nền được lựa chọn trước.
- Đánh giá trên nhiều seed và cùng ngân sách tính toán.
- Lưu đầy đủ lịch sử action, reward, metric và hyperparameter.

### Ngoài phạm vi phiên bản đầu

- Tìm kiếm đồng thời kiến trúc mạng (Neural Architecture Search).
- Thay đổi số lớp, số kênh hoặc cấu trúc backbone trong một episode.
- Dùng tập test làm tín hiệu phản hồi.
- Tự động huấn luyện lại từ phản hồi người dùng trên website.
- Điều khiển toàn bộ hyperparameter ngay trong thí nghiệm đầu tiên.

## 4. Thiết kế thí nghiệm tổng quát

```text
Cấu hình ban đầu + kiến trúc YAML, trọng số ngẫu nhiên theo seed
                    |
                    v
          Huấn luyện K epoch/segment
                    |
                    v
       Thu thập train/validation metrics
                    |
                    v
       State -> RL Agent -> Action mới
                    |
                    v
        Cập nhật hyperparameter an toàn
                    |
                    +------ lặp đến hết episode
                    |
                    v
       Chọn checkpoint bằng validation
                    |
                    v
          Đánh giá một lần trên test
```

Một episode là một lần huấn luyện YOLO với tổng số epoch cố định. Agent không ra quyết định ở từng batch mà ra quyết định sau mỗi segment, dự kiến 5 epoch. Thiết kế theo segment giảm nhiễu và giảm chi phí tương tác.

## 5. Mô hình hóa bài toán RL

### 5.1. Environment

Environment bao quanh trainer YOLO và cung cấp giao diện tương tự Gymnasium:

```python
observation, info = env.reset(seed=seed)
observation, reward, terminated, truncated, info = env.step(action)
```

Mỗi `step(action)` thực hiện:

1. Kiểm tra và ánh xạ action thành hyperparameter hợp lệ.
2. Huấn luyện mô hình thêm `segment_epochs`.
3. Đánh giá trên tập validation.
4. Chuẩn hóa metric thành state mới.
5. Tính reward từ mức cải thiện và chi phí.
6. Lưu checkpoint, action, metric và thời gian chạy.

### 5.2. State

State ban đầu nên nhỏ, có thể chuẩn hóa về khoảng ổn định:

| Nhóm | Thành phần |
|---|---|
| Tiến độ | epoch hiện tại / tổng epoch |
| Train | box loss, class loss, objectness/DFL loss, tổng loss |
| Validation | precision, recall, mAP50, mAP50-95 |
| Vật thể nhỏ | AP_small hoặc recall_small nếu pipeline hỗ trợ |
| Tổng quát hóa | khoảng cách train loss và validation loss/metric |
| Tối ưu | learning rate, weight decay, momentum hiện tại |
| Lịch sử | thay đổi metric và reward của 1-3 segment gần nhất |
| Tài nguyên | thời gian đã dùng / ngân sách thời gian |

Không đưa metric của test set vào state.

### 5.3. Action

Triển khai theo hai giai đoạn để hạn chế không gian tìm kiếm.

#### Giai đoạn A: action rời rạc

Agent chọn hệ số nhân cho từng hyperparameter:

| Hyperparameter | Action đề xuất |
|---|---|
| Learning rate | `x0.5`, `x0.8`, giữ nguyên, `x1.2`, `x1.5` |
| Weight decay | `x0.8`, giữ nguyên, `x1.2` |
| Momentum | `-0.02`, giữ nguyên, `+0.02` |
| Augmentation strength | giảm, giữ nguyên, tăng |

Mọi giá trị phải được clip trong miền cấu hình cho phép. Không thay đổi batch size giữa episode vì dễ làm sai so sánh và gây lỗi bộ nhớ.

#### Giai đoạn B: action liên tục

Chỉ thực hiện sau khi Giai đoạn A hoạt động ổn định. Agent phát ra vector trong `[-1, 1]`, sau đó ánh xạ thành phần trăm thay đổi của learning rate, weight decay, momentum và augmentation strength.

### 5.4. Reward

Reward mặc định được tính trên validation:

```text
reward = 0.45 * delta_mAP50_95
       + 0.25 * delta_recall
       + 0.20 * delta_AP_small
       + 0.10 * delta_mAP50
       - lambda_time * normalized_training_time
       - lambda_overfit * overfit_penalty
       - lambda_jump * unsafe_action_penalty
```

Trong đó:

- `delta_*` là mức thay đổi so với segment trước, giúp agent học ảnh hưởng của action.
- `overfit_penalty` tăng khi train loss tiếp tục giảm nhưng validation metric suy giảm.
- `unsafe_action_penalty` áp dụng khi action bị clip mạnh hoặc gây NaN/OOM.
- Có thể thêm terminal bonus dựa trên metric cuối episode để tránh agent chỉ tối ưu cải thiện ngắn hạn.

Tất cả thành phần reward phải được chuẩn hóa trên cùng thang đo. Trọng số reward là giả thuyết ban đầu và phải có ablation, không được khẳng định là tối ưu trước khi thử nghiệm.

### 5.5. Kết thúc episode

Episode kết thúc khi xảy ra một trong các điều kiện:

- Đạt tổng số epoch quy định.
- Hết ngân sách thời gian/GPU.
- Loss xuất hiện NaN hoặc diverge không thể phục hồi.
- Không cải thiện sau số segment kiên nhẫn đã cấu hình.

## 6. Lựa chọn thuật toán

### Thuật toán chính: PPO

PPO được chọn cho phiên bản chính vì tương đối ổn định, hỗ trợ action rời rạc hoặc liên tục và có cơ chế giới hạn cập nhật policy.

### Đối chứng RL đơn giản

- Contextual bandit hoặc epsilon-greedy cho action rời rạc.
- Random policy để xác minh môi trường và reward.

Nếu số episode thực tế quá ít để PPO học ổn định, contextual bandit sẽ được dùng làm phương án chính thức thay vì cố sử dụng một thuật toán RL phức tạp nhưng thiếu dữ liệu tương tác.

## 7. Baseline và nguyên tắc so sánh công bằng

Các phương pháp cần so sánh:

1. Cấu hình YOLO mặc định hoặc cấu hình supervised tốt nhất hiện có.
2. Manual tuning hiện tại của dự án.
3. Random search.
4. Optuna/TPE.
5. RL-HPO của KB3.

Mọi phương pháp phải dùng:

- Cùng kiến trúc và model initialization từ đầu theo seed.
- Cùng train/validation/test split.
- Cùng augmentation search space.
- Cùng giới hạn số trial, epoch hoặc GPU-hours.
- Cùng quy tắc early stopping và chọn checkpoint.
- Cùng tập seed, tối thiểu 3 seed cho kết quả cuối.

Không được so một trial RL tốt nhất với trung bình nhiều trial của baseline. Cần báo cáo cả best, mean, standard deviation và tổng chi phí.

## 8. Chỉ số đánh giá

### Chất lượng mô hình

- `mAP50-95` — chỉ số chính.
- `mAP50`.
- Precision, recall và F1-score.
- `AP_small` và/hoặc `recall_small`.
- Metric theo từng lớp và confusion matrix.

### Hiệu quả tối ưu

- Best validation metric theo GPU-hour.
- Số trial/episode để đạt một ngưỡng metric.
- Area under the optimization curve.
- Tỷ lệ episode lỗi hoặc diverge.
- Độ ổn định giữa các seed.

### Khả năng triển khai web

- Test mAP và recall.
- Latency trên một ảnh.
- FPS với batch size 1.
- Dung lượng checkpoint.
- RAM/VRAM sử dụng.
- Khả năng export ONNX và chạy bằng ONNX Runtime.

## 9. Quản lý dữ liệu và chống rò rỉ

- Chốt một manifest chứa đường dẫn ảnh, nhãn, split và checksum.
- Giữ nguyên split giữa KB1, KB2 và KB3.
- Nếu nhiều ảnh đến từ cùng ảnh gốc/video/nguồn gần nhau, phải group trước khi chia split.
- Train set dùng cập nhật trọng số YOLO.
- Validation set dùng tạo state, reward, early stopping và chọn checkpoint.
- Test set chỉ dùng sau khi khóa policy, search space và cấu hình cuối.
- Mọi lần truy cập test phải được ghi log để tránh vô tình tune theo test.

## 10. Cấu trúc thư mục dự kiến

```text
kb3_hyperparameter_optimization/
├── README.md
├── configs/
│   ├── kb3_default.yaml
│   ├── search_space.yaml
│   └── experiment_budget.yaml
├── docs/
│   └── KB3_IMPLEMENTATION_PLAN.md
├── envs/
│   ├── __init__.py
│   └── yolo_hpo_env.py
├── agents/
│   ├── __init__.py
│   ├── ppo_agent.py
│   └── bandit_agent.py
├── adapters/
│   ├── __init__.py
│   └── trainer_adapter.py
├── baselines/
│   ├── random_search.py
│   └── optuna_search.py
├── tests/
│   ├── test_action_space.py
│   ├── test_reward.py
│   ├── test_environment.py
│   └── test_resume.py
├── train_agent.py
├── evaluate_policy.py
├── compare_hpo.py
└── export_results.py
```

Chỉ `docs/` được tạo ở bước lập kế hoạch hiện tại. Các thành phần mã nguồn sẽ được tạo theo từng giai đoạn sau khi xác nhận trainer và cấu hình dùng chung với KB1/KB2.

## 11. Thiết kế cấu hình

Tất cả tham số phải nằm trong YAML, không hard-code trong agent:

```yaml
experiment:
  seed: 42
  total_epochs: 100
  segment_epochs: 5
  max_gpu_hours: 12

objective:
  primary_metric: map50_95
  weights:
    delta_map50_95: 0.45
    delta_recall: 0.25
    delta_ap_small: 0.20
    delta_map50: 0.10

search_space:
  lr0: [0.0001, 0.02]
  weight_decay: [0.0001, 0.002]
  momentum: [0.80, 0.98]
  augmentation_strength: [0.0, 1.0]

safety:
  terminate_on_nan: true
  max_oom_retries: 1
  rollback_on_failure: true
```

Các miền giá trị trên chỉ là giá trị khởi tạo; phải đối chiếu với trainer thực tế trước khi chạy thí nghiệm đầy đủ.

## 12. Log, checkpoint và khả năng tái lập

Mỗi run cần lưu:

- Git commit, phiên bản thư viện, GPU và CUDA.
- Dataset manifest/checksum.
- Model YAML, initialization từ đầu và seed.
- State, raw metric, normalized state, action và reward từng step.
- Hyperparameter trước và sau action.
- Thời gian train/eval, GPU-hour và peak VRAM.
- Checkpoint YOLO tốt nhất theo validation.
- Checkpoint agent và optimizer của agent.
- Lý do episode kết thúc.

Checkpoint phải hỗ trợ resume mà không làm mất optimizer state, scheduler state, epoch, policy state và lịch sử reward.

## 13. Các giai đoạn triển khai

### Giai đoạn 0 — Kiểm kê và khóa baseline

- Xác nhận dataset, class mapping và split đang dùng.
- Xác nhận model nền, trainer và checkpoint khởi tạo.
- Chạy lại một supervised baseline tối thiểu 3 seed.
- Đo thời gian một epoch và ước lượng ngân sách KB3.
- Chốt metric chính và quy tắc chọn checkpoint.

**Đầu ra:** baseline tái lập được, manifest dữ liệu và bảng ngân sách.

### Giai đoạn 1 — Tách trainer adapter

- Xây dựng API train theo segment.
- Cho phép cập nhật hyperparameter hợp lệ giữa hai segment.
- Trả về metric thống nhất, độc lập với backend YOLO.
- Hỗ trợ save, rollback và resume checkpoint.

**Đầu ra:** trainer có thể chạy tuần tự nhiều segment mà kết quả nhất quán.

### Giai đoạn 2 — Xây dựng environment

- Định nghĩa observation/action space.
- Cài đặt reset/step/termination.
- Chuẩn hóa state và reward.
- Thêm xử lý NaN, OOM và action ngoài miền.
- Chạy random policy end-to-end.

**Đầu ra:** environment vượt qua smoke test và sinh log đầy đủ.

### Giai đoạn 3 — Baseline HPO

- Cài random search.
- Cài Optuna/TPE.
- Chạy cùng search space và ngân sách dự kiến cho RL.
- Xuất optimization curve và best configuration.

**Đầu ra:** mốc đối chứng bắt buộc trước khi đánh giá RL.

### Giai đoạn 4 — Agent RL

- Bắt đầu bằng action rời rạc và bandit/random agent.
- Cài PPO khi environment đã ổn định.
- Huấn luyện policy trên các episode độc lập.
- Đánh giá policy deterministic trên seed chưa dùng khi train agent.

**Đầu ra:** policy checkpoint và lịch điều chỉnh hyperparameter.

### Giai đoạn 5 — Ablation và robustness

- Reward có/không có `AP_small`.
- Reward tuyệt đối so với reward dùng delta metric.
- Segment 3, 5 và 10 epoch nếu ngân sách cho phép.
- Chỉ điều chỉnh learning rate so với điều chỉnh nhiều tham số.
- Có/không có time penalty và overfit penalty.
- Kiểm tra ít nhất 3 seed.

**Đầu ra:** giải thích thành phần nào thực sự tạo cải thiện.

### Giai đoạn 6 — Đánh giá cuối và so sánh KB1–KB3

- Khóa policy và cấu hình trước khi chạy test.
- Đánh giá baseline, KB1, KB2, KB3 bằng cùng evaluator.
- Đo accuracy, small-object performance, latency, FPS và model size.
- Lập Pareto frontier giữa chất lượng và tốc độ.
- Chọn model cho web bằng tiêu chí công bố trước.

**Đầu ra:** bảng so sánh cuối, kết luận và model deployment candidate.

## 14. Kiểm thử

### Unit test

- Mapping action không vượt giới hạn.
- Chuẩn hóa state không sinh NaN/Inf.
- Reward có dấu đúng khi metric tăng hoặc giảm.
- Không có trường test metric trong observation/reward.
- Terminal và truncation hoạt động đúng.

### Integration test

- Chạy 2 segment trên dữ liệu nhỏ.
- Save/resume cho kết quả tương đương chạy liên tục trong sai số cho phép.
- Rollback sau action gây lỗi.
- Random agent hoàn thành một episode.
- Log có đủ state/action/reward/hyperparameter.

### Reproducibility test

- Cùng seed và cấu hình cho kết quả gần tương đương.
- Khác seed được ghi nhận đúng trong metadata.
- Dataset checksum không thay đổi giữa các phương pháp.

## 15. Ma trận thí nghiệm tối thiểu

| Mã | Phương pháp | Search/Policy | Seed | Mục đích |
|---|---|---|---:|---|
| E0 | Fixed baseline | Cấu hình mặc định | 3 | Mốc supervised |
| E1 | Manual tuning | Cấu hình dự án hiện tại | 3 | Mốc thực tế |
| E2 | Random search | Cùng search space | >=3 | Đối chứng tìm kiếm |
| E3 | Optuna/TPE | Cùng search space | >=3 | Baseline HPO mạnh |
| E4 | Bandit | Action rời rạc | >=3 | RL đơn giản |
| E5 | PPO | Action rời rạc | >=3 | Phương pháp KB3 chính |
| E6 | PPO ablation | Reward/action rút gọn | >=3 | Giải thích đóng góp |

Số trial cụ thể được quyết định sau Giai đoạn 0 dựa trên thời gian một epoch và ngân sách GPU. Không được đặt số trial của một phương pháp cao hơn phương pháp khác nếu báo cáo là so sánh cùng ngân sách.

## 16. Tiêu chí chọn mô hình cho web demo

Không mặc định chọn model có mAP cao nhất. Model triển khai phải thỏa các ràng buộc tối thiểu về latency và bộ nhớ, sau đó mới xếp hạng chất lượng.

Điểm tham khảo:

```text
deployment_score = 0.40 * normalized_mAP50_95
                 + 0.25 * normalized_recall
                 + 0.20 * normalized_AP_small
                 + 0.15 * normalized_FPS
```

Ngoài điểm số, model phải:

- Export ONNX thành công.
- Suy luận batch size 1 ổn định.
- Trả về bounding box, class và confidence để tích hợp web.
- Đạt giới hạn latency được xác định trên máy triển khai thực tế.

Nếu model tốt nhất về độ chính xác quá chậm, chọn model nằm trên Pareto frontier và giải thích sự đánh đổi giữa chất lượng với tốc độ.

## 17. Rủi ro và phương án xử lý

| Rủi ro | Ảnh hưởng | Xử lý |
|---|---|---|
| RL quá tốn GPU | Không đủ episode | Thu nhỏ model/dataset proxy, dùng segment, bandit trước PPO |
| Reward nhiễu | Policy không hội tụ | Dùng delta, smoothing, normalization và nhiều seed |
| Agent khai thác validation | Kết quả test kém | Giới hạn episode, giữ test kín, đánh giá seed mới |
| Không công bằng với Optuna | Kết luận thiếu tin cậy | Khóa search space và GPU-hour trước thí nghiệm |
| Thay hyperparameter làm trainer lỗi | Mất run | Clip action, validate, checkpoint và rollback |
| PPO không hơn baseline | Đóng góp yếu | Báo cáo trung thực; phân tích chi phí và dùng bandit nếu phù hợp hơn |
| Khó tái lập | Không bảo vệ được kết quả | Lưu seed, config, manifest, version và toàn bộ trajectory |

## 18. Tiêu chí hoàn thành KB3

KB3 được xem là hoàn thành khi:

- Environment và agent chạy end-to-end, có test và resume.
- Có baseline fixed, random search và Optuna/TPE trong cùng ngân sách.
- Có kết quả tối thiểu 3 seed cho thí nghiệm chính.
- Test set chỉ được dùng sau khi khóa cấu hình cuối.
- Có bảng metric, optimization curve, chi phí GPU và ablation.
- Có checkpoint tốt nhất và cấu hình tái lập đầy đủ.
- Có kết luận rõ RL-HPO tốt hơn, tương đương hay kém hơn baseline nào.
- Có một model ứng viên đã kiểm tra khả năng export để tích hợp web demo.

## 19. Thứ tự công việc đề xuất

1. Kiểm kê pipeline và khóa baseline.
2. Chốt ngân sách, search space và tiêu chí đánh giá trước khi chạy.
3. Tách trainer adapter hỗ trợ huấn luyện theo segment.
4. Hoàn thiện environment và random-agent smoke test.
5. Chạy random search và Optuna/TPE.
6. Huấn luyện bandit/PPO trong cùng ngân sách.
7. Chạy multi-seed và ablation.
8. Đánh giá test một lần cho cấu hình đã khóa.
9. So sánh với KB1, KB2 và lựa chọn model web.
10. Export model, lưu báo cáo và hoàn thiện tài liệu khóa luận.

