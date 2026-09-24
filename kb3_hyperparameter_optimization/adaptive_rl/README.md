# KB3-B — Adaptive RL

Agent quan sát state huấn luyện và điều chỉnh hyperparameter trước mỗi segment.

```text
State -> Action -> train K epoch -> validation -> Reward -> state mới
```

Các phương pháp:

- `random_schedule`: thay đổi ngẫu nhiên sau mỗi segment, không học từ reward.
- `bandit`: học giá trị trung bình của action nhưng chưa dùng state.
- `PPO`: policy phụ thuộc state, được cập nhật bằng reward.

Random Schedule là đối chứng quan trọng: nếu PPO không tốt hơn nó, chưa có đủ bằng chứng rằng policy đã học được quy luật điều chỉnh hữu ích.

