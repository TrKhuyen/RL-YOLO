# Đường dẫn checkpoint cho lần chạy lại trên dữ liệu sạch

Ngày kiểm tra: 2026-09-26. Dữ liệu dùng chung: pre-data/data/v2i_cleanned.
Các đường dẫn dưới đây tính từ thư mục gốc repository.

| Kịch bản | Đầu vào | Đầu ra checkpoint |
|---|---|---|
| KB1 giai đoạn supervised | Pretrained YOLO gốc | kb1_reward_guided_training/checkpoint_based/<model>/weights/best.pt |
| KB1 reward guided qua train_rl.py | Best supervised tương ứng | kb1_reward_guided_training/checkpoint_reward_guide_trainning/<model>_seed<seed>_rl_best.pt |
| KB1 screening qua run_screening.py | Best supervised tương ứng | kb1_reward_guided_training/checkpoint_reward_guide_trainning/screening_seed42_v1/<model>/<native_only hoặc kb1b>/best.pt |
| KB2 sinh feedback | Best supervised của KB1 | kb2_preference_optimization/feedback_data_clean/<model>_train.jsonl và file summary cùng tên |
| KB2 feedback fine tuning | Best supervised KB1 và feedback cùng model | kb2_preference_optimization/checkpoint_preference_optimization/<model>_feedback_last_best.pt |
| KB2 các level RL cũ | Best supervised KB1 | kb2_preference_optimization/checkpoint_preference_optimization/<model>_rl_l*_best.pt; level 2 có hậu tố l2_anchor_dpo_best.pt |
| KB3-A và KB3-B | Pretrained YOLO gốc | kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/traditional_hpo/ và kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/adaptive_rl/ |

KB1 evaluate.py mặc định đọc checkpoint từ screening; dùng --checkpoint-source train_rl để đánh giá output của train_rl.py. Kết quả hai nguồn được lưu riêng trong results/canonical_clean/<nguồn>/<split>.

Đã xóa 217 checkpoint .pt do các lần chạy cũ sinh ra. Theo yêu cầu, 8 file .pt pretrained YOLO gốc cũng đã được xóa sau khi kiểm tra checksum. Hiện chỉ còn weights.pt trong .venv của torchmetrics; đây không phải checkpoint thí nghiệm. Thư mục output mới hiện chưa có checkpoint. Không dùng các báo cáo metric lịch sử để kết luận cho split mới.

Audit theo source ID trên split mới: train 2411, validation 479, test 239 ảnh; không phát hiện nhóm source ID trùng giữa các split. Audit này không thay thế kiểm tra trùng nội dung ảnh khác ID. Chi tiết: kb1_reward_guided_training/results/dataset_audit_clean/summary.json.
