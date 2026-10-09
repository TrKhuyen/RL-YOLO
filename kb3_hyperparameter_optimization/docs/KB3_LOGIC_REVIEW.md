# Review logic KB3-A và KB3-B

## Kết luận

Theo yêu cầu train từ đầu, worker hiện chỉ chấp nhận kiến trúc `.yaml`/`.yml`
và đặt `pretrained=False`. Mỗi episode/trial khởi tạo độc lập bằng seed.
File `run_all_kb3.sh` chạy các phương pháp và đánh giá lại trên seed mới.

Hai kịch bản hiện đã tách đúng về mặt phương pháp:

- **KB3-A – Traditional HPO:** mỗi trial chọn một cấu hình tuyệt đối trước khi train, giữ nguyên cấu hình trong toàn bộ run và dùng validation để chọn cấu hình tốt nhất.
- **KB3-B – Adaptive RL:** một episode giữ liên tục model, optimizer, scheduler và EMA; agent thay đổi hyperparameter sau mỗi segment dựa trên state và reward.

## Các lỗi logic đã sửa

1. **Reward segment đầu từng dùng metric giả.** Backend hiện dựng đúng model ngẫu nhiên với số lớp dataset và seed, đo validation epoch 0 trước segment đầu. Loss validation được đo bằng forward không gradient; không cập nhật trọng số trong bước này.
2. **Metric cuối từng bị dùng thay cho metric tốt nhất.** Kết quả hiện lưu riêng `metrics` của checkpoint tốt nhất và `final_metrics` của trạng thái cuối. Chất lượng dùng best; chi phí thời gian dùng final.
3. **HPO từng chấm checkpoint cuối.** Random Search và Optuna hiện nhận `best_metrics`/`best_checkpoint` từ worker để chọn trial.
4. **Checkpoint policy chưa resume đầy đủ.** PPO lưu model, optimizer và RNG của Torch/CUDA; bandit lưu bảng giá trị, số lượt chọn và RNG. Metadata lưu episode kế tiếp, lịch sử và snapshot cấu hình, đồng thời từ chối resume khi cấu hình thay đổi.
5. **Checkpoint và metric có thể không cùng model.** Environment ưu tiên cặp `best_checkpoint`/`best_metrics` do worker trả về, thay vì ghép checkpoint tốt nhất với metric cuối.
6. **So sánh chi phí từng bị thấp hơn thực tế.** `compare_hpo.py` hiện lấy `elapsed_seconds` từ `final_metrics`, kể cả khi checkpoint tốt nhất xuất hiện sớm.
7. **Scheduler/warmup từng ghi đè tham số RL.** Tham số optimizer và `scheduler.base_lrs` được cập nhật sau restore; mọi phương pháp tắt warmup/LR decay/close-mosaic để action có hiệu lực trong toàn segment.
8. **Checkpoint upstream từng chỉ resume từ EMA và optimizer fp16.** Worker bổ sung live model, optimizer và EMA đủ precision, scheduler vào checkpoint để khôi phục ở segment kế tiếp. `nbs=batch` tránh gradient đang tích lũy bị mất.
9. **Best epoch lệch một và mAP=0 không có checkpoint.** Epoch của best lấy từ checkpoint thật; epoch 0 chỉ làm mốc reward, segment thành công đầu tiên luôn cung cấp ứng viên đã train, kể cả mAP vẫn bằng 0.
10. **Chi phí canonical evaluation chưa tính đủ.** Worker tính elapsed sau khi canonical evaluation xong và cộng chi phí validation epoch 0 trong episode RL.

## Quy tắc thí nghiệm bắt buộc

- Search KB3-A có thể dùng chung một seed để giảm nhiễu khi chọn cấu hình; sau đó phải khóa cấu hình và đánh giá lại trên các seed chưa dùng.
- Policy KB3-B phải được khóa và chạy deterministic trên đúng các seed đánh giá của KB3-A.
- Không dùng test set để tính reward, chọn trial, chọn policy hoặc chọn checkpoint. Chỉ đánh giá test một lần sau khi khóa phương pháp.
- Nếu reward có trọng số `AP_small`, command Ultralytics phải bật `--canonical-eval`; nếu không, `AP_small` bằng 0 và thành phần reward đó không có tác dụng.
- So sánh phải báo cáo cả chất lượng và chi phí thực tế. Early stopping không được coi là cùng ngân sách epoch nếu một phương pháp đã dùng ít epoch hơn; cần kèm GPU-hour hoặc elapsed time.

## Phạm vi đã kiểm chứng

Unit test bao phủ contract command worker, continuity epoch, reward, action clipping, best-vs-final metrics, HPO cố định và RNG resume. Việc xác nhận kết luận nghiên cứu vẫn cần chạy ma trận nhiều seed trên GPU/dataset thật; backend mô phỏng chỉ kiểm tra logic phần mềm.

Integration test tùy chọn `KB3_TEST_ULTRALYTICS=1` kiểm tra initial validation,
hai segment train thật, resume optimizer/EMA và LR/momentum do action mới chọn.
Worker hiện tái tạo dataloader và seed RNG mỗi process; chưa chứng minh batch
trajectory tương đương một run không ngắt đoạn. So sánh nghiên cứu cần thêm
đối chứng fixed segmented khi muốn tách riêng ảnh hưởng của segment boundary.
