# Review ngân sách KB3 v4

**Review lịch sử v4.** Luồng mặc định hiện tại là [A–TPE bốn lượt](KB3_A_TPE_PROTOCOL.md).
B tạm dừng; xem [báo cáo đầy đủ](KB3_B_FUTURE_RESEARCH_REPORT.md).

V3 đặt 128 HPO + 128 PPO trước khi có bằng chứng cần ngân sách đó. Tổng 324 detector,
97.200 epoch/model gồm cả tuning/final quá lớn đối với GPU hiện có. 128 là lựa chọn
cấu hình, không được suy ra từ lý thuyết PPO hay độ phức tạp bốn hyperparameter.
V4 giữ vai trò YAML/checkpoint KB1 và cách train liên tục; thay cách cấp ngân sách.

## Luồng hiện tại

1. Đánh giá read-only checkpoint supervised KB1 bằng canonical evaluator.
2. Search đợt đầu: 4 HPO trial và 4 PPO episode. Mỗi detector vẫn scratch, horizon
   tối đa 300. Search dùng min_epochs=100, patience=50, min_delta=.0005 theo mAP50–95
   canonical validation; dừng tại boundary 5 epoch để PPO có terminal transition.
3. Dừng trước held-out tuning/final/test. Đọc `search_progress.json`, `epochs.json`,
   `decisions.json`, cập nhật PPO và chi phí thực. Learning episode scores của PPO
   là diagnostics của chính sách đang thay đổi, không phải phép so sánh policy đã khóa.
4. Nếu cần khám phá thêm, resume lên tổng 8 mỗi bên. Có thể yêu cầu mốc lớn hơn
   bằng `--search-target` (bội số của 4) trước khi chốt, vẫn giữ toàn bộ kết quả.
   Mặc định không tự mở rộng quá 8. Kết quả HPO và PPO optimizer,
   policy RNG, cursor, history/rollout được giữ. Chạy 4 rồi 8 có cùng lịch update PPO
   như chạy thẳng 8, với force update ở candidate boundary 4/8.
5. Khi chọn chốt, `--stage evaluate` khóa số lượt search trước khi dùng tuning seeds.
   Ở mốc 4: một ứng viên HPO và một policy; ở mốc 8: hai ứng viên mỗi bên.
   Mỗi ứng viên chạy đủ 300 epoch trên 2 tuning seeds mới, chọn theo mean validation.
6. Default, HPO, random-schedule và frozen PPO chạy đủ 300 epoch trên cùng 3 final
   detector seeds mới. Chọn canonical best từng lượt, báo mean/std và paired differences.
7. `--stage test` chỉ inference trên checkpoint đã khóa bằng validation, không train.

`run_all_kb3.sh` mặc định chỉ đến bước 3. `--finalize` là lựa chọn rõ ràng để tự chạy
tiếp bước 5–7. Không mở rộng search sau khi đã bắt đầu tuning: kết quả held-out không
được dùng để điều khiển ngân sách cùng thí nghiệm. Đổi protocol/source/data phải tạo
thư mục mới; v4 không nạp policy/detector cũ của v3.

## Chi phí và ý nghĩa

| Mốc | Search | Tuning | Final | Tối đa epoch |
|---|---:|---:|---:|---:|
| Đợt đầu, chưa chốt | 8 | 0 | 0 | 2.400 |
| Chốt ở 4 mỗi bên | 8 | 4 | 12 | 7.200 |
| Chốt ở 8 mỗi bên | 16 | 8 | 12 | 10.800 |
| V3 trước đây | 256 | 48 | 20 | 97.200 |

Số 4/8 là điểm kiểm tra và trần thực dụng ban đầu, không phải số lượt tối ưu được
chứng minh. Bốn episode có thể mới chỉ tạo vài PPO updates; không kết luận RL hội tụ
hay thua HPO chỉ từ đợt này. Điều kiện dừng có thể bỏ lỡ model cải thiện muộn; tuning
và final đủ 300 giúp kiểm chứng ứng viên còn lại, không khắc phục việc bỏ sót ứng viên.
Lịch LR của search vẫn theo horizon 300, không nén lịch xuống thời điểm dừng thực tế.
Search dừng trước pha close-mosaic có thể chưa trải qua pha này; tuning/final luôn có.
Đây là đánh đổi để sàng lọc trong ngân sách thực tế, không bảo đảm điểm tốt nhất tuyệt đối.

A/B có cùng số lượt và quy tắc dừng search, nhưng thời gian GPU thực có thể khác.
`search_progress.json` và `comparison.json` ghi actual epochs/hours. Tuning/final dùng
cùng horizon và seed; chỉ có một lần học policy PPO, ba final seeds là ba detector
khác nhau dưới cùng policy, không phải ba lần học RL độc lập. KB1 pretrained vẫn
là mốc chất lượng lịch sử; nhóm scratch default mới giúp đo riêng tác dụng tối ưu.

## Validation

- Unit/regression suite: 68 tests, 67 pass, 1 optional legacy integration skipped.
- Train thật trên ảnh giả 64 px: canonical early stop ở epoch 4/8, giữ best epoch 2,
  hoàn tất terminal reward; cùng cấu hình final không dừng sớm và chạy đủ 8 epoch.
- Staged search 4 → 8 giữ kết quả đã xong; policy tensors/actions/RNG bằng direct 8.
- Gián đoạn giữa hai episode tiếp tục đúng target cũ, không lặp episode đã commit.
- Selection lock ngăn mở rộng sau held-out tuning; budgets HPO/PPO tuning bằng nhau.
- Mở rộng vượt mốc 8 không train lại các lượt đã hoàn thành; số ứng viên tuning
  của A/B vẫn bằng nhau, cùng seed. Kiểm tra cả xung đột seed ở target lớn hơn.
- End-to-end smoke và preflight dataset thực được lưu trong các log v4; không bắt
  đầu lượt train nghiên cứu 300 epoch trong quá trình sửa.

Nguồn lý thuyết: [PPO paper](https://arxiv.org/abs/1707.06347) mô tả luân phiên thu thập
trải nghiệm và cập nhật objective, không quy định 128 detector episodes cho bài toán này.
[Ultralytics train documentation](https://docs.ultralytics.com/modes/train/) mô tả
epochs/patience; v4 dùng stopping riêng theo canonical mAP thay cho native fitness.
Implementation được đối chiếu với source Ultralytics 8.4.163 đang cài trong workspace.

Lệnh chạy một dòng dành cho PowerShell nằm trong [protocol](KB3_QUALITY_PROTOCOL.md).
