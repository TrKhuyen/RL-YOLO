# Đối chứng ghép cặp native/DPO từ TPE — 16/11/2026

Sáu lượt chạy mới: seed42/43/44 × λ0/λ0,03. Cùng TPE trial2, bank, stream trong mỗi seed, mask cv3, native background0,5, AdamW LR1e−6; LR giảm10 lần từ update1001; eval mỗi250 tới2000 rồi mỗi1000;15.000 update mỗi lượt.

Tiêu chí chính khóa trước: Δ mAP tại step500. Best và last15000 là tiêu chí bổ sung. Best dùng min_delta1e−4 fraction và cho phép step0; không thay tiêu chí sau khi xem số liệu.

| Phép so | Seed | Step native / DPO | Native mAP (%) | DPO mAP (%) | Δ DPO−native, điểm % | Δ FP | Δ FN |
|---|---:|---|---:|---:|---:|---:|---:|
| Cùng step500 — tiêu chí chính | 42 | 500 / 500 | 38.661698 | 38.713256 | +0.051558 | +0 | +0 |
| Best theo cùng selector/grid | 42 | 250 / 500 | 38.685122 | 38.713256 | +0.028133 | -20 | +8 |
| Cùng step15000 | 42 | 15000 / 15000 | 38.551301 | 38.516697 | -0.034603 | +2 | +0 |
| Cùng step500 — tiêu chí chính | 43 | 500 / 500 | 38.671300 | 38.685235 | +0.013936 | -1 | +0 |
| Best theo cùng selector/grid | 43 | 500 / 500 | 38.671300 | 38.685235 | +0.013936 | -1 | +0 |
| Cùng step15000 | 43 | 15000 / 15000 | 38.643399 | 38.644430 | +0.001031 | +0 | +1 |
| Cùng step500 — tiêu chí chính | 44 | 500 / 500 | 38.672855 | 38.673171 | +0.000316 | +0 | +0 |
| Best theo cùng selector/grid | 44 | 500 / 500 | 38.672855 | 38.673171 | +0.000316 | +0 | +0 |
| Cùng step15000 | 44 | 15000 / 15000 | 38.592610 | 38.588393 | -0.004217 | +1 | +1 |

| Phép so | Δ mAP trung bình ± SD mẫu, điểm % | Seed DPO cao hơn |
|---|---:|---:|
| Cùng step500 — tiêu chí chính | +0.021936 ± 0.026541 | 3/3 |
| Best theo cùng selector/grid | +0.014128 ± 0.013910 | 3/3 |
| Cùng step15000 | -0.012596 ± 0.019238 | 1/3 |

## Diễn giải

Tại step500, DPO cao hơn native ở 3/3 seed, thấp hơn ở 0/3 seed, Δ trung bình +0.021936 điểm phần trăm.
Đây là đối chứng can thiệp trọng số DPO trong recipe cố định; chênh lệch ghép cặp kiểm tác dụng thêm DPO trên validation này. Không quy toàn bộ gain hybrid−TPE cho DPO: native continuation là đối chứng cần so.
Nếu dấu thay đổi giữa seed hoặc mức tăng rất nhỏ, không khẳng định lợi ích DPO ổn định. Nếu DPO thắng cả ba seed, bằng chứng chỉ áp dụng cho ba continuation seed từ một TPE và validation đã dùng phát triển; chưa chứng minh tổng quát hóa độc lập hoặc ưu thế mọi metric.
Seed42 và step500 từng được dùng phát triển recipe; khóa phép so trước campaign mới không biến validation này thành holdout mới. SD là độ lệch chuẩn mẫu qua ba chênh lệch ghép cặp, không phải khoảng tin cậy hoặc kiểm định ý nghĩa thống kê.

## Kiểm chứng

Đã kiểm lại config/stream/hash, selector và lịch LR của sáu run. Gradient auxiliary của native bằng0 ở các mốc diagnostic. Đã nạp lại18 checkpoint theo ba phép so × ba seed × hai arm và đánh giá đủ479 ảnh/1887GT; metric chính và per-class khớp log. Frozen-state giữ nguyên. File DPO reloaded có source_gt_transitions so trực tiếp với native tại cùng phép so.

[Kết quả máy đọc được](paired_results.json), [CSV](paired_results.csv), [xác minh](verification.json), [quỹ đạo](validation.csv), [kế hoạch khóa trước](plan.json).
