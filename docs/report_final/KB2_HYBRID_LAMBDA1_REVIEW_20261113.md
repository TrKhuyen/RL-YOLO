# Hybrid KB2 — thử λ=1 so với λ=0,03

Giữ TPE trial2, bank773 cặp, seed42, background0,5, β0,1, AdamW LR1e−6; LR giảm sau1000, eval mỗi250 tới2000 rồi mỗi1000, budget15000 update.
Stream ảnh/pair được xác minh giống hệt; chỉ dpo_weight thay đổi. Không đánh giá test hoặc thay model chốt.

| Checkpoint | Step | mAP50–95 (%) | AP50 (%) | AP75 (%) | Precision (%) | Recall (%) | F1 (%) | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TPE_step0 | 0 | 38.5097 | 50.4306 | 43.4945 | 50.8763 | 55.3789 | 53.0322 | 1045 | 1009 | 842 |
| lambda003_best | 500 | 38.7133 | 50.6287 | 43.7080 | 52.6208 | 54.2660 | 53.4307 | 1024 | 922 | 863 |
| lambda003_last | 15000 | 38.5167 | 50.3684 | 43.5123 | 53.8916 | 53.2061 | 53.5467 | 1004 | 859 | 883 |
| lambda100_best | 1000 | 38.6985 | 50.5825 | 43.6869 | 52.2205 | 54.2130 | 53.1981 | 1023 | 936 | 864 |
| lambda100_last | 15000 | 38.5783 | 50.4260 | 43.6601 | 52.9657 | 53.9481 | 53.4523 | 1018 | 904 | 869 |
| lambda100_best_positive_update_by_raw_AP | 1000 | 38.6985 | 50.5825 | 43.6869 | 52.2205 | 54.2130 | 53.1981 | 1023 | 936 | 864 |

Best λ1 ở step1000; so best λ0,03 ở step500: ΔmAP=-0.0148 điểm phần trăm, ΔP=-0.4002, ΔR=-0.0530, ΔF1=-0.2326; FP+14, FN+1.

Ở cùng update15000, ΔmAP λ1−λ0,03 = +0.0616 điểm phần trăm. Do đó λ0,03 nhỉnh hơn theo selected-best, trong khi λ1 cao hơn về mAP khi chỉ so last15k.

Đã nạp lại best/last λ1 và chấm đủ479 ảnh/1887GT; các metric khớp log validation (sai lệch bằng0), đồng thời xác nhận frozen-state và checkpoint SHA256.

| Batch đầu, cùng model/ảnh/pair | λ=0,03 | λ=1 |
|---|---:|---:|
| Native loss | 2.964051 | 2.964051 |
| λ × DPO loss | 0.020794 | 0.693147 |
| Native gradient norm | 3.825481 | 3.825481 |
| Weighted-DPO gradient norm | 0.082913 | 2.763752 |
| Weighted-DPO/native norm ratio | 2.1674% | 72.2459% |

Batch đầu: weighted-DPO norm λ1/λ0,03=33.333334, khớp tỷ lệ1/0,03 trước khi cộng native gradient/clipping/AdamW. Không coi đó là tỷ lệ update weights.

Selector cho phép TPE step0 thắng nếu continuation không cải thiện đủ min_delta1e−4. Dòng best_positive_update chỉ là AP cao nhất trong các checkpoint đã train, không thay selector.

Một seed continuation và validation phát triển; kết quả chưa xác lập λ tối ưu phổ quát.

![So sánh theo update](assets/kb2_hybrid_lambda1_20261113/lambda_comparison.png)

Artifact: [review.json](assets/kb2_hybrid_lambda1_20261113/review.json), [summary.csv](assets/kb2_hybrid_lambda1_20261113/summary.csv).
