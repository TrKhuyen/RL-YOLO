# Biểu đồ quá trình train Hybrid HPO–DPO

Nguồn tạo: notebooks/plot_hybrid_training_20261113.ipynb — Run All.
Không train lại, không load weights, không inference.

- 01_hpo_search: objective canonical giữa các trial, không phải đường train liên tục.
- 02_hpo_all_trials: native mAP/train loss/val loss theo epoch của từng trial độc lập.
- 03_hpo_focus_trial: loss thành phần, native metrics và actual LR của trial 2.
- 04_kb2_training_losses: native, DPO, weighted-DPO và total loss tính lại từ log.
- 05_kb2_validation_map: canonical mAP cả run và zoom 0–2000 update.
- 06_kb2_operating_metrics: P/R/F1 và TP/FP/FN tại confidence0,25.
- 07_kb2_diagnostics: LR, gradient, preference margin, foreground/background cls.
- 08_hybrid_two_stage_overview: hai panel riêng, ghi đúng evaluator và trục.

HPO winner: trial 2, selected epoch 139.
KB2 selected: update 500, mAP 38.7133% trong budget 15000.
HPO native metric KHÁC canonical KB2; không ghép hai đường để kết luận gain.
Metric mỗi checkpoint reload nằm trong checkpoints_canonical_reload.csv, nếu có.
Loss smoothing: trailing mean 15 log records, không smoothing validation.
Không có KB2 validation loss trong log; không tự dựng dữ liệu đó.

Mỗi hình được xuất theo SAVE_FORMATS; có các CSV để vẽ lại.
provenance.json ghi config, protocol và SHA256 các log đầu vào.
