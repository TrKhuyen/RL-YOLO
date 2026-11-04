# KB2 — đối chiếu các lượt chạy với S0, 01/11/2026

Theo mục tiêu đã làm rõ: ưu tiên cải thiện chất lượng detector, lấy mAP50–95 làm metric chính; precision/recall/FP/FN mô tả đánh đổi. Không bắt buộc mọi metric cùng tăng. Đây là tổng hợp các thí nghiệm YOLOv8n gần đây đã có full validation; không tính smoke/kiểm kỹ thuật hoặc các kiến trúc trong báo cáo lịch sử15/10.

Tất cả bảng dùng479 ảnh validation/1.887GT; precision/recall/FP/FN tại confidence0,25, IoU0,5. mAP tính từ predictions confidence0,001, NMS0,6, cap300. ΔmAP là điểm phần trăm, ΔFN dương là bỏ sót nhiều hơn S0.

Hai nhóm dùng cùng checkpoint supervised S0 nhưng chế độ số học khác nhau: S0 legacy mAP37,8084%/FN867, S0 conditional TF32-off mAP37,8218%/FN866. Chỉ tính delta với S0 tương ứng, không xếp hạng trực tiếp hai nhóm như một thí nghiệm cùng protocol.

## Nhóm conditional gần đây: checkpoint đã train ở cuối lượt

| Lượt chạy | Seed | Step / checkpoint | mAP50–95 (%) | Precision (%) | Recall (%) | FP | FN | ΔmAP (điểm %) | ΔFN |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | — | 0 / supervised | 37,8218 | 50,9227 | 54,1070 | 984 | 866 | 0 | 0 |
| Conditional background 1 / native | 42 | 2000 / last | 37,8882 | 54,3430 | 51,7223 | 820 | 911 | 0,0665 | +45 |
| Conditional background 1 / dpo | 42 | 2000 / last | 37,8863 | 54,3430 | 51,7223 | 820 | 911 | 0,0645 | +45 |
| Conditional background 1 / native | 42 | 15000 / last | 38,0195 | 57,1174 | 51,0334 | 723 | 924 | 0,1977 | +58 |
| Conditional background 1 / dpo | 42 | 15000 / last | 38,0096 | 57,0836 | 51,0334 | 724 | 924 | 0,1878 | +58 |
| Conditional background 0.5 / native | 42 | 2000 / last | 38,0150 | 51,4445 | 53,7891 | 958 | 872 | 0,1933 | +6 |
| Conditional background 0.5 / dpo | 42 | 2000 / last | 38,0141 | 51,4184 | 53,7891 | 959 | 872 | 0,1924 | +6 |
| Conditional background 0.5 / native | 42 | 15000 / last | 38,0099 | 53,2730 | 53,0472 | 878 | 886 | 0,1882 | +20 |
| Conditional background 0.5 / dpo | 42 | 15000 / last | 38,0072 | 53,1881 | 53,0472 | 881 | 886 | 0,1854 | +20 |

Các dòng trên là last.pt đã train. Tất cả best.pt của nhóm này vẫn có trọng số S0 do selector cũ yêu cầu micro recall>=S0; không có nghĩa S0 có mAP cao nhất. Các điểm trung gian không được selector nhận thường chỉ có metrics, không có checkpoint lưu riêng.

## Nhóm scorer F1/F2 và recovery trước đó

F1/F2 trong tên là scorer preference. F1/F2 dùng best checkpoint theo mAP của từng seed, không phải trung bình hoặc checkpoint cuối lượt. Recovery pilot lấy điểm2k; recovery long ghi riêng best14k và last15k. Bảng này không so cùng ngân sách train/selection với nhóm conditional.

| Lượt chạy | Seed | Step / checkpoint | mAP50–95 (%) | Precision (%) | Recall (%) | FP | FN | ΔmAP (điểm %) | ΔFN |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | — | 0 / supervised | 37,8084 | 50,8982 | 54,0541 | 984 | 867 | 0 | 0 |
| DPO scorer F1 | 42 | 9000 / best_AP | 38,2432 | 57,3547 | 51,2454 | 719 | 920 | 0,4347 | +53 |
| DPO scorer F2 | 42 | 9000 / best_AP | 38,2570 | 57,3887 | 51,2454 | 718 | 920 | 0,4486 | +53 |
| DPO scorer F1 | 43 | 9000 / best_AP | 38,2147 | 57,3468 | 51,0864 | 717 | 923 | 0,4063 | +56 |
| DPO scorer F2 | 43 | 9000 / best_AP | 38,2150 | 57,3897 | 51,0334 | 715 | 924 | 0,4066 | +57 |
| DPO scorer F1 | 44 | 14000 / best_AP | 38,2989 | 57,7456 | 50,7684 | 701 | 929 | 0,4905 | +62 |
| DPO scorer F2 | 44 | 10000 / best_AP | 38,2361 | 57,3731 | 50,9274 | 714 | 926 | 0,4277 | +59 |
| Recovery pilot control | 42 | 2000 / step_2000 | 37,9698 | 55,2961 | 51,4573 | 785 | 916 | 0,1614 | +49 |
| Recovery pilot selector | 42 | 2000 / step_2000 | 37,9583 | 55,2961 | 51,4573 | 785 | 916 | 0,1499 | +49 |
| Recovery pilot balanced | 42 | 2000 / step_2000 | 37,9690 | 55,2961 | 51,4573 | 785 | 916 | 0,1606 | +49 |
| Recovery balanced long | 42 | 14000 / best_AP | 38,2458 | 57,5721 | 50,7684 | 706 | 929 | 0,4374 | +62 |
| Recovery balanced long | 42 | 15000 / last | 38,2174 | 57,7550 | 50,7154 | 700 | 930 | 0,4090 | +63 |

## So DPO và native-only ở cùng FP984 trong nhóm conditional

Đây là cutoff được quét hậu nghiệm trên validation, không đổi threshold chính hoặc checkpoint selection. S0 có FN866 ở budget984.

| Recipe | Bước | FN DPO | FN native-only | DPO hơn native bao nhiêu GT |
|---|---:|---:|---:|---:|
| Background1 | 2.000 | 864 | 864 | 0 |
| Background1 | 15.000 | 860 | 860 | 0 |
| Background0,5 | 2.000 | 862 | 864 | 2 |
| Background0,5 | 15.000 | 859 | 860 | 1 |

## Đánh giá theo mục tiêu cải thiện metric

- Mọi checkpoint train được liệt kê có mAP cao hơn S0 tương ứng. Các model cũng có precision cao hơn và FP thấp hơn, kèm recall thấp hơn/FN cao hơn tại0,25.
- Trong nhóm F1/F2/recovery đã lưu best theo AP, F1 seed44 best14k có AP cao nhất38,2989%, tăng0,4905 điểm so S0 của nhóm; đây là checkpoint thực tế đã lưu. Nó chưa chứng minh lợi ích riêng DPO vì không có native-only đối chứng khớp cho thí nghiệm này.
- Trong các endpoint conditional được liệt kê, native-only background1/15k có AP cao nhất38,0195%, DPO background0,5/2k cao nhất trong bốn endpoint DPO38,0141%. Khác biệt giữa các endpoint rất nhỏ, chưa xác nhận ổn định qua nhiều seed.
- Không nhầm bảng endpoint với AP cao nhất từng log: DPO background1/14k từng log38,0692%, nhưng không lưu checkpoint14k riêng khi recall gate từ chối; không coi trọng số đó sẵn có để dùng.
- Có thể ghi nhận cải thiện validation so S0 theo mục tiêu AP/precision/FP. Chưa đủ bằng chứng DPO tạo lợi ích thêm lớn so native-only, hoặc cải thiện tổng quát trên test độc lập. Không tuyên bố model tốt hơn mọi mặt.
- Việc chuyển tiêu chí đánh giá chính sang mAP là mục tiêu nghiên cứu mới được làm rõ. Tổng hợp này không sửa code selector, không ghi đè checkpoint, không train lại hoặc chọn threshold mới cho deployment.

## Nguồn số liệu

- [CSV từng lượt và delta](assets/kb2_s0_comparison_20261101/runs.csv).
- [F1/F2: từng seed](assets/kb2_f2_20261028/per_seed.csv).
- [Recovery pilot](KB2_RECOVERY_PILOT_20261029.md), [recovery long](KB2_RECOVERY_LONG_REVIEW_20261030.md).
- [Conditional pilot](KB2_CONDITIONAL_PILOT_REVIEW_20261031.md), [conditional long](KB2_CONDITIONAL_LONG_REVIEW_20261031.md).
- [Background0,5 pilot](KB2_NATIVE_BACKGROUND_PILOT_20261031.md), [background0,5 long](KB2_NATIVE_BACKGROUND_LONG_REVIEW_20261031.md).
