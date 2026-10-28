# Lưu trữ KB2 trước thiết kế 26/10/2026

Các file ở đây là source/tài liệu lịch sử (gồm các chỉnh sửa đã có trong
workspace trước đợt thiết kế lại). Chúng được giữ để tránh mất lịch sử, không
phải các entrypoint hiện hành. Không chạy các script cũ để tạo run mới.

Theo yêu cầu người dùng, checkpoint và feedback cũ trong riêng KB2 đã xóa.
`cleanup_manifest.json` ghi đường dẫn, kích thước và SHA256 của từng file đã
xóa. `results/` giữ result JSON/metrics JSONL nhỏ phục vụ truy nguyên báo cáo;
không giữ bản sao trọng số. Không xóa checkpoint supervised KB1 hoặc data chung.

Objective hiện hành là `structured_detection_dpo_v1`, xem [../README.md](../README.md).
Các source cũ phụ thuộc relative imports/paths của bố cục cũ; archive không
được bảo đảm chạy độc lập trong vị trí mới và không được discover test mặc định.
