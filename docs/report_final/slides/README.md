# Slide báo cáo tiến độ RL-YOLO

- `BAO_CAO_TIEN_DO_RL_YOLO_20261103.pptx`: bản trình chiếu gồm 32 slide chính và 4 phụ lục. Text, ô bảng và sơ đồ là đối tượng chỉnh sửa được; lời thuyết trình và nguồn chi tiết nằm trong Notes.
- `BAO_CAO_TIEN_DO_RL_YOLO_20261103.pdf`: bản xem nhanh, xuất từ HTML cùng dữ liệu bố cục.
- `BAO_CAO_TIEN_DO_RL_YOLO_20261103.html`: bản xem trong trình duyệt; không cần kết nối mạng.
- `TONG_QUAN_SLIDE.png`: ảnh tổng quan 36 slide.
- `build_slides.py`: mã sinh PowerPoint/HTML, dùng Python 3 và Pillow.

## Cần bổ sung trước khi báo cáo

1. Slide 01: tên nhóm, thành viên, GVHD và ngày trình bày.
2. Slide 04: biểu đồ phân bố class, ảnh dữ liệu và nhận xét của nhóm.
3. Slide 07: ảnh gốc, ảnh sau augmentation và nhận xét.
4. Slide 31: hiện là kế hoạch website; cập nhật nếu đã có kết quả triển khai thực tế.

Nguồn: kịch bản `../KICH_BAN_SLIDE_BAO_CAO_TIEN_DO_20261103.md` và ba report được dẫn ở đó. KB1 có test; KB2 là validation; KB3 là snapshot 15/10/2026. Không coi số liệu KB3 là tiến độ chạy trực tiếp ngày 03/11.

## Chỉnh sửa và tạo lại

Mở PowerPoint để chỉnh sửa trực tiếp, hoặc thay nội dung trong script rồi chạy:

```bash
python3 docs/report_final/slides/build_slides.py
```

Script ghi lại PPTX và HTML, không ghi lại PDF/ảnh tổng quan. Chỉnh sửa PowerPoint trực tiếp không được đồng bộ ngược vào script/HTML/PDF.

Font dùng: **DejaVu Sans**, có hỗ trợ tiếng Việt. Nếu máy trình chiếu thiếu font, cài font hoặc dùng PDF để giữ bố cục. PDF và ảnh tổng quan đã được rà trực quan; cấu trúc XML và toàn bộ đường dẫn relationship trong PPTX đã được kiểm tra. Chưa render lại PPTX bằng Microsoft PowerPoint/LibreOffice trong môi trường này.
