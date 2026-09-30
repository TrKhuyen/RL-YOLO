# Kiểm tra nguồn gốc 8 file pretrained YOLO

Trạng thái sau audit: theo yêu cầu người dùng, cả 8 file trong bảng đã được xóa khỏi workspace. Bảng checksum là bằng chứng trước khi xóa; các lần chạy sau sẽ tải lại khi cần. Một file weights.pt thuộc torchmetrics vẫn nằm trong .venv.

Ngày kiểm tra: 2026-09-26. So SHA256 toàn tệp và từng tensor storage trong ZIP checkpoint.

| File | SHA256 tại máy | Kết quả |
|---|---|---|
| yolo11n.pt | 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1 | Trùng toàn tệp với Ultralytics/YOLO11 |
| yolo11s.pt | 85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5 | Trùng toàn tệp với Ultralytics/YOLO11 |
| yolo26n.pt | 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef | Trùng toàn tệp với Ultralytics/YOLO26 |
| kb1_reward_guided_training/yolo26n.pt | 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef | Bản sao trùng toàn tệp |
| kb1_reward_guided_training/yolov5s.pt | 8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee | Trùng toàn tệp với Ultralytics/YOLOv5 |
| yolov8n.pt | f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 | 359/359 tensor storage trùng SHA256 với asset YOLOv8n v0.0.0; metadata khác |
| kb1_reward_guided_training/yolov8n.pt | f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 | Bản sao trùng toàn tệp của file trên |
| yolov8s.pt | 1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a | 359/359 tensor storage trùng SHA256 với asset YOLOv8s v0.0.0; metadata khác |

Nguồn đối chiếu: [Ultralytics YOLOv5](https://huggingface.co/Ultralytics/YOLOv5/blob/main/yolov5s.pt), [YOLO11n](https://huggingface.co/Ultralytics/YOLO11/blob/main/yolo11n.pt), [YOLO11s](https://huggingface.co/Ultralytics/YOLO11/blob/main/yolo11s.pt), [YOLO26n](https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26n.pt), [Ultralytics assets v0.0.0](https://github.com/ultralytics/assets/releases/tag/v0.0.0).

Metadata YOLOv8 tại máy ghi version=8.0.0.dev0, epoch=-1, model COCO 80 lớp. SHA256 toàn tệp khác bản v0.0.0 vì phần metadata và cách lưu khác; tất cả tensor storage dùng cho trọng số đều khớp. Repository không lưu manifest tải xuống và không theo dõi file .pt bằng Git, nên không thể chứng minh chính xác lần tải hoặc thời điểm tệp được ghi lại. Về nội dung tensor, các file này là trọng số pretrained ban đầu, không phải checkpoint đã fine-tune trên PlantDoc.
