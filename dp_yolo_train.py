"""
dp_yolo_train.py – Wrapper để train DP-YOLO với tất cả patches được áp dụng.

Thay vì gọi trực tiếp `python yolov5/train.py`, dùng script này để:
1. Apply DP-YOLO patches (modules, W3F_MPDIoU loss, PSA label assignment)
2. Sau đó chạy YOLOv5 training với các patches đã active

Cách dùng:
    python dp_yolo_train.py \\
        --weights yolov5s.pt \\
        --cfg models/dp_yolo/dp_yolo.yaml \\
        --data configs/pest.yaml \\
        --hyp configs/hyp.pest.yaml \\
        --epochs 300 --batch-size 32 \\
        --optimizer SGD --device 0 \\
        --project checkpoints --name dp_yolo --exist-ok

Tất cả CLI arguments được truyền thẳng sang yolov5/train.py.
"""

import sys
from pathlib import Path

# ── 1. Thêm project root vào path để import models.dp_yolo ──────────────────
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── 2. Thêm yolov5/ vào path ─────────────────────────────────────────────────
_YOLOV5_ROOT = _HERE / 'yolov5'
if not _YOLOV5_ROOT.exists():
    print(f"[ERROR] yolov5/ not found at {_YOLOV5_ROOT}")
    print("  → git clone https://github.com/ultralytics/yolov5.git")
    sys.exit(1)

if str(_YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(_YOLOV5_ROOT))

# ── 3. Apply DP-YOLO patches TRƯỚC KHI import bất kỳ thứ gì từ YOLOv5 ───────
print("=" * 60)
print(" DP-YOLO Training Wrapper")
print("=" * 60)

from models.dp_yolo.patch_yolov5 import patch
if not patch():
    print("[FATAL] Patch failed. Aborting.")
    sys.exit(1)

# ── 4. Chạy YOLOv5 training với patches đã active ───────────────────────────
# Import sau khi patch để các tên module đã được đăng ký
import train as yolo_train   # yolov5/train.py

if __name__ == '__main__':
    opt = yolo_train.parse_opt()
    yolo_train.main(opt)
