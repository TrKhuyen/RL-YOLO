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
import importlib.util
from pathlib import Path

# ── 1. Project root (kb1_reward_guided_training/) vào sys.path TRƯỚC yolov5/ ─────────────────────
# Lý do: cả kb1_reward_guided_training/ và yolov5/ đều có thư mục models/.
# Nếu yolov5/ đứng trước, `from models.dp_yolo...` sẽ lỗi.
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── 2. Thêm yolov5/ vào cuối sys.path (SAU project root) ────────────────────
_YOLOV5_ROOT = _HERE / 'yolov5'
if not _YOLOV5_ROOT.exists():
    print(f"[ERROR] yolov5/ not found at {_YOLOV5_ROOT}")
    print("  → git clone https://github.com/ultralytics/yolov5.git")
    sys.exit(1)

if str(_YOLOV5_ROOT) not in sys.path:
    sys.path.append(str(_YOLOV5_ROOT))  # append, không insert

# ── 3. Apply DP-YOLO patches TRƯỚC KHI import bất kỳ thứ gì từ YOLOv5 ───────
print("=" * 60)
print(" DP-YOLO Training Wrapper")
print("=" * 60)

# Load patch_yolov5 bằng đường dẫn tuyệt đối (tránh xung đột models/ namespace)
_patch_file = _HERE / 'models' / 'dp_yolo' / 'patch_yolov5.py'
_spec = importlib.util.spec_from_file_location('dp_yolo_patch', str(_patch_file))
_patch_mod = importlib.util.module_from_spec(_spec)
sys.modules['dp_yolo_patch'] = _patch_mod
_spec.loader.exec_module(_patch_mod)

if not _patch_mod.patch():
    print("[FATAL] Patch failed. Aborting.")
    sys.exit(1)

# ── 4. Chạy YOLOv5 training với patches đã active ───────────────────────────
# Import sau khi patch để các tên module đã được đăng ký
import train as yolo_train   # yolov5/train.py

# YOLOv5's AMP probe deep-copies the model and runs AutoShape inference twice
# at 640 px. That probe is unstable for torchvision deformable convolutions on
# Windows and may terminate the process before training starts. Keep DP-YOLO
# in FP32; standard YOLOv5 training still uses its normal AMP check.
yolo_train.check_amp = lambda model: False
_init_seeds = yolo_train.init_seeds
yolo_train.init_seeds = lambda seed=0, deterministic=True: _init_seeds(seed, deterministic=False)

if __name__ == '__main__':
    opt = yolo_train.parse_opt()
    try:
        yolo_train.main(opt)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
