"""
models/dp_yolo/patch_yolov5.py – Đăng ký DP-YOLO custom modules vào YOLOv5.

Chạy TRƯỚC KHI train DP-YOLO:
    python models/dp_yolo/patch_yolov5.py

Script này sẽ:
1. Import các module (D2C3, D3C3, PTCSP, C3Ghost, GhostBottleneck) từ modules.py
2. Đăng ký chúng vào yolov5/models/common.py (thêm vào globals)
3. Đăng ký vào yolov5/models/yolo.py (vào parse_model)

Sau khi patch, YOLOv5 có thể đọc dp_yolo.yaml với các tên module mới.

Lưu ý: Chỉ cần chạy 1 lần, hoặc gọi patch() trong train script.
"""

import sys
from pathlib import Path


def patch():
    """
    Patch runtime: inject custom modules vào YOLOv5 module registry.
    Gọi trong train_supervised.py trước khi train dp_yolo.
    """
    # Thêm yolov5 vào sys.path nếu chưa có
    yolov5_root = Path('yolov5')
    if str(yolov5_root) not in sys.path:
        sys.path.insert(0, str(yolov5_root))

    # Import YOLOv5 modules
    import models.common as common
    import models.yolo   as yolo_module

    # Import DP-YOLO modules
    dp_root = Path(__file__).parent
    if str(dp_root.parent.parent) not in sys.path:
        sys.path.insert(0, str(dp_root.parent.parent))

    from models.dp_yolo.modules import (
        D2C3, D3C3, PTCSP, C3Ghost, GhostBottleneck, GhostConv,
        DCNv2, DCNv3, TransformerLayer,
    )

    # Đăng ký vào common module (YOLOv5 tìm module theo tên trong common)
    for name, cls in [
        ('D2C3',            D2C3),
        ('D3C3',            D3C3),
        ('PTCSP',           PTCSP),
        ('C3Ghost',         C3Ghost),
        ('GhostBottleneck', GhostBottleneck),
        ('GhostConv',       GhostConv),
        ('DCNv2',           DCNv2),
        ('DCNv3',           DCNv3),
        ('TransformerLayer', TransformerLayer),
    ]:
        setattr(common, name, cls)

    print(f"  DP-YOLO modules patched into yolov5/models/common.py:")
    print(f"    D2C3, D3C3, PTCSP, C3Ghost, GhostBottleneck, DCNv2, DCNv3")

    return True


if __name__ == '__main__':
    ok = patch()
    if ok:
        print("\n  Kiểm tra patch bằng cách import:")
        import models.common as c
        print(f"    D2C3   OK: {hasattr(c, 'D2C3')}")
        print(f"    D3C3   OK: {hasattr(c, 'D3C3')}")
        print(f"    PTCSP  OK: {hasattr(c, 'PTCSP')}")
        print(f"    C3Ghost OK: {hasattr(c, 'C3Ghost')}")
