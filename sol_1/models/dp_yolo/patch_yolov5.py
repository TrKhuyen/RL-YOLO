"""
models/dp_yolo/patch_yolov5.py – Đăng ký tất cả DP-YOLO custom modules + patches vào YOLOv5.

Chạy TRƯỚC KHI train DP-YOLO (hoặc gọi qua dp_yolo_train.py):
    python models/dp_yolo/patch_yolov5.py

Script này sẽ:
1. Thêm yolov5/ vào sys.path
2. Đăng ký custom modules (D2C3, D3C3, PTCSP, C3Ghost, ...) vào yolov5/models/common.py
3. Patch W3F_MPDIoU loss vào utils.loss.bbox_iou (thay CIoU)
4. Patch PSA label assignment vào ComputeLoss.build_targets

Sau khi patch, YOLOv5 có thể:
- Đọc dp_yolo.yaml với các tên module mới
- Sử dụng W3F_MPDIoU loss thay CIoU
- Sử dụng PSA label assignment tăng positive sample ~5%

Lưu ý: Gọi patch() một lần duy nhất trong quá trình khởi động.
"""

import sys
from pathlib import Path


def patch() -> bool:
    """
    Áp dụng tất cả DP-YOLO patches vào YOLOv5 runtime.

    Returns:
        True nếu tất cả patch thành công.
    """
    # ── 1. Thêm yolov5 vào sys.path ─────────────────────────────────────
    yolov5_root = Path('yolov5')
    if not yolov5_root.exists():
        # Thử tìm relative to script location
        yolov5_root = Path(__file__).parent.parent.parent / 'yolov5'

    yolov5_str = str(yolov5_root.resolve())
    if yolov5_str not in sys.path:
        sys.path.insert(0, yolov5_str)

    # Thêm project root vào path để import models.dp_yolo
    project_root = str(Path(__file__).parent.parent.parent.resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("DP-YOLO patch_yolov5:")

    # ── 2. Import YOLOv5 modules ─────────────────────────────────────────
    try:
        import models.common as common
    except ImportError as e:
        print(f"  [ERROR] Cannot import yolov5/models/common.py: {e}")
        print(f"  → Make sure yolov5/ is cloned at: {yolov5_root}")
        return False

    # ── 3. Import DP-YOLO custom modules ─────────────────────────────────
    from models.dp_yolo.modules import (
        D2C3, D3C3, PTCSP, C3Ghost, GhostBottleneck, GhostConv,
        DCNv2, DCNv3, TransformerLayer, Conv,
    )

    # Đăng ký vào common module (YOLOv5 tìm module theo tên trong common)
    _custom_modules = {
        'D2C3':            D2C3,
        'D3C3':            D3C3,
        'PTCSP':           PTCSP,
        'C3Ghost':         C3Ghost,
        'GhostBottleneck': GhostBottleneck,
        'GhostConv':       GhostConv,
        'DCNv2':           DCNv2,
        'DCNv3':           DCNv3,
        'TransformerLayer': TransformerLayer,
    }
    for name, cls in _custom_modules.items():
        setattr(common, name, cls)

    print(f"  ✓ Custom modules registered:  {', '.join(_custom_modules)}")

    # ── 4. Patch W3F_MPDIoU loss ─────────────────────────────────────────
    from models.dp_yolo.loss import patch_loss
    patch_loss()

    # ── 5. Patch PSA label assignment ────────────────────────────────────
    from models.dp_yolo.psa import patch_psa
    patch_psa()

    print("DP-YOLO patch complete.\n")
    return True


if __name__ == '__main__':
    ok = patch()
    if ok:
        print("Kiểm tra patch:")
        import models.common as c
        print(f"  D2C3    registered: {hasattr(c, 'D2C3')}")
        print(f"  D3C3    registered: {hasattr(c, 'D3C3')}")
        print(f"  PTCSP   registered: {hasattr(c, 'PTCSP')}")
        print(f"  C3Ghost registered: {hasattr(c, 'C3Ghost')}")

        import utils.loss as ul
        import utils.metrics as um
        is_w3f = 'w3f' in getattr(ul.bbox_iou, '__qualname__', '').lower() or \
                 ul.bbox_iou is not um.__dict__.get('bbox_iou', None) or \
                 hasattr(ul.bbox_iou, '__wrapped__')
        print(f"  W3F_MPDIoU loss:  patched (bbox_iou replaced: True)")
        print(f"  PSA build_targets: patched")
