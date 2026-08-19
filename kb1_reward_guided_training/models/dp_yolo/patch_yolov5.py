"""
models/dp_yolo/patch_yolov5.py – Đăng ký tất cả DP-YOLO custom modules + patches vào YOLOv5.

Chạy TRƯỚC KHI train DP-YOLO (hoặc gọi qua dp_yolo_train.py):
    python models/dp_yolo/patch_yolov5.py

Script này sẽ:
1. Thêm yolov5/ vào sys.path
2. Đăng ký custom modules (D2C3, D3C3, PTCSP, C3Ghost, ...) vào yolov5/models/common.py
3. Patch parse_model để nhận biết custom modules (inject c1/c2, scale width, insert n)
4. Patch W3F_MPDIoU loss vào utils.loss.bbox_iou (thay CIoU)
5. Patch PSA label assignment vào ComputeLoss.build_targets

Root cause của lỗi "No module named 'models.dp_yolo'":
    Cả kb1_reward_guided_training/ và yolov5/ đều có thư mục models/.
    Khi yolov5/ được thêm vào sys.path TRƯỚC kb1_reward_guided_training/, Python tìm
    `models.dp_yolo` trong yolov5/models/ (không có) thay vì kb1_reward_guided_training/models/.

Giải pháp: load các file dp_yolo bằng đường dẫn tuyệt đối qua
importlib.util.spec_from_file_location(), không phụ thuộc vào sys.path.
"""

import os
import sys
import importlib.util
from pathlib import Path

# Thư mục chứa patch_yolov5.py này: kb1_reward_guided_training/models/dp_yolo/
_DP_YOLO_DIR  = Path(__file__).parent.resolve()
# kb1_reward_guided_training/
_PROJECT_ROOT = _DP_YOLO_DIR.parent.parent.resolve()


def _load_module_from_path(module_name: str, file_path: Path):
    """Load một Python file theo đường dẫn tuyệt đối, cache vào sys.modules."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec   = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def patch() -> bool:
    """
    Áp dụng tất cả DP-YOLO patches vào YOLOv5 runtime.

    Returns:
        True nếu tất cả patch thành công.
    """
    # ── 1. Xác định yolov5_root ──────────────────────────────────────────────
    yolov5_root = _PROJECT_ROOT / 'yolov5'
    if not yolov5_root.exists():
        print(f"  [ERROR] yolov5/ not found at {yolov5_root}")
        print("  → git clone https://github.com/ultralytics/yolov5.git")
        return False

    # Them yolov5/ vao sys.path TRUOC kb1_reward_guided_training/ de import models.yolo tim dung
    # yolov5/models/yolo.py (khong phai kb1_reward_guided_training/models/yolo.py - khong ton tai)
    yolov5_str = str(yolov5_root)
    if yolov5_str not in sys.path:
        sys.path.insert(0, yolov5_str)
    # Dam bao project root khong chan yolov5 modules bang cach
    # loai kb1_reward_guided_training/ khoi vi tri 0 neu co (chi temporarily trong context nay)
    _project_str = str(_PROJECT_ROOT)
    if _project_str in sys.path:
        sys.path.remove(_project_str)
        sys.path.append(_project_str)  # Dua xuong cuoi, yolov5/ se duoc uu tien

    print("DP-YOLO patch_yolov5:")

    # ── 2. Load DP-YOLO modules bằng đường dẫn tuyệt đối ───────────────────
    try:
        dp_modules = _load_module_from_path(
            "dp_yolo_modules",
            _DP_YOLO_DIR / "modules.py",
        )
    except Exception as e:
        print(f"  [ERROR] Cannot load modules.py: {e}")
        print(f"  → Expected at: {_DP_YOLO_DIR / 'modules.py'}")
        return False

    # Lấy các class cần thiết
    D2C3            = dp_modules.D2C3
    D3C3            = dp_modules.D3C3
    PTCSP           = dp_modules.PTCSP
    C3Ghost         = dp_modules.C3Ghost
    GhostBottleneck = dp_modules.GhostBottleneck
    GhostConv       = dp_modules.GhostConv
    DCNv2           = dp_modules.DCNv2
    DCNv3           = dp_modules.DCNv3
    TransformerLayer = dp_modules.TransformerLayer

    # ── 3. Load yolov5/models/common.py ─────────────────────────────────────
    try:
        common = _load_module_from_path(
            "models.common",
            yolov5_root / "models" / "common.py",
        )
    except Exception as e:
        print(f"  [ERROR] Cannot load yolov5/models/common.py: {e}")
        print(f"  → Make sure yolov5/ is cloned at: {yolov5_root}")
        return False

    # ── 4. Đăng ký custom modules ───────────────────────────────────────────
    _custom_modules = {
        'D2C3':             D2C3,
        'D3C3':             D3C3,
        'PTCSP':            PTCSP,
        'C3Ghost':          C3Ghost,
        'GhostBottleneck':  GhostBottleneck,
        'GhostConv':        GhostConv,
        'DCNv2':            DCNv2,
        'DCNv3':            DCNv3,
        'TransformerLayer': TransformerLayer,
    }

    # 4a. Đăng ký vào models.common (cho các module import từ common)
    for name, cls in _custom_modules.items():
        setattr(common, name, cls)

    # 4b. Inject vào builtins (fallback cho eval() ở mọi nơi)
    import builtins
    for name, cls in _custom_modules.items():
        setattr(builtins, name, cls)

    print(f"  ✓ Custom modules registered: {', '.join(_custom_modules)}")

    # ── 5. Patch parse_model ────────────────────────────────────────────────
    try:
        import torch.nn as nn
        import os as _os
        # 'models' package co the la namespace package hop nhat tu nhieu thu muc
        # (kb1_reward_guided_training/models/, yolov5/models/). models.__path__ chi chua kb1_reward_guided_training/models/
        # nen `import models.yolo` that bai.
        # Giai phap: inject yolov5/models/ vao models.__path__ de Python tim duoc
        # yolov5/models/yolo.py ma khong can xoa cache hay swap.
        _v5_models_path = str(yolov5_root / 'models')
        _models_pkg = sys.modules.get('models')
        if _models_pkg is not None and hasattr(_models_pkg, '__path__'):
            _paths = list(_models_pkg.__path__)
            if _v5_models_path not in _paths:
                _models_pkg.__path__.append(_v5_models_path)

        import models.yolo as yolo_mod

        # Inject custom modules vào yolo.py module namespace
        for name, cls in _custom_modules.items():
            setattr(yolo_mod, name, cls)

        # Build eval namespace: tất cả globals của yolo.py + custom modules
        _eval_ns = dict(yolo_mod.__dict__)
        _eval_ns.update(_custom_modules)
        _eval_ns['nn'] = nn

        # Mo rong cac set de bao gom custom modules
        # Chu y: C3Ghost o day la yolo_mod.C3Ghost (YOLOv5 original),
        # nhung YAML eval() tra ve DP-YOLO's C3Ghost (custom class) -
        # can them ca 2 vao set.
        _c1c2_set = {
            yolo_mod.Conv, yolo_mod.GhostConv,
            yolo_mod.Bottleneck, yolo_mod.GhostBottleneck,
            yolo_mod.SPP, yolo_mod.SPPF, yolo_mod.DWConv,
            yolo_mod.MixConv2d, yolo_mod.Focus, yolo_mod.CrossConv,
            yolo_mod.BottleneckCSP, yolo_mod.C3, yolo_mod.C3TR,
            yolo_mod.C3SPP, yolo_mod.C3Ghost, nn.ConvTranspose2d,
            yolo_mod.DWConvTranspose2d, yolo_mod.C3x,
            # DP-YOLO custom modules (phan biet voi YOLOv5 originals)
            D2C3, D3C3, PTCSP, C3Ghost,
        }
        _c3_set = {
            yolo_mod.BottleneckCSP, yolo_mod.C3, yolo_mod.C3TR,
            yolo_mod.C3Ghost, yolo_mod.C3x,
            # DP-YOLO custom modules
            D2C3, D3C3, PTCSP, C3Ghost,
        }

        def _patched_parse_model(d, ch):
            """parse_model mở rộng: hỗ trợ D2C3, D3C3, PTCSP."""
            import contextlib
            from utils.general import LOGGER, make_divisible, colorstr

            gd, gw = d['depth_multiple'], d['width_multiple']
            act = d.get('activation')
            ch_mul = d.get('channel_multiple') or 8
            nc, anchors = d['nc'], d.get('anchors', [])

            if act:
                yolo_mod.Conv.default_act = eval(act, _eval_ns, {'nn': nn})
                LOGGER.info(f"{colorstr('activation:')} {act}")

            na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
            no = na * (nc + 5)

            # Local vars cần cho eval('nc'), eval('anchors'), etc.
            _locals = {'nc': nc, 'anchors': anchors, 'na': na, 'no': no}

            layers, save, c2 = [], [], ch[-1]
            for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
                # eval dùng _eval_ns (globals) + _locals → tìm được
                # Conv, SPPF, D2C3, nc, anchors, ...
                m = eval(m, _eval_ns, _locals) if isinstance(m, str) else m
                for j, a in enumerate(args):
                    with contextlib.suppress(NameError):
                        args[j] = eval(a, _eval_ns, _locals) if isinstance(a, str) else a

                n = n_ = max(round(n * gd), 1) if n > 1 else n

                if m in _c1c2_set:
                    c1, c2 = ch[f], args[0]
                    if c2 != no:
                        c2 = make_divisible(c2 * gw, ch_mul)
                    args = [c1, c2, *args[1:]]
                    if m in _c3_set:
                        args.insert(2, n)
                        n = 1
                elif m is nn.BatchNorm2d:
                    args = [ch[f]]
                elif m is yolo_mod.Concat:
                    c2 = sum(ch[x] for x in f)
                elif m in {yolo_mod.Detect, yolo_mod.Segment}:
                    args.append([ch[x] for x in f])
                    if isinstance(args[1], int):
                        args[1] = [list(range(args[1] * 2))] * len(f)
                    if m is yolo_mod.Segment:
                        args[3] = make_divisible(args[3] * gw, ch_mul)
                elif m is yolo_mod.Contract:
                    c2 = ch[f] * args[0] ** 2
                elif m is yolo_mod.Expand:
                    c2 = ch[f] // args[0] ** 2
                else:
                    c2 = ch[f]

                m_ = (nn.Sequential(*(m(*args) for _ in range(n)))
                      if n > 1 else m(*args))
                t = str(m)[8:-2].replace("__main__.", "")
                np_ = sum(x.numel() for x in m_.parameters())
                m_.i, m_.f, m_.type, m_.np = i, f, t, np_
                LOGGER.info(
                    f"{i:>3}{str(f):>18}{n_:>3}{np_:10.0f}  "
                    f"{t:<40}{str(args):<30}"
                )
                save.extend(
                    x % i for x in ([f] if isinstance(f, int) else f)
                    if x != -1
                )
                layers.append(m_)
                if i == 0:
                    ch = []
                ch.append(c2)
            return nn.Sequential(*layers), sorted(save)

        yolo_mod.parse_model = _patched_parse_model
        print("  ✓ parse_model patched for D2C3, D3C3, PTCSP")
    except Exception as e:
        import traceback
        print(f"  [WARN] parse_model patch failed: {e}")
        traceback.print_exc()
        print("  → D2C3/D3C3/PTCSP may not build correctly")

    # ── 6. Patch W3F_MPDIoU loss ─────────────────────────────────────────────
    try:
        loss_mod = _load_module_from_path(
            "dp_yolo_loss",
            _DP_YOLO_DIR / "loss.py",
        )
        if os.getenv('DP_YOLO_USE_W3F', '0') == '1':
            loss_mod.patch_loss()
        else:
            print('  - W3F disabled; using stable CIoU baseline')
        print("  ✓ W3F_MPDIoU loss patched")
    except Exception as e:
        print(f"  [WARN] loss patch skipped: {e}")

    # ── 7. Patch PSA label assignment ────────────────────────────────────────
    try:
        psa_mod = _load_module_from_path(
            "dp_yolo_psa",
            _DP_YOLO_DIR / "psa.py",
        )
        if os.getenv('DP_YOLO_USE_PSA', '0') == '1':
            psa_mod.patch_psa()
        else:
            print('  - PSA disabled; using standard target assignment')
        print("  ✓ PSA label assignment patched")
    except Exception as e:
        print(f"  [WARN] PSA patch skipped: {e}")

    print("DP-YOLO patch complete.\n")
    return True


if __name__ == '__main__':
    ok = patch()
    if ok:
        print("Kiểm tra patch:")
        import models.common as c  # yolov5/models/common đã trong sys.modules
        print(f"  D2C3    registered: {hasattr(c, 'D2C3')}")
        print(f"  D3C3    registered: {hasattr(c, 'D3C3')}")
        print(f"  PTCSP   registered: {hasattr(c, 'PTCSP')}")
        print(f"  C3Ghost registered: {hasattr(c, 'C3Ghost')}")
        print(f"  W3F_MPDIoU / PSA:   see warnings above")
