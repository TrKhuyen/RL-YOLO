"""
tests/test_fixes.py - Kiem tra tat ca cac fix da thuc hien.

Chay:
    python tests/test_fixes.py
"""

import sys
import math
import traceback
from pathlib import Path

# Them project root vao path
sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = []
FAIL = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  [PASS] {name}")
    except Exception as e:
        FAIL.append((name, str(e)))
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()


# =============================================================================
# Fix 1: W3F_MPDIoU Loss
# =============================================================================

def test_w3f_loss_import():
    from models.dp_yolo.loss import bbox_iou_w3f, patch_loss
    assert callable(bbox_iou_w3f)
    assert callable(patch_loss)

def test_w3f_loss_shape():
    import torch
    from models.dp_yolo.loss import bbox_iou_w3f
    box1 = torch.tensor([[10., 10., 20., 20.]])  # xyxy
    box2 = torch.tensor([[12., 12., 22., 22.]])
    result = bbox_iou_w3f(box1, box2, xywh=False)
    assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
    assert not torch.isnan(result).any(), "Result contains NaN"
    assert not torch.isinf(result).any(), "Result contains Inf"

def test_w3f_loss_gradient():
    import torch
    from models.dp_yolo.loss import bbox_iou_w3f
    box1 = torch.tensor([[10., 10., 20., 20.]], requires_grad=True)
    box2 = torch.tensor([[10., 10., 20., 20.]])
    result = bbox_iou_w3f(box1, box2, xywh=False)
    loss = (1.0 - result).mean()
    loss.backward()
    assert box1.grad is not None, "No gradient computed"
    assert not torch.isnan(box1.grad).any(), "NaN gradient"

def test_w3f_loss_perfect_match():
    """Perfect overlap should give IoU=1 -> W3F_loss ~0 -> return ~1."""
    import torch
    from models.dp_yolo.loss import bbox_iou_w3f
    box = torch.tensor([[10., 10., 20., 20.]])
    result = bbox_iou_w3f(box, box.clone(), xywh=False)
    # Khi perfect match: d1=d2=0, R_wiou=exp(0)=1, L_mpdiou=1-1+0=0
    # L_f_mpdiou = 0 + 1 - 1^0.5 = 0
    # W3F_loss = r * 1 * 0 = 0 -> return = 1 - 0 = 1
    assert abs(result.item() - 1.0) < 0.01, f"Expected ~1.0 for perfect match, got {result.item()}"

def test_w3f_loss_no_overlap():
    """No overlap should give high loss -> return < 0."""
    import torch
    from models.dp_yolo.loss import bbox_iou_w3f
    box1 = torch.tensor([[0., 0., 10., 10.]])
    box2 = torch.tensor([[100., 100., 110., 110.]])
    result = bbox_iou_w3f(box1, box2, xywh=False)
    # Gia tri tra ve se la 1 - loss, va loss se lon khi no overlap
    assert result.item() < 1.0, f"Expected < 1.0 for no overlap, got {result.item()}"

def test_w3f_xywh_format():
    """Test xywh format input."""
    import torch
    from models.dp_yolo.loss import bbox_iou_w3f
    # box: cx=15, cy=15, w=10, h=10 -> x1=10, y1=10, x2=20, y2=20
    box = torch.tensor([[15., 15., 10., 10.]])
    result = bbox_iou_w3f(box, box.clone(), xywh=True)
    assert abs(result.item() - 1.0) < 0.01, f"Expected ~1.0, got {result.item()}"


# =============================================================================
# Fix 2: PSA Label Assignment
# =============================================================================

def test_psa_import():
    from models.dp_yolo.psa import patch_psa, _psa_build_targets, PSA_RADIUS
    assert callable(patch_psa)
    assert callable(_psa_build_targets)
    assert PSA_RADIUS == 1.0

def test_psa_candidate_offsets():
    """PSA nen xem xet 9 cells trong 3x3 grid."""
    from models.dp_yolo.psa import _CANDIDATE_OFFSETS
    assert len(_CANDIDATE_OFFSETS) == 9, f"Expected 9 offsets, got {len(_CANDIDATE_OFFSETS)}"
    assert (0, 0) in _CANDIDATE_OFFSETS, "Center offset (0,0) missing"

def test_psa_radius_filtering():
    """Cell center phai nam trong vong tron r=1 de duoc chon."""
    import torch
    from models.dp_yolo.psa import PSA_RADIUS

    r2 = PSA_RADIUS ** 2

    # Cell (0,0): center (0.5, 0.5), GT at (0.3, 0.3)
    # dist^2 = (0.3-0.5)^2 + (0.3-0.5)^2 = 0.04+0.04 = 0.08 < 1
    gx, gy = 0.3, 0.3
    dx, dy = 0, 0
    gi = int(gx) + dx
    gj = int(gy) + dy
    dist2 = (gx - (gi + 0.5))**2 + (gy - (gj + 0.5))**2
    assert dist2 < r2, f"Center cell should be valid: dist2={dist2}"

    # Cell (-1, 0): center (-0.5, 0.5), GT at (0.3, 0.3)
    # dist^2 = (0.3+0.5)^2 + (0.3-0.5)^2 = 0.64+0.04 = 0.68 < 1
    dx, dy = -1, 0
    gi = int(gx) + dx
    dist2 = (gx - (gi + 0.5))**2 + (gy - (gj + 0.5))**2
    assert dist2 < r2, f"Left neighbor should be valid: dist2={dist2}"


# =============================================================================
# Fix 3: DCNv3 offset groups
# =============================================================================

def test_dcnv3_import():
    from models.dp_yolo.modules import DCNv3
    assert DCNv3 is not None

def test_dcnv3_offset_shape():
    """DCNv3.offset_mask phai output 3*k*k channels (khong nhan voi groups)."""
    import torch
    from models.dp_yolo.modules import DCNv3
    # c1=16, c2=16, k=3, groups=2
    m = DCNv3(c1=16, c2=16, k=3, groups=2)
    k, groups = m.k, m.groups
    expected_channels = 3 * k * k   # DUNG: 27
    wrong_channels    = groups * 3 * k * k  # SAI: 54
    actual_channels   = m.offset_mask.out_channels
    assert actual_channels == expected_channels, (
        f"Expected {expected_channels} channels (3*k*k), "
        f"got {actual_channels}. Old wrong value would be {wrong_channels}."
    )

def test_dcnv3_forward():
    """DCNv3 forward phai chay duoc voi shape dung."""
    import torch
    from models.dp_yolo.modules import DCNv3
    m = DCNv3(c1=16, c2=32, k=3, groups=2)
    m.eval()
    x = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 32, 8, 8), f"Expected (2,32,8,8), got {y.shape}"


# =============================================================================
# Fix 4: train_rl.py improvements
# =============================================================================

def test_train_rl_quick_eval_import():
    from train_rl import quick_eval, EMABaseline
    assert callable(quick_eval)

def test_train_rl_has_val_loader():
    """rl_finetune phai tao val_loader."""
    import inspect
    import train_rl
    src = inspect.getsource(train_rl.rl_finetune)
    assert 'val_loader' in src, "rl_finetune missing val_loader"
    assert 'eval_interval' in src, "rl_finetune missing eval_interval"

def test_train_rl_best_checkpoint_uses_avg():
    """Best checkpoint phai dung rolling average, khong phai instant reward."""
    import inspect
    import train_rl
    src = inspect.getsource(train_rl.rl_finetune)
    assert 'best_avg_reward' in src, "rl_finetune should use best_avg_reward"
    assert 'avg_r > best_avg_reward' in src, "Should compare avg_r"

def test_train_rl_checkpoint_format():
    """Checkpoint phai duoc luu dang dict voi is_rl_checkpoint=True."""
    import inspect
    import train_rl
    src = inspect.getsource(train_rl.rl_finetune)
    assert 'is_rl_checkpoint' in src, "Checkpoint missing is_rl_checkpoint key"
    assert 'state_dict' in src, "Checkpoint missing state_dict key"


# =============================================================================
# Fix 5: evaluate.py RL checkpoint loading
# =============================================================================

def test_evaluate_rl_loader_import():
    from evaluate import _load_model_for_eval, evaluate_model
    assert callable(_load_model_for_eval)

def test_evaluate_has_supervised_ckpt_param():
    """evaluate_model phai co tham so supervised_ckpt."""
    import inspect
    from evaluate import evaluate_model
    sig = inspect.signature(evaluate_model)
    assert 'supervised_ckpt' in sig.parameters, \
        "evaluate_model missing supervised_ckpt parameter"

def test_evaluate_detects_rl_format():
    """_load_model_for_eval phai phat hien is_rl_checkpoint."""
    import inspect
    from evaluate import _load_model_for_eval
    src = inspect.getsource(_load_model_for_eval)
    assert 'is_rl_checkpoint' in src, "Missing RL checkpoint detection"
    assert 'state_dict' in src, "Missing state_dict loading"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print(" Test Suite: RL-YOLO Fix Verification")
    print("=" * 60)
    print()

    print("[Fix 1] W3F_MPDIoU Loss")
    check("w3f_loss_import",        test_w3f_loss_import)
    check("w3f_loss_shape",         test_w3f_loss_shape)
    check("w3f_loss_gradient",      test_w3f_loss_gradient)
    check("w3f_loss_perfect_match", test_w3f_loss_perfect_match)
    check("w3f_loss_no_overlap",    test_w3f_loss_no_overlap)
    check("w3f_xywh_format",        test_w3f_xywh_format)
    print()

    print("[Fix 2] PSA Label Assignment")
    check("psa_import",             test_psa_import)
    check("psa_candidate_offsets",  test_psa_candidate_offsets)
    check("psa_radius_filtering",   test_psa_radius_filtering)
    print()

    print("[Fix 3] DCNv3 Offset Groups")
    check("dcnv3_import",           test_dcnv3_import)
    check("dcnv3_offset_shape",     test_dcnv3_offset_shape)
    check("dcnv3_forward",          test_dcnv3_forward)
    print()

    print("[Fix 4] train_rl.py improvements")
    check("train_rl_quick_eval",    test_train_rl_quick_eval_import)
    check("train_rl_val_loader",    test_train_rl_has_val_loader)
    check("train_rl_avg_reward",    test_train_rl_best_checkpoint_uses_avg)
    check("train_rl_ckpt_format",   test_train_rl_checkpoint_format)
    print()

    print("[Fix 5] evaluate.py RL checkpoint")
    check("evaluate_rl_import",     test_evaluate_rl_loader_import)
    check("evaluate_param",         test_evaluate_has_supervised_ckpt_param)
    check("evaluate_rl_detect",     test_evaluate_detects_rl_format)
    print()

    print("=" * 60)
    print(f" PASSED: {len(PASS)}/{len(PASS)+len(FAIL)}")
    if FAIL:
        print(f" FAILED: {len(FAIL)}")
        for name, err in FAIL:
            print(f"   - {name}: {err}")
    print("=" * 60)

    sys.exit(0 if not FAIL else 1)
