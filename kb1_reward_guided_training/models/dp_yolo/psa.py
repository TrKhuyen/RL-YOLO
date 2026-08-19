"""
models/dp_yolo/psa.py  –  PSA (Petal-like Sample Amplification) label assignment.

Thay thế build_targets mặc định của YOLOv5 bằng phiên bản mở rộng dùng hình tròn
bán kính r=1 (đơn vị grid) để chọn positive sample.

Nguyên tắc:
  - Standard YOLOv5: anchor tại cell (gi, gj) là positive nếu tâm GT nằm trong
    ±0.5 grid unit từ biên cell (5 cell: center + 4 neighbors).
  - PSA r=1: anchor là positive nếu tâm GT nằm trong hình tròn bán kính r=1
    xung quanh tâm cell, cho phép tối đa 9 cell thay vì 5.
  - Kết quả: hai vòng tròn từ hai cell kề nhau chồng lên nhau tạo hình dạng
    "cánh hoa" → tăng ~5% positive sample (từ tài liệu DP-YOLO).

Hàm `patch_psa()` monkey-patch `ComputeLoss.build_targets` trong utils.loss.

Tham khảo:
  Paper DP-YOLO: Wang et al., Applied Sciences 2023, Section 3.1 PSA.
"""

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình PSA
# ─────────────────────────────────────────────────────────────────────────────

PSA_RADIUS: float = 1.0   # bán kính hình tròn (đơn vị grid)

# Tất cả offset (dx, dy) trong hộp 3×3 quanh containing cell.
# Các cell ngoài vòng tròn sẽ bị loại bỏ bởi điều kiện distance.
_CANDIDATE_OFFSETS = [
    (dx, dy)
    for dx in range(-1, 2)
    for dy in range(-1, 2)
]  # 9 candidates: đủ để bao phủ r=1


# ─────────────────────────────────────────────────────────────────────────────
# PSA build_targets
# ─────────────────────────────────────────────────────────────────────────────

def _psa_build_targets(self, p, targets):
    """
    PSA-extended build_targets: tất cả cell trong vòng tròn bán kính PSA_RADIUS
    từ tâm GT đều là positive nếu anchor ratio thỏa mãn anchor_t.

    Thay thế ``ComputeLoss.build_targets`` qua monkey-patch.

    Signature giống hệt hàm gốc của YOLOv5:
        self:    ComputeLoss instance
        p:       list[Tensor] – predictions mỗi scale
        targets: Tensor (nt, 6) [image, class, cx, cy, w, h] normalized
    """
    na, nt = self.na, targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []

    # ── Trường hợp batch không có GT ────────────────────────────────────
    if nt == 0:
        for _ in range(self.nl):
            _empty = torch.zeros(0, dtype=torch.long, device=targets.device)
            tcls.append(_empty)
            tbox.append(torch.zeros(0, 4, device=targets.device))
            indices.append((_empty, _empty, _empty, _empty))
            anch.append(torch.zeros(0, 2, device=targets.device))
        return tcls, tbox, indices, anch

    # Gắn anchor index vào targets: (na, nt, 7) [image,cls,cx,cy,w,h,anch_idx]
    gain = torch.ones(7, device=targets.device, dtype=torch.float32)
    ai   = torch.arange(na, device=targets.device, dtype=torch.float32)
    ai   = ai.view(na, 1).repeat(1, nt)
    targets_aug = torch.cat(
        (targets.float().repeat(na, 1, 1), ai[..., None]), dim=2
    )  # (na, nt, 7)

    r2 = PSA_RADIUS ** 2   # so sánh distance² < r² thay vì sqrt

    for i in range(self.nl):
        anchors, shape = self.anchors[i], p[i].shape
        # gain: scale cx,cy,w,h sang grid coordinates của layer i
        # shape = (B, na, H, W, 5+nc) → shape[[3,2,3,2]] = [W, H, W, H]
        gain[2:6] = torch.tensor(
            [shape[3], shape[2], shape[3], shape[2]],
            device=targets.device, dtype=torch.float32,
        )

        # Scale targets sang grid coords
        t = targets_aug * gain            # (na, nt, 7)

        # ── Anchor ratio test ────────────────────────────────────────────
        # Loại bỏ (anchor, target) pairs mà tỉ lệ w/h không phù hợp
        ratio = t[..., 4:6] / anchors[:, None]   # (na, nt, 2)
        j     = torch.max(ratio, 1.0 / ratio).max(2).values < self.hyp['anchor_t']
        t     = t[j]        # (valid, 7): các cặp hợp lệ

        if len(t) == 0:
            _empty = torch.zeros(0, dtype=torch.long, device=targets.device)
            tcls.append(_empty)
            tbox.append(torch.zeros(0, 4, device=targets.device))
            indices.append((_empty, _empty, _empty, _empty))
            anch.append(torch.zeros(0, 2, device=targets.device))
            continue

        b    = t[:, 0].long()    # batch index
        c    = t[:, 1].long()    # class
        a    = t[:, 6].long()    # anchor index
        gxy  = t[:, 2:4]         # GT center (x, y) in grid coords
        gwh  = t[:, 4:6]         # GT w, h in grid coords

        H, W = shape[2], shape[3]

        # Accumulators cho layer i
        col_b, col_c, col_a = [], [], []
        col_gj, col_gi      = [], []
        col_tbox, col_anch  = [], []

        for dx, dy in _CANDIDATE_OFFSETS:
            # Candidate cell index: floor(GT center) + offset
            gi_cand = gxy[:, 0].long() + dx   # x-index (column, max W-1)
            gj_cand = gxy[:, 1].long() + dy   # y-index (row,    max H-1)

            # Khoảng cách từ tâm candidate cell đến tâm GT
            # Tâm cell (gi, gj) = (gi + 0.5, gj + 0.5)
            cx_cell = gi_cand.float() + 0.5
            cy_cell = gj_cand.float() + 0.5
            dist2   = (gxy[:, 0] - cx_cell) ** 2 + (gxy[:, 1] - cy_cell) ** 2

            # Điều kiện valid: trong vòng tròn PSA VÀ trong bounds grid
            valid = (
                (dist2 < r2) &
                (gi_cand >= 0) & (gi_cand < W) &
                (gj_cand >= 0) & (gj_cand < H)
            )

            if not valid.any():
                continue

            gi_v   = gi_cand[valid].clamp(0, W - 1)
            gj_v   = gj_cand[valid].clamp(0, H - 1)
            gxy_v  = gxy[valid]
            gwh_v  = gwh[valid]
            a_v    = a[valid]

            col_b.append(b[valid])
            col_c.append(c[valid])
            col_a.append(a_v)
            col_gi.append(gi_v)
            col_gj.append(gj_v)

            # tbox: (dx_from_cell, dy_from_cell, w, h)
            gij_v = torch.stack([gi_v.float(), gj_v.float()], dim=1)
            col_tbox.append(torch.cat((gxy_v - gij_v, gwh_v), dim=1))
            col_anch.append(anchors[a_v])

        if not col_b:
            # Không có cell nào hợp lệ (hiếm gặp, xảy ra khi tất cả nằm ngoài bounds)
            _empty = torch.zeros(0, dtype=torch.long, device=targets.device)
            tcls.append(_empty)
            tbox.append(torch.zeros(0, 4, device=targets.device))
            indices.append((_empty, _empty, _empty, _empty))
            anch.append(torch.zeros(0, 2, device=targets.device))
            continue

        tcls.append(torch.cat(col_c))
        tbox.append(torch.cat(col_tbox))
        indices.append((
            torch.cat(col_b),    # batch idx
            torch.cat(col_a),    # anchor idx
            torch.cat(col_gj),   # gj = y-index (row)
            torch.cat(col_gi),   # gi = x-index (col)
        ))
        anch.append(torch.cat(col_anch))

    return tcls, tbox, indices, anch


# ─────────────────────────────────────────────────────────────────────────────
# Patch function
# ─────────────────────────────────────────────────────────────────────────────

def patch_psa() -> bool:
    """
    Monkey-patch ``ComputeLoss.build_targets`` với PSA version.

    Gọi SAU KHI yolov5 đã được thêm vào sys.path.

    Returns:
        True nếu patch thành công, False nếu import thất bại.
    """
    try:
        import utils.loss as _loss_mod
        _loss_mod.ComputeLoss.build_targets = _psa_build_targets
        print(
            f"  ✓ PSA label assignment patched  →  "
            f"ComputeLoss.build_targets  (radius={PSA_RADIUS})"
        )
        return True
    except (ImportError, AttributeError) as e:
        print(f"  [WARN] patch_psa failed: {e}")
        print("  → Continuing with standard YOLOv5 label assignment.")
        return False
