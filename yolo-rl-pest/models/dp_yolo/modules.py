"""
models/dp_yolo/modules.py – Custom modules cho DP-YOLO.

Cung cấp:
  D2C3  – C3 block với DCNv2 deformable convolution (stage 1-3 backbone)
  D3C3  – C3 block với DCNv3-style group deformable conv (stage 4, "3+1")
  PTCSP – Parallel CNN + Transformer CSP block (neck P2)
  C3Ghost / GhostBottleneck – đã có trong YOLOv5 nhưng re-export để tiện

Yêu cầu: torchvision >= 0.8 (có DeformConv2d) hoặc dcnv2 package
"""

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def autopad(k, p=None):
    """Tự động tính padding để output_size == input_size / stride."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard Conv + BN + SiLU (giống YOLOv5 Conv)."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p),
                              dilation=d, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = (self.default_act if act is True
                     else act if isinstance(act, nn.Module)
                     else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# ─────────────────────────────────────────────────────────────────────────────
# Deformable Conv v2  (DCNv2)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from torchvision.ops import DeformConv2d as _TorchDeformConv2d
    _HAVE_DEFORM = True
except ImportError:
    _HAVE_DEFORM = False
    warnings.warn("torchvision.ops.DeformConv2d not found. "
                  "Falling back to regular Conv2d for D2C3/D3C3. "
                  "Install torchvision >= 0.8 for full DP-YOLO.")


class DCNv2(nn.Module):
    """
    Deformable Convolution v2.

    Tính offset + mask bằng 1 conv phụ, sau đó chạy DeformConv2d.
    offset shape : (B, 2*kh*kw, H, W)
    mask  shape  : (B, kh*kw,   H, W)  → sigmoid → [0,1]
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1,
                 p: Optional[int] = None, g: int = 1):
        super().__init__()
        p = autopad(k, p)
        self.stride = s
        self.k = k

        if _HAVE_DEFORM:
            self.dcn = _TorchDeformConv2d(
                c1, c2, kernel_size=k, stride=s, padding=p,
                groups=g, bias=False,
            )
        else:
            # Fallback: plain Conv2d (không có deformable effect)
            self.dcn = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)

        # Offset + mask predictor
        self.offset_mask = nn.Conv2d(
            c1, 3 * k * k,  # 2*k*k offset + k*k mask
            kernel_size=k, stride=s, padding=p, bias=True,
        )
        nn.init.constant_(self.offset_mask.weight, 0.0)
        nn.init.constant_(self.offset_mask.bias,   0.0)

        self.bn  = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        om = self.offset_mask(x)
        offset = om[:, :2 * self.k * self.k]
        mask   = om[:, 2 * self.k * self.k:].sigmoid()
        if _HAVE_DEFORM:
            out = self.dcn(x, offset, mask)
        else:
            out = self.dcn(x)
        return self.act(self.bn(out))


# ─────────────────────────────────────────────────────────────────────────────
# D2C3 – C3 block với DCNv2
# ─────────────────────────────────────────────────────────────────────────────

class D2Bottleneck(nn.Module):
    """Bottleneck: 1×1 Conv → DCNv2 3×3 → 1×1 Conv, với shortcut."""
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1,
                 e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_,  1, 1)
        self.cv2 = DCNv2(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class D2C3(nn.Module):
    """
    C3 block thay thế tất cả 3×3 Conv bằng DCNv2.
    Dùng cho stage 1, 2, 3 của backbone DP-YOLO.
    """
    def __init__(self, c1: int, c2: int, n: int = 1,
                 shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m   = nn.Sequential(
            *[D2Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# D3C3 – C3 block với DCNv3 (group deformable, "3+1" strategy)
# ─────────────────────────────────────────────────────────────────────────────

class DCNv3(nn.Module):
    """
    DCNv3-style: group deformable conv với nhiều nhóm offset độc lập.

    Cách tiếp cận "3+1": 3 bottleneck dùng Conv thường, 1 bottleneck
    cuối mới dùng deformable → tránh overfitting ở feature map nhỏ.

    Nếu không có torchvision DeformConv2d, fallback về depthwise+pointwise.
    """
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1,
                 groups: int = 4):
        super().__init__()
        assert c1 % groups == 0, f"c1 ({c1}) phải chia hết cho groups ({groups})"
        p = autopad(k)

        if _HAVE_DEFORM:
            self.dcn = _TorchDeformConv2d(
                c1, c2, kernel_size=k, stride=s, padding=p,
                groups=groups, bias=False,
            )
        else:
            # Fallback: depthwise + pointwise
            self.dcn = nn.Sequential(
                nn.Conv2d(c1, c1, k, s, p, groups=c1, bias=False),
                nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            )

        self.offset_mask = nn.Conv2d(
            c1, groups * 3 * k * k,
            kernel_size=k, stride=s, padding=p, bias=True,
        )
        nn.init.constant_(self.offset_mask.weight, 0.0)
        nn.init.constant_(self.offset_mask.bias,   0.0)

        self.bn  = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.groups = groups
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAVE_DEFORM:
            om     = self.offset_mask(x)
            g      = self.groups
            k2     = self.k * self.k
            offset = om[:, :2 * g * k2]
            mask   = om[:, 2 * g * k2:].sigmoid()
            out = self.dcn(x, offset, mask)
        else:
            out = self.dcn(x)
        return self.act(self.bn(out))


class D3Bottleneck(nn.Module):
    """Bottleneck với DCNv3 ở conv 3×3."""
    def __init__(self, c1: int, c2: int, shortcut: bool = True,
                 groups: int = 4, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DCNv3(c_, c2, 3, 1, groups=groups)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class D3C3(nn.Module):
    """
    C3 block với chiến lược "3+1":
    - Bottleneck 0..n-2: Conv thường
    - Bottleneck n-1:    DCNv3

    Dùng cho stage 4 của backbone.
    """
    def __init__(self, c1: int, c2: int, n: int = 1,
                 shortcut: bool = True, g: int = 4, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        # Tất cả bottleneck trước dùng Conv thường
        plain = [D2Bottleneck(c_, c_, shortcut, g=1, e=1.0)
                 for _ in range(max(0, n - 1))]
        # Bottleneck cuối dùng DCNv3
        deform = [D3Bottleneck(c_, c_, shortcut, groups=min(g, c_), e=1.0)]
        self.m = nn.Sequential(*plain, *deform)

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# PTCSP – Parallel CNN + Transformer CSP block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerLayer(nn.Module):
    """
    1 layer Transformer đơn giản:
    Multi-head Self-Attention → Add & Norm → FFN → Add & Norm.
    """
    def __init__(self, c: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(c, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        self.ffn   = nn.Sequential(
            nn.Linear(c, 4 * c),
            nn.GELU(),
            nn.Linear(4 * c, c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → flatten → attention → reshape
        B, C, H, W = x.shape
        seq = x.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.norm1(seq + attn_out)
        seq = self.norm2(seq + self.ffn(seq))
        return seq.permute(0, 2, 1).view(B, C, H, W)


class PTCSP(nn.Module):
    """
    Parallel CNN + Transformer CSP (PTCSP) – dùng ở P2 neck.

    Cấu trúc:
    ┌─ cv1 ─→ [CNN branch: n × Conv 3×3] ─────────────────┐
    │                                                       concat → cv3
    └─ cv2 ─→ [Transformer branch: n × TransformerLayer] ──┘

    Ưu điểm: CNN nắm local texture, Transformer nắm global context.
    Đặc biệt quan trọng ở P2 (resolution cao, object nhỏ).
    """
    def __init__(self, c1: int, c2: int, n: int = 1,
                 shortcut: bool = False, num_heads: int = 4, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        # CNN branch
        self.cnn_branch = nn.Sequential(
            *[Conv(c_, c_, 3, 1) for _ in range(n)]
        )
        # Transformer branch
        self.tr_branch = nn.Sequential(
            *[TransformerLayer(c_, num_heads=min(num_heads, c_ // 8))
              for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn = self.cnn_branch(self.cv1(x))
        tr  = self.tr_branch(self.cv2(x))
        return self.cv3(torch.cat([cnn, tr], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# C3Ghost / GhostBottleneck  (neck DP-YOLO)
# ─────────────────────────────────────────────────────────────────────────────

class GhostConv(nn.Module):
    """Ghost Convolution: 1×1 cheap op + depthwise conv."""
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1,
                 g: int = 1, act: bool = True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], dim=1)


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck (MobileNetV3-style)."""
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),
            Conv(c_, c_, k, s, None, c_, act=False) if s == 2 else nn.Identity(),
            GhostConv(c_, c2, 1, 1, act=False),
        )
        self.shortcut = (
            nn.Sequential(
                Conv(c1, c1, k, s, None, c1, act=False),
                Conv(c1, c2, 1, 1, None, 1,  act=False),
            ) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class C3Ghost(nn.Module):
    """C3 block với GhostBottleneck thay thế Bottleneck thường."""
    def __init__(self, c1: int, c2: int, n: int = 1,
                 shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m   = nn.Sequential(
            *[GhostBottleneck(c_, c_) for _ in range(n)]
        )

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))
