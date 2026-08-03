"""
augment.py – Group Augmentation cho GRPO-style RL Fine-tuning.

Ý tưởng từ GRPO (DeepSeek):
  LLM:  Sinh G text responses cho 1 prompt → tính group reward
  YOLO: Sinh G augmented views cho 1 ảnh  → tính group reward

Augmentation càng diverse → reward variance trong group càng lớn →
group advantage signal càng rõ ràng → training càng ổn định.

Cấu trúc:
  view 0 = ảnh gốc (không augment) → reward cao nhất (reference)
  view 1 = augment nhẹ (flip, brightness)
  view 2 = augment vừa (rotation, crop)
  view 3 = augment mạnh (blur, noise, occlusion)

Lưu ý normalize:
  DataLoader normalize ảnh với ImageNet mean/std.
  Các phép augment pixel-level (brightness, contrast, noise) cần
  hoạt động trên [0,1] → hàm này tự denorm/renorm trước/sau.
"""

import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# ImageNet normalization constants (phải khớp với dataloader.py)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _denorm(images: torch.Tensor) -> torch.Tensor:
    """Chuyển tensor normalized (ImageNet) → [0, 1] pixel space."""
    mean = _MEAN.to(images.device)
    std  = _STD.to(images.device)
    return (images * std + mean).clamp(0.0, 1.0)


def _renorm(images: torch.Tensor) -> torch.Tensor:
    """Chuyển [0, 1] pixel space → normalized (ImageNet)."""
    mean = _MEAN.to(images.device)
    std  = _STD.to(images.device)
    return (images - mean) / std


# =============================================================================
# 1. Per-view Augmentation (cấp độ tăng dần theo g_idx)
# =============================================================================

def augment_view(images: torch.Tensor, g_idx: int) -> torch.Tensor:
    """
    Sinh 1 augmented view tương ứng với group index g_idx.

    Cấp độ augmentation:
      g_idx=0: gốc (không augment) – reference view
      g_idx=1: nhẹ (flip ngẫu nhiên, brightness)
      g_idx=2: vừa (flip + rotation nhỏ + contrast)
      g_idx=3: mạnh (blur + noise + color jitter)

    Args:
        images: Tensor (B, 3, H, W), đã normalized với ImageNet stats
        g_idx:  chỉ số view trong group (0 ≤ g_idx < G)

    Returns:
        Tensor (B, 3, H, W), normalized ImageNet – cùng format với input
    """
    if g_idx == 0:
        return images.clone()

    aug = images.clone()

    # ── Level 1: Flip (không ảnh hưởng pixel values) ───────────────────────
    if g_idx >= 1:
        if random.random() > 0.5:
            aug = TF.hflip(aug)

    # ── Level 2: Rotation + Contrast ───────────────────────────────────────
    # rotate không cần denorm (chỉ spatial transform)
    # adjust_contrast cần pixel space [0,1] → denorm/renorm
    if g_idx >= 2:
        if random.random() > 0.4:
            aug = TF.vflip(aug)
        angle = random.uniform(-10, 10)
        aug = TF.rotate(aug, angle)
        # Contrast: hoạt động trên pixel space
        aug_px = _denorm(aug)
        factor = random.uniform(0.7, 1.3)
        aug_px = TF.adjust_contrast(aug_px, factor)
        aug = _renorm(aug_px)

    # ── Level 3: Blur + Noise + Brightness ─────────────────────────────────
    # Tất cả phép toán pixel-level → denorm một lần, xử lý, rồi renorm
    if g_idx >= 3:
        aug_px = _denorm(aug)
        # Gaussian blur giả lập motion blur UAV
        if random.random() > 0.4:
            kernel_size = random.choice([3, 5])
            aug_px = _gaussian_blur(aug_px, kernel_size)
        # Gaussian noise giả lập sensor noise
        if random.random() > 0.4:
            noise = torch.randn_like(aug_px) * random.uniform(0.02, 0.06)
            aug_px = (aug_px + noise).clamp(0.0, 1.0)
        # Brightness
        factor = random.uniform(0.6, 1.4)
        aug_px = TF.adjust_brightness(aug_px, factor)
        aug = _renorm(aug_px)

    return aug.contiguous()


def _gaussian_blur(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Áp dụng Gaussian blur bằng F.avg_pool2d (không cần scipy).
    Nhanh hơn torchvision GaussianBlur vì dùng avg pooling xấp xỉ.
    """
    padding = kernel_size // 2
    blurred = F.avg_pool2d(
        images,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    )
    return blurred


# =============================================================================
# 2. Tạo G views cho toàn batch
# =============================================================================

def create_group_views(
    images:    torch.Tensor,
    G:         int = 4,
    device:    str = 'cuda',
) -> list[torch.Tensor]:
    """
    Tạo G augmented views từ batch images.

    Args:
        images: Tensor (B, 3, H, W)
        G:      số views (thường G=4)
        device: 'cuda' hoặc 'cpu'

    Returns:
        list of G Tensors, mỗi Tensor (B, 3, H, W)
    """
    views = []
    for g in range(G):
        aug_images = augment_view(images.cpu(), g_idx=g)
        views.append(aug_images.to(device))
    return views


# =============================================================================
# 3. Validate diversity (debug helper)
# =============================================================================

def compute_view_diversity(views: list[torch.Tensor]) -> dict:
    """
    Tính mức độ khác nhau giữa các views trong group.
    Dùng để debug: nếu diversity quá thấp → augmentation chưa đủ.

    Returns:
        dict với 'mean_pixel_diff' và 'max_pixel_diff'
    """
    G = len(views)
    diffs = []
    ref = views[0].float()
    for g in range(1, G):
        diff = (views[g].float() - ref).abs().mean().item()
        diffs.append(diff)

    return {
        'mean_pixel_diff': sum(diffs) / len(diffs) if diffs else 0.0,
        'max_pixel_diff':  max(diffs)              if diffs else 0.0,
        'G':               G,
    }
