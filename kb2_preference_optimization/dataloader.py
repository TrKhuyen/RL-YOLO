"""
dataloader.py – Dataset và DataLoader cho bài toán phát hiện sâu bệnh.

Format:
- Ảnh: JPG/PNG, bất kỳ kích thước
- Nhãn: YOLO txt format (class cx cy w h, tọa độ chuẩn hóa [0,1])
- Output: (images Tensor B×3×H×W, targets list[dict])
  Mỗi target dict: {'boxes': Tensor(N,4) xyxy absolute, 'labels': Tensor(N,)}

Tại sao cần custom DataLoader thay vì dùng YOLOv5/Ultralytics built-in?
→ RL training loop cần targets ở format dict chuẩn (xyxy absolute)
  để tính reward trực tiếp, không phụ thuộc vào từng framework.
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────────────────────
# 1. Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 640) -> A.Compose:
    """
    Augmentation cho training: đa dạng điều kiện thực địa UAV.
    Bao gồm: biến thiên ánh sáng, nhiễu cảm biến, rotation, flip, blur.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=114),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),       # nhiễu cảm biến UAV
        A.MotionBlur(blur_limit=5, p=0.2),                 # motion blur khi bay
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=114, p=0.4),
        A.CLAHE(clip_limit=2.0, p=0.2),                    # tăng tương phản vùng tối
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,   # loại box bị cắt quá nhiều
    ))


def get_val_transforms(img_size: int = 640) -> A.Compose:
    """Validation: chỉ resize + normalize, không augment."""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=114),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
    ))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PestDataset(Dataset):
    """
    Dataset phát hiện sâu bệnh, đọc ảnh và nhãn theo YOLO format.

    Cấu trúc thư mục:
        root/images/train/*.jpg
        root/labels/train/*.txt   ← YOLO format: class cx cy w h (normalized)
    """

    def __init__(self, root: str, split: str = 'train',
                 img_size: int = 640, transforms=None):
        self.img_size = img_size
        self.transforms = transforms

        # Fallback path resolution for dataset location
        root_path = Path(root)
        if not root_path.exists():
            for candidate in [
                Path('../pre-data/data/v2i'),
                Path('pre-data/data/v2i'),
                Path('../pre-data/data'),
                Path('pre-data/data'),
            ]:
                if candidate.exists():
                    root_path = candidate
                    break
        if (root_path / 'v2i').exists() and not (root_path / 'train').exists() and not (root_path / 'images').exists():
            root_path = root_path / 'v2i'
        root = str(root_path)

        # Fallback: map split 'val' to 'valid' if 'val' doesn't exist but 'valid' does
        if split == 'val' and not (Path(root) / 'val').exists() and not (Path(root) / 'images' / 'val').exists():
            if (Path(root) / 'valid').exists() or (Path(root) / 'images' / 'valid').exists():
                split = 'valid'
        elif split == 'valid' and not (Path(root) / 'valid').exists() and not (Path(root) / 'images' / 'valid').exists():
            if (Path(root) / 'val').exists() or (Path(root) / 'images' / 'val').exists():
                split = 'val'

        # Try structure: root / images / split
        img_dir   = Path(root) / 'images' / split
        label_dir = Path(root) / 'labels' / split

        # Fallback: try structure: root / split / images
        if not img_dir.exists():
            img_dir   = Path(root) / split / 'images'
            label_dir = Path(root) / split / 'labels'

        # Lấy tất cả file ảnh
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        self.img_paths = sorted([
            p for p in img_dir.iterdir() if p.suffix.lower() in exts
        ])

        # Map ảnh → label (cùng tên, khác extension)
        self.label_paths = []
        for img_path in self.img_paths:
            lbl = label_dir / (img_path.stem + '.txt')
            self.label_paths.append(lbl if lbl.exists() else None)

        print(f"[PestDataset] {split}: {len(self.img_paths)} images")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        lbl_path = self.label_paths[idx]

        # ── Load image ───────────────────────────────────────────────────
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        # ── Load labels ──────────────────────────────────────────────────
        bboxes, class_labels = [], []
        if lbl_path is not None and lbl_path.stat().st_size > 0:
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        # Clamp để tránh box tràn ra ngoài sau augment
                        cx = np.clip(cx, 0.0, 1.0)
                        cy = np.clip(cy, 0.0, 1.0)
                        w  = np.clip(w,  0.0, 1.0)
                        h  = np.clip(h,  0.0, 1.0)
                        bboxes.append([cx, cy, w, h])
                        class_labels.append(cls)

        # ── Augmentation ─────────────────────────────────────────────────
        if self.transforms is not None:
            try:
                result = self.transforms(
                    image=img,
                    bboxes=bboxes,
                    class_labels=class_labels,
                )
                img_t       = result['image']            # Tensor (3, H, W)
                bboxes      = list(result['bboxes'])
                class_labels = list(result['class_labels'])
            except Exception:
                # Fallback: resize sạch không augment nếu transform lỗi
                img_t = torch.from_numpy(
                    cv2.resize(img, (self.img_size, self.img_size))
                    .transpose(2, 0, 1).astype(np.float32) / 255.0
                )
        else:
            img_t = torch.from_numpy(
                cv2.resize(img, (self.img_size, self.img_size))
                .transpose(2, 0, 1).astype(np.float32) / 255.0
            )

        # ── Convert YOLO (cx cy w h normalized) → xyxy absolute ──────────
        _, H, W = img_t.shape
        boxes_xyxy = []
        for (cx, cy, bw, bh) in bboxes:
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            boxes_xyxy.append([x1, y1, x2, y2])

        target = {
            'boxes':  torch.tensor(boxes_xyxy,   dtype=torch.float32)
                      if boxes_xyxy else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(class_labels, dtype=torch.long)
                      if class_labels else torch.zeros(0,     dtype=torch.long),
            'image_id': torch.tensor([idx]),
        }

        return img_t, target


# ─────────────────────────────────────────────────────────────────────────────
# 3. Collate function
# ─────────────────────────────────────────────────────────────────────────────

def pest_collate_fn(batch):
    """
    Custom collate: stack images thành batch Tensor, giữ targets là list[dict].
    Cần thiết vì mỗi ảnh có số lượng boxes khác nhau.
    """
    images  = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


# ─────────────────────────────────────────────────────────────────────────────
# 4. DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_pest_dataloader(
    root:       str,
    split:      str = 'train',
    batch_size: int = 16,
    img_size:   int = 640,
    num_workers:int = 4,
    shuffle:    Optional[bool] = None,
) -> DataLoader:
    """
    Factory function trả về DataLoader sẵn dùng cho RL training / evaluation.

    Args:
        root:        Path đến thư mục dataset (chứa images/ và labels/)
        split:       'train', 'val', hoặc 'test'
        batch_size:  Số ảnh mỗi batch
        img_size:    Kích thước ảnh đầu vào (img_size × img_size)
        num_workers: Số worker song song
        shuffle:     None → tự động (train=True, val/test=False)

    Returns:
        DataLoader trả về (images: Tensor, targets: list[dict])
    """
    if shuffle is None:
        shuffle = (split == 'train')

    transforms = (get_train_transforms(img_size) if split == 'train'
                  else get_val_transforms(img_size))

    dataset = PestDataset(root=root, split=split,
                          img_size=img_size, transforms=transforms)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pest_collate_fn,
        pin_memory=True,
        drop_last=(split == 'train'),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else 'data/pest'
    loader = get_pest_dataloader(root, split='train', batch_size=4)
    images, targets = next(iter(loader))
    print(f"images: {images.shape}")          # (4, 3, 640, 640)
    print(f"targets[0]: {targets[0]}")        # dict với boxes, labels
    for t in targets:
        print(f"  boxes: {t['boxes'].shape}, labels: {t['labels']}")
