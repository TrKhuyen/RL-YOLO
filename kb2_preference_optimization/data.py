"""Strict deterministic replay of offline preferences on the clean dataset."""
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from kb1_reward_guided_training.run_provenance import DATA_ROOT

TRANSFORM_VERSION = 'longest_max_size_center_pad114_rgb_unit_v1'


class DetectionDataset(Dataset):
    def __init__(self, split='train', image_size=640, root=DATA_ROOT, classes=28):
        if split not in ('train', 'valid', 'test'):
            raise ValueError('split must be train, valid or test')
        if image_size < 64 or image_size % 32:
            raise ValueError('image_size must be >=64 and divisible by 32')
        self.root, self.split = Path(root).resolve(), split
        self.image_size, self.classes = image_size, classes
        self.paths = sorted((self.root / split / 'images').glob('*.jpg'))
        if not self.paths:
            raise ValueError(f'Empty split: {self.root / split}')
        self.transforms = A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT, fill=114),
            A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.), max_pixel_value=255.),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', clip=True, label_fields=['class_labels']))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label_path = self.root / self.split / 'labels' / (path.stem + '.txt')
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Unreadable image: {path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, labels = [], []
        for line in label_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f'Malformed label: {label_path}')
            cls, *box = map(float, parts)
            if (not np.isfinite([cls, *box]).all() or cls != int(cls)
                    or not 0 <= cls < self.classes or min(box) < 0
                    or max(box) > 1 or min(box[2:]) <= 0):
                raise ValueError(f'Invalid class/box: {label_path}')
            boxes.append(box)
            labels.append(int(cls))
        # Fail on transform errors; never silently change the input distribution.
        transformed = self.transforms(image=image, bboxes=boxes, class_labels=labels)
        xywh = torch.tensor(np.asarray(transformed['bboxes']), dtype=torch.float32).reshape(-1, 4)
        xyxy = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2,
                          xywh[:, :2] + xywh[:, 2:] / 2), -1) * self.image_size
        target = {'boxes': xyxy, 'labels': torch.tensor(transformed['class_labels'], dtype=torch.long),
                  'image_id': index, 'image_path': str(path.relative_to(self.root))}
        return transformed['image'], target


def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def detection_loader(split='valid', image_size=640, batch_size=8):
    return DataLoader(DetectionDataset(split, image_size), batch_size=batch_size,
                      shuffle=False, num_workers=0, collate_fn=collate)
