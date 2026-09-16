"""Dataset helpers for feedback-guided YOLO fine-tuning.

Stored predictions were produced on deterministic, non-augmented images. They
are used as image-level signals and sampling weights; native YOLO loss still
learns from augmented ground-truth boxes.
"""
import json
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dataloader import PestDataset, get_train_transforms, pest_collate_fn
from feedback import FEEDBACK_TYPES, SCHEMA_VERSION


DEFAULT_ERROR_WEIGHTS = {
    'matched': -0.10,
    'wrong_class': 1.00,
    'bad_localization': 0.75,
    'false_positive': 0.50,
    'duplicate': 0.50,
    'missed': 1.25,
}

OBJECT_FEEDBACK_CODES = {
    'none': 0, 'wrong_class': 1, 'bad_localization': 2, 'missed': 3,
}


def load_feedback_records(path):
    """Load and strictly validate one JSONL feedback record per image."""
    records = {}
    with open(path, encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get('schema_version') != SCHEMA_VERSION:
                raise ValueError(
                    f'{path}:{line_number}: unsupported schema '
                    f'{record.get("schema_version")!r}'
                )
            image_id = int(record['image_id'])
            if image_id in records:
                raise ValueError(f'{path}:{line_number}: duplicate image_id {image_id}')
            feedback = record.get('feedback', {})
            missing = set(FEEDBACK_TYPES) - set(feedback)
            if missing:
                raise ValueError(f'{path}:{line_number}: missing feedback types {sorted(missing)}')
            records[image_id] = record
    if not records:
        raise ValueError(f'No feedback records found in {path}')
    return records


def feedback_vector(record):
    return torch.tensor(
        [len(record['feedback'][kind]) for kind in FEEDBACK_TYPES],
        dtype=torch.float32,
    )


def feedback_difficulty(record, error_weights=None):
    """Compute a size-normalized image difficulty score from explicit errors."""
    weights = DEFAULT_ERROR_WEIGHTS if error_weights is None else error_weights
    counts = {kind: len(record['feedback'][kind]) for kind in FEEDBACK_TYPES}
    scale = max(1, int(record.get('num_ground_truths', 0)))
    raw = sum(float(weights.get(kind, 0.0)) * count
              for kind, count in counts.items())
    return max(0.0, raw / scale)


def object_feedback_codes(record, gt_indices):
    by_gt = {}
    for item in record['feedback']['missed']:
        by_gt[int(item['gt_index'])] = OBJECT_FEEDBACK_CODES['missed']
    for item in record['feedback']['bad_localization']:
        if item.get('gt_index') is not None:
            by_gt[int(item['gt_index'])] = OBJECT_FEEDBACK_CODES['bad_localization']
    for item in record['feedback']['wrong_class']:
        if item.get('gt_index') is not None:
            by_gt[int(item['gt_index'])] = OBJECT_FEEDBACK_CODES['wrong_class']
    return torch.tensor([by_gt.get(int(index), 0) for index in gt_indices],
                        dtype=torch.long)


class FeedbackDataset(Dataset):
    """Attach frozen teacher feedback to each item of a detection dataset."""

    def __init__(self, dataset, feedback_path, error_weights=None,
                 sampling_strength=1.0, max_sampling_weight=5.0):
        self.dataset = dataset
        self.records = load_feedback_records(feedback_path)
        self.error_weights = error_weights
        self.sampling_strength = float(sampling_strength)
        self.max_sampling_weight = float(max_sampling_weight)
        if self.sampling_strength < 0:
            raise ValueError('sampling_strength must be non-negative')
        if self.max_sampling_weight < 1:
            raise ValueError('max_sampling_weight must be at least 1')

        expected, actual = set(range(len(dataset))), set(self.records)
        if expected != actual:
            missing = sorted(expected - actual)[:10]
            extra = sorted(actual - expected)[:10]
            raise ValueError(
                f'Feedback/dataset mismatch: missing={missing}, extra={extra}, '
                f'dataset={len(expected)}, feedback={len(actual)}')
        self.sampling_weights = torch.tensor([
            min(self.max_sampling_weight,
                1.0 + self.sampling_strength * feedback_difficulty(self.records[i], error_weights))
            for i in range(len(dataset))
        ], dtype=torch.double)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        record = self.records[index]
        target['feedback_counts'] = feedback_vector(record)
        target['feedback_difficulty'] = torch.tensor(
            feedback_difficulty(record, self.error_weights), dtype=torch.float32)
        target['feedback_preferences'] = record.get('preferences', [])
        target['object_feedback_codes'] = object_feedback_codes(
            record, target.get('gt_indices', range(len(target['boxes']))))
        return image, target

    def summary(self):
        counts = Counter()
        for record in self.records.values():
            for kind in FEEDBACK_TYPES:
                counts[kind] += len(record['feedback'][kind])
        return {
            'num_images': len(self),
            'feedback_counts': dict(counts),
            'sampling_weight_min': float(self.sampling_weights.min()),
            'sampling_weight_mean': float(self.sampling_weights.mean()),
            'sampling_weight_max': float(self.sampling_weights.max()),
        }


def get_feedback_dataloader(root, feedback_path, batch_size=16, img_size=640,
                            num_workers=4, sampling_strength=1.0,
                            max_sampling_weight=5.0, error_weights=None,
                            seed=42):
    transforms = get_train_transforms(img_size)
    if hasattr(transforms, 'set_random_seed'):
        transforms.set_random_seed(seed)
    base = PestDataset(root=root, split='train', img_size=img_size,
                       transforms=transforms)
    dataset = FeedbackDataset(
        base, feedback_path, error_weights=error_weights,
        sampling_strength=sampling_strength,
        max_sampling_weight=max_sampling_weight)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        dataset.sampling_weights, num_samples=len(dataset), replacement=True,
        generator=generator)
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, collate_fn=pest_collate_fn,
        pin_memory=torch.cuda.is_available(), generator=generator)
