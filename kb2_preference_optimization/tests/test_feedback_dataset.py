import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

KB2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KB2_DIR))

from feedback import FEEDBACK_TYPES, SCHEMA_VERSION
from feedback_dataset import (FeedbackDataset, feedback_difficulty,
                              feedback_vector, load_feedback_records,
                              object_feedback_codes)


class DummyDataset(Dataset):
    def __init__(self, size): self.size = size
    def __len__(self): return self.size
    def __getitem__(self, index):
        return torch.zeros(3, 8, 8), {
            'image_id': torch.tensor([index]), 'image_path': f'image_{index}.jpg',
            'boxes': torch.zeros((0, 4)), 'labels': torch.zeros(0, dtype=torch.long)}


def make_record(image_id=0, counts=None, num_ground_truths=1):
    counts = counts or {}
    return {
        'schema_version': SCHEMA_VERSION, 'image_id': image_id,
        'image_path': f'image_{image_id}.jpg',
        'num_ground_truths': num_ground_truths, 'num_predictions': 0,
        'feedback': {kind: [{} for _ in range(counts.get(kind, 0))]
                     for kind in FEEDBACK_TYPES},
        'preferences': [{'chosen': 1}] if image_id == 0 else []}


class FeedbackDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self): self.temp_dir.cleanup()

    def write_jsonl(self, name, records):
        path = self.root / name
        path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        return path

    def test_load_and_vector_order(self):
        record = make_record(0, {'matched': 2, 'wrong_class': 1, 'missed': 3})
        self.assertEqual(list(load_feedback_records(self.write_jsonl('f.jsonl', [record]))), [0])
        self.assertEqual(feedback_vector(record).tolist(), [2, 1, 0, 0, 0, 3])

    def test_rejects_bad_schema_missing_type_empty_and_duplicate(self):
        record = make_record(); record['schema_version'] = '0.9'
        with self.assertRaisesRegex(ValueError, 'unsupported schema'):
            load_feedback_records(self.write_jsonl('version.jsonl', [record]))
        record = make_record(); record['feedback'].pop('missed')
        with self.assertRaisesRegex(ValueError, 'missing feedback types'):
            load_feedback_records(self.write_jsonl('missing.jsonl', [record]))
        with self.assertRaisesRegex(ValueError, 'No feedback records'):
            load_feedback_records(self.write_jsonl('empty.jsonl', []))
        with self.assertRaisesRegex(ValueError, 'duplicate image_id'):
            load_feedback_records(self.write_jsonl('dup.jsonl', [make_record(), make_record()]))

    def test_difficulty_normalization_and_floor(self):
        self.assertAlmostEqual(feedback_difficulty(
            make_record(0, {'missed': 2, 'wrong_class': 1}, 2)), 1.75)
        self.assertEqual(feedback_difficulty(make_record(0, {'matched': 20}, 2)), 0.0)
        self.assertAlmostEqual(feedback_difficulty(
            make_record(0, {'false_positive': 2}, 0)), 1.0)

    def test_object_feedback_mapping_uses_priority_and_augmented_indices(self):
        record = make_record(0, {'missed': 3}, 3)
        record['feedback']['missed'] = [
            {'gt_index': 0}, {'gt_index': 1}, {'gt_index': 2}]
        record['feedback']['bad_localization'] = [{'gt_index': 1}]
        record['feedback']['wrong_class'] = [{'gt_index': 2}]
        codes = object_feedback_codes(record, torch.tensor([2, 0, 1]))
        self.assertEqual(codes.tolist(), [1, 3, 2])

    def test_dataset_attachment_and_weight_bounds(self):
        path = self.write_jsonl('f.jsonl', [
            make_record(0, {'matched': 1}), make_record(1, {'missed': 20})])
        dataset = FeedbackDataset(DummyDataset(2), path, sampling_strength=2, max_sampling_weight=3)
        self.assertEqual(dataset.sampling_weights.tolist(), [1.0, 3.0])
        image, target = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 8, 8))
        self.assertEqual(target['feedback_counts'].tolist(), [1, 0, 0, 0, 0, 0])
        self.assertEqual(target['feedback_preferences'], [{'chosen': 1}])
        self.assertIsInstance(WeightedRandomSampler(dataset.sampling_weights, 2, True), WeightedRandomSampler)

    def test_rejects_mismatch_and_invalid_settings(self):
        path = self.write_jsonl('one.jsonl', [make_record()])
        with self.assertRaisesRegex(ValueError, 'Feedback/dataset mismatch'):
            FeedbackDataset(DummyDataset(2), path)
        with self.assertRaisesRegex(ValueError, 'sampling_strength'):
            FeedbackDataset(DummyDataset(1), path, sampling_strength=-1)
        with self.assertRaisesRegex(ValueError, 'max_sampling_weight'):
            FeedbackDataset(DummyDataset(1), path, max_sampling_weight=.5)

    def test_real_feedback_covers_real_train_dataset(self):
        feedback_path = KB2_DIR / 'feedback_data' / 'yolov8n_train.jsonl'
        data_root = KB2_DIR.parent / 'pre-data' / 'data' / 'v2i'
        if not feedback_path.exists() or not data_root.exists():
            self.skipTest('Real artifacts unavailable')
        from dataloader import PestDataset, get_val_transforms
        base = PestDataset(str(data_root), 'train', 640, get_val_transforms(640))
        dataset = FeedbackDataset(base, feedback_path)
        summary = dataset.summary()
        self.assertEqual(len(dataset), 1722)
        self.assertEqual(summary['feedback_counts']['matched'], 6050)
        self.assertEqual(summary['feedback_counts']['missed'], 1124)
        self.assertGreaterEqual(summary['sampling_weight_min'], 1.0)
        self.assertLessEqual(summary['sampling_weight_max'], 5.0)

    def test_real_loader_is_reproducible_for_same_seed(self):
        feedback_path = KB2_DIR / 'feedback_data' / 'yolov8n_train.jsonl'
        data_root = KB2_DIR.parent / 'pre-data' / 'data' / 'v2i'
        if not feedback_path.exists() or not data_root.exists():
            self.skipTest('Real artifacts unavailable')
        from feedback_dataset import get_feedback_dataloader
        kwargs = dict(root=str(data_root), feedback_path=str(feedback_path),
                      batch_size=2, img_size=64, num_workers=0, seed=123)
        first = next(iter(get_feedback_dataloader(**kwargs)))
        second = next(iter(get_feedback_dataloader(**kwargs)))
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertEqual([int(t['image_id']) for t in first[1]],
                         [int(t['image_id']) for t in second[1]])
        for target in first[1]:
            self.assertEqual(len(target['object_feedback_codes']),
                             len(target['boxes']))

    def test_gt_indices_survive_augmentation_and_align_with_boxes(self):
        data_root = KB2_DIR.parent / 'pre-data' / 'data' / 'v2i'
        if not data_root.exists():
            self.skipTest('Real dataset unavailable')
        from dataloader import PestDataset, get_train_transforms
        transforms = get_train_transforms(128)
        transforms.set_random_seed(321)
        dataset = PestDataset(str(data_root), 'train', 128, transforms)
        for index in range(20):
            _, target = dataset[index]
            self.assertEqual(len(target['boxes']), len(target['labels']))
            self.assertEqual(len(target['boxes']), len(target['gt_indices']))
            self.assertTrue((target['gt_indices'] >= 0).all())
            self.assertEqual(len(target['gt_indices'].unique()),
                             len(target['gt_indices']))


if __name__ == '__main__': unittest.main(verbosity=2)
