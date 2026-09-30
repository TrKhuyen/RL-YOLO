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


def make_record(image_id=0, statuses=None, false_positives=0):
    statuses = list(statuses or [])
    feedback = {kind: [] for kind in FEEDBACK_TYPES}
    for gi, kind in enumerate(statuses):
        feedback[kind].append({'gt_index': gi})
    feedback['false_positive'] = [{} for _ in range(false_positives)]
    return {
        'schema_version': SCHEMA_VERSION, 'image_id': image_id,
        'image_path': f'image_{image_id}.jpg',
        'num_ground_truths': len(statuses), 'num_predictions': 0,
        'gt_status': statuses, 'feedback': feedback,
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
        record = make_record(0, ['matched', 'matched', 'wrong_class', 'missed', 'missed', 'missed'])
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
        record = make_record(0, ['matched'])
        record['gt_status'][0] = 'missed'
        with self.assertRaisesRegex(ValueError, 'inconsistent matched GT status'):
            load_feedback_records(self.write_jsonl('inconsistent.jsonl', [record]))

    def test_difficulty_normalization_and_floor(self):
        self.assertAlmostEqual(feedback_difficulty(
            make_record(0, ['missed', 'wrong_class'])), 1.125)
        self.assertEqual(feedback_difficulty(make_record(0, ['matched', 'matched'])), 0.0)
        self.assertAlmostEqual(feedback_difficulty(
            make_record(0, [], 2)), 1.0)

    def test_object_feedback_mapping_uses_exclusive_augmented_indices(self):
        record = make_record(0, ['missed', 'bad_localization', 'wrong_class'])
        codes = object_feedback_codes(record, torch.tensor([2, 0, 1]))
        self.assertEqual(codes.tolist(), [1, 3, 2])
        record['feedback']['wrong_class'].append({'gt_index': 0})
        self.assertEqual(object_feedback_codes(record, [0]).tolist(), [3])

    def test_dataset_attachment_and_weight_bounds(self):
        path = self.write_jsonl('f.jsonl', [
            make_record(0, ['matched']), make_record(1, ['missed'])])
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

    def real_feedback_fixture(self):
        """Build schema-2 records from actual train labels without model inference."""
        data_root = KB2_DIR.parent / 'pre-data' / 'data' / 'v2i_cleanned'
        if not data_root.exists():
            self.skipTest('Real train dataset unavailable')
        from dataloader import PestDataset
        base = PestDataset(str(data_root), 'train', 64)
        records = []
        for image_id, (image_path, label_path) in enumerate(
                zip(base.img_paths, base.label_paths)):
            count = 0
            if label_path is not None:
                count = sum(len(line.split()) == 5
                            for line in label_path.read_text().splitlines())
            record = make_record(image_id, ['missed'] * count)
            record['image_path'] = str(image_path)
            records.append(record)
        return data_root, base, self.write_jsonl('real_train_fixture.jsonl', records)

    def test_schema_feedback_covers_real_train_dataset(self):
        _, base, feedback_path = self.real_feedback_fixture()
        dataset = FeedbackDataset(base, feedback_path)
        summary = dataset.summary()
        self.assertEqual(len(dataset), len(base))
        self.assertEqual(summary['feedback_counts'].get('missed', 0),
                         sum(len(record['gt_status'])
                             for record in dataset.records.values()))
        self.assertGreaterEqual(summary['sampling_weight_min'], 1.0)
        self.assertLessEqual(summary['sampling_weight_max'], 5.0)
        for image_id in (0, len(base) // 2, len(base) - 1):
            _, target = dataset[image_id]
            self.assertEqual(len(target['object_feedback_codes']),
                             len(target['boxes']))

    def test_real_loader_is_reproducible_for_same_seed(self):
        data_root, _, feedback_path = self.real_feedback_fixture()
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
        data_root = KB2_DIR.parent / 'pre-data' / 'data' / 'v2i_cleanned'
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
