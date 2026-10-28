"""Canonical comparison and immutable deterministic input contract."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from kb1_reward_guided_training.canonical_eval import evaluate_adapter as kb1_evaluate
from kb2_preference_optimization.data import DetectionDataset
from kb2_preference_optimization.metrics import evaluate_adapter
from kb2_preference_optimization.tests.test_structured_dpo import prediction


class FakeAdapter:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, images, *args, **kwargs):
        return self.predictions[:len(images)]

    def eval_mode(self):
        pass

    def forward_with_grad(self, images, **kwargs):
        return self.predict(images)


class MetricTests(unittest.TestCase):
    def test_perfect_metrics_and_same_ap_contract_as_kb1(self):
        gt = prediction([[0, 0, 20, 20], [30, 30, 60, 60]], [0, 1])
        loader = [(torch.zeros(1, 3, 64, 64), [gt])]
        adapter = FakeAdapter([gt])
        report = evaluate_adapter(adapter, loader)
        canonical = kb1_evaluate(adapter, loader, device='cpu', class_metrics=True)
        for key in ('mAP50_95', 'mAP50', 'AR300', 'precision', 'operating_recall', 'f1'):
            self.assertAlmostEqual(report[key], canonical[key], places=6, msg=key)
        for key in ('mAP50_95', 'mAP75', 'f1', 'matched_mean_iou', 'geometry_gt_coverage'):
            self.assertAlmostEqual(report[key], 1., places=6, msg=key)
        self.assertEqual(report['per_class']['0']['gt_support'], 1)
        self.assertEqual(report['num_images'], 1)

    def test_false_positives_false_negatives_and_undefined_iou(self):
        gt = prediction([[0, 0, 20, 20]], [0])
        pred = prediction([[30, 30, 40, 40]], [1])
        report = evaluate_adapter(FakeAdapter([pred]), [(torch.zeros(1, 3, 64, 64), [gt])])
        self.assertEqual((report['tp'], report['fp'], report['fn']), (0, 1, 1))
        self.assertEqual(report['fp_per_image'], 1.)
        self.assertEqual(report['fn_rate'], 1.)
        self.assertIsNone(report['matched_mean_iou'])

    def test_partial_evaluation_is_explicit(self):
        gt = prediction([[0, 0, 20, 20]], [0])
        report = evaluate_adapter(FakeAdapter([gt, gt]),
                                  [(torch.zeros(2, 3, 64, 64), [gt, gt])], image_limit=1)
        self.assertEqual(report['num_images'], 1)
        self.assertTrue(report['partial_evaluation'])


class DatasetTests(unittest.TestCase):
    def test_replay_is_deterministic_and_missing_labels_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'train/images').mkdir(parents=True)
            (root / 'train/labels').mkdir()
            image = np.full((40, 80, 3), 200, dtype=np.uint8)
            cv2.imwrite(str(root / 'train/images/a.jpg'), image)
            label = root / 'train/labels/a.txt'
            label.write_text('1 0.5 0.5 0.5 0.5\n')
            ds = DetectionDataset('train', 64, root, classes=2)
            x, target = ds[0]
            y, replay = ds[0]
            self.assertTrue(torch.equal(x, y))
            self.assertTrue(torch.equal(target['boxes'], replay['boxes']))
            self.assertTrue(torch.allclose(target['boxes'], torch.tensor([[16., 24., 48., 40.]])))
            self.assertTrue(0 <= float(x.min()) <= float(x.max()) <= 1)
            label.unlink()
            with self.assertRaises(FileNotFoundError):
                ds[0]

    def test_invalid_label_fails_without_silent_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'train/images').mkdir(parents=True)
            (root / 'train/labels').mkdir()
            cv2.imwrite(str(root / 'train/images/a.jpg'), np.zeros((64, 64, 3), dtype=np.uint8))
            (root / 'train/labels/a.txt').write_text('29 0.5 0.5 0.5 0.5\n')
            ds = DetectionDataset('train', 64, root)
            with self.assertRaises(ValueError):
                ds[0]


if __name__ == '__main__':
    unittest.main()
