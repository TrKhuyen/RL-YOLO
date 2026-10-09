"""YOLO26 integration without downloading weights or running full training."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from ultralytics.nn.tasks import DetectionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.ultralytics_adapter import UltralyticsAdapter
from train_rl import CHECKPOINTS, load_adapter
from train_supervised import MODELS, MODEL_BATCH


class Yolo26Tests(unittest.TestCase):
    def test_dual_head_native_loss_and_canonical_score_gradients(self):
        torch.set_num_threads(2)
        model = DetectionModel('yolo26n.yaml', nc=3, verbose=False)
        model.args = {}
        model.end2end = True  # Also handle checkpoints saved with NMS-free inference.
        self.assertTrue(model.end2end)
        with patch('ultralytics.YOLO', return_value=SimpleNamespace(model=model)):
            adapter = load_adapter('yolo26n', 'unused.pt', 'cpu')
        self.assertIsInstance(adapter, UltralyticsAdapter)
        self.assertFalse(model.end2end)
        self.assertEqual(model.model[-1].reg_max, 1)
        self.assertIsNotNone(model.model[-1].one2one_cv2)
        adapter.freeze_except_detection_head()
        images = torch.rand(1, 3, 64, 64)
        targets = [{'boxes': torch.tensor([[10., 10., 45., 45.]]),
                    'labels': torch.tensor([1])}]
        loss, items = adapter.native_detection_loss(images, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('l1_loss', items)
        loss.backward()
        for branch in (model.model[-1].cv3, model.model[-1].one2one_cv3):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in branch.parameters()))
        model.zero_grad(set_to_none=True)
        predictions = adapter.forward_with_grad(images, conf_thres=0., max_det=5)
        self.assertGreater(len(predictions[0]['scores']), 0)
        predictions[0]['scores'].sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.model[-1].cv3.parameters()))
        self.assertTrue(all(p.grad is None for p in model.model[-1].one2one_cv3.parameters()))

    def test_supervised_and_continuation_registration(self):
        self.assertEqual(MODELS['yolo26n']['weights'], 'yolo26n.pt')
        self.assertEqual(MODEL_BATCH['yolo26n'], 16)
        self.assertEqual(CHECKPOINTS['yolo26n'].parent.parent.name, 'yolo26n')
