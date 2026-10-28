"""Exercise YOLO26's dense feedback candidates and native dual-head loss."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from ultralytics.nn.tasks import DetectionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_rl import CHECKPOINTS, load_adapter
from train_feedback import parse_args


class Yolo26Tests(unittest.TestCase):
    def test_dense_candidates_and_native_loss_keep_gradients(self):
        torch.set_num_threads(2)
        model = DetectionModel('yolo26n.yaml', nc=3, verbose=False)
        model.args = {}
        model.end2end = True
        with patch('ultralytics.YOLO', return_value=SimpleNamespace(model=model)):
            adapter = load_adapter('yolo26n', 'unused.pt', 'cpu')
        self.assertFalse(model.end2end)
        images = torch.rand(1, 3, 64, 64)
        raw = adapter.raw_predictions(images)
        self.assertEqual(tuple(raw.shape), (1, 7, 84))
        self.assertTrue(((raw[:, 4:] >= 0) & (raw[:, 4:] <= 1)).all())
        raw[:, 4:].sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.model[-1].cv3.parameters()))
        model.zero_grad(set_to_none=True)
        predictions = adapter.forward_with_grad(images, conf_thres=0.)
        self.assertGreater(len(predictions[0]['scores']), 0)
        predictions[0]['scores'].sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.model[-1].cv3.parameters()))
        model.zero_grad(set_to_none=True)
        targets = [{'boxes': torch.tensor([[10., 10., 45., 45.]]),
                    'labels': torch.tensor([1])}]
        loss, items = adapter.supervised_loss(images, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('l1_loss', items)
        loss.backward()
        for branch in (model.model[-1].cv3, model.model[-1].one2one_cv3):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in branch.parameters()))

    def test_feedback_cli_and_checkpoint_registration(self):
        self.assertEqual(parse_args(['--model', 'yolo26n']).model, 'yolo26n')
        self.assertEqual(CHECKPOINTS['yolo26n'].parent.parent.name, 'yolo26n')
