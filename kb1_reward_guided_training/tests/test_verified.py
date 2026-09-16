'''Behavioral regressions for the verified KB1 workflow.'''
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.common import canonical_postprocess
from canonical_eval import evaluate_adapter, _operating_point_counts
from dataloader import get_val_transforms
from run_verified import SeededDataset, StepBatches
from train_rl import match_aware_objective


class Checks(unittest.TestCase):
    def test_nms_keeps_other_classes_and_score_gradient(self):
        boxes = torch.tensor([[0., 0., 20., 20.]] * 3)
        scores = torch.tensor([.9, .8, .7], requires_grad=True)
        labels = torch.tensor([0, 0, 1])
        ids, _, _ = canonical_postprocess(boxes, scores, labels, .001, .6)
        self.assertEqual(ids.tolist(), [0, 2])
        scores[ids].sum().backward()
        self.assertEqual(scores.grad.tolist(), [1., 0., 1.])

    def test_matching_duplicate_wrong_class_and_empty(self):
        p = {'boxes': torch.tensor([[0., 0., 20., 20.]] * 3),
             'scores': torch.tensor([.9, .8, .7]), 'labels': torch.tensor([0, 0, 1])}
        t = {'boxes': torch.tensor([[0., 0., 20., 20.]]), 'labels': torch.tensor([0])}
        c = _operating_point_counts([p], [t])
        self.assertEqual(c[0], [1, 1, 0])
        self.assertEqual(c[1], [0, 1, 0])
        empty = {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)}
        self.assertEqual(_operating_point_counts([p], [empty])[0], [0, 2, 0])

    def test_reward_proxy_gradient_direction(self):
        scores = torch.tensor([.7, .6], requires_grad=True)
        p = {'boxes': torch.tensor([[0., 0., 20., 20.], [40., 40., 50., 50.]]),
             'scores': scores, 'labels': torch.tensor([0, 0]),
             'policy_log_prob': scores.mean().log(), 'max_score_all': scores.max()}
        t = {'boxes': torch.tensor([[0., 0., 20., 20.]]), 'labels': torch.tensor([0])}
        _, loss = match_aware_objective([p], [t])
        loss.backward()
        self.assertLess(scores.grad[0], 0)  # Gradient descent raises TP confidence.
        self.assertGreater(scores.grad[1], 0)  # And lowers FP confidence.

    def test_border_rounding_and_letterbox(self):
        transform = get_val_transforms()
        result = transform(image=np.zeros((100, 200, 3), dtype=np.uint8),
                           bboxes=[[.5, .02, .2, .040001]], class_labels=[0])
        self.assertEqual(tuple(result['image'].shape), (3, 640, 640))
        self.assertEqual(len(result['bboxes']), 1)
        self.assertAlmostEqual(float(result['bboxes'][0][1]), .260000125, places=5)

    def test_resume_batches_and_augmentations(self):
        all_batches = list(StepBatches(17, 4, 0, 10, 42))
        resumed = list(StepBatches(17, 4, 5, 10, 42))
        self.assertEqual(all_batches[5:], resumed)
        a, b = SeededDataset(42), SeededDataset(42)
        image_a, target_a = a[(2, 0)]
        a[(2, 1)]
        image_b, target_b = b[(2, 0)]
        torch.testing.assert_close(image_a, image_b, atol=0, rtol=0)
        torch.testing.assert_close(target_a['boxes'], target_b['boxes'], atol=0, rtol=0)

    def test_perfect_evaluator_and_bad_input(self):
        target = {'boxes': torch.tensor([[20., 20., 100., 100.]]), 'labels': torch.tensor([0])}
        class Adapter:
            def eval_mode(self):
                pass
            def forward_with_grad(self, images, **kwargs):
                return [{**target, 'scores': torch.tensor([.9])}]
        result = evaluate_adapter(Adapter(), [(torch.zeros(1, 3, 640, 640), [target])],
                                  device='cpu')
        self.assertAlmostEqual(result['mAP50_95'], 1.0)
        self.assertAlmostEqual(result['AR300'], 1.0)
        with self.assertRaises(ValueError):
            evaluate_adapter(Adapter(), [(torch.full((1, 3, 2, 2), 2.), [target])], device='cpu')


if __name__ == '__main__':
    unittest.main()

