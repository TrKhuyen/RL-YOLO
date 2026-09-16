import sys
import unittest
from pathlib import Path

import torch

KB2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KB2_DIR))

from feedback_loss import (blend_native_losses, combine_projected_gradients,
                           compute_hybrid_loss,
                           dynamic_object_feedback_codes_from_predictions,
                           normalized_difficulty_weights,
                           object_feedback_loss_from_predictions)


class FakeAdapter:
    def __init__(self): self.scale = torch.tensor(2.0, requires_grad=True)
    def supervised_loss(self, images, targets):
        loss = images.mean() * self.scale
        return loss, torch.tensor([loss.detach()])


class FeedbackLossTests(unittest.TestCase):
    def targets(self, values):
        return [{'feedback_difficulty': torch.tensor(v)} for v in values]

    def test_weights_are_neutral_or_mean_one_and_bounded(self):
        self.assertEqual(normalized_difficulty_weights(self.targets([0, 0])).tolist(), [1, 1])
        weights = normalized_difficulty_weights(self.targets([0, 1, 100]), max_weight=3)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(float(weights[2]), float(weights[0]))
        self.assertLessEqual(float(weights.max()), 3.0)

    def test_weights_reject_invalid_values(self):
        with self.assertRaisesRegex(ValueError, 'finite and non-negative'):
            normalized_difficulty_weights(self.targets([-1]))
        with self.assertRaisesRegex(ValueError, 'finite and non-negative'):
            normalized_difficulty_weights(self.targets([float('nan')]))
        with self.assertRaisesRegex(ValueError, 'max_weight'):
            normalized_difficulty_weights(self.targets([1]), .5)

    def test_blend_endpoints_and_validation(self):
        base = torch.tensor(2.0)
        losses = torch.tensor([1.0, 3.0])
        weights = torch.tensor([.5, 1.5])
        total0, weighted = blend_native_losses(base, losses, weights, 0)
        total1, _ = blend_native_losses(base, losses, weights, 1)
        self.assertEqual(float(total0), 2.0)
        self.assertAlmostEqual(float(weighted), 2.5)
        self.assertAlmostEqual(float(total1), 2.5)
        with self.assertRaisesRegex(ValueError, 'feedback_alpha'):
            blend_native_losses(base, losses, weights, 1.1)

    def test_hybrid_loss_keeps_gradient_and_alpha_zero_is_native(self):
        images = torch.tensor([[[[1.0]]], [[[3.0]]]])
        targets = self.targets([0, 2])
        adapter = FakeAdapter()
        native, _ = compute_hybrid_loss(adapter, images, targets, feedback_alpha=0)
        self.assertAlmostEqual(float(native.detach()), 4.0)
        total, details = compute_hybrid_loss(adapter, images, targets, feedback_alpha=.25)
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertIsNotNone(adapter.scale.grad)
        self.assertGreater(float(details['weights'][1]), float(details['weights'][0]))

    def test_alpha_zero_does_not_require_feedback_fields(self):
        images = torch.ones(2, 1, 1, 1)
        total, details = compute_hybrid_loss(
            FakeAdapter(), images, [{}, {}], feedback_alpha=0)
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(details['weights'].tolist(), [1, 1])

    def test_object_loss_targets_class_and_box_and_keeps_gradient(self):
        # B=1, channels=4 box + 2 classes, anchors=2.
        predictions = torch.tensor([[
            [1., 6.], [1., 6.], [2., 2.], [2., 2.],
            [.9, .1], [.1, .9],
        ]], requires_grad=True)
        target = {'boxes': torch.tensor([[0., 0., 2., 2.]]),
                  'labels': torch.tensor([0]),
                  'object_feedback_codes': torch.tensor([1])}
        good = object_feedback_loss_from_predictions(predictions, [target])
        worse = predictions.detach().clone().requires_grad_(True)
        worse.data[0, 4, 0] = .1
        bad = object_feedback_loss_from_predictions(worse, [target])
        self.assertLess(float(good), float(bad))
        good.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())

        target['object_feedback_codes'] = torch.tensor([2])
        perfect_box = object_feedback_loss_from_predictions(
            predictions.detach(), [target])
        shifted = predictions.detach().clone()
        shifted[0, 0, 0] = 2.
        shifted_box = object_feedback_loss_from_predictions(shifted, [target])
        self.assertLess(float(perfect_box), float(shifted_box))

    def test_dynamic_feedback_classifies_current_error_types(self):
        target = {'boxes': torch.tensor([[0., 0., 2., 2.]]),
                  'labels': torch.tensor([0])}
        def code(cx, class0, class1):
            prediction = torch.tensor([[[cx], [1.], [2.], [2.],
                                        [class0], [class1]]])
            return int(dynamic_object_feedback_codes_from_predictions(
                prediction, [target])[0][0])
        self.assertEqual(code(1., .9, .1), 0)
        self.assertEqual(code(1., .1, .9), 1)
        self.assertEqual(code(2.5, .9, .1), 2)
        self.assertEqual(code(1., .1, .1), 3)

    def test_gradient_projection_preserves_native_and_removes_conflict(self):
        native = [torch.tensor([1., 0.])]
        conflicting = [torch.tensor([-1., 2.])]
        combined, info = combine_projected_gradients(
            native, conflicting, alpha=.5)
        projected_feedback = (combined[0] - native[0]) / .5
        self.assertTrue(info['gradient_projected'])
        self.assertAlmostEqual(float((native[0] * projected_feedback).sum()),
                               0.0, places=6)
        self.assertTrue(torch.equal(combined[0], torch.tensor([1., 1.])))

        aligned, info = combine_projected_gradients(
            native, [torch.tensor([2., 1.])], alpha=.5)
        self.assertFalse(info['gradient_projected'])
        self.assertTrue(torch.equal(aligned[0], torch.tensor([2., .5])))

    def test_gradient_projection_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            combine_projected_gradients([torch.ones(1)], [torch.ones(1)], -1)
        with self.assertRaisesRegex(ValueError, 'equal length'):
            combine_projected_gradients([torch.ones(1)], [], 1)


if __name__ == '__main__': unittest.main(verbosity=2)
