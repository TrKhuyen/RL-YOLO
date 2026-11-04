"""Preserve positive assignment/gradients while weighting only background slots."""
import os
import unittest

import torch
from torch.nn import functional as F

from kb2_preference_optimization.conditional_bg.loss import BackgroundBCE, BackgroundAdapter


class BackgroundLossTests(unittest.TestCase):
    def test_slot_weighting_preserves_foreground_zero_class_and_normalization(self):
        z = torch.tensor([[[.2, -.5], [-1., 2.], [1., -.1]]], requires_grad=True)
        target = torch.tensor([[[.8, 0.], [0., 0.], [0., .6]]])
        fg = torch.tensor([[True, False, True]])
        base = F.binary_cross_entropy_with_logits(z, target, reduction='none')
        criterion = BackgroundBCE(.5)
        criterion.set_foreground(fg)
        weighted = criterion(z, target)
        expected = base * torch.tensor([[[1.], [.5], [1.]]])
        torch.testing.assert_close(weighted, expected, rtol=0, atol=0)
        grad0 = torch.autograd.grad(base.sum()/target.sum(), z, retain_graph=True)[0]
        grad = torch.autograd.grad(weighted.sum()/target.sum(), z)[0]
        torch.testing.assert_close(grad[fg], grad0[fg], rtol=0, atol=0)
        torch.testing.assert_close(grad[~fg], .5*grad0[~fg], rtol=0, atol=0)
        self.assertAlmostEqual(float(criterion.last_parts['native_target_score_sum']), 1.4, places=6)
        with self.assertRaises(ValueError):
            criterion(z, target)  # A previous assignment cannot leak into the next batch.

    def test_identity_and_empty_foreground(self):
        for alpha in (0., .5, 1.):
            z = torch.randn(2, 3, 4, requires_grad=True)
            target = torch.zeros_like(z)
            criterion = BackgroundBCE(alpha)
            criterion.set_foreground(torch.zeros(2, 3, dtype=torch.bool))
            weighted = criterion(z, target)
            base = F.binary_cross_entropy_with_logits(z, target, reduction='none')
            torch.testing.assert_close(weighted, alpha*base, rtol=0, atol=0)
            self.assertEqual(float(criterion.last_parts['native_target_score_sum']), 1.)
            self.assertEqual(float(criterion.last_parts['native_cls_foreground']), 0.)

    def test_optional_class_weight_diagnostics_and_all_foreground(self):
        z = torch.randn(1, 3, 2)
        y = torch.full_like(z, .25)
        cw = torch.tensor([[[.5, 2.]]])
        criterion = BackgroundBCE(.5, cls_gain=.7, class_weights=cw)
        criterion.set_foreground(torch.ones(1, 3, dtype=torch.bool))
        output = criterion(z, y)
        base = F.binary_cross_entropy_with_logits(z, y, reduction='none')
        torch.testing.assert_close(output, base, rtol=0, atol=0)
        torch.testing.assert_close(criterion.last_parts['native_cls_weighted_total'],
                                   (base*cw).sum()*.7/y.sum())

    def test_reject_invalid_weight_and_missing_assignment(self):
        for alpha in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                BackgroundBCE(alpha)
        with self.assertRaises(ValueError):
            BackgroundBCE(.5)(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))

    @unittest.skipUnless(os.environ.get('KB2_RUN_GPU_TESTS') == '1', 'Opt-in real S0 GPU loss test')
    def test_real_s0_loss_identity_assignment_and_output_gradients(self):
        from kb2_preference_optimization.artifacts import source_checkpoint
        from kb2_preference_optimization.conditional.policy import ConditionalAdapter, configure_numerics
        from kb2_preference_optimization.data import DetectionDataset, collate
        configure_numerics()
        dataset = DetectionDataset('train')
        images, targets = collate([dataset[i] for i in [923, 1317, 1807, 1933]])
        images = images.cuda()
        values = []
        for alpha in (None, 1., .5):
            adapter = (ConditionalAdapter(source_checkpoint('yolov8n'), 'cuda') if alpha is None
                       else BackgroundAdapter(source_checkpoint('yolov8n'), 'cuda', alpha))
            if alpha is None:
                adapter.model.criterion = adapter.model.init_criterion()
            criterion = adapter.model.criterion
            captured = {}

            def assignment(module, inputs, output):
                captured['fg'] = output[3].detach().clone()
                captured['target'] = output[2].detach().clone()

            def bce_inputs(module, inputs):
                captured['logits'] = inputs[0]

            criterion.assigner.register_forward_hook(assignment)
            criterion.bce.register_forward_pre_hook(bce_inputs)
            frozen = adapter.frozen_digest()
            loss = adapter.native_loss(images, targets)
            params = [p for p in adapter.model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, [captured['logits']]+params)
            values.append({'loss': loss.detach().cpu(),
                           'fg': captured['fg'].cpu(), 'target': captured['target'].cpu(),
                           'output_grad': grads[0].cpu(),
                           'parameter_grad': torch.cat([g.flatten().cpu() for g in grads[1:]])})
            self.assertEqual(frozen, adapter.frozen_digest())
            if alpha is not None:
                self.assertGreater(adapter.native_parts()['native_foreground_slots'], 0)
            del adapter, criterion, params, grads, loss, captured
        original, identity, half = values
        for key in original:
            torch.testing.assert_close(identity[key], original[key], rtol=0, atol=0)
        for key in ('fg', 'target'):
            torch.testing.assert_close(half[key], original[key], rtol=0, atol=0)
        fg = original['fg']
        torch.testing.assert_close(half['output_grad'][fg], original['output_grad'][fg], rtol=0, atol=0)
        torch.testing.assert_close(half['output_grad'][~fg], .5*original['output_grad'][~fg], rtol=0, atol=0)
        self.assertLess(float(half['loss']), float(original['loss']))


if __name__ == '__main__':
    unittest.main()
