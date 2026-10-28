"""Regression tests for CUDA batch1-cache versus same-batch DPO scoring."""
import math
import os
import unittest

import torch

from kb2_preference_optimization.policy import DetectionDistribution, dpo_loss
from kb2_preference_optimization.train_dpo import pair_losses


class SameBatchScoringTests(unittest.TestCase):
    def setUp(self):
        self.classes = torch.tensor([[.3, -.2], [.1, .4]], requires_grad=True)
        self.boxes = torch.zeros(2, 4, 4, requires_grad=True)
        self.reference_classes = self.classes.detach().clone().requires_grad_(True)
        self.reference_boxes = self.boxes.detach().clone().requires_grad_(True)
        self.policy = DetectionDistribution(self.classes, self.boxes)
        self.reference = DetectionDistribution(self.reference_classes, self.reference_boxes)
        chosen = {'slots': 2, 'indices': [0], 'classes': [0], 'bins': [[1, 1, 1, 1]]}
        rejected = {'slots': 2, 'indices': [1], 'classes': [1], 'bins': [[2, 2, 2, 2]]}
        self.record = {'chosen': chosen, 'rejected': rejected,
                       'reference_chosen_logp': float(self.reference.log_prob(chosen).detach()),
                       'reference_rejected_logp': float(self.reference.log_prob(rejected).detach())}

    def test_stale_cache_cannot_bias_initial_loss_or_margin(self):
        stale = {**self.record, 'reference_chosen_logp': self.record['reference_chosen_logp'] + 2.}
        loss, diagnostic = pair_losses([self.policy], [self.reference], [stale], .1)
        self.assertEqual(diagnostic['relative_margin_max_abs'], 0.)
        self.assertAlmostEqual(float(loss), math.log(2), places=12)
        self.assertAlmostEqual(diagnostic['cached_reference_gap_error_mean'], -2., places=12)
        self.assertAlmostEqual(diagnostic['cached_reference_gap_error_max_abs'], 2., places=12)
        loss.backward()
        self.assertIsNone(self.reference_classes.grad)
        self.assertIsNone(self.reference_boxes.grad)
        self.assertGreater(float(self.classes.grad.abs().sum()), 0.)
        self.assertGreater(float(self.boxes.grad.abs().sum()), 0.)

    def test_live_reference_controls_loss_after_policy_changes(self):
        offset = torch.zeros_like(self.classes)
        offset[0, 0] = .4  # Increase only the chosen action's class logit.
        policy = DetectionDistribution(self.classes + offset, self.boxes)
        loss, diagnostic = pair_losses([policy], [self.reference], [self.record], .1)
        expected, _ = dpo_loss(policy.log_prob(self.record['chosen']), policy.log_prob(self.record['rejected']),
                               self.reference.log_prob(self.record['chosen']),
                               self.reference.log_prob(self.record['rejected']), .1)
        self.assertAlmostEqual(float(loss), float(expected), places=12)
        self.assertGreater(diagnostic['relative_margin'], 0.)
        self.assertLess(float(loss), math.log(2))

    def test_incomplete_batch_is_rejected(self):
        for policies, references, records in (([], [], []), ([self.policy], [], [self.record]),
                                               ([], [self.reference], [self.record])):
            with self.assertRaises(ValueError):
                pair_losses(policies, references, records, .1)


@unittest.skipUnless(os.environ.get('KB2_RUN_GPU') == '1', 'opt-in actual CUDA checkpoint integration')
class GPUReferenceScoringTests(unittest.TestCase):
    def test_initial_policy_matches_reference_at_actual_training_batch_size(self):
        from kb2_preference_optimization.adapter import DFLAdapter
        from kb2_preference_optimization.artifacts import source_checkpoint, tensor_digest
        from kb2_preference_optimization.data import DetectionDataset
        self.assertTrue(torch.cuda.is_available(), 'Run this test outside the GPU-restricted sandbox')
        policy = DFLAdapter(source_checkpoint('yolov8n'), 'cuda')
        reference = DFLAdapter(source_checkpoint('yolov8n'), 'cuda')
        reference.model.requires_grad_(False)
        before = tensor_digest(reference.model)
        dataset = DetectionDataset('train', 640)
        examples = [dataset[i] for i in range(4)]
        images = torch.stack([e[0] for e in examples]).cuda()
        targets = [e[1] for e in examples]
        records = []
        with torch.no_grad():
            for index, image in enumerate(images):
                single, _, _, _ = reference.distributions(image[None])
                generator = torch.Generator(device='cuda').manual_seed(42 + index)
                chosen, rejected = single[0].sample(generator), single[0].sample(generator)
                records.append({'chosen': chosen, 'rejected': rejected,
                                'reference_chosen_logp': float(single[0].log_prob(chosen)),
                                'reference_rejected_logp': float(single[0].log_prob(rejected))})
        native = policy.native_loss(images, targets)
        pd, _, _, _ = policy.distributions(images)
        with torch.no_grad():
            rd, _, _, _ = reference.distributions(images)
        loss, diagnostic = pair_losses(pd, rd, records, .1)
        self.assertLessEqual(diagnostic['relative_margin_max_abs'], 1e-5)
        self.assertAlmostEqual(float(loss), math.log(2), places=7)
        (native + .01 * loss).backward()
        self.assertTrue(all(p.grad is None for p in reference.model.parameters()))
        self.assertEqual(tensor_digest(reference.model), before)


if __name__ == '__main__':
    unittest.main()
