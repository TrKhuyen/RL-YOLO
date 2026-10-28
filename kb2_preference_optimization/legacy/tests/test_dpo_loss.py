"""Online DPO: GT labels current candidates; frozen reference only scores them."""
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feedback_loss import (candidate_selection_log_probs,
                           dpo_preference_loss_from_predictions, select_preference_pairs)


class DpoLossTests(unittest.TestCase):
    def setUp(self):
        self.reference = torch.tensor([[[1., 2.2, 8.], [1., 1., 8.],
                                        [2., 2., 2.], [2., 2., 2.],
                                        [.2, .9, .4], [.8, .1, .6]]])
        self.targets = [{'boxes': torch.tensor([[0., 0., 2., 2.]]),
                         'labels': torch.tensor([0])}]

    def test_action_policy_is_normalized_over_anchors(self):
        logp = candidate_selection_log_probs(self.reference)
        torch.testing.assert_close(logp.logsumexp(-1), torch.zeros(1, 2))
        saturated = self.reference.half()
        saturated[0, 4, :2] = torch.tensor([0., 1.], dtype=torch.float16)
        self.assertTrue(torch.isfinite(candidate_selection_log_probs(saturated)).all())

    def test_equal_reference_gives_log2_but_nonzero_policy_gradient(self):
        policy = self.reference.clone().requires_grad_(True)
        reference = self.reference.clone().requires_grad_(True)
        loss, details = dpo_preference_loss_from_predictions(
            policy, reference, self.targets)
        self.assertEqual(details['pair_count'], 1)
        self.assertAlmostEqual(float(loss.detach()), math.log(2), places=6)
        self.assertAlmostEqual(float(details['relative_margin']), 0., places=6)
        loss.backward()
        self.assertLess(float(policy.grad[0, 4, 0]), 0)
        self.assertGreater(float(policy.grad[0, 4, 1]), 0)
        self.assertIsNone(reference.grad)
        self.assertTrue(torch.equal(policy.grad[:, :4], torch.zeros_like(policy.grad[:, :4])))

    def test_current_gt_quality_reverses_pair_and_gradient_when_boxes_swap(self):
        policy = self.reference.clone()
        # Reference chose 0. CURRENT geometry makes 1 good and 0 inferior:
        # the old implementation incorrectly continued reinforcing anchor 0.
        policy[0, 0, 0], policy[0, 0, 1] = 2.2, 1.
        policy[0, 4, 0], policy[0, 4, 1] = .8, .3
        policy.requires_grad_(True)
        loss, details = dpo_preference_loss_from_predictions(
            policy, self.reference, self.targets, beta=.5)
        self.assertEqual(select_preference_pairs(policy, self.targets).tolist(), [[0, 0, 1, 0]])
        self.assertEqual(details['pair_count'], 1)
        self.assertAlmostEqual(float(details['chosen_iou']), 1.)
        self.assertAlmostEqual(float(details['rejected_iou']), .25, places=5)
        gap_policy = torch.logit(torch.tensor(.3)) - torch.logit(torch.tensor(.8))
        gap_ref = torch.logit(torch.tensor(.9)) - torch.logit(torch.tensor(.2))
        expected = -torch.nn.functional.logsigmoid(.5 * (gap_policy - gap_ref))
        torch.testing.assert_close(loss.detach(), expected)
        loss.backward()
        self.assertGreater(float(policy.grad[0, 4, 0]), 0)
        self.assertLess(float(policy.grad[0, 4, 1]), 0)

    def test_reference_boxes_do_not_choose_labels_or_change_loss(self):
        policy = self.reference.clone().requires_grad_(True)
        loss, details = dpo_preference_loss_from_predictions(policy, self.reference, self.targets)
        moved_reference = self.reference.clone()
        moved_reference[:, :2, :] += 1000
        same_loss, same_details = dpo_preference_loss_from_predictions(
            policy, moved_reference, self.targets)
        torch.testing.assert_close(loss, same_loss)
        for name in details:
            if name == 'pair_count':
                self.assertEqual(details[name], same_details[name])
            else:
                torch.testing.assert_close(details[name], same_details[name])

    def test_reference_confidence_cannot_override_gt_preference(self):
        policy = self.reference.clone().requires_grad_(True)
        reference = self.reference.clone()
        reference[0, 4, :2] = torch.tensor([.9, .2])
        loss, details = dpo_preference_loss_from_predictions(policy, reference, self.targets)
        self.assertEqual(details['pair_count'], 1)
        self.assertGreater(float(details['quality_gap']), 0.)
        loss.backward()
        self.assertLess(float(policy.grad[0, 4, 0]), 0.)
        self.assertGreater(float(policy.grad[0, 4, 1]), 0.)

    def test_reference_pair_is_skipped_when_current_policy_has_no_valid_pair(self):
        policy = self.reference.clone()
        policy[:, :2, :] += 1000
        policy.requires_grad_(True)
        loss, details = dpo_preference_loss_from_predictions(policy, self.reference, self.targets)
        self.assertEqual(details['pair_count'], 0)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue(torch.equal(policy.grad, torch.zeros_like(policy.grad)))

    def test_ranks_two_good_boxes_by_gt_iou_and_ignores_exact_ties(self):
        policy = self.reference.clone()
        policy[0, 0, 1] = 1.5  # IoU .6: inferior to A=1.0, but both >= .5.
        policy.requires_grad_(True)
        loss, details = dpo_preference_loss_from_predictions(policy, self.reference, self.targets)
        self.assertEqual(details['pair_count'], 1)
        self.assertAlmostEqual(float(details['chosen_iou']), 1.)
        self.assertAlmostEqual(float(details['rejected_iou']), .6, places=5)
        loss.backward()
        self.assertLess(float(policy.grad[0, 4, 0]), 0.)
        self.assertGreater(float(policy.grad[0, 4, 1]), 0.)
        # The old pairwise experiment still uses its original .5 cutoff.
        self.assertEqual(len(select_preference_pairs(policy, self.targets)), 0)
        tied = policy.detach().clone()
        tied[0, 0, 1] = 1.
        _, tied_details = dpo_preference_loss_from_predictions(tied, self.reference, self.targets)
        self.assertEqual(tied_details['pair_count'], 0)

    def test_no_pair_does_not_invent_a_negative_for_another_gt(self):
        targets = [{'boxes': torch.tensor([[0., 0., 2., 2.], [1.2, 0., 3.2, 2.]]),
                    'labels': torch.tensor([0, 0])}]
        policy = self.reference.clone().requires_grad_(True)
        loss, details = dpo_preference_loss_from_predictions(policy, self.reference, targets)
        self.assertEqual(details['pair_count'], 0)
        loss.backward()
        self.assertTrue(torch.equal(policy.grad, torch.zeros_like(policy.grad)))

    def test_invalid_beta_shape_and_probabilities_are_rejected(self):
        for beta in (0., -1., float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'beta'):
                dpo_preference_loss_from_predictions(
                    self.reference, self.reference, self.targets, beta)
        with self.assertRaisesRegex(ValueError, 'shapes'):
            dpo_preference_loss_from_predictions(
                self.reference[:, :, :2], self.reference, self.targets)
        for value in (-.1, 1.1, float('nan')):
            invalid = self.reference.clone()
            invalid[0, 4, 0] = value
            with self.assertRaisesRegex(ValueError, 'probabilities'):
                candidate_selection_log_probs(invalid)


if __name__ == '__main__':
    unittest.main(verbosity=2)
