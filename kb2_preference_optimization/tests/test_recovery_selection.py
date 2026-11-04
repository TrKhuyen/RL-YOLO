"""GT identities, tradeoff gates, and marginal probabilities in the pilot."""
import unittest

import torch

from kb1_reward_guided_training.canonical_eval import _operating_point_counts
from kb2_preference_optimization.recovery.diagnostics import canonical_gt_ids
from kb2_preference_optimization.recovery.selection import (
    action_distance, balanced_probabilities, eligible_pairs, is_probe_image, matched_ids,
)
from kb2_preference_optimization.tests.test_structured_dpo import prediction


def action(classes):
    return {'slots': 3, 'indices': list(range(len(classes))), 'classes': classes,
            'bins': [[1, 1, 1, 1] for _ in classes]}


def quality(score, tp, fp):
    return {'score': score, 'tp': tp, 'fp': fp}


class RecoveryTests(unittest.TestCase):
    def test_matching_uses_gt_identity_despite_different_prediction_coordinates(self):
        gt = prediction([[10, 10, 50, 50], [100, 100, 140, 140]], [0, 1])
        a = prediction([[11, 10, 49, 50], [102, 101, 141, 140]], [0, 1])
        b = prediction([[9, 12, 50, 49]], [0])
        self.assertEqual(matched_ids(a, gt), {0, 1})
        self.assertEqual(matched_ids(b, gt), {0})

    def test_recovery_requires_no_gt_loss_and_no_extra_fp(self):
        q = [quality(.9, 2, 0), quality(.7, 1, 0)]
        r, m = eligible_pairs(q, [{0, 1}, {0}], [0, 1], [action([0, 1]), action([0])])
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]['rescued_classes'], [1])
        self.assertEqual(r[0]['lost_gt'], [])
        q[0]['fp'] = 1
        self.assertEqual(eligible_pairs(q, [{0, 1}, {0}], [0, 1], [action([0, 1]), action([0])])[0], [])

    def test_equal_tp_but_replacing_a_gt_is_not_recovery_or_maintenance(self):
        r, m = eligible_pairs([quality(.9, 2, 0), quality(.7, 2, 0)],
                              [{0, 1}, {0, 2}], [0, 0, 1], [action([0, 0]), action([0, 1])])
        self.assertEqual((r, m), ([], []))

    def test_all_six_pairs_are_checked_instead_of_only_extreme_scores(self):
        q = [quality(.95, 1, 0), quality(.8, 2, 0), quality(.6, 1, 0), quality(.3, 0, 5)]
        sets = [{2}, {0, 1}, {0}, set()]
        r, _ = eligible_pairs(q, sets, [0, 1, 2], [action([2]), action([0, 1]), action([0]), action([])])
        self.assertTrue(any(p['chosen_index'] == 1 and p['rejected_index'] == 2 for p in r))
        self.assertTrue(all(not p['lost_gt'] for p in r))

    def test_action_distance_is_not_inferred_from_matched_gt(self):
        a, b = action([0]), action([0])
        b['bins'][0][0] = 2
        self.assertEqual(action_distance(a, b), 1)
        self.assertEqual(action_distance(a, action([])), 5)

    def test_class_balancing_handles_multi_class_pairs_without_duplicate_rows(self):
        rows = [{'selection': {'group': 'recovery', 'rescued_classes': c}} for c in ([0], [0], [1], [0, 1])]
        rows += [{'selection': {'group': 'maintenance', 'rescued_classes': []}} for _ in range(4)]
        p, config = balanced_probabilities(rows)
        self.assertAlmostEqual(float(p.sum()), 1.)
        self.assertAlmostEqual(float(p[:4].sum()), .5)
        self.assertGreater(float(p[2]), float(p[0]))
        self.assertGreater(float(p[3]), float(p[2]))
        self.assertLessEqual(config['max_relative_repeat'], 5+1e-10)

    def test_repeat_cap_changes_infeasible_group_fraction(self):
        rows = [{'selection': {'group': 'recovery', 'rescued_classes': [0]}}]
        rows += [{'selection': {'group': 'maintenance', 'rescued_classes': []}} for _ in range(99)]
        p, config = balanced_probabilities(rows)
        self.assertAlmostEqual(float(p.sum()), 1.)
        self.assertLessEqual(float(p.max()), .05+1e-12)
        self.assertLess(config['expected_recovery_fraction'], .5)

    def test_canonical_identity_matching_agrees_with_counts_in_crowded_scene(self):
        gt = prediction([[0, 0, 10, 10], [4, 0, 14, 10], [30, 0, 40, 10]], [0, 0, 1])
        pred = prediction([[1, 0, 11, 10], [0, 0, 8, 10], [30, 0, 40, 10]], [0, 0, 0])
        matches = canonical_gt_ids(pred, gt)
        counts = _operating_point_counts([pred], [gt], .25, .5)
        self.assertEqual(len(matches), sum(v[0] for v in counts.values()))
        self.assertNotIn(2, matches)
        self.assertEqual(is_probe_image(42), is_probe_image(42))


if __name__ == '__main__':
    unittest.main()
