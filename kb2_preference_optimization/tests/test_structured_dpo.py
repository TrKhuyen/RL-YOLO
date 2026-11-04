"""Probability, matching and real-YOLO gradient checks for the new objective."""
import copy
import itertools
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from kb2_preference_optimization.adapter import DFLAdapter
from kb2_preference_optimization.policy import (
    DetectionDistribution, PolicyConfig, dpo_loss, render_action,
)
from kb2_preference_optimization.quality import QualityConfig, match_boxes, quality_score, select_pair


def prediction(boxes, labels=None, scores=None):
    return {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.tensor(labels if labels is not None else [0] * len(boxes), dtype=torch.long),
            'scores': torch.tensor(scores if scores is not None else [.9] * len(boxes))}


class ProbabilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_complete_action_probability_sums_to_one(self):
        distribution = DetectionDistribution(torch.tensor([[.7]]), torch.randn(1, 4, 2))
        actions = [{'slots': 1, 'indices': [], 'classes': [], 'bins': []}]
        actions += [{'slots': 1, 'indices': [0], 'classes': [0], 'bins': [list(bins)]}
                    for bins in itertools.product(range(2), repeat=4)]
        total = sum(distribution.log_prob(a).exp() for a in actions)
        self.assertAlmostEqual(float(total), 1., places=6)

    def test_background_and_coordinate_likelihood_are_included(self):
        dist = DetectionDistribution(torch.zeros(2, 1), torch.zeros(2, 4, 2))
        empty = {'slots': 2, 'indices': [], 'classes': [], 'bins': []}
        emitted = {'slots': 2, 'indices': [1], 'classes': [0], 'bins': [[0, 1, 0, 1]]}
        self.assertAlmostEqual(float(dist.log_prob(empty)), -2 * math.log(2), places=6)
        self.assertAlmostEqual(float(dist.log_prob(emitted)), -6 * math.log(2), places=6)

    def test_sampling_is_reproducible_but_produces_distinct_actions(self):
        dist = DetectionDistribution(torch.zeros(20, 2), torch.zeros(20, 4, 4))
        a = dist.sample(torch.Generator().manual_seed(1))
        self.assertEqual(a, dist.sample(torch.Generator().manual_seed(1)))
        self.assertNotEqual(a, dist.sample(torch.Generator().manual_seed(2)))

    def test_dpo_initial_loss_and_class_regression_gradient_direction(self):
        classes = torch.zeros(1, 2, requires_grad=True)
        boxes = torch.zeros(1, 4, 2, requires_grad=True)
        chosen = {'slots': 1, 'indices': [0], 'classes': [0], 'bins': [[1, 1, 1, 1]]}
        rejected = {'slots': 1, 'indices': [0], 'classes': [1], 'bins': [[0, 0, 0, 0]]}
        dist = DetectionDistribution(classes, boxes)
        a, b = dist.log_prob(chosen), dist.log_prob(rejected)
        ref_a, ref_b = a.detach().clone(), b.detach().clone()
        loss, relative = dpo_loss(a, b, ref_a, ref_b)
        self.assertAlmostEqual(float(loss), math.log(2), places=12)
        self.assertEqual(float(relative), 0.)
        loss.backward()
        self.assertLess(float(classes.grad[0, 0]), 0)
        self.assertGreater(float(classes.grad[0, 1]), 0)
        self.assertTrue((boxes.grad[:, :, 1] < 0).all())
        self.assertTrue((boxes.grad[:, :, 0] > 0).all())
        with torch.no_grad():
            classes -= .1 * classes.grad
            boxes -= .1 * boxes.grad
        new = DetectionDistribution(classes, boxes)
        updated, _ = dpo_loss(new.log_prob(chosen), new.log_prob(rejected), ref_a, ref_b)
        self.assertLess(float(updated), float(loss))

    def test_cached_double_reference_does_not_lose_precision(self):
        a = torch.tensor(-12345.123456789, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(-12346.987654321, dtype=torch.float64, requires_grad=True)
        loss, margin = dpo_loss(a, b, float(a), float(b))
        self.assertEqual(float(margin), 0.)
        self.assertAlmostEqual(float(loss), math.log(2), places=12)

    def test_reference_is_detached(self):
        a = torch.tensor(2., requires_grad=True)
        b = torch.tensor(1., requires_grad=True)
        r = torch.tensor(3., requires_grad=True)
        loss, _ = dpo_loss(a, b, r, r)
        loss.backward()
        self.assertIsNone(r.grad)

    def test_invalid_actions_and_config_are_rejected(self):
        dist = DetectionDistribution(torch.zeros(2, 1), torch.zeros(2, 4, 2))
        for action in (
            {'slots': 3, 'indices': [], 'classes': [], 'bins': []},
            {'slots': 2, 'indices': [0, 0], 'classes': [0, 0], 'bins': [[0]*4, [0]*4]},
            {'slots': 2, 'indices': [0], 'classes': [0], 'bins': [[2]*4]},
            {'slots': 2, 'indices': [0], 'classes': [1], 'bins': [[0]*4]},
        ):
            with self.assertRaises(ValueError):
                dist.log_prob(action)
        for value in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                PolicyConfig(box_temperature=value)

    def test_renderer_replays_fixed_coordinates_independent_of_policy_logits(self):
        action = {'slots': 1, 'indices': [0], 'classes': [0], 'bins': [[1, 1, 2, 2]]}
        a = DetectionDistribution(torch.zeros(1, 1), torch.zeros(1, 4, 3))
        b = DetectionDistribution(torch.randn(1, 1), torch.randn(1, 4, 3))
        anchors, strides, scores = torch.tensor([[3., 3.]]), torch.tensor([[8.]]), torch.tensor([[.9]])
        ra = render_action(a, action, anchors, strides, scores, 64)
        rb = render_action(b, action, anchors, strides, scores, 64)
        self.assertTrue(torch.equal(ra['boxes'], torch.tensor([[16., 16., 40., 40.]])))
        for key in ra:
            self.assertTrue(torch.equal(ra[key], rb[key]))


class QualityTests(unittest.TestCase):
    def test_f2_prefers_recovered_gt_with_moderate_extra_false_positives(self):
        # Ten GT with exact geometry at all ten IoU thresholds. A finds nine
        # with three FP; B finds eight with zero FP. F1 and F2 disagree.
        boxes = [[20*i, 0, 20*i+10, 10] for i in range(10)]
        gt = prediction(boxes)
        a = prediction(boxes[:9] + [[300+20*i, 0, 310+20*i, 10] for i in range(3)])
        b = prediction(boxes[:8])
        f1_a, f1_b = [quality_score(p, gt, QualityConfig(f_beta=1.)) for p in (a, b)]
        f2_a, f2_b = [quality_score(p, gt) for p in (a, b)]
        self.assertLess(f1_a['score'], f1_b['score'])
        self.assertGreater(f2_a['score'], f2_b['score'])
        self.assertAlmostEqual(f2_a['mean_f2'], 45 / 52)
        self.assertAlmostEqual(f2_b['mean_f2'], 40 / 48)
        self.assertEqual((f2_a['tp'], f2_a['fp'], f2_a['fn']), (9, 3, 1))
        self.assertAlmostEqual(f1_a['score'], .8 * 18 / 22 + .2 * .9)
        self.assertEqual(QualityConfig().to_dict()['f_beta'], 2.)

    def test_f2_still_penalizes_false_positives_and_invalid_beta(self):
        gt = prediction([[0, 0, 10, 10]])
        spam = prediction([[0, 0, 10, 10]] + [[20+i*20, 0, 30+i*20, 10] for i in range(20)])
        self.assertLess(quality_score(spam, gt)['score'], quality_score(gt, gt)['score'])
        for beta in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                QualityConfig(f_beta=beta)

    def test_perfect_output_beats_missing_wrong_class_duplicate_and_loose_boxes(self):
        gt = prediction([[0, 0, 10, 10], [20, 20, 30, 30]], [0, 1])
        perfect = quality_score(gt, gt)['score']
        self.assertAlmostEqual(perfect, 1.)
        bad = [prediction([[0, 0, 10, 10]], [0]),
               prediction([[0, 0, 10, 10], [20, 20, 30, 30]], [1, 0]),
               prediction([[0, 0, 10, 10], [20, 20, 30, 30], [0, 0, 10, 10]], [0, 1, 0]),
               prediction([[0, 0, 13, 13], [20, 20, 33, 33]], [0, 1])]
        for output in bad:
            self.assertLess(quality_score(output, gt)['score'], perfect)

    def test_matching_maximizes_cardinality_before_iou(self):
        # P0 matches either GT, P1 only GT0; greedy matching could lose a TP.
        gt = prediction([[0, 0, 10, 10], [4, 0, 14, 10]])
        output = prediction([[1, 0, 11, 10], [0, 0, 8, 10]])
        self.assertEqual(len(match_boxes(output, gt, .5)), 2)

    def test_quality_does_not_depend_on_confidence(self):
        gt = prediction([[0, 0, 10, 10]])
        p = prediction([[0, 0, 9, 9]], scores=[.01])
        q = quality_score(p, gt)
        p['scores'].fill_(.99)
        self.assertEqual(quality_score(p, gt), q)

    def test_empty_cases_and_preference_ties(self):
        empty, gt = prediction([]), prediction([[0, 0, 10, 10]])
        self.assertEqual(quality_score(empty, empty)['score'], 1.)
        self.assertEqual(quality_score(empty, gt)['score'], 0.)
        self.assertLess(quality_score(gt, empty)['score'], 1.)
        self.assertIsNone(select_pair([gt, gt], [{'score': .8}, {'score': .7}], .02))
        self.assertIsNone(select_pair([gt, empty], [{'score': .8}, {'score': .79}], .02))
        self.assertEqual(select_pair([gt, empty], [{'score': 1.}, {'score': 0.}], .02), (0, 1))


class RealYOLOTests(unittest.TestCase):
    def test_dfl_policy_native_loss_and_reference_immutability(self):
        from ultralytics.nn.tasks import DetectionModel
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = DetectionModel('yolov8n.yaml', nc=3, verbose=False)
        model.args = {}
        with patch('kb2_preference_optimization.adapter.YOLO', return_value=SimpleNamespace(model=model)):
            adapter = DFLAdapter('unused.pt', 'cpu')
        reference = copy.deepcopy(adapter)
        reference.model.requires_grad_(False)
        before = {k: v.clone() for k, v in reference.model.state_dict().items()}
        policy_before = {k: v.clone() for k, v in adapter.model.state_dict().items()}
        images = torch.rand(2, 3, 64, 64)
        targets = [prediction([[8, 8, 40, 40]], [1]) for _ in range(2)]
        distributions, _, _, _ = adapter.distributions(images)
        ref, _, _, _ = reference.distributions(images)
        # A freshly initialized detector can sample background everywhere;
        # explicitly construct two supported actions to test regression signal.
        chosen = {'slots': distributions[0].slots, 'indices': [0],
                  'classes': [1], 'bins': [[2, 2, 2, 2]]}
        rejected = {'slots': distributions[0].slots, 'indices': [0],
                    'classes': [2], 'bins': [[3, 3, 3, 3]]}
        preference, _ = dpo_loss(distributions[0].log_prob(chosen), distributions[0].log_prob(rejected),
                                 ref[0].log_prob(chosen), ref[0].log_prob(rejected))
        preference.backward()
        for head in (adapter.head.cv2, adapter.head.cv3):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))
        adapter.model.zero_grad(set_to_none=True)
        native = adapter.native_loss(images, targets)
        self.assertTrue(torch.isfinite(native))
        native.backward()
        optimizer = torch.optim.AdamW([p for p in adapter.model.parameters() if p.requires_grad], lr=1e-5)
        optimizer.step()
        for key, value in reference.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertTrue(all(p.grad is None for p in reference.model.parameters()))
        for key, value in adapter.model.state_dict().items():
            if 'running_' in key or 'num_batches_tracked' in key or not key.startswith('model.22.'):
                self.assertTrue(torch.equal(value, policy_before[key]), key)
        self.assertTrue(any(not torch.equal(v, policy_before[k]) for k, v in adapter.model.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
