import copy
import unittest

import torch
from torch.nn import functional as F

from kb2_preference_optimization.candidate_rank.pairs import build_rank_pairs, build_rank_context, judge_swap, render_swap
from kb2_preference_optimization.candidate_rank.policy import rank_loss


class CandidateRankingTests(unittest.TestCase):
    def test_pair_uses_real_suppressed_box_and_improves_localization(self):
        boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 7., 10.]])
        logits = torch.logit(torch.tensor([[.4, .01], [.7, .01]]))
        target = {'boxes': boxes[:1].clone(), 'labels': torch.tensor([0])}
        context, records, diagnostic, _ = build_rank_pairs(boxes, logits, target)
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual((r['good_slot'], r['bad_slot']), (0, 1))
        self.assertAlmostEqual(r['evidence']['chosen_target_iou'], 1.)
        self.assertAlmostEqual(r['evidence']['rejected_target_iou'], .7, places=6)
        before = copy.deepcopy(context)
        a = render_swap(context, 0, 1, True)
        b = render_swap(context, 0, 1, False)
        self.assertEqual(context, before)
        torch.testing.assert_close(a['boxes'][0], boxes[0], rtol=0, atol=0)
        torch.testing.assert_close(b['boxes'][0], boxes[1], rtol=0, atol=0)
        self.assertAlmostEqual(float(a['scores'][0]), .4, places=6)
        self.assertAlmostEqual(float(b['scores'][0]), .7, places=6)

    def test_whole_image_rejects_candidate_suppressed_by_neighbor(self):
        boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 6.5, 10.], [2., 0., 12., 10.]])
        logits = torch.logit(torch.tensor([[.4, .01], [.7, .01], [.9, .01]]))
        target = {'boxes': boxes[[0, 2]].clone(), 'labels': torch.tensor([0, 0])}
        context, records, diagnostic, _ = build_rank_pairs(boxes, logits, target)
        self.assertEqual(records, [])
        valid, _ = judge_swap(context, target, 0, 1, 0)
        self.assertFalse(valid)
        self.assertGreater(diagnostic['gate_rejected'], 0)

    def test_no_pair_without_existing_better_box(self):
        boxes = torch.tensor([[0., 0., 7., 10.]])
        logits = torch.logit(torch.tensor([[.7, .01]]))
        target = {'boxes': torch.tensor([[0., 0., 10., 10.]]), 'labels': torch.tensor([0])}
        _, records, _, rows = build_rank_pairs(boxes, logits, target)
        self.assertEqual(records, [])
        self.assertEqual(rows[0]['status'], 'no_better_suppressed_candidate')

    def test_cached_context_preserves_noncontiguous_original_slot_ids(self):
        target = {'boxes': torch.tensor([[0., 0., 10., 10.]]), 'labels': torch.tensor([0])}
        source = {'slots': [8123, 21, 90], 'boxes': [[0, 0, 10, 10], [0, 0, 7, 10], [0, 0, 10, 10]],
                  'scores': [.4, .7, .1], 'labels': [0, 0, 0], 'active': [True, True, False]}
        before = copy.deepcopy(source)
        _, pairs, _, _ = build_rank_context(source, target)
        self.assertEqual((pairs[0]['good_slot'], pairs[0]['bad_slot']), (8123, 21))
        self.assertEqual(source, before)

    def test_normalized_XOR_policy_and_reference_DPO_gradient(self):
        z = torch.tensor([[[.3], [-.7]]], requires_grad=True)
        ref = torch.tensor([[[.1], [-.5]]], requires_grad=True)
        record = [{'batch_index': 0, 'class_id': 0, 'good_slot': 0, 'bad_slot': 1}]
        loss, margin = rank_loss(z, ref, record)
        p, p0 = z.sigmoid(), ref.sigmoid()
        a, b = p[0, 0, 0]*(1-p[0, 1, 0]), (1-p[0, 0, 0])*p[0, 1, 0]
        a0, b0 = p0[0, 0, 0]*(1-p0[0, 1, 0]), (1-p0[0, 0, 0])*p0[0, 1, 0]
        torch.testing.assert_close(a/(a+b), torch.sigmoid(z[0, 0, 0]-z[0, 1, 0]))
        torch.testing.assert_close(margin[0], torch.log(a/b)-torch.log(a0/b0))
        torch.testing.assert_close(loss, F.softplus(-.1*margin).mean())
        loss.backward()
        self.assertLess(float(z.grad[0, 0, 0]), 0)
        self.assertGreater(float(z.grad[0, 1, 0]), 0)
        self.assertIsNone(ref.grad)
        with self.assertRaises(ValueError):
            rank_loss(z, ref, [{**record[0], 'good_slot': 1}])


if __name__ == '__main__':
    unittest.main()
