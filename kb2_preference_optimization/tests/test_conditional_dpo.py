"""Scientific contracts of conditional keep/drop DPO, independent of training."""
import copy
import math
import os
import unittest

import torch
from torch.nn import functional as F

from kb1_reward_guided_training.canonical_eval import _operating_point_counts
from kb2_preference_optimization.conditional.bank import sampling_probabilities
from kb2_preference_optimization.conditional.evaluation import choose_probes, eligible, evaluate
from kb2_preference_optimization.conditional.pairs import PairConfig, build_pairs, judge, matches, render
from kb2_preference_optimization.conditional.policy import pair_loss


def target(boxes, labels):
    return {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.tensor(labels, dtype=torch.long)}


def context(boxes, scores, labels, active=None):
    return {'slots': list(range(len(boxes))), 'boxes': boxes, 'scores': scores,
            'labels': labels, 'active': active or [True] * len(boxes)}


class ConditionalDPOTests(unittest.TestCase):
    def test_loss_equals_normalized_bernoulli_dpo_and_gradient_sign(self):
        z = torch.tensor([[-80., 80.], [2., -3.]], requires_grad=True).reshape(2, 1, 2)
        ref = torch.tensor([[-81., 82.], [1., -1.]]).reshape(2, 1, 2)
        records = [{'batch_index': 0, 'slot': 0, 'class_id': 0, 'chosen_keep': True},
                   {'batch_index': 1, 'slot': 0, 'class_id': 1, 'chosen_keep': False}]
        loss, margin, selected, selected_ref = pair_loss(z, ref, records)
        signs = torch.tensor([1., -1.])
        gap = signs * (F.logsigmoid(selected) - F.logsigmoid(-selected)
                       - F.logsigmoid(selected_ref) + F.logsigmoid(-selected_ref))
        torch.testing.assert_close(margin, gap)
        torch.testing.assert_close(loss, F.softplus(-.1 * gap).mean())
        gradients = torch.autograd.grad(loss, z)[0]
        self.assertLess(float(gradients[0, 0, 0]), 0)
        self.assertGreater(float(gradients[1, 0, 1]), 0)
        self.assertEqual(float(gradients[0, 0, 1]), 0)

    def test_initial_reference_margin_is_zero(self):
        z = torch.randn(4, 10, 3)
        records = [{'batch_index': 3, 'slot': 9, 'class_id': 2, 'chosen_keep': True}]
        loss, margin, _, _ = pair_loss(z, z.clone(), records)
        self.assertEqual(float(margin[0]), 0)
        self.assertAlmostEqual(float(loss), math.log(2), places=6)

    def test_low_confidence_keep_recovers_true_FN_without_fabricating_box_or_score(self):
        gt = target([[0, 0, 10, 10], [20, 0, 30, 10]], [0, 0])
        boxes = gt['boxes'].clone()
        logits = torch.logit(torch.tensor([[.8, .01], [.2, .01]]))
        ctx, pairs, diagnostic = build_pairs(boxes, logits, gt)
        recovery = next(r for r in pairs if r['group'] == 'recovery')
        self.assertEqual(diagnostic['s0_gt'], [0])
        self.assertEqual(recovery['target_gt'], 1)
        kept = render(ctx, recovery['slot'], True)
        self.assertAlmostEqual(float(kept['scores'][-1]), .2, places=6)
        torch.testing.assert_close(kept['boxes'][-1], boxes[1])
        self.assertEqual(matches(kept, gt)[0], {0, 1})
        self.assertEqual(matches(kept, gt, .25)[0], {0})

    def test_common_context_can_contain_errors_and_is_immutable(self):
        ctx = context([[0, 0, 10, 10], [20, 0, 30, 10], [50, 0, 60, 10]],
                      [.8, .2, .9], [0, 0, 1], [True, False, True])
        snapshot = copy.deepcopy(ctx)
        a, b = render(ctx, 1, True), render(ctx, 1, False)
        self.assertEqual(ctx, snapshot)
        self.assertEqual(set(a['slots'].tolist()) - set(b['slots'].tolist()), {1})
        self.assertIn(2, b['slots'].tolist())  # Common background error remains common.
        valid, _ = judge(ctx, target([[0, 0, 10, 10], [20, 0, 30, 10]], [0, 0]), 1, True, 1)
        self.assertTrue(valid)

    def test_keep_rejected_if_common_candidate_already_covers_target(self):
        gt = target([[0, 0, 10, 10]], [0])
        ctx = context([[0, 0, 7, 10], [3, 0, 10, 10]], [.8, .2], [0, 0], [True, False])
        valid, _ = judge(ctx, gt, 1, True, 0)
        self.assertFalse(valid)

    def test_whole_image_gate_rejects_NMS_loss_of_neighbor(self):
        gt = target([[0, 0, 10, 10], [1, 0, 11, 10]], [0, 0])
        ctx = context(gt['boxes'].tolist(), [.8, .95], [0, 0], [True, False])
        valid, evidence = judge(ctx, gt, 1, True, 1)
        self.assertFalse(valid)
        self.assertEqual(evidence['rejected_gt'], [0])
        self.assertEqual(evidence['chosen_gt'], [1])

    def test_wrong_class_and_localization_errors_are_not_background_negatives(self):
        gt = target([[0, 0, 10, 10]], [0])
        boxes = torch.tensor([[0, 0, 10, 10], [6, 0, 16, 10], [30, 0, 40, 10]], dtype=torch.float32)
        logits = torch.logit(torch.tensor([[.1, .9], [.8, .01], [.8, .01]]))
        _, pairs, _ = build_pairs(boxes, logits, gt)
        negatives = [r for r in pairs if not r['chosen_keep']]
        self.assertEqual([r['slot'] for r in negatives], [2])

    def test_duplicate_drop_requires_real_FP_reduction_without_GT_loss(self):
        gt = target([[0, 0, 10, 10]], [0])
        boxes = torch.tensor([[0, 0, 7, 10], [3, 0, 10, 10]], dtype=torch.float32)
        logits = torch.logit(torch.tensor([[.8, .01], [.7, .01]]))
        _, pairs, _ = build_pairs(boxes, logits, gt)
        duplicate = next(r for r in pairs if r['group'] == 'duplicate')
        self.assertFalse(duplicate['chosen_keep'])
        self.assertEqual(duplicate['evidence']['chosen_gt'], [0])
        self.assertEqual(duplicate['evidence']['chosen_fp'], 0)
        self.assertEqual(duplicate['evidence']['rejected_fp'], 1)

    def test_matching_counts_agree_with_canonical_in_crowded_multiclass_image(self):
        gt = target([[0, 0, 10, 10], [1, 0, 11, 10], [40, 0, 50, 10]], [0, 0, 1])
        prediction = {'boxes': torch.tensor([[0, 0, 10, 10], [0, 0, 9, 10], [40, 0, 50, 10], [70, 0, 80, 10]], dtype=torch.float32),
                      'scores': torch.tensor([.9, .8, .2, .7]), 'labels': torch.tensor([0, 0, 1, 1])}
        found, fp, _ = matches(prediction, gt, .25)
        canonical = _operating_point_counts([prediction], [gt])
        self.assertEqual(len(found), sum(v[0] for v in canonical.values()))
        self.assertEqual(fp, sum(v[1] for v in canonical.values()))

    def test_sampler_limits_image_exposure_even_with_many_pairs_same_GT(self):
        records = [{'image_id': i, 'target_gt': 0, 'slot': j, 'class_id': i % 3,
                    'group': 'recovery' if i == 0 else 'background', 'chosen_keep': i == 0}
                   for i in range(40) for j in range(100 if i == 0 else 1)]
        weights, metadata = sampling_probabilities(records)
        image_mass = [sum(float(w) for r, w in zip(records, weights) if r['image_id'] == i) for i in range(40)]
        self.assertAlmostEqual(float(weights.sum()), 1., places=12)
        self.assertLessEqual(max(image_mass), 5 / 40 + 1e-12)
        self.assertLess(metadata['expected_keep_fraction'], .5)
        self.assertTrue(torch.all(weights > 0))

    def test_checkpoint_gate_protects_recall_FP_and_AP(self):
        source = {'operating_recall': .5, 'fp': 100, 'mAP50_95': .38}
        self.assertTrue(eligible(source, source))
        self.assertFalse(eligible({**source, 'operating_recall': .49}, source))
        self.assertFalse(eligible({**source, 'fp': 106}, source))
        self.assertFalse(eligible({**source, 'mAP50_95': .378}, source))

    def test_small_probe_budget_covers_all_decision_groups(self):
        records = [{'group': 'background', 'class_id': c} for c in range(28)]
        records += [{'group': group, 'class_id': 0} for group in ('recovery', 'preservation', 'duplicate')]
        selected = choose_probes(records, 4)
        self.assertEqual({r['group'] for r in selected}, {'background', 'recovery', 'preservation', 'duplicate'})

    def test_validation_distinguishes_rescued_GT_from_lost_neighbor_at_equal_TP(self):
        gt = target([[0, 0, 10, 10], [20, 0, 30, 10]], [0, 1])
        source = {'boxes': gt['boxes'][0:1], 'labels': gt['labels'][0:1], 'scores': torch.tensor([.8])}
        policy = {'boxes': gt['boxes'][1:2], 'labels': gt['labels'][1:2], 'scores': torch.tensor([.9])}

        class Adapter:
            device = torch.device('cpu')

            def predict(self, images, *args):
                return [policy]

        metrics, _ = evaluate(Adapter(), [(torch.zeros(1, 3, 64, 64), [gt])], [source])
        transitions = metrics['source_gt_transitions']
        self.assertEqual((transitions['recovered'], transitions['lost']), (1, 1))
        self.assertEqual(transitions['net_recovery'], 0)
        self.assertEqual(transitions['per_class']['0']['lost'], 1)
        self.assertEqual(transitions['per_class']['1']['recovered'], 1)

    @unittest.skipUnless(os.environ.get('KB2_RUN_GPU_TESTS') == '1', 'Opt-in real S0/GPU integration')
    def test_real_Adam_step_preserves_geometry_and_all_frozen_state(self):
        from kb2_preference_optimization.artifacts import source_checkpoint, tensor_digest
        from kb2_preference_optimization.conditional.policy import ConditionalAdapter, configure_numerics
        from kb2_preference_optimization.data import DetectionDataset
        configure_numerics()
        policy = ConditionalAdapter(source_checkpoint('yolov8n'), 'cuda')
        reference = ConditionalAdapter(source_checkpoint('yolov8n'), 'cuda')
        reference.model.requires_grad_(False)
        reference_digest = tensor_digest(reference.model)
        frozen = policy.frozen_digest()
        data = DetectionDataset('train')
        pairs = []
        for index in range(min(64, len(data))):
            image, gt = data[index]
            images = image[None].cuda()
            with torch.no_grad():
                initial_logits, initial_boxes = reference.raw(images)
            _, pairs, _ = build_pairs(initial_boxes[0], initial_logits[0], gt)
            if pairs:
                break
        self.assertTrue(pairs)
        native = policy.native_loss(images, [gt])
        logits, boxes = policy.raw(images)
        records = [{**pairs[0], 'batch_index': 0}]
        loss, margin, _, _ = pair_loss(logits, initial_logits, records)
        self.assertEqual(float(margin.abs().max()), 0.)
        optimizer = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=1e-6)
        (native + .01 * loss).backward()
        optimizer.step()
        with torch.no_grad():
            _, after_boxes = policy.raw(images)
        torch.testing.assert_close(after_boxes, initial_boxes, atol=0, rtol=0)
        self.assertEqual(policy.frozen_digest(), frozen)
        self.assertEqual(tensor_digest(reference.model), reference_digest)


if __name__ == '__main__':
    unittest.main()
