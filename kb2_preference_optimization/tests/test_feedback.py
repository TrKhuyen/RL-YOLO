import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feedback import build_feedback_record
from feedback_dataset import feedback_difficulty, object_feedback_codes


def record(boxes, labels, scores, gt_boxes, gt_labels):
    prediction = {
        'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        'labels': torch.tensor(labels, dtype=torch.long),
        'scores': torch.tensor(scores, dtype=torch.float32),
    }
    target = {
        'boxes': torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4),
        'labels': torch.tensor(gt_labels, dtype=torch.long),
        'image_id': torch.tensor([0]),
    }
    return build_feedback_record(prediction, target)


class MatchingTests(unittest.TestCase):
    def test_correct_match_wins_over_wrong_class_and_duplicate(self):
        result = record(
            [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]],
            [1, 2, 1], [.8, .99, .7],
            [[0, 0, 10, 10]], [1])
        self.assertEqual(result['gt_status'], ['matched'])
        self.assertEqual(len(result['feedback']['matched']), 1)
        self.assertEqual(len(result['feedback']['duplicate']), 1)
        self.assertEqual(len(result['feedback']['missed']), 0)
        self.assertEqual(object_feedback_codes(result, [0]).tolist(), [0])
        self.assertAlmostEqual(feedback_difficulty(result), 0.9)

    def test_nearby_ground_truths_match_correct_classes(self):
        result = record(
            [[0, 0, 10, 10], [2, 0, 12, 10]],
            [1, 2], [.9, .8],
            [[0, 0, 10, 10], [2, 0, 12, 10]], [1, 2])
        self.assertEqual(result['gt_status'], ['matched', 'matched'])
        self.assertEqual({item['gt_index'] for item in result['feedback']['matched']}, {0, 1})

    def test_matching_maximizes_covered_nearby_ground_truths(self):
        result = record(
            [[2, 0, 12, 10], [0, 0, 10, 10]],
            [1, 1], [.9, .8],
            [[0, 0, 10, 10], [4, 0, 14, 10]], [1, 1])
        self.assertEqual(result['gt_status'], ['matched', 'matched'])
        mapping = {item['gt_index']: item['prediction_index']
                   for item in result['feedback']['matched']}
        self.assertEqual(mapping, {0: 1, 1: 0})

    def test_wrong_class_and_bad_localization_are_not_missed(self):
        result = record(
            [[0, 0, 10, 10], [20, 0, 30, 10]],
            [2, 1], [.9, .8],
            [[0, 0, 10, 10], [25, 0, 35, 10]], [1, 1])
        self.assertEqual(result['gt_status'], ['wrong_class', 'bad_localization'])
        self.assertEqual(len(result['feedback']['missed']), 0)
        self.assertEqual(object_feedback_codes(result, [1, 0]).tolist(), [2, 1])

    def test_preference_never_rejects_primary_match_of_other_gt(self):
        result = record(
            [[0, 0, 10, 10], [20, 0, 30, 10], [1, 0, 11, 10]],
            [1, 1, 1], [.8, .99, .5],
            [[0, 0, 10, 10], [20, 0, 30, 10]], [1, 1])
        self.assertEqual(result['gt_status'], ['matched', 'matched'])
        self.assertEqual(len(result['preferences']), 1)
        self.assertEqual(result['preferences'][0]['rejected_prediction_index'], 2)


if __name__ == '__main__':
    unittest.main()
