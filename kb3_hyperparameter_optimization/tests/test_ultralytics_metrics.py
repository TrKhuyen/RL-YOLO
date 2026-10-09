import unittest

from kb3_hyperparameter_optimization.adapters.ultralytics_worker import _row_metrics


class LossMetricsTests(unittest.TestCase):
    def test_dfl_and_yolo26_l1_are_both_counted(self):
        for regression in ('dfl_loss', 'l1_loss'):
            with self.subTest(regression=regression):
                row = {'metrics/precision(B)': .5, 'metrics/recall(B)': .6,
                       'metrics/mAP50(B)': .7, 'metrics/mAP50-95(B)': .4}
                for prefix in ('train', 'val'):
                    row.update({f'{prefix}/box_loss': 1., f'{prefix}/cls_loss': 2.,
                                f'{prefix}/{regression}': 3.})
                result = _row_metrics(row, 1, 2., 0.)
                self.assertEqual(result['train_loss'], 6.)
                self.assertEqual(result['val_loss'], 6.)

    def test_missing_regression_loss_fails(self):
        with self.assertRaises(KeyError):
            _row_metrics({'train/box_loss': 1., 'train/cls_loss': 2.}, 1, 0., 0.)
