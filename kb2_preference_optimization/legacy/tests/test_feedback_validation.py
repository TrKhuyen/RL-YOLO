import sys
import unittest
from pathlib import Path

KB2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KB2_DIR))

from feedback_validation import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
    def test_max_mode_improvement_patience_and_delta(self):
        stop = EarlyStopping(patience=2, min_delta=.01)
        self.assertEqual(stop.update(.50), (True, False))
        self.assertEqual(stop.update(.505), (False, False))
        self.assertEqual(stop.update(.506), (False, True))
        self.assertEqual(stop.update(.52), (True, False))
        self.assertAlmostEqual(stop.best, .52)

    def test_state_round_trip(self):
        first = EarlyStopping(patience=3)
        first.update(.5); first.update(.4)
        second = EarlyStopping(patience=3)
        second.load_state_dict(first.state_dict())
        self.assertEqual(second.best, .5)
        self.assertEqual(second.bad_evaluations, 1)

    def test_rejects_invalid_configuration_and_metric(self):
        for kwargs in ({'patience': 0}, {'min_delta': -1}, {'mode': 'sideways'}):
            with self.assertRaises(ValueError): EarlyStopping(**kwargs)
        with self.assertRaisesRegex(ValueError, 'finite'):
            EarlyStopping().update(float('nan'))


if __name__ == '__main__': unittest.main(verbosity=2)
