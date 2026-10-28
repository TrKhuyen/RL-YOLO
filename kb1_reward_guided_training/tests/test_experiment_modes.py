"""Contracts for the ablations proposed in the KB1 report."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_modes import objective_weights


class AblationModes(unittest.TestCase):
    def test_isolate_components_without_changing_existing_objectives(self):
        config = {'native_supervised_loss_weight': .25, 'supervised_loss_weight': .25,
                  'reward_loss_weight': .1, 'stability_loss_weight': .001}
        expected = {'native_only': (.25, 0, 0, .001), 'proxy_only': (.25, .25, 0, .001),
                    'reward_only': (.25, 0, .1, .001), 'kb1b': (.25, .25, .1, .001),
                    'kb1b_no_sp': (.25, .25, .1, 0)}
        for mode, values in expected.items():
            with self.subTest(mode=mode):
                actual = objective_weights(mode, config)
                self.assertEqual(tuple(actual[k] for k in ('native','proxy','reward','stability')), values)
        with self.assertRaises(ValueError):
            objective_weights('typo', config)
