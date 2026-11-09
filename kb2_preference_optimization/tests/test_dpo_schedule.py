import contextlib
import io
import unittest

from kb2_preference_optimization.hybrid_schedule.schedule import evaluation_steps, learning_rate_for_update
from scripts.run_kb2_dpo import parse_args


class DPOScheduleTests(unittest.TestCase):
    def test_drop_applies_to_update_1001_not_update_1000(self):
        self.assertEqual(learning_rate_for_update(1e-6, 1000, 1000, .1), 1e-6)
        self.assertEqual(learning_rate_for_update(1e-6, 1001, 1000, .1), 1e-7)
        self.assertEqual(learning_rate_for_update(1e-6, 15000, 1000, .1), 1e-7)

    def test_validation_density_never_changes_optimizer_schedule(self):
        updates = [1, 250, 500, 750, 1000, 1001, 1250, 2000, 15000]
        rates = [learning_rate_for_update(1e-6, step, 1000, .1) for step in updates]
        self.assertEqual(rates, [1e-6]*5 + [1e-7]*4)
        self.assertEqual(learning_rate_for_update(1e-6, 15000), 1e-6)

    def test_dense_grid_keeps_all_historical_checkpoints_and_final_partial_interval(self):
        grid = evaluation_steps(15000, 1000, 250, 2000)
        self.assertEqual(len(grid), 22)
        self.assertTrue(set(range(0, 15001, 1000)) <= set(grid))
        self.assertEqual([v for v in grid if v <= 2000], list(range(0, 2001, 250)))
        self.assertEqual(evaluation_steps(17, 1000, 5, 10), [0, 5, 10, 17])

    def test_cli_accepts_schedule_and_rejects_silent_unsupported_sources(self):
        args = parse_args(['--lr-drop-step', '1000', '--early-eval-interval', '250'])
        self.assertEqual(args.arms, ['dpo'])
        self.assertEqual((args.lr_drop_step, args.lr_drop_factor, args.early_eval_until), (1000, .1, 2000))
        for flags in (['--initialization', 's0', '--lr-drop-step', '1000'],
                      ['--lr-drop-factor', '0'], ['--early-eval-interval', '-1']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(flags)


if __name__ == '__main__':
    unittest.main()
