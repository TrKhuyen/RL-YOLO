import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_kb2_dpo import parse_args, run_hybrid, run_s0


class DPOEntrypointTests(unittest.TestCase):
    def test_default_commands_run_one_pipeline_and_keep_each_recipe(self):
        hybrid = parse_args([])
        standalone = parse_args(['--initialization', 's0'])
        self.assertEqual(hybrid.arms, ['dpo'])
        self.assertEqual((hybrid.steps, hybrid.dpo_weight, hybrid.background_weight), (15000, .1, .5))
        self.assertEqual(standalone.arms, ['dpo'])
        self.assertEqual((standalone.steps, standalone.dpo_weight), (2000, .01))

    def test_control_and_repeated_arms_are_rejected_before_model_loading(self):
        for values in (['native'], ['native', 'dpo'], ['bce'], ['dpo', 'dpo']):
            with self.subTest(values=values), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(['--arms', *values])
        self.assertEqual(parse_args(['--arms=dpo']).arms, ['dpo'])

    def test_zero_dpo_weight_cannot_silently_restore_native_only(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--dpo-weight', '0'])

    def test_s0_forwarding_removes_historical_default_control_arm(self):
        captured = []
        original = sys.argv
        with patch('kb2_preference_optimization.conditional.run.main',
                   side_effect=lambda: captured.extend(sys.argv[1:])):
            run_s0(parse_args(['--initialization', 's0', '--steps', '17']))
        self.assertIs(sys.argv, original)
        self.assertEqual(captured[captured.index('--arms')+1], 'dpo')
        self.assertNotIn('native', captured)
        self.assertEqual(captured[captured.index('--steps')+1], '17')

    def test_hybrid_writes_pipeline_scope_and_preserves_single_arm_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'new_run'
            args = parse_args(['--device', 'cpu', '--output-root', str(root)])
            aggregate = {'arms': {'dpo': {'best_step': 1000}}, 'S0': {'mAP50_95': .37},
                         'TPE': {'mAP50_95': .38}, 'interpretation': 'historical control wording'}
            def fake_execute(received, output, provenance):
                self.assertEqual(received.arms, ['dpo'])
                (output/'comparison.json').write_text(json.dumps(aggregate))
            provenance = {'source_tpe': {'trial': 2}, 'shared_S0': {}}
            with patch('kb2_preference_optimization.hybrid.source.source_provenance', return_value=provenance), \
                    patch('kb2_preference_optimization.hybrid.run.execute', side_effect=fake_execute):
                run_hybrid(args)
            result = json.loads((root/'completed.json').read_text())
            self.assertEqual(result, json.loads((root/'comparison.json').read_text()))
            self.assertEqual(result['arms'], aggregate['arms'])
            self.assertEqual(result['evaluation_models'], ['S0', 'TPE', 'KB2_DPO'])
            workflow = json.loads((root/'workflow.json').read_text())
            self.assertEqual(workflow['control_arms'], [])
            self.assertTrue(workflow['native_detection_loss_retained'])


if __name__ == '__main__':
    unittest.main()
