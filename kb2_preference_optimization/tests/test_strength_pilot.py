import json
import os
from pathlib import Path
import tempfile
import unittest

from kb2_preference_optimization.conditional_strength.selection import select_pareto


class StrengthSelectionTests(unittest.TestCase):
    def test_tradeoff_retains_source_and_improved_AP_but_removes_dominated_point(self):
        def row(step, ap, precision, recall):
            return {'step': step, 'metrics': {'mAP50_95': ap, 'precision': precision, 'operating_recall': recall}}
        history = [row(0, .38, .5, .54), row(1000, .39, .52, .53), row(2000, .385, .51, .52)]
        self.assertEqual(select_pareto(history), [0, 1000])

    def test_exact_ties_do_not_invent_improvement(self):
        m = {'mAP50_95': .38, 'precision': .5, 'operating_recall': .54}
        self.assertEqual(select_pareto([{'step': 0, 'metrics': m}, {'step': 1, 'metrics': m.copy()}]), [0, 1])

    @unittest.skipUnless(os.environ.get('KB2_RUN_GPU_TESTS') == '1', 'Opt-in actual trainer GPU smoke')
    def test_every_eval_checkpoint_saved_with_new_selection_and_immutable_state(self):
        import torch
        from kb2_preference_optimization.artifacts import sha256
        from kb2_preference_optimization.conditional_strength.train import train
        bank = 'kb2_preference_optimization/outputs/yolov8n_conditional_bank_20261031_v1/preferences'
        with tempfile.TemporaryDirectory(prefix='kb2-strength-test-') as temp:
            path = Path(temp) / 'run'
            result = train(bank, path, steps=2, eval_interval=1, eval_limit=8, audit_pairs=4, dpo_weight=.03)
            self.assertTrue(result['reference_unchanged'] and result['frozen_state_unchanged'])
            self.assertEqual(result['selection']['all_saved_steps'], [0, 1, 2])
            for step in (0, 1, 2):
                file = path / f'step_{step:06d}.pt'
                self.assertEqual(sha256(file), result['evaluation_checkpoint_sha256'][str(step)])
                checkpoint = torch.load(file, map_location='cpu', weights_only=False)
                self.assertEqual(checkpoint['step'], step)
                self.assertTrue(all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values()))
            rows = [json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['eligible'] for r in rows if r['event'] == 'validation'))
            self.assertIn('no recall gate', json.loads((path/'manifest.json').read_text())['config']['selection'])


if __name__ == '__main__':
    unittest.main()
