import sys
import tempfile
import unittest
from pathlib import Path

import torch

KB2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KB2_DIR))

from train_feedback import load_training_checkpoint, parse_args, save_training_checkpoint


class FakeAdapter:
    def __init__(self): self.model = torch.nn.Linear(2, 1)
    def state_dict(self): return self.model.state_dict()


class TrainFeedbackTests(unittest.TestCase):
    def test_checkpoint_round_trip_restores_model_optimizer_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            source = FakeAdapter()
            optimizer = torch.optim.AdamW(source.model.parameters(), lr=.01)
            before = {k: v.clone() for k, v in source.state_dict().items()}
            save_training_checkpoint(path, source, optimizer, 7, {'seed': 42})
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix('.pt.tmp').exists())

            target = FakeAdapter()
            target_optimizer = torch.optim.AdamW(target.model.parameters(), lr=.5)
            step, config, training_state = load_training_checkpoint(
                path, target, target_optimizer, 'cpu')
            self.assertEqual(step, 7)
            self.assertEqual(config['seed'], 42)
            self.assertEqual(training_state, {})
            for name, value in target.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]))
            self.assertAlmostEqual(target_optimizer.param_groups[0]['lr'], .01)

    def test_rejects_wrong_checkpoint_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.pt'
            torch.save({'training_method': 'old_dpo'}, path)
            adapter = FakeAdapter()
            optimizer = torch.optim.AdamW(adapter.model.parameters())
            with self.assertRaisesRegex(ValueError, 'not a feedback-guided checkpoint'):
                load_training_checkpoint(path, adapter, optimizer, 'cpu')

    def test_cli_defaults_and_alpha_validation(self):
        args = parse_args([])
        self.assertEqual(args.model, 'yolov8n')
        self.assertAlmostEqual(args.feedback_alpha, .10)
        with self.assertRaises(SystemExit):
            parse_args(['--feedback-alpha', '1.1'])


if __name__ == '__main__': unittest.main(verbosity=2)
