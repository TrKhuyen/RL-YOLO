import sys
import tempfile
import unittest
from pathlib import Path

import torch

KB2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KB2_DIR))

from train_feedback import (load_training_checkpoint, make_frozen_reference,
                            parse_args, save_training_checkpoint, DPO_PAIR_SOURCE,
                            DPO_OBJECTIVE_VERSION)


class FakeAdapter:
    def __init__(self): self.model = torch.nn.Linear(2, 1)
    def state_dict(self): return self.model.state_dict()
    def eval_mode(self): self.model.eval()


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

    def test_pairwise_checkpoint_records_method_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pairwise.pt'
            source = FakeAdapter()
            optimizer = torch.optim.AdamW(source.model.parameters())
            save_training_checkpoint(
                path, source, optimizer, 3, {'pairwise_alpha': .1},
                {'pairwise_pairs_seen': 12})
            data = torch.load(path, map_location='cpu', weights_only=False)
            self.assertEqual(data['training_method'],
                             'pairwise_preference_native_loss')
            self.assertEqual(data['training_state']['pairwise_pairs_seen'], 12)
            target = FakeAdapter()
            load_training_checkpoint(
                path, target,
                torch.optim.AdamW(target.model.parameters()), 'cpu')

    def test_rejects_wrong_checkpoint_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.pt'
            torch.save({'training_method': 'old_dpo'}, path)
            adapter = FakeAdapter()
            optimizer = torch.optim.AdamW(adapter.model.parameters())
            with self.assertRaisesRegex(ValueError, 'not a KB2 training checkpoint'):
                load_training_checkpoint(path, adapter, optimizer, 'cpu')

    def test_dpo_reference_is_independent_and_resume_checks_original_base(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = FakeAdapter()
            reference = make_frozen_reference(adapter)
            original = {k: v.clone() for k, v in reference.state_dict().items()}
            self.assertTrue(all(not p.requires_grad for p in reference.model.parameters()))
            self.assertFalse(reference.model.training)
            optimizer = torch.optim.AdamW(adapter.model.parameters(), lr=.01)
            optimizer.zero_grad()
            adapter.model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            for key, value in reference.state_dict().items():
                self.assertTrue(torch.equal(value, original[key]))
                self.assertNotEqual(value.data_ptr(), adapter.state_dict()[key].data_ptr())
            path = Path(directory) / 'dpo.pt'
            config = {'dpo_alpha': .1, 'dpo_beta': .1, 'base_checkpoint_sha256': 'base',
                      'dpo_pair_source': DPO_PAIR_SOURCE,
                      'dpo_objective_version': DPO_OBJECTIVE_VERSION}
            save_training_checkpoint(path, adapter, optimizer, 2, config, {'dpo_pairs_seen': 4})
            data = torch.load(path, weights_only=False)
            self.assertEqual(data['training_method'], 'candidate_selection_dpo_native_loss')
            step, _, state = load_training_checkpoint(path, adapter, optimizer, 'cpu', config)
            self.assertEqual(step, 2)
            self.assertEqual(state['dpo_pairs_seen'], 4)
            for changes in ({'base_checkpoint_sha256': 'different'}, {'dpo_beta': .2},
                            {'dpo_pair_source': 'frozen_reference_same_gt_v1'},
                            {'dpo_objective_version': 'v1'}):
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    load_training_checkpoint(path, adapter, optimizer, 'cpu', {**config, **changes})
            with self.assertRaisesRegex(ValueError, 'method'):
                load_training_checkpoint(path, adapter, optimizer, 'cpu', {'pairwise_alpha': .1})

    def test_legacy_dpo_cannot_resume_as_online_gt_dpo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pt'
            adapter = FakeAdapter()
            optimizer = torch.optim.AdamW(adapter.model.parameters())
            old = {'dpo_alpha': .1, 'dpo_pair_source': 'frozen_reference_same_gt_v1'}
            save_training_checkpoint(path, adapter, optimizer, 5, old)
            new = {**old, 'dpo_pair_source': DPO_PAIR_SOURCE,
                   'dpo_objective_version': DPO_OBJECTIVE_VERSION}
            with self.assertRaisesRegex(ValueError, 'dpo_pair_source'):
                load_training_checkpoint(path, adapter, optimizer, 'cpu', new)

    def test_dpo_cli_rejects_mixed_objectives(self):
        base = ['--dpo-alpha', '.1', '--feedback-alpha', '0', '--sampling-strategy', 'shuffle']
        self.assertEqual(parse_args(base).dpo_beta, .1)
        for extra in (['--pairwise-alpha', '.1'], ['--object-feedback-alpha', '.1'],
                      ['--dpo-beta', '0'], ['--dpo-alpha', 'nan']):
            with self.assertRaises(SystemExit):
                parse_args([*base, *extra])

    def test_cli_defaults_and_alpha_validation(self):
        args = parse_args([])
        self.assertEqual(args.model, 'yolov8n')
        self.assertAlmostEqual(args.feedback_alpha, .10)
        with self.assertRaises(SystemExit):
            parse_args(['--feedback-alpha', '1.1'])


if __name__ == '__main__': unittest.main(verbosity=2)
