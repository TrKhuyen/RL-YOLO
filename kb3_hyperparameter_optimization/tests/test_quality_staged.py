"""Search boundaries must preserve work, PPO state, and held-out separation."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from kb3_hyperparameter_optimization.config import SearchSpaceConfig
from kb3_hyperparameter_optimization.search_space import DiscreteSearchSpace
from kb3_hyperparameter_optimization.quality.detector import CanonicalSearchStopper, atomic_json, file_hash
from kb3_hyperparameter_optimization.quality.pipeline import Pipeline, parse_args, settings_from, budget
from kb3_hyperparameter_optimization.quality.ppo import QualityPPO
from kb3_hyperparameter_optimization.quality.state import STATE_NAMES, action_masks


class StagedSearchTests(unittest.TestCase):
    def setUp(self):
        self.space_config = SearchSpaceConfig()
        self.space = DiscreteSearchSpace(self.space_config)
        self.settings = settings_from(parse_args([]))
        self.calls = []

    def fake_train(self, **kwargs):
        output = Path(kwargs['output_dir'])
        self.calls.append(output.relative_to(output.parents[1]).as_posix())
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / 'selected_best.pt'
        checkpoint.write_bytes(str(kwargs['seed']).encode())
        actions = []
        agent = kwargs['agent']
        if agent is not None and kwargs['learn']:
            for step in range(3):
                state = (float(step) / 3,) * (len(STATE_NAMES) + len(self.space.names))
                action = agent.act(state, action_masks(self.space, self.space.initial_values()))
                actions.append(action)
                agent.observe(sum(action) / 10, step == 2)
        result = dict(seed=kwargs['seed'], failed=False, best_checkpoint=str(checkpoint),
                      checkpoint_sha256=file_hash(checkpoint), metrics={'map50_95': .1},
                      final_metrics={'epoch': 100}, actions=actions,
                      hyperparameters=kwargs['parameters'] or self.space.initial_values(), elapsed_seconds=1.)
        atomic_json(output / 'result.json', result)
        return result

    def pipeline(self, root, target=None):
        return Pipeline(SimpleNamespace(model='yolov8n', data_root='unused', search_target=target),
                        self.settings, self.space_config, root, root / 'data.yaml')

    def test_four_then_eight_matches_direct_eight_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('kb3_hyperparameter_optimization.quality.pipeline.train_detector', side_effect=self.fake_train):
            root = Path(directory)
            staged, direct = root / 'staged', root / 'direct'
            staged.mkdir()
            direct.mkdir()
            pipeline = self.pipeline(staged)
            self.assertEqual(pipeline.search(), 4)
            self.assertEqual(len(self.calls), 8)
            first = (staged / 'hpo/trial_0000/result.json').read_bytes()
            self.assertFalse((staged / 'hpo_selection.json').exists())
            self.assertFalse((staged / 'comparison.json').exists())
            self.assertEqual(pipeline.search(), 8)
            self.assertEqual(len(self.calls), 16)
            self.assertEqual((staged / 'hpo/trial_0000/result.json').read_bytes(), first)
            self.pipeline(direct, 8).search()
            staged_agent, staged_meta = QualityPPO.load(staged / 'ppo/latest.pt')
            direct_agent, direct_meta = QualityPPO.load(direct / 'ppo/latest.pt')
            self.assertEqual(staged_meta['next_episode'], 8)
            self.assertEqual(direct_meta['next_episode'], 8)
            self.assertTrue(torch.equal(staged_agent.rng.get_state(), direct_agent.rng.get_state()))
            for name, value in staged_agent.model.state_dict().items():
                self.assertTrue(torch.equal(value, direct_agent.model.state_dict()[name]), name)
            for branch, prefix in (('hpo', 'trial'), ('ppo', 'episode')):
                for i in range(8):
                    a = json.loads((staged / branch / f'{prefix}_{i:04d}/result.json').read_text())
                    b = json.loads((direct / branch / f'{prefix}_{i:04d}/result.json').read_text())
                    self.assertEqual(a['hyperparameters'], b['hyperparameters'])
                    self.assertEqual(a['actions'], b['actions'])
            count = len(self.calls)
            pipeline.search()
            self.assertEqual(len(self.calls), count)
            with self.assertRaisesRegex(ValueError, 'Cannot lower'):
                self.pipeline(staged, 4).search()

    def test_interruption_retries_original_batch_and_keeps_committed_policy(self):
        failed = False

        def interrupted(**kwargs):
            nonlocal failed
            if Path(kwargs['output_dir']).name == 'episode_0001' and not failed:
                failed = True
                raise RuntimeError('interrupted')
            return self.fake_train(**kwargs)

        with tempfile.TemporaryDirectory() as directory, \
             patch('kb3_hyperparameter_optimization.quality.pipeline.train_detector', side_effect=interrupted):
            root = Path(directory)
            pipeline = self.pipeline(root)
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                pipeline.search()
            self.assertEqual(json.loads((root / 'search_progress.json').read_text())['status'], 'running')
            self.assertEqual(QualityPPO.load(root / 'ppo/latest.pt')[1]['next_episode'], 1)
            self.assertEqual(pipeline.search(), 4)
            self.assertEqual(len(self.calls), 8)
            self.assertEqual(QualityPPO.load(root / 'ppo/latest.pt')[1]['next_episode'], 4)

    def test_selection_locks_search_before_held_out_seeds_and_matches_tuning_cost(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('kb3_hyperparameter_optimization.quality.pipeline.train_detector', side_effect=self.fake_train):
            root = Path(directory)
            pipeline = self.pipeline(root)
            pipeline.search()
            pipeline.select()
            self.assertEqual(json.loads((root / 'selection_lock.json').read_text())['target'], 4)
            self.assertEqual(len(self.calls), 12)  # 8 search + 2 HPO tuning + 2 policy tuning
            self.assertEqual(pipeline.search(), 4)
            with self.assertRaisesRegex(ValueError, 'Selection already started'):
                self.pipeline(root, 8).search()
            count = len(self.calls)
            pipeline.select()
            self.assertEqual(len(self.calls), count)

    def test_finalize_keeps_current_completed_batch_unless_extension_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('kb3_hyperparameter_optimization.quality.pipeline.train_detector', side_effect=self.fake_train):
            root = Path(directory)
            pipeline = self.pipeline(root)
            pipeline.search()
            pipeline.args.finalize = True
            self.assertEqual(pipeline.search(), 4)
            self.assertEqual(len(self.calls), 8)
            pipeline.args.search_target = 8
            self.assertEqual(pipeline.search(), 8)
            self.assertEqual(len(self.calls), 16)

    def test_final_retraining_disables_search_stopping_and_budget_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('kb3_hyperparameter_optimization.quality.pipeline.train_detector') as train:
            pipeline = self.pipeline(Path(directory))
            pipeline.detector('hpo/trial_0000', 42)
            self.assertEqual(train.call_args.kwargs['settings']['early_stopping']['patience'], 50)
            pipeline.detector('evaluation/default/seed_10001', 10001)
            self.assertIsNone(train.call_args.kwargs['settings']['early_stopping'])
            pipeline.detector('ppo/tuning_candidate_0004/seed_5001', 5001)
            self.assertIsNone(train.call_args.kwargs['settings']['early_stopping'])
        first, ceiling = budget(self.settings, 4), budget(self.settings)
        self.assertEqual(first['first_search_batch_detectors'], 8)
        self.assertEqual(first['maximum_all_epochs'], 7200)
        self.assertEqual(ceiling['maximum_all_epochs'], 10800)
        for target in ('3', '0'):
            with self.assertRaisesRegex(ValueError, '--search-target'):
                settings_from(parse_args(['--search-target', target]))
        settings_from(parse_args(['--search-target', '16']))
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            settings_from(parse_args(['--search-target', '5000']))

    def test_explicit_extension_beyond_eight_preserves_work_and_matched_tuning_budget(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('kb3_hyperparameter_optimization.quality.pipeline.train_detector', side_effect=self.fake_train):
            root = Path(directory)
            self.pipeline(root, 8).search()
            self.assertEqual(len(self.calls), 16)
            self.pipeline(root, 12).search()
            self.assertEqual(len(self.calls), 24)
            self.pipeline(root).search()  # no implicit growth beyond the automatic target
            self.assertEqual(len(self.calls), 24)
            self.pipeline(root).select()
            self.assertEqual(len(self.calls), 36)  # 3 candidates x 2 seeds for each branch
            self.assertEqual(budget(self.settings, 12)['hpo_tuning_detectors'], 6)
            self.assertEqual(budget(self.settings, 12)['ppo_tuning_detectors'], 6)

    def test_patience_uses_canonical_score_threshold_and_minimum_training(self):
        stop = CanonicalSearchStopper({'early_stopping': {'patience': 2, 'min_epochs': 4, 'min_delta': .01}})
        self.assertFalse(stop(1, .2))
        self.assertFalse(stop(2, .201))
        self.assertFalse(stop(3, .202))
        self.assertTrue(stop(4, .203))
        self.assertFalse(stop(5, .22))
        disabled = CanonicalSearchStopper({})
        self.assertFalse(disabled(300, .0))


if __name__ == '__main__':
    unittest.main()
