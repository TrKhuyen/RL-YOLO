"""A-only TPE must use completed objectives, preserve trials, and never invoke RL."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import optuna

from kb3_hyperparameter_optimization.config import SearchSpaceConfig
from kb3_hyperparameter_optimization.quality.detector import atomic_json, file_hash
from kb3_hyperparameter_optimization.quality.tpe_pipeline import (
    TPEExperiment, exclusive_run, parse_args, settings_from, validated_result)
from kb3_hyperparameter_optimization.quality.tpe_reporting import trajectory_diagnostics


class TPEPipelineTests(unittest.TestCase):
    def setUp(self):
        self.settings = settings_from(parse_args([]))
        self.space = SearchSpaceConfig()
        self.calls = []

    def experiment(self, root, target=4):
        return TPEExperiment(SimpleNamespace(model='yolov8n', data_root='unused', trials=target),
                             self.settings, self.space, root)

    def fake_train(self, **kwargs):
        self.assertIsNone(kwargs['agent'])
        self.assertFalse(kwargs['learn'])
        self.assertFalse(kwargs['random_schedule'])
        self.assertEqual(kwargs['settings']['early_stopping']['min_epochs'], 100)
        self.calls.append(kwargs)
        output = Path(kwargs['output_dir'])
        checkpoint = output / 'selected_best.pt'
        checkpoint.write_bytes(str(kwargs['parameters']).encode())
        score = .3 + kwargs['parameters']['lr0']
        metrics = dict(epoch=100, map50_95=score, map50=.5, ap_small=.04,
                       train_loss=3., val_loss=4.)
        atomic_json(output / 'epochs.json', [dict(metrics=metrics)])
        result = dict(failed=False, best_checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
                      seed=kwargs['seed'], hyperparameters=kwargs['parameters'], decisions=0,
                      initial_weights_sha256='same-initial-model', actual_epochs=100, elapsed_seconds=1.,
                      metrics=metrics, final_metrics=metrics, run_dir=str(output), stopped_early=True)
        atomic_json(output / 'result.json', result)
        return result

    def test_four_includes_exact_anchor_and_two_actual_tpe_proposals(self):
        original = optuna.samplers.TPESampler._sample
        calls = []

        def track(sampler, study, trial, *args, **kwargs):
            calls.append(trial.number)
            return original(sampler, study, trial, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, \
             patch('kb3_hyperparameter_optimization.quality.tpe_pipeline.train_detector', side_effect=self.fake_train), \
             patch.object(optuna.samplers.TPESampler, '_sample', track):
            experiment = self.experiment(Path(directory))
            records = experiment.search()
            self.assertEqual(len(self.calls), 4)
            self.assertEqual(set(calls), {2, 3})
            self.assertEqual(records[0]['result']['hyperparameters'], experiment.initial)
            self.assertEqual([r['proposal']['mode'] for r in records], ['kb1_anchor', 'startup_random', 'tpe', 'tpe'])
            for record in records:
                hp = record['result']['hyperparameters']
                self.assertEqual(record['result']['seed'], 42)
                self.assertEqual(hp['momentum'], .937)
                self.assertEqual(hp['augmentation_strength'], .5)
                self.assertTrue(.005 <= hp['lr0'] <= .015)
                self.assertTrue(.00025 <= hp['weight_decay'] <= .001)

    def test_resume_two_to_four_matches_direct_four_and_never_retrains_complete(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('kb3_hyperparameter_optimization.quality.tpe_pipeline.train_detector', side_effect=self.fake_train):
            root = Path(directory)
            staged, direct = root / 'staged', root / 'direct'
            staged.mkdir()
            direct.mkdir()
            self.experiment(staged, 2).search()
            first = (staged / 'tpe/trial_0000/result.json').read_bytes()
            resumed = self.experiment(staged).search()
            self.assertEqual(len(self.calls), 4)
            self.assertEqual((staged / 'tpe/trial_0000/result.json').read_bytes(), first)
            complete = self.experiment(direct).search()
            self.assertEqual([r['proposal'] for r in resumed], [r['proposal'] for r in complete])
            self.experiment(staged).search()
            self.assertEqual(len(self.calls), 8)
            with self.assertRaisesRegex(ValueError, 'Cannot lower'):
                self.experiment(staged, 2).search()

    def test_tpe_proposal_responds_to_objective_history(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = self.experiment(Path(directory))
            # Sufficient observations to form different good/bad density groups.
            history = [dict(result=dict(hyperparameters={'lr0': .005 + i * .0005,
                       'weight_decay': .00025 + i * .000025}, metrics={'map50_95': i / 30})) for i in range(20)]
            forward = experiment.propose(20, history)
            for i, record in enumerate(history):
                record['result']['metrics']['map50_95'] = (19 - i) / 30
            reverse = experiment.propose(20, history)
            self.assertNotEqual(forward['parameters'], reverse['parameters'])

    def test_interrupted_trial_preserves_proposal_and_archives_only_partial_work(self):
        failed = False

        def interrupted(**kwargs):
            nonlocal failed
            output = Path(kwargs['output_dir'])
            if output.name == 'trial_0001' and not failed:
                failed = True
                (output / 'epochs.json').write_text('[]')
                raise RuntimeError('power interruption')
            return self.fake_train(**kwargs)

        with tempfile.TemporaryDirectory() as directory, \
             patch('kb3_hyperparameter_optimization.quality.tpe_pipeline.train_detector', side_effect=interrupted):
            root = Path(directory)
            experiment = self.experiment(root)
            with self.assertRaisesRegex(RuntimeError, 'power interruption'):
                experiment.search()
            spec = (root / 'tpe/trial_0001/trial_spec.json').read_bytes()
            experiment.search()
            self.assertEqual((root / 'tpe/trial_0001/trial_spec.json').read_bytes(), spec)
            archives = list((root / 'tpe').glob('trial_0001.interrupted_*'))
            self.assertEqual(len(archives), 1)
            # An archived result must not be interpreted as an extra logical trial.
            (archives[0] / 'result.json').write_text('{}')
            experiment.search()
            self.assertEqual(len(self.calls), 4)

    def test_corrupt_checkpoint_and_changed_proposal_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('kb3_hyperparameter_optimization.quality.tpe_pipeline.train_detector', side_effect=self.fake_train):
            root = Path(directory)
            experiment = self.experiment(root, 1)
            record = experiment.search()[0]
            checkpoint = Path(record['result']['best_checkpoint'])
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'Invalid completed'):
                experiment.search()
            path = root / 'tpe/trial_0000/trial_spec.json'
            spec = json.loads(path.read_text())
            spec['seed'] += 1
            atomic_json(path, spec)
            with self.assertRaisesRegex(ValueError, 'Saved TPE proposal'):
                experiment.search()

    def test_evaluation_uses_existing_checkpoints_only_and_preserves_versioned_reports(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('kb3_hyperparameter_optimization.quality.tpe_pipeline.train_detector', side_effect=self.fake_train), \
             patch('kb3_hyperparameter_optimization.quality.tpe_pipeline.evaluate_reference', return_value=None):
            root = Path(directory)
            two = self.experiment(root, 2)
            two.search()
            two.evaluate()
            original = (root / 'reports/trials_0002/evaluation.json').read_bytes()
            four = self.experiment(root)
            four.search()
            report = four.evaluate()
            self.assertEqual(len(self.calls), 4)
            self.assertFalse(report['extra_training_performed'])
            self.assertFalse(report['test_evaluated'])
            self.assertEqual(report['independent_detector_seeds'], 1)
            self.assertEqual((root / 'reports/trials_0002/evaluation.json').read_bytes(), original)
            repeated = (root / 'evaluation.json').read_bytes()
            four.evaluate()
            self.assertEqual((root / 'evaluation.json').read_bytes(), repeated)

    def test_same_output_cannot_be_locked_by_two_process_handles(self):
        with tempfile.TemporaryDirectory() as directory:
            with exclusive_run(directory):
                with self.assertRaises(OSError):
                    with exclusive_run(directory):
                        self.fail('Second lock was incorrectly acquired')
            with exclusive_run(directory):
                pass

    def test_invalid_budget_and_old_ab_config_rejected(self):
        with self.assertRaisesRegex(ValueError, '--trials'):
            settings_from(parse_args(['--trials', '0']))
        with self.assertRaisesRegex(ValueError, 'A-only TPE'):
            settings_from(parse_args(['--config', 'kb3_hyperparameter_optimization/configs/kb3_quality.yaml']))

    def test_curve_decline_is_diagnostic_and_exact_five_epoch_window_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / 'epochs.json', [dict(metrics=dict(epoch=i, map50_95=.3, val_loss=10-i))
                                               for i in range(2, 7)])
            record = dict(result=dict(run_dir=str(root), metrics={'epoch': 1, 'map50_95': .4},
                                      final_metrics={'map50_95': .3}, actual_epochs=6))
            diagnostic = trajectory_diagnostics(record)
            self.assertEqual(diagnostic['first_sustained_map_decline_epoch'], 2)
            self.assertEqual(diagnostic['minimum_val_loss_epoch'], 6)
            self.assertIn('not proof', diagnostic['interpretation'])


if __name__ == '__main__':
    unittest.main()
