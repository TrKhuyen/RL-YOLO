import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from kb2_preference_optimization.hybrid.source import resolve_tpe, assert_provenance
from kb2_preference_optimization.hybrid.run import baseline_signature


class HybridSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='kb2-hybrid-source-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        trial = self.root / 'tpe/trial_0001'
        trial.mkdir(parents=True)
        self.checkpoint = trial / 'selected_best.pt'
        self.checkpoint.write_bytes(b'immutable selected checkpoint fixture')
        digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        parameters = {'lr0': .005, 'weight_decay': .001}
        rows = [{'trial': i, 'scores': {'mAP50_95': .3 + i*.01}, 'parameters': parameters,
                 'seed': 0, 'checkpoint_sha256': digest} for i in range(2)]
        self.evaluation = {'protocol': 'kb3_tpe_v2_kb1_pretrained', 'metrics_split': 'validation',
                           'requested_trials': 2, 'completed_trials': 2, 'best_trial': 1,
                           'best_checkpoint_sha256': digest,
                           'best_checkpoint': r'D:\old\host\selected_best.pt', 'trials': rows}
        manifest = {'protocol': self.evaluation['protocol'], 'model': 'yolov8n',
                    'settings': {'kb1_reference': {'dataset_sha256': 'dataset'}}}
        result = {'failed': False, 'initialization': 'pretrained', 'pretrained_backbone_verified': True,
                  'hyperparameters': parameters, 'checkpoint_sha256': digest, 'seed': 0,
                  'canonical_best_scores': rows[1]['scores'], 'checkpoint_selection': 'native_best'}
        self.write('evaluation.json', self.evaluation)
        self.write('manifest.json', manifest)
        self.write('tpe/trial_0001/result.json', result)
        self.write('tpe/trial_0001/trial_spec.json', {'number': 1, 'parameters': parameters, 'seed': 0})

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def test_relocated_windows_path_uses_verified_local_selected_weights(self):
        source = resolve_tpe(self.root)
        self.assertEqual(Path(source['checkpoint']), self.checkpoint)
        self.assertEqual(source['trial'], 1)

    def test_checkpoint_byte_change_is_rejected(self):
        self.checkpoint.write_bytes(b'different weights')
        with self.assertRaisesRegex(ValueError, 'hash disagreement'):
            resolve_tpe(self.root)

    def test_incomplete_search_is_rejected(self):
        self.evaluation['requested_trials'] = 3
        self.write('evaluation.json', self.evaluation)
        with self.assertRaisesRegex(ValueError, 'complete'):
            resolve_tpe(self.root)

    def test_nonwinning_record_is_rejected(self):
        self.evaluation['trials'][0]['scores']['mAP50_95'] = .9
        self.write('evaluation.json', self.evaluation)
        with self.assertRaisesRegex(ValueError, 'maximum'):
            resolve_tpe(self.root)

    def test_S0_bank_cannot_be_relabelled_as_TPE_bank(self):
        old = {'model': 'yolov8n', 'source_sha256': 'S0', 'objective_version': 'v0',
               'transform_version': 'transform', 'code_sha256': {}, 'dataset': {'sha256': 'dataset'},
               'runtime': {'versions': {}, 'ultralytics_code_sha256': {}}}
        current = copy.deepcopy(old)
        current['source_sha256'] = 'TPE'
        current['source_kind'] = 'kb3_tpe_winner'
        with self.assertRaisesRegex(ValueError, 'source_sha256'):
            assert_provenance(old, current)

    def test_reference_metadata_changes_are_rejected_even_with_same_checkpoint(self):
        current = {'model': 'yolov8n', 'source_sha256': 'TPE', 'objective_version': 'v0',
                   'transform_version': 'transform', 'code_sha256': {}, 'dataset': {'sha256': 'dataset'},
                   'runtime': {'versions': {}, 'ultralytics_code_sha256': {}}, 'source_tpe': {'trial': 1}}
        recorded = copy.deepcopy(current)
        recorded['source_tpe']['trial'] = 2
        with self.assertRaisesRegex(ValueError, 'source_tpe'):
            assert_provenance(recorded, current)


class HybridUpdateParityTests(unittest.TestCase):
    def test_baseline_comparison_ignores_elapsed_time_but_detects_metric_changes(self):
        first = {'mAP50_95': .38, 'fp': 10, 'evaluation_wall_seconds': 1.}
        second = {**first, 'evaluation_wall_seconds': 2.}
        self.assertEqual(baseline_signature(first), baseline_signature(second))
        second['fp'] = 11
        self.assertNotEqual(baseline_signature(first), baseline_signature(second))

    def test_hybrid_update_loop_preserves_verified_strength_updates(self):
        root = Path(__file__).resolve().parents[1]
        def update_loop(path):
            tree = ast.parse(path.read_text())
            trainer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
            loop = next(n for n in trainer.body if isinstance(n, ast.For)
                        and isinstance(n.target, ast.Name) and n.target.id == 'step')
            return ast.dump(loop, include_attributes=False)
        self.assertEqual(update_loop(root / 'conditional_strength/train.py'),
                         update_loop(root / 'hybrid/train.py'))


if __name__ == '__main__':
    unittest.main()
