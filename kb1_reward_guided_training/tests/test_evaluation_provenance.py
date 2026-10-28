"""Evaluation must reject incomplete or mismatched experiments before inference."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation_provenance as ep


class EvaluationGuards(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / 'source.pt'
        self.source.write_bytes(b'source')
        self.best = self.root / 'best.pt'
        self.dataset = {'sha256': 'data', 'counts': {'test': 2}, 'source_id_policy': 'ids'}
        self.supervised = {'dataset': self.dataset, 'dp_flags': {'w3f': False, 'psa': False}}
        self.manifest = {'model': 'yolo26n', 'source_sha256': ep.sha256(self.source),
            'source_manifest': self.supervised, 'mode': 'kb1b', 'seed': 42,
            'protocol': {'conf': .001, 'nms_iou': .6, 'max_det': 300,
                         'operating_conf': .25, 'operating_iou': .5,
                         'image_size': 640, 'score': 'mAP50_95'}}
        self.metrics = {'mAP50': .5, 'mAP50_95': .4, 'AR300': .6}
        self.completed = {'status': 'complete', 'best_step': 500, 'best': self.metrics}
        for name, value in [('dataset_manifest', self.dataset), ('verify_supervised', self.supervised)]:
            patcher = patch.object(ep, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.write()

    def write(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))
        (self.root / 'completed.json').write_text(json.dumps(self.completed))
        torch.save({'metadata': self.manifest, 'step': 500, 'metrics': self.metrics,
                    'state_dict': {}}, self.best)

    def inspect(self, **kwargs):
        return ep.inspect_checkpoint('yolo26n', self.best, self.source, **kwargs)

    def test_accept_completed_and_report_real_seed(self):
        record, _ = self.inspect(expected_seed=42, expected_mode='kb1b')
        self.assertEqual(record['seed'], 42)
        self.assertEqual(record['checkpoint_sha256'], ep.sha256(self.best))

    def test_reject_missing_completion_and_wrong_source(self):
        (self.root / 'completed.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.inspect()
        self.manifest['source_sha256'] = 'wrong'
        self.write()
        with self.assertRaisesRegex(ValueError, 'source/model'):
            self.inspect()

    def test_reject_seed_mode_and_unselected_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.inspect(expected_seed=43)
        with self.assertRaisesRegex(ValueError, 'mode'):
            self.inspect(expected_mode='native_only')
        self.completed['best_step'] = 1000
        self.write()
        with self.assertRaisesRegex(ValueError, 'validation-selected'):
            self.inspect()

    def test_reject_completed_hash_and_changed_metadata(self):
        self.completed['best_sha256'] = 'wrong'
        self.write()
        with self.assertRaisesRegex(ValueError, 'checkpoint hash'):
            self.inspect()
        self.completed.pop('best_sha256')
        self.write()
        manifest = dict(self.manifest, seed=43)
        (self.root / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'metadata differs'):
            self.inspect()

    def test_final_selection_requires_fixed_budget_and_full_steps(self):
        self.manifest['config'] = {'fixed_budget': True}
        self.completed.update(termination='budget', steps_run=1000, last=self.metrics)
        self.write()
        final = self.root / 'last.pt'
        torch.save({'metadata': self.manifest, 'step': 1000, 'metrics': self.metrics,
                    'state_dict': {}}, final)
        self.completed['last_sha256'] = ep.sha256(final)
        (self.root / 'completed.json').write_text(json.dumps(self.completed))
        record, _ = ep.inspect_checkpoint('yolo26n', final, self.source, 42, 'kb1b')
        self.assertEqual(record['step'], 1000)
        self.completed['termination'] = 'early_stopping'
        (self.root / 'completed.json').write_text(json.dumps(self.completed))
        with self.assertRaisesRegex(ValueError, 'fixed-budget'):
            ep.inspect_checkpoint('yolo26n', final, self.source)

    def test_reject_nonfinite_metrics_and_changed_dataset(self):
        self.metrics['mAP50'] = float('nan')
        self.write()
        with self.assertRaisesRegex(ValueError, 'metric mismatch'):
            self.inspect()
        self.metrics['mAP50'] = .5
        self.manifest['source_manifest'] = {**self.supervised,
            'dataset': {**self.dataset, 'sha256': 'changed'}}
        self.write()
        with self.assertRaisesRegex(ValueError, 'Dataset changed'):
            self.inspect()


class EvaluationPublicationGuards(unittest.TestCase):
    def test_missing_input_does_not_create_output(self):
        from evaluate import run_comparison
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'new'
            with patch('evaluation_provenance.inspect_checkpoint', side_effect=FileNotFoundError('missing native run')):
                with self.assertRaises(FileNotFoundError):
                    run_comparison(model_filter='YOLO26n', output=output, device='cpu')
            self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self):
        from evaluate import run_comparison
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            sentinel = output / 'results_full.csv'
            sentinel.write_text('historical')
            with patch('evaluation_provenance.inspect_checkpoint', return_value=({}, {})):
                with self.assertRaises(FileExistsError):
                    run_comparison(model_filter='YOLO26n', output=output, device='cpu')
            self.assertEqual(sentinel.read_text(), 'historical')
