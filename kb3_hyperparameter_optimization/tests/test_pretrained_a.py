"""Verify KB1 YAML inheritance and actual transfer without a research run."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from kb3_hyperparameter_optimization.config import SearchSpaceConfig
from kb3_hyperparameter_optimization.search_space import DiscreteSearchSpace
from kb3_hyperparameter_optimization.quality.detector import file_hash
from kb3_hyperparameter_optimization.quality.pretrained_detector import train_detector, training_overrides, SELECTION
from kb3_hyperparameter_optimization.quality.pretrained_reference import resolve_pretrained_recipe
from kb3_hyperparameter_optimization.quality.tpe_pipeline import parse_args, settings_from, validated_result


class PretrainedContractTests(unittest.TestCase):
    def test_settings_inherit_actual_kb1_initialization_budget_seed_and_native_validation(self):
        settings = settings_from(parse_args([]))
        self.assertEqual(settings['initialization'], 'pretrained')
        self.assertEqual((settings['epochs'], settings['patience'], settings['seed']), (200, 30, 0))
        self.assertEqual(settings['pretrained_source']['checkpoint'], str(Path('yolov8n.pt').resolve()))
        self.assertEqual(settings['pretrained_source']['checkpoint_sha256'], file_hash('yolov8n.pt'))
        override = training_overrides(settings, DiscreteSearchSpace(SearchSpaceConfig()).initial_values())
        self.assertTrue(override['pretrained'])
        self.assertFalse(override['resume'])
        self.assertEqual(override['batch'], 16)
        self.assertEqual(override['warmup_epochs'], 3)
        self.assertEqual(override['iou'], .7)
        self.assertEqual(override['mixup'], .1)
        self.assertEqual(override['nbs'], 64)

    def test_changing_inherited_epoch_or_seed_is_rejected(self):
        config = yaml.safe_load(Path(parse_args([]).config).read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            for key, value in (('seed', 42), ('epochs', 300), ('patience', 50)):
                path.write_text(yaml.safe_dump({**config, key: value}))
                with self.assertRaisesRegex(ValueError, 'must inherit KB1 YAML'):
                    settings_from(parse_args(['--config', str(path)]))

    def test_scratch_results_and_kb1_best_initialization_are_rejected(self):
        settings = settings_from(parse_args([]))
        params = DiscreteSearchSpace(SearchSpaceConfig()).initial_values()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'best.pt'
            checkpoint.write_bytes(b'old scratch checkpoint')
            result = dict(best_checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
                          seed=0, hyperparameters=params, decisions=0, initialization='scratch')
            path = root / 'result.json'
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(RuntimeError, 'scratch initialization'):
                validated_result(path, parameters=params, seed=0)
            with self.assertRaisesRegex(ValueError, 'exact original pretrained'):
                train_detector(settings=settings, data_yaml='unused', data_root='unused',
                               model=checkpoint, seed=0, output_dir=root / 'output',
                               space_config=SearchSpaceConfig(), parameters=params)


@unittest.skipUnless(os.environ.get('KB3_TEST_PRETRAINED') == '1', 'optional tiny native pretrained integration')
class PretrainedTrainerTests(unittest.TestCase):
    def dataset(self, root):
        data = root / 'data'
        for split in ('train', 'valid', 'test'):
            (data / split / 'images').mkdir(parents=True)
            (data / split / 'labels').mkdir()
            for i in range(2):
                Image.new('RGB', (64, 64), (110 + i * 20, 130, 70)).save(data / split / 'images' / f'{i}.jpg')
                (data / split / 'labels' / f'{i}.txt').write_text('0 .5 .5 .2 .2\n')
        path = root / 'dataset.yaml'
        path.write_text(yaml.safe_dump(dict(path=data.as_posix(), train='train/images', val='valid/images',
                                           test='test/images', nc=28, names=[f'class_{i}' for i in range(28)])))
        return data, path

    def settings(self):
        settings = settings_from(parse_args(['--smoke']))
        settings.update(device=os.environ.get('KB3_TEST_DEVICE', 'cpu'), batch_size=2, img_size=64)
        settings['trainer_recipe'].update(nbs=2, close_mosaic=1)
        return settings

    def test_pretrained_backbone_is_loaded_trial_initialization_is_identical_and_sources_are_read_only(self):
        settings = self.settings()
        params = DiscreteSearchSpace(SearchSpaceConfig()).initial_values()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, yaml_path = self.dataset(root)
            external = root / 'kb1_best.pt'
            external.write_bytes(b'comparison checkpoint must not be loaded for training')
            reference_hash = file_hash(external)
            settings['kb1_reference'] = dict(checkpoint=str(external), checkpoint_sha256=reference_hash)
            before = file_hash(settings['pretrained_source']['checkpoint'])
            common = dict(settings=settings, data_yaml=yaml_path, data_root=data,
                          model=settings['pretrained_source']['checkpoint'], seed=settings['seed'],
                          space_config=SearchSpaceConfig())
            first = train_detector(**common, parameters=params, output_dir=root / 'anchor')
            second = train_detector(**common, parameters={**params, 'lr0': .008, 'weight_decay': .0007},
                                    output_dir=root / 'changed')
            self.assertEqual(first['initial_weights_sha256'], second['initial_weights_sha256'])
            self.assertTrue(first['pretrained_backbone_verified'])
            self.assertEqual(first['initialization'], 'pretrained')
            self.assertEqual(first['checkpoint_selection'], SELECTION)
            self.assertEqual(first['actual_epochs'], 3)
            self.assertEqual(second['actual_epochs'], 3)
            initial = json.loads((root / 'anchor/initial.json').read_text())
            self.assertGreater(initial['transferred_backbone_tensors'], 100)
            self.assertTrue(initial['training_overrides']['pretrained'])
            self.assertEqual(file_hash(external), reference_hash)
            self.assertEqual(file_hash(settings['pretrained_source']['checkpoint']), before)
            selected = validated_result(root / 'changed/result.json', parameters=second['hyperparameters'], seed=settings['seed'])
            self.assertEqual(selected['metrics']['epoch'], second['native_best_metrics']['epoch'])

    def test_native_early_stop_uses_patience_without_extra_final_eval_epoch(self):
        settings = self.settings()
        settings.update(epochs=8, patience=2)
        from ultralytics.engine.trainer import BaseTrainer

        def synthetic_validate(trainer):
            fitness = (.1, .2, .19, .18, .17, .16, .15, .14)[trainer.epoch]
            if trainer.best_fitness is None or fitness > trainer.best_fitness:
                trainer.best_fitness = fitness
            return {'metrics/precision(B)': .5, 'metrics/recall(B)': .5,
                    'metrics/mAP50(B)': .5, 'metrics/mAP50-95(B)': fitness,
                    'val/box_loss': 1., 'val/cls_loss': 1., 'val/dfl_loss': 1.}, fitness

        with tempfile.TemporaryDirectory() as directory, patch.object(BaseTrainer, 'validate', synthetic_validate):
            root = Path(directory)
            data, yaml_path = self.dataset(root)
            result = train_detector(settings=settings, data_yaml=yaml_path, data_root=data,
                        model=settings['pretrained_source']['checkpoint'], seed=settings['seed'],
                        output_dir=root / 'early', space_config=SearchSpaceConfig(),
                        parameters=DiscreteSearchSpace(SearchSpaceConfig()).initial_values())
            self.assertEqual(result['actual_epochs'], 4)
            self.assertTrue(result['stopped_early'])
            self.assertEqual(result['native_best_metrics']['epoch'], 2)
            self.assertEqual(len(json.loads((root / 'early/epochs.json').read_text())), 4)


if __name__ == '__main__':
    unittest.main()
