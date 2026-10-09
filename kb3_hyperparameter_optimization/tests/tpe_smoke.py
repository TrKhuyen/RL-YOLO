"""Four tiny real CPU trainings, staged resume, and inference-only reporting."""

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from PIL import Image
import yaml

from kb3_hyperparameter_optimization.quality.tpe_pipeline import main


def run():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = root / 'data'
        for split in ('train', 'valid', 'test'):
            (dataset / split / 'images').mkdir(parents=True)
            (dataset / split / 'labels').mkdir()
            for i in range(2):
                Image.new('RGB', (64, 64), (100 + i * 20, 100, 60)).save(dataset / split / 'images' / f'{i}.jpg')
                (dataset / split / 'labels' / f'{i}.txt').write_text('0 .5 .5 .3 .3\n')
        data_yaml = root / 'dataset.yaml'
        data_yaml.write_text(yaml.safe_dump(dict(nc=28, names=[f'class_{i}' for i in range(28)])))
        config = yaml.safe_load(Path('kb3_hyperparameter_optimization/configs/kb3_tpe.yaml').read_text())
        config.update(device='cpu', batch_size=2, img_size=64)
        config_yaml = root / 'tpe.yaml'
        config_yaml.write_text(yaml.safe_dump(config))
        output = root / 'results'
        common = ['--config', str(config_yaml), '--data-root', str(dataset), '--data-config', str(data_yaml),
                  '--output-dir', str(output), '--smoke', '--stage', 'all']
        main([*common, '--trials', '2'])
        first = (output / 'tpe/trial_0000/result.json').read_bytes()
        main([*common, '--trials', '4', '--resume'])
        assert (output / 'tpe/trial_0000/result.json').read_bytes() == first
        report = json.loads((output / 'evaluation.json').read_text())
        assert report['actual_epochs'] == 12 and report['completed_trials'] == 4
        assert report['rl_training_runs'] == 0 and not report['test_evaluated']
        assert report['kb1_pretrained_reference'] is None
        assert not (output / 'ppo').exists()
        assert len({r['initial_weights_sha256'] for r in
                   [json.loads(p.read_text()) for p in (output / 'tpe').glob('trial_*/result.json')]}) == 1
        with patch('kb3_hyperparameter_optimization.quality.tpe_pipeline.train_detector',
                   side_effect=AssertionError('Resume/evaluate must not train again')):
            main([*common, '--trials', '4', '--resume'])
            main([*common, '--trials', '4', '--resume', '--stage', 'evaluate'])
        snapshot = (output / 'starting_train.yaml').read_bytes()
        tampered = yaml.safe_load(snapshot)
        tampered['pretrained'] = True
        (output / 'starting_train.yaml').write_text(yaml.safe_dump(tampered))
        try:
            main([*common, '--resume', '--check-only'])
        except ValueError as error:
            assert 'starting_train.yaml changed' in str(error)
        else:
            raise AssertionError('Tampered initialization accepted')
        print('A-TPE REAL CPU SMOKE PASS: four trials/12 epochs, identical initial weights, two-to-four resume, no RL/test/extra training.')


if __name__ == '__main__':
    run()
