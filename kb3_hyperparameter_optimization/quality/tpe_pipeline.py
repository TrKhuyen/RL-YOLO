"""KB3-A only: four local TPE trials, a KB1 reference, and validation diagnostics."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import optuna
import yaml

from ..run_all import MODELS, ROOT, _preflight
from ..search_space import DiscreteSearchSpace
from .detector import atomic_json, file_hash, train_detector, training_overrides
from .reference import read_recipe, prepare_reference, evaluate_reference
from .tpe_reporting import write_evaluation


PROTOCOL = 'kb3_tpe_v1_kb1_start'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='kb3_hyperparameter_optimization/configs/kb3_tpe.yaml')
    parser.add_argument('--space-config', default='kb3_hyperparameter_optimization/configs/kb3_default.yaml')
    parser.add_argument('--model', choices=('all', *MODELS), default='yolov8n')
    parser.add_argument('--stage', choices=('baseline', 'search', 'evaluate', 'all'), default='all')
    parser.add_argument('--trials', type=int, help='Cumulative detector trials; default four, not four extra on resume')
    parser.add_argument('--data-root', default='pre-data/data/v2i_cleanned')
    parser.add_argument('--data-config', default='kb1_reward_guided_training/configs/pest.yaml')
    parser.add_argument('--output-dir')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Synthetic integration only; not a research result')
    return parser.parse_args(argv)


def settings_from(args):
    settings = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if (settings.get('protocol') != PROTOCOL or settings.get('method') != 'tpe'
            or settings.get('rl_enabled') is not False):
        raise ValueError('This entry point requires the A-only TPE protocol, with rl_enabled=false')
    if settings.get('recipe_source') != 'kb1_supervised' or settings.get('initialization') != 'scratch':
        raise ValueError('Use the actual KB1 recipe, with scratch detector initialization')
    raw, recipe, _ = read_recipe(args.model)
    settings['trainer_recipe'] = recipe
    settings['smoke'] = args.smoke
    for target, source in (('warmup_epochs', 'warmup_epochs'), ('lrf', 'lrf'),
                           ('batch_size', 'batch'), ('img_size', 'imgsz')):
        if settings.get(target) is None:
            settings[target] = raw[source]
    if args.smoke:
        settings.update(epochs=3, segment_epochs=1, warmup_epochs=0,
                        search_patience=0, search_min_epochs=0, search_min_delta=0.0)
        settings['trainer_recipe'] = {**recipe, 'nbs': settings['batch_size'], 'close_mosaic': 1}
    for name in ('trials', 'epochs', 'segment_epochs', 'batch_size', 'img_size',
                 'n_startup_trials', 'n_ei_candidates'):
        if not isinstance(settings.get(name), int) or settings[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if args.trials is not None and args.trials <= 0:
        raise ValueError('--trials must be positive')
    for name in ('seed', 'sampler_seed', 'search_patience', 'search_min_epochs'):
        if not isinstance(settings.get(name), int) or settings[name] < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if not 0 <= settings['search_min_epochs'] <= settings['epochs']:
        raise ValueError('search_min_epochs must not exceed the maximum horizon')
    if not 0 <= settings['warmup_epochs'] <= settings['segment_epochs'] <= settings['epochs']:
        raise ValueError('Warmup must complete within the shared stopping segment')
    if not math.isfinite(settings['search_min_delta']) or settings['search_min_delta'] < 0:
        raise ValueError('search_min_delta must be finite and nonnegative')
    if settings['workers'] != 0 or not 0 < settings['lrf'] <= 1:
        raise ValueError('Use workers=0 and lrf in (0,1]')
    settings['device'] = str(settings['device'])
    if settings['device'] != 'cpu' and not settings['device'].isdigit():
        raise ValueError('device must be cpu or one GPU index')
    if set(settings.get('local_factors', {})) != {'lr0', 'weight_decay'}:
        raise ValueError('The initial small experiment searches only lr0 and weight_decay')
    for name, factors in settings['local_factors'].items():
        if (not isinstance(factors, list) or len(factors) != 2
                or not all(math.isfinite(x) for x in factors) or not 0 < factors[0] < 1 < factors[1]):
            raise ValueError(f'{name}: local factors must satisfy 0 < lower < 1 < upper')
    return settings


def local_distributions(settings, space):
    distributions = {}
    for name, (lower, upper) in settings['local_factors'].items():
        spec = space.parameters[name]
        low, high = max(spec.minimum, spec.initial * lower), min(spec.maximum, spec.initial * upper)
        if not 0 < low <= spec.initial <= high or low == high:
            raise ValueError(f'{name}: the local log search must contain the actual KB1 starting value')
        distributions[name] = optuna.distributions.FloatDistribution(low, high, log=True)
    return distributions


def fingerprint(args, settings, space):
    sources = []
    for package in ('kb3_hyperparameter_optimization', 'kb1_reward_guided_training'):
        for path in sorted((ROOT / package).rglob('*.py')):
            if 'tests' not in path.parts and 'yolov5' not in path.parts:
                sources.append((str(path.relative_to(ROOT)), file_hash(path)))
    root = Path(args.data_root).resolve()
    dataset = []
    for split in ('train', 'valid', 'test'):
        for path in sorted((root / split).rglob('*')):
            if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.txt'}:
                dataset.append((str(path.relative_to(root)), file_hash(path)))
    dependencies = {name: version(name) for name in ('optuna', 'torch', 'torchvision', 'ultralytics',
                    'torchmetrics', 'faster-coco-eval', 'pycocotools', 'numpy', 'albumentations')}
    identity = dict(protocol=PROTOCOL, settings=settings, space=asdict(space), model=args.model,
                    architecture=MODELS[args.model], sources=sources, dataset=dataset, dependencies=dependencies,
                    data_config_sha256=file_hash(args.data_config),
                    objective='canonical_validation_mAP50_95_best_trained_epoch',
                    detector_seed_rule='same seed for every trial',
                    sampler_seed_rule='sampler_seed + logical trial number, conditioned on completed history',
                    test_used_for_selection=False)
    identity['sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


@contextmanager
def exclusive_run(output):
    """An OS lock prevents two resume processes from training the same trial."""
    handle = (Path(output) / '.run.lock').open('a+b')
    try:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if sys.platform == 'win32':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        handle.close()


def validated_result(path, *, parameters, seed):
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    checkpoint = Path(result['best_checkpoint'])
    if (result.get('failed') or not checkpoint.is_file()
            or file_hash(checkpoint) != result['checkpoint_sha256']):
        raise RuntimeError(f'Invalid completed result or checkpoint: {path}')
    if result['hyperparameters'] != parameters or result['seed'] != seed or result['decisions'] != 0:
        raise RuntimeError('Completed detector does not match the fixed TPE trial contract')
    score = result['metrics']['map50_95']
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise RuntimeError('Invalid canonical validation objective')
    return result


class TPEExperiment:
    def __init__(self, args, settings, space, output):
        self.args, self.settings, self.space, self.output = args, settings, space, Path(output)
        self.initial = DiscreteSearchSpace(space).initial_values()
        self.distributions = local_distributions(settings, space)
        self.target = args.trials if args.trials is not None else settings['trials']

    def records(self, target=None):
        records = []
        for number in range(self.target if target is None else target):
            directory = self.output / 'tpe' / f'trial_{number:04d}'
            spec = json.loads((directory / 'trial_spec.json').read_text(encoding='utf-8'))
            result = validated_result(directory / 'result.json', parameters=spec['parameters'], seed=self.settings['seed'])
            records.append(dict(number=number, proposal=spec, result=result))
        if len({r['result']['initial_weights_sha256'] for r in records}) != 1:
            raise RuntimeError('Trial initial weights differ despite the common detector seed')
        return records

    def propose(self, number, history):
        sampler_seed = self.settings['sampler_seed'] + number
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(
            seed=sampler_seed, n_startup_trials=self.settings['n_startup_trials'],
            n_ei_candidates=self.settings['n_ei_candidates']))
        for record in history:
            params = {name: record['result']['hyperparameters'][name] for name in self.distributions}
            study.add_trial(optuna.trial.create_trial(params=params, distributions=self.distributions,
                                                      value=record['result']['metrics']['map50_95']))
        if number == 0:
            study.enqueue_trial({name: self.initial[name] for name in self.distributions})
        trial = study.ask()
        if trial.number != number:
            raise RuntimeError('Logical trial history is not contiguous')
        parameters = {**self.initial, **{name: trial.suggest_float(name, d.low, d.high, log=d.log)
                                       for name, d in self.distributions.items()}}
        return dict(number=number, sampler_seed=sampler_seed, seed=self.settings['seed'], parameters=parameters,
                    mode='kb1_anchor' if number == 0 else 'startup_random' if number < self.settings['n_startup_trials'] else 'tpe',
                    history_objectives=[r['result']['metrics']['map50_95'] for r in history])

    def search(self):
        history = []
        directories = [p for p in (self.output / 'tpe').glob('trial_*/result.json')
                       if re.fullmatch(r'trial_\d+', p.parent.name)]
        if any(int(p.parent.name.split('_')[1]) >= self.target for p in directories):
            raise ValueError('Cannot lower the cumulative trial budget below completed work')
        for number in range(self.target):
            directory = self.output / 'tpe' / f'trial_{number:04d}'
            directory.mkdir(parents=True, exist_ok=True)
            spec_path = directory / 'trial_spec.json'
            expected = self.propose(number, history)
            if spec_path.exists():
                spec = json.loads(spec_path.read_text(encoding='utf-8'))
                if spec != expected:
                    raise ValueError('Saved TPE proposal differs from the reproducible completed history')
            else:
                spec = expected
                atomic_json(spec_path, spec)
            result_path = directory / 'result.json'
            if result_path.exists():
                print(f'SKIP verified completed TPE trial {number}', flush=True)
            else:
                if any(p.name not in {'trial_spec.json'} for p in directory.iterdir()):
                    archived = directory.with_name(directory.name + '.interrupted_' +
                        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
                    if not archived.resolve().is_relative_to(self.output.resolve()):
                        raise ValueError('Archive must remain inside this experiment')
                    directory.rename(archived)
                    directory.mkdir()
                    atomic_json(spec_path, spec)
                atomic_json(self.output / 'status.json', dict(protocol=PROTOCOL, stage='search', status='running',
                            active_trial=number, completed_trials=len(history), requested_trials=self.target,
                            parameters=spec['parameters'], epoch_log=str(directory / 'epochs.json')))
                settings = {**self.settings, 'early_stopping': dict(patience=self.settings['search_patience'],
                            min_epochs=self.settings['search_min_epochs'], min_delta=self.settings['search_min_delta'])}
                train_detector(settings=settings, data_yaml=self.output / 'dataset.yaml', data_root=self.args.data_root,
                               model=MODELS[self.args.model], seed=self.settings['seed'], output_dir=directory,
                               space_config=self.space, parameters=spec['parameters'],
                               agent=None, learn=False, random_schedule=False)
            result = validated_result(result_path, parameters=spec['parameters'], seed=self.settings['seed'])
            if history and result['initial_weights_sha256'] != history[0]['result']['initial_weights_sha256']:
                raise RuntimeError('Initial detector weights changed between fixed-seed trials')
            history.append(dict(number=number, proposal=spec, result=result))
            atomic_json(self.output / 'trial_history.json', dict(protocol=PROTOCOL, trials=history,
                        completed_trials=len(history), requested_trials=self.target, rl_training_runs=0,
                        actual_epochs=sum(r['result']['actual_epochs'] for r in history),
                        detector_hours=sum(r['result']['elapsed_seconds'] for r in history) / 3600,
                        test_used=False))
        return history

    def evaluate(self):
        records = self.records()
        reference = evaluate_reference(self.settings, self.args.data_root, self.output)
        return write_evaluation(self.output, records, reference, self.settings, protocol=PROTOCOL)


def run_models(args):
    if args.resume and not args.output_dir:
        raise ValueError('--resume requires the original output directory')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = Path(args.output_dir or ROOT / 'kb3_hyperparameter_optimization' /
                'checkpoint_hyperparameter_optimization' / 'quality' / f'tpe_all_{stamp}').resolve()
    commands = []
    for model in MODELS:
        command = [sys.executable, '-u', '-m', 'kb3_hyperparameter_optimization.quality.tpe_pipeline',
                   '--model', model, '--stage', args.stage, '--config', args.config, '--space-config', args.space_config,
                   '--data-root', args.data_root, '--data-config', args.data_config, '--output-dir', str(root / model)]
        if args.trials is not None:
            command += ['--trials', str(args.trials)]
        for name in ('resume', 'smoke'):
            if getattr(args, name):
                command.append('--' + name)
        commands.append((model, command))
    for model, command in commands:
        print(f'PREFLIGHT A-TPE: {model}', flush=True)
        subprocess.run([*command, '--check-only'], cwd=ROOT, check=True)
    if args.check_only:
        print('ALL MODELS A-TPE PREFLIGHT PASS; no output or training.', flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    status = dict(protocol=PROTOCOL, stage=args.stage, results={})
    for model, command in commands:
        status['active_model'] = model
        atomic_json(root / 'models_status.json', status)
        subprocess.run(command, cwd=ROOT, check=True)
        status['results'][model] = 'complete'
    status['active_model'] = None
    atomic_json(root / 'models_status.json', status)


def main(argv=None):
    args = parse_args(argv)
    if args.model == 'all':
        return run_models(args)
    settings = settings_from(args)
    target = args.trials if args.trials is not None else settings['trials']
    print(json.dumps(dict(protocol=PROTOCOL, model=args.model, stage=args.stage, requested_trials=target,
                         maximum_epochs=target * settings['epochs'], extra_evaluation_training_runs=0,
                         rl_training_runs=0, test_evaluations=0), indent=2), flush=True)
    preflight = SimpleNamespace(config=args.space_config, with_optuna=True, backend='command', device=settings['device'],
                                data_root=args.data_root, data_config=args.data_config)
    config, data = _preflight(preflight, [args.model])
    if len(data['names']) != 28:
        raise ValueError('Canonical pest evaluation requires 28 classes')
    space = prepare_reference(args, settings, config.search_space)
    distributions = local_distributions(settings, space)
    identity = fingerprint(args, settings, space)
    if args.resume and not args.output_dir:
        raise ValueError('--resume requires the original output directory')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(args.output_dir or ROOT / 'kb3_hyperparameter_optimization' /
                  'checkpoint_hyperparameter_optimization' / 'quality' / f'tpe_{stamp}' / args.model).resolve()
    start = dict(model=MODELS[args.model], data=str(output / 'dataset.yaml'), seed=settings['seed'],
                 **training_overrides(settings, DiscreteSearchSpace(space).initial_values()))
    snapshots = {'dataset.yaml': data, 'kb1_recipe.yaml': settings['trainer_recipe'], 'starting_train.yaml': start}
    if args.resume:
        old = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        if old['sha256'] != identity['sha256']:
            raise ValueError('TPE code/data/dependencies/recipe changed, or this is an older A/B protocol; use a new run')
        for filename, expected in snapshots.items():
            if yaml.safe_load((output / filename).read_text(encoding='utf-8')) != expected:
                raise ValueError(f'{filename} changed since the experiment was created')
    elif output.exists() and any(p.name != '.run.lock' for p in output.iterdir()):
        raise ValueError('Use a new empty output directory or --resume')
    if args.check_only:
        print(f'A-TPE preflight PASS identity={identity["sha256"]}; no training started.', flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output):
        if not args.resume:
            if any(p.name != '.run.lock' for p in output.iterdir()):
                raise ValueError('Output was populated by another process; use --resume')
            atomic_json(output / 'manifest.json', identity)
            for filename, payload in snapshots.items():
                (output / filename).write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
            atomic_json(output / 'train_contract.json', dict(protocol=PROTOCOL, initialization='scratch',
                        kb1_checkpoint_role='comparison_only', initial_parameters=DiscreteSearchSpace(space).initial_values(),
                        local_search={k: dict(low=v.low, high=v.high, log=v.log) for k, v in distributions.items()},
                        frozen_parameters=['momentum', 'augmentation_strength'],
                        stopping=dict(metric='canonical_validation_mAP50_95', patience=settings['search_patience'],
                                      min_epochs=settings['search_min_epochs'], min_delta=settings['search_min_delta']),
                        native_patience_role='disabled; the canonical stopper controls search',
                        detector_seed=settings['seed'], test_used=False))
        experiment = TPEExperiment(args, settings, space, output)
        stages = ('baseline', 'search', 'evaluate') if args.stage == 'all' else (args.stage,)
        for stage in stages:
            atomic_json(output / 'status.json', dict(protocol=PROTOCOL, stage=stage, status='running', requested_trials=target))
            try:
                if stage == 'baseline':
                    evaluate_reference(settings, args.data_root, output)
                elif stage == 'search':
                    experiment.search()
                else:
                    experiment.evaluate()
            except BaseException as error:
                atomic_json(output / 'status.json', dict(protocol=PROTOCOL, stage=stage, status='interrupted', reason=str(error),
                                                       requested_trials=target))
                raise
            atomic_json(output / 'status.json', dict(protocol=PROTOCOL, stage=stage, status='complete', requested_trials=target))
    print(f'A-TPE stage complete: {output}', flush=True)


if __name__ == '__main__':
    main()
