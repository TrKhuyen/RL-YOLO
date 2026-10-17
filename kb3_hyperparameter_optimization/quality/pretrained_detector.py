"""A-only native KB1 training: original pretrained source, fixed hyperparameters."""

from copy import deepcopy
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import math
import torch

from ..adapters.ultralytics_worker import _canonical_metrics
from ..search_space import DiscreteSearchSpace
from .detector import atomic_json, file_hash, weights_hash, isolated_evaluation_rng


SELECTION = 'kb1_native_validation_fitness_best_ema'


def training_overrides(settings, parameters):
    if parameters['augmentation_strength'] != .5:
        raise ValueError('A-only trials must keep the actual KB1 augmentation recipe')
    return {
        **settings['trainer_recipe'], 'pretrained': True, 'resume': False,
        'epochs': settings['epochs'], 'patience': settings['patience'], 'seed': settings['seed'],
        'batch': settings['batch_size'], 'imgsz': settings['img_size'],
        'device': settings['device'], 'workers': settings['workers'],
        'warmup_epochs': settings['warmup_epochs'], 'lrf': settings['lrf'],
        'lr0': parameters['lr0'], 'momentum': parameters['momentum'],
        'weight_decay': parameters['weight_decay'],
        # Artifact/plot settings do not change optimization or native selection.
        'save': True, 'save_period': -1, 'plots': False,
    }


def train_detector(*, settings, data_yaml, data_root, model, seed, output_dir,
                   space_config, parameters, agent=None, learn=False, random_schedule=False):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import init_seeds

    if agent is not None or learn or random_schedule:
        raise ValueError('This trainer is A-only; policy/action schedules are not allowed')
    if settings['initialization'] != 'pretrained' or seed != settings['seed']:
        raise ValueError('Use the inherited KB1 pretrained initialization and detector seed')
    source = settings['pretrained_source']
    checkpoint = Path(source['checkpoint'])
    if Path(model).resolve() != checkpoint.resolve() or file_hash(checkpoint) != source['checkpoint_sha256']:
        raise ValueError('Trial must load the exact original pretrained checkpoint from KB1 YAML')
    reference = settings.get('kb1_reference')
    if reference and (checkpoint.resolve() == Path(reference['checkpoint']).resolve()
                      or file_hash(checkpoint) == reference['checkpoint_sha256']):
        raise ValueError('KB1 best checkpoint is comparison-only')
    initial = DiscreteSearchSpace(space_config).initial_values()
    if set(parameters) != set(initial) or any(not math.isfinite(value) for value in parameters.values()):
        raise ValueError('A-only hyperparameters must be complete and finite')
    for name, spec in space_config.parameters.items():
        if not spec.minimum <= parameters[name] <= spec.maximum:
            raise ValueError(f'{name} is outside the declared bounds')
    if parameters['momentum'] != initial['momentum'] or parameters['augmentation_strength'] != initial['augmentation_strength']:
        raise ValueError('Only lr0 and weight_decay may differ from the KB1 starting recipe')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise ValueError('Completed detector result already exists')
    started = time.perf_counter()
    history, best_row = [], None
    initial_hash = None
    best_ema_hash = None
    source_weights = YOLO(str(checkpoint))
    head_prefix = f'model.{len(source_weights.model.model)-1}.'
    source_backbone = {name: tensor.detach().cpu().clone()
                       for name, tensor in source_weights.model.state_dict().items()
                       if not name.startswith(head_prefix)}
    overrides = training_overrides(settings, parameters)
    eval_args = SimpleNamespace(dataset_root=str(Path(data_root).resolve()), batch=settings['batch_size'],
                                imgsz=settings['img_size'], workers=0, device=settings['device'])

    class PretrainedTrainer(DetectionTrainer):
        def setup_model(self):
            # Match the declared seed after first-use integration imports, before
            # constructing the 28-class head; other native training logic is intact.
            init_seeds(seed, deterministic=self.args.deterministic)
            return super().setup_model()

        def final_eval(self):
            self._a_final_evaluation = True
            try:
                return super().final_eval()
            finally:
                self._a_final_evaluation = False

    def setup(trainer):
        nonlocal initial_hash
        for name in ('pretrained', 'resume', 'epochs', 'patience', 'seed', 'batch', 'imgsz',
                     'optimizer', 'lr0', 'momentum', 'weight_decay', 'warmup_epochs', 'lrf'):
            if getattr(trainer.args, name) != overrides[name]:
                raise RuntimeError(f'Native trainer changed inherited/requested {name}')
        if trainer.model.yaml.get('nc') != 28:
            raise RuntimeError('Native dataset setup did not create a 28-class detector')
        actual = trainer.model.state_dict()
        if not source_backbone or any(name not in actual or not torch.equal(actual[name].detach().cpu(), tensor)
                                      for name, tensor in source_backbone.items()):
            raise RuntimeError('Pretrained backbone was not transferred into the live detector')
        initial_hash = weights_hash(trainer.model)
        atomic_json(output / 'initial.json', dict(initialization='pretrained', seed=seed,
                    live_weights_sha256=initial_hash, pretrained_source=source,
                    pretrained_backbone_verified=True, transferred_backbone_tensors=len(source_backbone), hyperparameters=parameters,
                    training_overrides=overrides, actual_training_args=vars(trainer.args)))

    def epoch_end(trainer):
        nonlocal best_row, best_ema_hash
        epoch = trainer.epoch + 1
        # Native final_eval emits an extra callback after revalidating best.pt.
        if getattr(trainer, '_a_final_evaluation', False):
            return
        if epoch > settings['epochs'] or (history and epoch != history[-1]['metrics']['epoch'] + 1):
            raise RuntimeError('Native epoch log is not contiguous')
        if trainer.batch_size != settings['batch_size']:
            raise RuntimeError('Native trainer changed batch size after OOM; freeze a new recipe before retrying')
        native = trainer.metrics
        loss_items = trainer.tloss.values() if isinstance(trainer.tloss, dict) else trainer.tloss
        metrics = dict(epoch=epoch, train_loss=sum(float(v) for v in loss_items),
                       val_loss=sum(float(v) for k, v in native.items() if k.startswith('val/') and k.endswith('_loss')),
                       precision=float(native['metrics/precision(B)']), recall=float(native['metrics/recall(B)']),
                       map50=float(native['metrics/mAP50(B)']), map50_95=float(native['metrics/mAP50-95(B)']),
                       ap_small=None, elapsed_seconds=time.perf_counter()-started)
        fitness = float(trainer.fitness)
        if not all(math.isfinite(v) for v in (*[v for v in metrics.values() if v is not None], fitness)):
            raise RuntimeError('Nonfinite native training metrics')
        row = dict(metrics=metrics, metrics_protocol='ultralytics_native_validation', native_fitness=fitness,
                   hyperparameters=parameters, optimizer_lr=[float(g['lr']) for g in trainer.optimizer.param_groups])
        history.append(row)
        if fitness == trainer.best_fitness:
            best_row = deepcopy(row)
            # Native save_model serializes EMA in fp16, including its buffers.
            best_ema_hash = weights_hash(deepcopy(trainer.ema.ema).half().cpu())
        atomic_json(output / 'epochs.json', history)
        print(f'KB3-A PRETRAINED seed={seed} epoch={epoch}/{settings["epochs"]} '
              f'native_mAP={metrics["map50_95"]:.5f} native_fitness={fitness:.5f}', flush=True)

    source_weights.add_callback('on_pretrain_routine_end', setup)
    source_weights.add_callback('on_fit_epoch_end', epoch_end)
    try:
        source_weights.train(trainer=PretrainedTrainer, data=str(Path(data_yaml).resolve()),
                             project=str(output), name='native', exist_ok=True, **overrides)
        trainer = source_weights.trainer
        if not history or best_row is None or initial_hash is None:
            raise RuntimeError('Native training did not produce a complete best checkpoint record')
        selected = output / 'selected_best.pt'
        shutil.copy2(trainer.best, selected)
        payload = torch.load(selected, map_location='cpu', weights_only=False)
        best_model = payload.get('ema') or payload.get('model')
        if weights_hash(best_model) != best_ema_hash:
            raise RuntimeError('Selected native checkpoint does not match the recorded best EMA epoch')
        with isolated_evaluation_rng():
            best_scores = _canonical_metrics(selected, eval_args, seed)
            final_scores = _canonical_metrics(trainer.last, eval_args, seed)

        def canonical_metrics(row, scores):
            for name in ('precision', 'recall', 'mAP50', 'mAP50_95', 'APs'):
                if not math.isfinite(float(scores[name])) or not 0 <= float(scores[name]) <= 1:
                    raise RuntimeError(f'Invalid canonical checkpoint score: {name}')
            return {**row['metrics'], 'precision': float(scores['precision']), 'recall': float(scores['recall']),
                    'map50': float(scores['mAP50']), 'map50_95': float(scores['mAP50_95']),
                    'ap_small': float(scores['APs'])}

        if file_hash(checkpoint) != source['checkpoint_sha256']:
            raise RuntimeError('Original pretrained source changed during training')
        if reference and file_hash(reference['checkpoint']) != reference['checkpoint_sha256']:
            raise RuntimeError('Comparison-only KB1 best checkpoint changed')
        actual_epochs = history[-1]['metrics']['epoch']
        result = dict(seed=seed, failed=False, initialization='pretrained', pretrained_source=source,
                      pretrained_backbone_verified=True, initial_weights_sha256=initial_hash,
                      hyperparameters=parameters, final_hyperparameters=parameters, decisions=0,
                      best_checkpoint=str(selected), checkpoint_sha256=file_hash(selected),
                      checkpoint=str(trainer.last), metrics=canonical_metrics(best_row, best_scores),
                      final_metrics=canonical_metrics(history[-1], final_scores),
                      canonical_best_scores=best_scores, canonical_final_scores=final_scores,
                      native_best_metrics=best_row['metrics'], checkpoint_selection=SELECTION,
                      trajectory_metrics_protocol='ultralytics_native_validation',
                      trainer_recipe=settings['trainer_recipe'], maximum_horizon=settings['epochs'],
                      actual_epochs=actual_epochs, early_stopping=bool(settings['patience']),
                      stopped_early=actual_epochs < settings['epochs'],
                      stopping_rule=dict(metric='ultralytics_native_validation_fitness', patience=settings['patience']),
                      elapsed_seconds=time.perf_counter()-started, run_dir=str(output))
        atomic_json(output / 'selected_best.json', dict(metrics=result['metrics'], canonical_scores=best_scores,
                    native_metrics=best_row['metrics'], selection=SELECTION,
                    sha256=result['checkpoint_sha256'], checkpoint=str(selected)))
        atomic_json(output / 'decisions.json', [])
        atomic_json(output / 'result.json', result)
        return result
    finally:
        trainer = getattr(source_weights, 'trainer', None)
        for name in ('train_loader', 'test_loader'):
            loader = getattr(trainer, name, None)
            if hasattr(loader, 'close'):
                loader.close()
