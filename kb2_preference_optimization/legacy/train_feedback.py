"""KB2 native YOLO fine-tuning with feedback, pairwise or candidate DPO."""
import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from feedback_dataset import get_feedback_dataloader
from feedback_loss import (apply_gradient_conflict_control, compute_hybrid_loss,
                           compute_pairwise_preference_loss, compute_dpo_preference_loss,
                           DPO_PAIR_SOURCE, DPO_OBJECTIVE_VERSION)
from feedback_validation import EarlyStopping, evaluate_adapter
from dataloader import get_pest_dataloader
from train_rl import CHECKPOINTS, load_adapter

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DPO_POLICY = 'gt_class_anchor_softmax_v1'


def training_method(config):
    if config.get('dpo_alpha', 0) > 0:
        return 'candidate_selection_dpo_native_loss'
    if config.get('pairwise_alpha', 0) > 0:
        return 'pairwise_preference_native_loss'
    return 'feedback_guided_native_loss'


def make_frozen_reference(adapter):
    reference = copy.deepcopy(adapter)
    reference.model.requires_grad_(False)
    reference.eval_mode()
    return reference


def validate_objective_args(args):
    for name in ('pairwise_alpha', 'pairwise_margin', 'dpo_alpha'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and non-negative')
    if not math.isfinite(args.dpo_beta) or args.dpo_beta <= 0:
        raise ValueError('dpo_beta must be finite and positive')
    if args.dpo_alpha and (args.pairwise_alpha or args.feedback_alpha
                           or args.object_feedback_alpha or args.gradient_conflict_control
                           or args.sampling_strategy != 'shuffle'):
        raise ValueError('DPO requires shuffle sampling and no other auxiliary loss')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_training_checkpoint(path, adapter, optimizer, step, config,
                             training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save({
        'format_version': 1,
        'training_method': training_method(config),
        'step': int(step),
        'state_dict': adapter.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': dict(config),
        'training_state': dict(training_state or {}),
    }, temporary)
    temporary.replace(path)
    return path


def load_training_checkpoint(path, adapter, optimizer, device, expected_config=None):
    data = torch.load(path, map_location=device, weights_only=False)
    if data.get('training_method') not in (
            'feedback_guided_native_loss', 'pairwise_preference_native_loss',
            'candidate_selection_dpo_native_loss'):
        raise ValueError(f'{path} is not a KB2 training checkpoint')
    if expected_config is not None:
        if data['training_method'] != training_method(expected_config):
            raise ValueError('Resume training method differs from the requested objective')
        if expected_config.get('dpo_alpha', 0):
            # Rebuild the reference from the ORIGINAL base, never resumed policy.
            for key in ('model', 'base_checkpoint_sha256', 'feedback_sha256',
                        'reference_checkpoint_sha256',
                        'dpo_alpha', 'dpo_beta', 'dpo_policy', 'dpo_pair_source',
                        'dpo_objective_version',
                        'img_size', 'batch_size', 'seed', 'lr', 'weight_decay'):
                if data.get('config', {}).get(key) != expected_config.get(key):
                    raise ValueError(f'DPO resume provenance/config mismatch: {key}')
    adapter.model.load_state_dict(data['state_dict'], strict=True)
    optimizer.load_state_dict(data['optimizer'])
    return int(data['step']), data.get('config', {}), data.get('training_state', {})


def train(args):
    validate_objective_args(args)
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError('steps and batch-size must be positive')
    if args.gradient_conflict_control and args.object_feedback_alpha <= 0:
        raise ValueError('gradient conflict control requires positive object feedback alpha')
    if args.gradient_conflict_control and args.feedback_alpha != 0:
        raise ValueError('gradient conflict control requires feedback alpha 0')
    if args.gradient_conflict_control and args.pairwise_alpha != 0:
        raise ValueError('gradient conflict control cannot combine pairwise loss')
    if args.pairwise_alpha < 0 or args.pairwise_margin < 0:
        raise ValueError('pairwise alpha and margin must be non-negative')
    set_seed(args.seed)
    checkpoint = Path(args.checkpoint or CHECKPOINTS[args.model]).resolve()
    feedback_path = Path(args.feedback).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not feedback_path.exists():
        raise FileNotFoundError(feedback_path)
    summary_path = feedback_path.with_suffix('.summary.json')
    if not summary_path.is_file():
        raise FileNotFoundError(f'Missing feedback provenance: {summary_path}')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    expected = {
        'model': args.model,
        'split': 'train',
        'checkpoint_sha256': sha256(checkpoint),
        'feedback_sha256': sha256(feedback_path),
        'matching_policy': 'max_cardinality_gt_v2_1',
        'data_root': str(Path(args.data_root).resolve()),
        'img_size': args.img_size,
    }
    from feedback import SCHEMA_VERSION
    expected['schema_version'] = SCHEMA_VERSION
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f'Feedback provenance mismatch for {key}: {summary_path}')

    adapter = load_adapter(args.model, str(checkpoint), args.device)
    if not callable(getattr(adapter, 'supervised_loss', None)):
        raise NotImplementedError(
            f'{args.model} adapter does not implement supervised_loss for feedback fine-tuning')
    # Copy before resume and before constructing native loss. No random draws,
    # so initialization of a reference does not change the shared data ordering.
    reference = make_frozen_reference(adapter) if args.dpo_alpha else None
    parameters = [p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    stopper = EarlyStopping(args.patience, args.min_delta)
    training_state = {}

    if args.sampling_strategy == 'feedback':
        loader = get_feedback_dataloader(
            args.data_root, str(feedback_path), batch_size=args.batch_size,
            img_size=args.img_size, num_workers=args.workers,
            sampling_strength=args.sampling_strength,
            max_sampling_weight=args.max_sampling_weight, seed=args.seed)
    else:
        if args.feedback_alpha != 0:
            raise ValueError('shuffle sampling requires --feedback-alpha 0')
        loader = get_pest_dataloader(
            args.data_root, split='train', batch_size=args.batch_size,
            img_size=args.img_size, num_workers=args.workers, shuffle=True)
        if hasattr(loader.dataset.transforms, 'set_random_seed'):
            loader.dataset.transforms.set_random_seed(args.seed)
    val_loader = None
    if args.eval_interval:
        val_loader = get_pest_dataloader(
            args.data_root, split='val', batch_size=args.val_batch_size,
            img_size=args.img_size, num_workers=args.workers, shuffle=False)
    config = vars(args).copy()
    config.update({
        'base_checkpoint': str(checkpoint),
        'base_checkpoint_sha256': expected['checkpoint_sha256'],
        'feedback_path': str(feedback_path),
        'feedback_sha256': sha256(feedback_path),
        'feedback_schema_version': summary['schema_version'],
        'validation_confidence': args.val_conf,
        'dpo_policy': DPO_POLICY if reference is not None else None,
        'dpo_pair_source': DPO_PAIR_SOURCE if reference is not None else None,
        'dpo_objective_version': DPO_OBJECTIVE_VERSION if reference is not None else None,
        'reference_checkpoint_sha256': expected['checkpoint_sha256'] if reference is not None else None,
    })
    if args.resume:
        start_step, _, training_state = load_training_checkpoint(
            args.resume, adapter, optimizer, args.device, config)
        stopper.load_state_dict(training_state.get('early_stopping', {}))
    output = Path(args.output).resolve()
    best_output = output.with_name(f'{output.stem}_best{output.suffix}')
    metrics_path = output.with_name(f'{output.stem}_metrics.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        metrics_path.write_text('', encoding='utf-8')
    def record_metrics(event):
        with metrics_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
    pairwise_pairs_seen = training_state.get('pairwise_pairs_seen', 0)
    pairwise_active_steps = training_state.get('pairwise_active_steps', 0)
    dpo_pairs_seen = training_state.get('dpo_pairs_seen', 0)
    dpo_active_steps = training_state.get('dpo_active_steps', 0)
    dpo_diagnostics = training_state.get('dpo_diagnostics', {})
    def current_state(**extra):
        return {'early_stopping': stopper.state_dict(),
                'pairwise_pairs_seen': pairwise_pairs_seen,
                'pairwise_active_steps': pairwise_active_steps,
                'dpo_pairs_seen': dpo_pairs_seen, 'dpo_active_steps': dpo_active_steps,
                'dpo_diagnostics': dpo_diagnostics, **extra}
    data_iterator = iter(loader)
    print(f'[Feedback FT] {args.model} | steps={args.steps} | alpha={args.feedback_alpha} '
          f'| pairwise_alpha={args.pairwise_alpha} '
          f'| dpo_alpha={args.dpo_alpha} beta={args.dpo_beta}')
    print(f'  feedback images={len(loader.dataset)} | start_step={start_step}')
    if reference is not None:
        print(f'  DPO objective={DPO_OBJECTIVE_VERSION} | pairs={DPO_PAIR_SOURCE} '
              '| reference=frozen KB1 scores only')

    if val_loader is not None and start_step == 0 and args.eval_before_train:
        metrics = evaluate_adapter(
            adapter, val_loader, args.device, args.val_conf, args.val_iou)
        stopper.update(metrics['mAP50-95'])
        state = current_state(validation=metrics)
        save_training_checkpoint(best_output, adapter, optimizer, 0, config, state)
        record_metrics({'event': 'validation', 'step': 0, **metrics})
        print('  val step=0 mAP50-95={:.5f} mAP50={:.5f} AP_small={:.5f} '
              'recall={:.5f}'.format(
                  metrics['mAP50-95'], metrics['mAP50'], metrics['AP_small'],
                  metrics['Recall']))
        print(f'  best: {best_output}')

    last_step = start_step
    stopped_early = False
    for step in range(start_step + 1, args.steps + 1):
        last_step = step
        try:
            images, targets = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            images, targets = next(data_iterator)
        images = images.to(args.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if args.gradient_conflict_control:
            loss, details = apply_gradient_conflict_control(
                adapter, images, targets, args.object_feedback_alpha,
                args.object_cls_weight, args.object_box_weight,
                args.dynamic_object_feedback)
        else:
            loss, details = compute_hybrid_loss(
                adapter, images, targets, args.feedback_alpha,
                args.max_feedback_loss_weight, args.object_feedback_alpha,
                args.object_cls_weight, args.object_box_weight,
                args.dynamic_object_feedback)
        pairwise_loss = loss.detach() * 0
        dpo_loss = loss.detach() * 0
        pair_count = 0
        if args.pairwise_alpha:
            pairwise_loss, pair_count = compute_pairwise_preference_loss(
                adapter, images, targets, args.pairwise_margin)
            loss = loss + args.pairwise_alpha * pairwise_loss
            pairwise_pairs_seen += pair_count
            pairwise_active_steps += int(pair_count > 0)
        if reference is not None:
            dpo_loss, diagnostic = compute_dpo_preference_loss(
                adapter, reference, images, targets, args.dpo_beta)
            loss = loss + args.dpo_alpha * dpo_loss
            dpo_count = diagnostic['pair_count']
            dpo_pairs_seen += dpo_count
            dpo_active_steps += int(dpo_count > 0)
            dpo_diagnostics = {key: int(value) if key == 'pair_count' else float(value)
                               for key, value in diagnostic.items()}
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at step {step}: {loss}')
        if not args.gradient_conflict_control:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f'non-finite gradient at step {step}')
        optimizer.step()

        if step == 1 or step % args.log_interval == 0:
            print(f'  step={step:6d} total={loss.item():.5f} '
                  f'native={details["native_loss"].item():.5f} '
                  f'feedback={details["feedback_loss"].item():.5f} '
                  f'pairwise={pairwise_loss.item():.5f} pairs={pair_count} '
                  f'dpo={dpo_loss.item():.5f} dpo_pairs={dpo_diagnostics.get("pair_count", 0):.0f} '
                  f'grad={float(grad_norm):.4f}')
            if reference is not None:
                print(f'    GT IoU chosen={dpo_diagnostics["chosen_iou"]:.4f} '
                      f'rejected={dpo_diagnostics["rejected_iou"]:.4f} '
                      f'policy_win={dpo_diagnostics["policy_win_rate"]:.1%} '
                      f'relative_win={dpo_diagnostics["relative_win_rate"]:.1%}')
            record_metrics({'event': 'train', 'step': step,
                            'total_loss': float(loss.detach()),
                            'native_loss': float(details['native_loss']),
                            'pairwise_loss': float(pairwise_loss.detach()),
                            'dpo_loss': float(dpo_loss.detach()),
                            'grad_norm': float(grad_norm),
                            'dpo_diagnostics': dpo_diagnostics})
        if args.save_interval and step % args.save_interval == 0:
            save_training_checkpoint(output.with_name(
                f'{output.stem}_step{step}{output.suffix}'),
                adapter, optimizer, step, config,
                current_state())

        if val_loader is not None and step % args.eval_interval == 0:
            metrics = evaluate_adapter(
                adapter, val_loader, args.device, args.val_conf, args.val_iou)
            improved, should_stop = stopper.update(metrics['mAP50-95'])
            metric_line = (
                '  val step={} mAP50-95={:.5f} mAP50={:.5f} '
                'AP_small={:.5f} recall={:.5f} best={:.5f}'
            ).format(step, metrics['mAP50-95'], metrics['mAP50'],
                     metrics['AP_small'], metrics['Recall'], stopper.best)
            print(metric_line)
            state = current_state(validation=metrics)
            record_metrics({'event': 'validation', 'step': step, **metrics})
            if improved:
                save_training_checkpoint(
                    best_output, adapter, optimizer, step, config, state)
                print(f'  best: {best_output}')
            if should_stop:
                print(f'  early stop: {stopper.bad_evaluations} evaluations without improvement')
                stopped_early = True
                break

    saved = save_training_checkpoint(
        output, adapter, optimizer, last_step, config,
        current_state(stopped_early=stopped_early))
    print(f'  saved: {saved}')
    return saved


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Feedback-guided native YOLO fine-tuning')
    parser.add_argument('--model', default='yolov8n', choices=list(CHECKPOINTS))
    parser.add_argument('--checkpoint')
    parser.add_argument('--feedback')
    parser.add_argument('--data-root', default=str(REPO_ROOT / 'pre-data/data/v2i_cleanned'))
    parser.add_argument('--output')
    parser.add_argument('--resume')
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--feedback-alpha', type=float, default=.10)
    parser.add_argument('--sampling-strategy', choices=['shuffle', 'feedback'], default='feedback')
    parser.add_argument('--sampling-strength', type=float, default=1.0)
    parser.add_argument('--max-sampling-weight', type=float, default=5.0)
    parser.add_argument('--max-feedback-loss-weight', type=float, default=3.0)
    parser.add_argument('--pairwise-alpha', type=float, default=0.0)
    parser.add_argument('--pairwise-margin', type=float, default=0.1)
    parser.add_argument('--dpo-alpha', type=float, default=0.0)
    parser.add_argument('--dpo-beta', type=float, default=0.1)
    parser.add_argument('--object-feedback-alpha', type=float, default=0.0)
    parser.add_argument('--object-cls-weight', type=float, default=1.0)
    parser.add_argument('--object-box-weight', type=float, default=1.0)
    parser.add_argument('--dynamic-object-feedback', action='store_true')
    parser.add_argument('--gradient-conflict-control', action='store_true')
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=2000)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--val-batch-size', type=int, default=8)
    parser.add_argument('--val-conf', type=float, default=.001)
    parser.add_argument('--val-iou', type=float, default=.45)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--no-eval-before-train', dest='eval_before_train',
                        action='store_false')
    parser.set_defaults(eval_before_train=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.feedback is None:
        args.feedback = str(SCRIPT_DIR / 'feedback_data_clean' / f'{args.model}_train.jsonl')
    if args.output is None:
        args.output = str(SCRIPT_DIR / 'checkpoint_preference_optimization' / f'{args.model}_feedback_last.pt')
    if not 0 <= args.feedback_alpha <= 1:
        parser.error('--feedback-alpha must be in [0, 1]')
    try:
        validate_objective_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == '__main__':
    train(parse_args())
