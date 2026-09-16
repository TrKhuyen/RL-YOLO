"""Feedback-guided fine-tuning for YOLO using native detection losses."""
import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import torch

from feedback_dataset import get_feedback_dataloader
from feedback_loss import apply_gradient_conflict_control, compute_hybrid_loss
from feedback_validation import EarlyStopping, evaluate_adapter
from dataloader import get_pest_dataloader
from train_rl import CHECKPOINTS, load_adapter

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


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
        'training_method': 'feedback_guided_native_loss',
        'step': int(step),
        'state_dict': adapter.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': dict(config),
        'training_state': dict(training_state or {}),
    }, temporary)
    temporary.replace(path)
    return path


def load_training_checkpoint(path, adapter, optimizer, device):
    data = torch.load(path, map_location=device, weights_only=False)
    if data.get('training_method') != 'feedback_guided_native_loss':
        raise ValueError(f'{path} is not a feedback-guided checkpoint')
    adapter.model.load_state_dict(data['state_dict'], strict=True)
    optimizer.load_state_dict(data['optimizer'])
    return int(data['step']), data.get('config', {}), data.get('training_state', {})


def train(args):
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError('steps and batch-size must be positive')
    if args.gradient_conflict_control and args.object_feedback_alpha <= 0:
        raise ValueError('gradient conflict control requires positive object feedback alpha')
    if args.gradient_conflict_control and args.feedback_alpha != 0:
        raise ValueError('gradient conflict control requires feedback alpha 0')
    set_seed(args.seed)
    checkpoint = Path(args.checkpoint or CHECKPOINTS[args.model]).resolve()
    feedback_path = Path(args.feedback).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not feedback_path.exists():
        raise FileNotFoundError(feedback_path)

    adapter = load_adapter(args.model, str(checkpoint), args.device)
    parameters = [p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    stopper = EarlyStopping(args.patience, args.min_delta)
    if args.resume:
        start_step, _, training_state = load_training_checkpoint(
            args.resume, adapter, optimizer, args.device)
        stopper.load_state_dict(training_state.get('early_stopping', {}))

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
        'feedback_path': str(feedback_path),
        'feedback_sha256': sha256(feedback_path),
    })
    output = Path(args.output).resolve()
    best_output = output.with_name(f'{output.stem}_best{output.suffix}')
    data_iterator = iter(loader)
    print(f'[Feedback FT] {args.model} | steps={args.steps} | alpha={args.feedback_alpha}')
    print(f'  feedback images={len(loader.dataset)} | start_step={start_step}')

    if val_loader is not None and start_step == 0 and args.eval_before_train:
        metrics = evaluate_adapter(
            adapter, val_loader, args.device, args.val_conf, args.val_iou)
        stopper.update(metrics['mAP50-95'])
        state = {'early_stopping': stopper.state_dict(), 'validation': metrics}
        save_training_checkpoint(best_output, adapter, optimizer, 0, config, state)
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
                  f'grad={float(grad_norm):.4f}')
        if args.save_interval and step % args.save_interval == 0:
            save_training_checkpoint(output.with_name(
                f'{output.stem}_step{step}{output.suffix}'),
                adapter, optimizer, step, config,
                {'early_stopping': stopper.state_dict()})

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
            state = {'early_stopping': stopper.state_dict(), 'validation': metrics}
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
        {'early_stopping': stopper.state_dict(), 'stopped_early': stopped_early})
    print(f'  saved: {saved}')
    return saved


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Feedback-guided native YOLO fine-tuning')
    parser.add_argument('--model', default='yolov8n', choices=list(CHECKPOINTS))
    parser.add_argument('--checkpoint')
    parser.add_argument('--feedback', default=str(SCRIPT_DIR / 'feedback_data/yolov8n_train.jsonl'))
    parser.add_argument('--data-root', default=str(REPO_ROOT / 'pre-data/data/v2i'))
    parser.add_argument('--output', default=str(SCRIPT_DIR / 'feedback_checkpoints/yolov8n_feedback_last.pt'))
    parser.add_argument('--resume')
    parser.add_argument('--steps', type=int, default=1000)
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
    parser.add_argument('--object-feedback-alpha', type=float, default=0.0)
    parser.add_argument('--object-cls-weight', type=float, default=1.0)
    parser.add_argument('--object-box-weight', type=float, default=1.0)
    parser.add_argument('--dynamic-object-feedback', action='store_true')
    parser.add_argument('--gradient-conflict-control', action='store_true')
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=500)
    parser.add_argument('--eval-interval', type=int, default=500)
    parser.add_argument('--val-batch-size', type=int, default=8)
    parser.add_argument('--val-conf', type=float, default=.25)
    parser.add_argument('--val-iou', type=float, default=.45)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--no-eval-before-train', dest='eval_before_train',
                        action='store_false')
    parser.set_defaults(eval_before_train=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if not 0 <= args.feedback_alpha <= 1:
        parser.error('--feedback-alpha must be in [0, 1]')
    return args


if __name__ == '__main__':
    train(parse_args())
