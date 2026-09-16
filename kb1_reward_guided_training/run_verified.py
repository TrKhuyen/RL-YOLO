'''Verified, resumable KB1 seed screening. Run each model in a fresh process.'''
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler

from canonical_eval import evaluate_adapter
from dataloader import PestDataset, get_train_transforms, get_pest_dataloader, pest_collate_fn
from train_rl import (CHECKPOINTS, DATA_ROOT, PROJECT_ROOT, load_adapter,
                      EMABaseline, match_aware_objective, l2sp_loss)
from reward import detection_composite_reward

PROTOCOL = {
    'id': 'kb1_canonical_v2_ar300', 'split': 'val',
    'annotation_policy': 'clip box corners to image bounds before transforms; fail on errors',
    'image_size': 640, 'input': 'RGB float32 /255; longest-side resize, centered pad114',
    'conf': 0.001, 'nms_iou': 0.60, 'max_det': 300,
    'class_policy': 'single best class per candidate, class-aware NMS',
    'operating_conf': 0.25, 'operating_iou': 0.5,
    'ap_iou': '0.50:0.05:0.95', 'area_space': 'resized padded 640x640',
    'score': '0.4*mAP50+0.4*mAP50_95+0.2*AR300',
    'bn': 'eval/frozen statistics', 'dfl': 'fixed',
    'dp_w3f': False, 'dp_psa': False,
}


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def atomic_save(path, data):
    tmp = path.with_suffix('.tmp')
    torch.save(data, tmp)
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class SeededDataset(Dataset):
    def __init__(self, seed):
        self.seed = seed
        self.data = PestDataset(DATA_ROOT, 'train', transforms=get_train_transforms(640))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        epoch, idx = item
        self.data.transforms.set_random_seed(self.seed + epoch * len(self) + idx)
        return self.data[idx]


class StepBatches(Sampler):
    def __init__(self, size, batch_size, start, stop, seed):
        self.size, self.batch_size = size, batch_size
        self.start, self.stop, self.seed = start, stop, seed
        if size < batch_size:
            raise ValueError('Dataset is smaller than one batch')

    def __iter__(self):
        epoch_size = self.size // self.batch_size
        previous, order = None, None
        for step in range(self.start, self.stop):
            epoch, offset = divmod(step, epoch_size)
            if epoch != previous:
                order = torch.randperm(self.size, generator=torch.Generator().manual_seed(
                    self.seed + epoch)).tolist()
                previous = epoch
            yield [(epoch, idx) for idx in order[
                offset * self.batch_size:(offset + 1) * self.batch_size]]

    def __len__(self):
        return self.stop - self.start


def worker_init(_):
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def score(m):
    return 0.4 * m['mAP50'] + 0.4 * m['mAP50_95'] + 0.2 * m['AR300']


def evaluate(adapter, loader):
    result = evaluate_adapter(adapter, loader, class_metrics=True)
    result.pop('recall')  # Legacy alias is intentionally excluded from new artifacts.
    return result


def parity_check(adapter, images, path, model_name, source):
    with torch.no_grad():
        before = adapter.forward_with_grad(images, conf_thres=0.001, iou_thres=0.60)
    atomic_save(path, {'state_dict': adapter.state_dict()})
    state = torch.load(path, map_location='cpu', weights_only=False)
    rebuilt = load_adapter(model_name, str(source), 'cuda')
    rebuilt.load_state_dict(state['state_dict'], strict=True)
    with torch.no_grad():
        after = rebuilt.forward_with_grad(images, conf_thres=0.001, iou_thres=0.60)
    for a, b in zip(before, after):
        for key in ('boxes', 'scores', 'labels'):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def run(args):
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    os.environ['DP_YOLO_USE_W3F'] = '0'
    os.environ['DP_YOLO_USE_PSA'] = '0'
    cfg = yaml.safe_load((PROJECT_ROOT / 'configs/hyp.rl.yaml').read_text(encoding='utf-8'))
    cfg.update(seed=42, steps=args.steps, workers=args.workers, mode=args.mode,
               eval_interval=args.eval_interval, batch_size=args.batch_size)
    if args.steps < 2 or args.steps % 2 or args.eval_interval % 2:
        raise ValueError('Steps and eval interval must be positive multiples of accumulation=2')
    source = CHECKPOINTS[args.model]
    out = Path(args.output).resolve() / args.model / args.mode
    out.mkdir(parents=True, exist_ok=True)
    metadata = {'model': args.model, 'seed': 42, 'mode': args.mode, 'config': cfg,
                'code_sha256': {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in
                    [Path(__file__), PROJECT_ROOT / 'canonical_eval.py',
                     PROJECT_ROOT / 'dataloader.py', PROJECT_ROOT / 'train_rl.py',
                     PROJECT_ROOT / 'reward.py', *sorted((PROJECT_ROOT / 'adapters').glob('*.py'))]},
                'protocol': PROTOCOL, 'source': str(source), 'source_sha256': sha256(source)}
    manifest = out / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != metadata:
        raise ValueError('Run configuration changed: select a new output directory')
    atomic_json(manifest, metadata)
    if (out / 'completed.json').exists():
        print('Already completed:', out, flush=True)
        return
    atomic_json(out / 'status.json', {'status': 'checking'})
    print('Loading', args.model, args.mode, flush=True)
    adapter = load_adapter(args.model, str(source), 'cuda')
    adapter.freeze_except_detection_head()
    named = [(n, p) for n, p in adapter.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    reference = {n: p.detach().clone() for n, p in named}
    frozen = {n: p.detach().cpu().clone() for n, p in adapter.named_parameters()
              if not p.requires_grad}
    buffers = {n: v.detach().cpu().clone() for n, v in adapter.state_dict().items()
               if 'running_' in n or 'num_batches_tracked' in n}
    optimizer = torch.optim.Adam(params, lr=cfg['lr'])
    baseline = EMABaseline(cfg['ema_alpha'])
    val = get_pest_dataloader(DATA_ROOT, 'val', batch_size=16, num_workers=args.workers, seed=42)
    images, _ = next(iter(val))
    parity_check(adapter, images.to('cuda'), out / 'step0.pt', args.model, source)
    start, best_score, patience_score, stale, best_step = 0, -math.inf, -math.inf, 0, 0
    baseline_path = out / 'baseline.json'
    if baseline_path.exists():
        initial = json.loads(baseline_path.read_text(encoding='utf-8'))
    else:
        initial = evaluate(adapter, val)
        atomic_json(baseline_path, initial)
    best_score = patience_score = score(initial)
    last_path = out / 'last.pt'
    if last_path.exists():
        saved = torch.load(last_path, map_location='cpu', weights_only=False)
        adapter.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        start, best_score, patience_score, stale, best_step = (
            saved[k] for k in ('step', 'best_score', 'patience_score', 'stale', 'best_step'))
        baseline.__dict__.update(saved['ema'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:
        atomic_save(out / 'best.pt', {'state_dict': adapter.state_dict(), 'step': 0,
                                     'metrics': initial, 'metadata': metadata})
    print('Step0 parity PASS; baseline', initial['mAP50'], initial['mAP50_95'], flush=True)
    dataset = SeededDataset(42)
    stop = start if stale >= cfg['early_stopping_patience'] else args.steps
    batches = StepBatches(len(dataset), args.batch_size, start, stop, 42)
    train = DataLoader(dataset, batch_sampler=batches, num_workers=args.workers,
                       collate_fn=pest_collate_fn, pin_memory=True,
                       worker_init_fn=worker_init, persistent_workers=args.workers > 0)
    optimizer.zero_grad(set_to_none=True)
    step = start
    started = time.perf_counter()
    for step, (images, targets) in enumerate(train, start + 1):
        images = images.to('cuda', non_blocking=True)
        warmup = cfg['warmup_steps']
        if step <= warmup:
            lr = cfg['min_lr'] + (cfg['lr'] - cfg['min_lr']) * step / warmup
        else:
            progress = (step - warmup) / max(args.steps - warmup, 1)
            lr = cfg['min_lr'] + 0.5 * (cfg['lr'] - cfg['min_lr']) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = lr
        native, _ = adapter.native_detection_loss(images, targets)
        loss = cfg['native_supervised_loss_weight'] * native
        reward_value, proxy_value = None, None
        if args.mode == 'kb1b':
            preds = adapter.forward_with_grad(images, conf_thres=cfg['conf_thres'],
                                             iou_thres=cfg['iou_thres'])
            rewards = detection_composite_reward(
                preds, targets, iou_thresholds=cfg['reward_iou_thresholds'],
                precision_weight=cfg['precision_weight'], recall_weight=cfg['recall_weight'],
                iou_weight=cfg['iou_weight']).to('cuda')
            advantage = baseline.normalized_advantage(rewards, cfg['advantage_clip'])
            log_prob, proxy = match_aware_objective(
                preds, targets, cfg['iou_threshold'], cfg['tp_weight'],
                cfg['fp_weight'], cfg['fn_weight'])
            loss = loss + cfg['reward_loss_weight'] * -(log_prob * advantage.detach()).mean()
            loss = loss + cfg['supervised_loss_weight'] * proxy
            reward_value, proxy_value = rewards.mean().item(), proxy.item()
        loss = loss + cfg['stability_loss_weight'] * l2sp_loss(named, reference)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step}')
        (loss / 2).backward()
        if step % 2 == 0:
            grads = [p.grad for p in params if p.grad is not None]
            if not grads or not any(bool(g.abs().max() > 0) for g in grads):
                raise RuntimeError('No nonzero trainable gradient')
            norm = torch.nn.utils.clip_grad_norm_(params, cfg['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == start + 2:
            for n, p in adapter.named_parameters():
                if n in frozen:
                    torch.testing.assert_close(p.detach().cpu(), frozen[n], atol=0, rtol=0)
            if not any(not torch.equal(p.detach(), reference[n]) for n, p in named):
                raise RuntimeError('Optimizer did not change the head')
            print('Gradient/frozen parameters/update checks PASS', flush=True)
        if step % 50 == 0:
            log = {'step': step, 'loss': loss.item(), 'native': native.item(),
                   'reward': reward_value, 'proxy': proxy_value, 'lr': lr,
                   'seconds': time.perf_counter() - started}
            with open(out / 'train.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(log) + '\n')
            print(args.model, args.mode, log, flush=True)
        if step % args.eval_interval == 0 or step == args.steps:
            m = evaluate(adapter, val)
            current = score(m)
            if current > best_score:
                best_score, best_step = current, step
                atomic_save(out / 'best.pt', {'state_dict': adapter.state_dict(), 'step': step,
                                             'metrics': m, 'metadata': metadata})
            if current > patience_score + cfg['early_stopping_min_delta']:
                patience_score, stale = current, 0
            else:
                stale += 1
            for n, v in buffers.items():
                torch.testing.assert_close(adapter.state_dict()[n].cpu(), v, atol=0, rtol=0)
            record = {'step': step, 'metrics': m, 'best_step': best_step, 'stale': stale}
            with open(out / 'validation.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
            atomic_save(last_path, {'state_dict': adapter.state_dict(),
                'optimizer': optimizer.state_dict(), 'step': step, 'best_score': best_score,
                'patience_score': patience_score, 'stale': stale, 'best_step': best_step,
                'ema': baseline.__dict__, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all()})
            atomic_json(out / 'status.json', {'status': 'training', 'step': step,
                                              'best_step': best_step, 'score': current})
            print('Validation', step, m['mAP50'], m['mAP50_95'], 'best', best_step, flush=True)
            if stale >= cfg['early_stopping_patience']:
                break
    best = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    adapter.load_state_dict(best['state_dict'], strict=True)
    final = evaluate(adapter, val)
    for key in ('mAP50', 'mAP50_95', 'AR300'):
        if abs(final[key] - best['metrics'][key]) > 1e-6:
            raise RuntimeError(f'Best checkpoint reload mismatch: {key}')
    atomic_json(out / 'completed.json', {'status': 'complete', 'steps_run': step,
                'best_step': best['step'], 'baseline': initial, 'best': final,
                'termination': 'early_stopping' if step < args.steps else 'budget',
                'checks': ['step0_strict_reload', 'finite_gradients', 'head_updated',
                           'frozen_parameters', 'bn_buffers', 'best_strict_reload']})
    atomic_json(out / 'status.json', {'status': 'complete', 'step': step})
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=list(CHECKPOINTS))
    p.add_argument('--mode', required=True, choices=['native_only', 'kb1b'])
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--eval-interval', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    run(p.parse_args())
