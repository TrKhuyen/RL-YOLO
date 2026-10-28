"""Fine-tune a supervised YOLO with native + whole-action DPO loss."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import OBJECTIVE_VERSION
from .adapter import DFLAdapter, SUPPORTED_MODELS
from .artifacts import (append_jsonl, atomic_checkpoint, atomic_json, seed_everything,
                        sha256, source_checkpoint, source_provenance, tensor_digest)
from .data import detection_loader
from .metrics import evaluate_adapter
from .policy import dpo_loss, render_action
from .preferences import PreferenceDataset, collate_preferences
from .quality import quality_score


REFERENCE_SCORING_VERSION = 'same_batch_reference_v1'


def pair_losses(distributions, reference_distributions, records, beta):
    """Score immutable actions with policy/reference from the SAME input batch.

    Generation-time batch1 likelihoods remain audit evidence only: numerical
    differences between CUDA batch shapes must not become a preference signal.
    """
    if not records or len(distributions) != len(records) or len(reference_distributions) != len(records):
        raise ValueError('Policy/reference distributions must cover every pair in the batch')
    chosen = torch.stack([d.log_prob(r['chosen']) for d, r in zip(distributions, records)])
    rejected = torch.stack([d.log_prob(r['rejected']) for d, r in zip(distributions, records)])
    with torch.no_grad():
        ref_chosen = torch.stack([d.log_prob(r['chosen'])
                                  for d, r in zip(reference_distributions, records)])
        ref_rejected = torch.stack([d.log_prob(r['rejected'])
                                    for d, r in zip(reference_distributions, records)])
        cached_gap = chosen.new_tensor([r['reference_chosen_logp'] - r['reference_rejected_logp']
                                        for r in records])
        cache_error = (ref_chosen - ref_rejected) - cached_gap
    loss, relative = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta)
    return loss, {'relative_margin': float(relative.mean()),
                  'relative_margin_max_abs': float(relative.abs().max()),
                  'relative_win_rate': float((relative > 0).double().mean()),
                  'absolute_chosen_win_rate': float((chosen > rejected).double().mean()),
                  'policy_logp_gap': float((chosen - rejected).detach().mean()),
                  'cached_reference_gap_error_mean': float(cache_error.mean()),
                  'cached_reference_gap_error_max_abs': float(cache_error.abs().max())}


@torch.no_grad()
def preference_audit(policy, reference, dataset, beta, count=8, seed=1729, draws=2,
                     verify_cache=False):
    """Fixed TRAIN audit; diagnostic only, never checkpoint selection."""
    count = min(count, len(dataset))
    if count <= 0:
        return {}
    summary = {'pairs': count, 'scope': 'fixed_train_audit_not_independent_validation'}
    losses, margins, wins, quality_values, detection_counts, empty = [], [], [], [], [], []
    for index in range(count):
        image, target, record = dataset[index]
        pd, anchors, strides, _ = policy.distributions(image[None], dataset.policy_config)
        rd, _, _, priorities = reference.distributions(image[None], dataset.policy_config)
        if verify_cache:
            for side in ('chosen', 'rejected'):
                current = float(rd[0].log_prob(record[side]))
                if abs(current - record[f'reference_{side}_logp']) > .002:
                    raise ValueError('Cached reference log-probability does not replay')
                rendered = render_action(rd[0], record[side], anchors, strides,
                                         priorities[0], dataset.base.image_size)
                saved = record[f'{side}_detection']
                if (not torch.equal(rendered['labels'], torch.tensor(saved['labels'], dtype=torch.long))
                        or not torch.allclose(rendered['boxes'], torch.tensor(saved['boxes']).reshape(-1, 4).float(), atol=1e-4)):
                    raise ValueError('Stored whole output does not replay under fixed reference renderer')
                score = quality_score(rendered, target, dataset.quality_config)['score']
                if abs(score - record[f'{side}_quality']['score']) > 1e-7:
                    raise ValueError('Stored preference quality does not replay against TRAIN GT')
        loss, diagnostic = pair_losses(pd, rd, [record], beta)
        losses.append(float(loss))
        margins.append(diagnostic['relative_margin'])
        wins.append(diagnostic['absolute_chosen_win_rate'])
        generator = torch.Generator(device=policy.device).manual_seed(seed + index)
        for _ in range(draws):
            action = pd[0].sample(generator)
            rendered = render_action(pd[0], action, anchors, strides, priorities[0], dataset.base.image_size)
            quality_values.append(quality_score(rendered, target, dataset.quality_config)['score'])
            detection_counts.append(len(rendered['boxes']))
            empty.append(not len(rendered['boxes']))
    summary.update(dpo_loss=sum(losses) / count, relative_margin=sum(margins) / count,
                   absolute_chosen_win_rate=sum(wins) / count,
                   sampled_mean_quality=sum(quality_values) / len(quality_values),
                   sampled_mean_detections=sum(detection_counts) / len(detection_counts),
                   sampled_empty_rate=sum(empty) / len(empty))
    return summary


def gradient_diagnostics(native, preference, parameters, weight):
    native_grads = torch.autograd.grad(native, parameters, retain_graph=True, allow_unused=True)
    dpo_grads = torch.autograd.grad(preference * weight, parameters, retain_graph=True, allow_unused=True)
    native_sq = native.new_zeros(())
    dpo_sq, dot = native_sq.clone(), native_sq.clone()
    for a, b in zip(native_grads, dpo_grads):
        if a is not None:
            native_sq = native_sq + a.detach().square().sum()
        if b is not None:
            dpo_sq = dpo_sq + b.detach().square().sum()
        if a is not None and b is not None:
            dot = dot + (a.detach() * b.detach()).sum()
    return {'native_grad_norm': float(native_sq.sqrt()), 'weighted_dpo_grad_norm': float(dpo_sq.sqrt()),
            'weighted_dpo_native_norm_ratio': float((dpo_sq / native_sq.clamp_min(1e-30)).sqrt()),
            'gradient_cosine': float(dot / (native_sq * dpo_sq).sqrt().clamp_min(1e-30))}


def train(args):
    for name in ('lr', 'beta', 'dpo_weight', 'grad_clip'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if (not math.isfinite(args.weight_decay) or args.weight_decay < 0
            or min(args.steps, args.batch_size, args.eval_interval, args.patience,
                   args.log_interval, args.gradient_interval, args.audit_images) < 1
            or args.eval_limit < 0):
        raise ValueError('Invalid optimizer/budget/diagnostic parameters')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a fresh run directory: {output}')
    seed_everything(args.seed)
    provenance = source_provenance(args.model)
    dataset = PreferenceDataset(args.preferences, provenance)
    policy = DFLAdapter(source_checkpoint(args.model), args.device)
    reference = DFLAdapter(source_checkpoint(args.model), args.device)
    reference.model.requires_grad_(False)
    reference_digest = tensor_digest(reference.model)
    if reference_digest != dataset.manifest['reference_state_sha256']:
        raise ValueError('Reference state differs from preference generation')
    parameters = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=collate_preferences, generator=generator)
    valid = detection_loader('valid', dataset.base.image_size, args.val_batch_size)
    pilot = bool(args.eval_limit or dataset.manifest['limit'] or dataset.base.image_size != 640)
    config = {**vars(args), 'image_size': dataset.base.image_size, 'head_only': True,
              'native_weight': 1., 'policy': dataset.policy_config.to_dict(),
              'reference_scoring': REFERENCE_SCORING_VERSION,
              'preference_manifest_sha256': sha256(dataset.folder / 'manifest.json'),
              'pairs_sha256': dataset.manifest['pairs_sha256'], 'pilot': pilot}
    output.mkdir(parents=True)
    manifest = {'provenance': provenance, 'config': config,
                'reference_state_sha256': reference_digest,
                'trainable_parameters': sum(p.numel() for p in parameters)}
    atomic_json(output / 'manifest.json', manifest)
    metrics_path = output / 'metrics.jsonl'
    stopper = {'best': -math.inf, 'best_step': 0, 'bad_evaluations': 0}
    last_validation = {}

    def checkpoint(step):
        return {'objective_version': OBJECTIVE_VERSION, 'state_dict': policy.model.state_dict(),
                'optimizer': optimizer.state_dict(), 'step': step, 'manifest': manifest,
                'metrics': last_validation, 'stopper': dict(stopper),
                'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                        'torch': torch.get_rng_state(), 'loader': generator.get_state(),
                        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
                'resume_supported': False}

    def validate(step, verify_cache=False):
        nonlocal last_validation
        last_validation = evaluate_adapter(policy, valid, args.device, args.eval_limit)
        audit = preference_audit(policy, reference, dataset, args.beta, args.audit_images,
                                 seed=args.seed + 10000, verify_cache=verify_cache)
        score = last_validation['mAP50_95']
        if not math.isfinite(score):
            raise ValueError('Nonfinite validation AP')
        improved = score > stopper['best'] + 1e-4
        if improved:
            stopper.update(best=score, best_step=step, bad_evaluations=0)
            atomic_checkpoint(output / 'best.pt', checkpoint(step))
        else:
            stopper['bad_evaluations'] += 1
        record = {'event': 'validation', 'step': step, 'metrics': last_validation,
                  'train_preference_audit': audit, 'best': improved}
        append_jsonl(metrics_path, record)
        print(json.dumps(record), flush=True)
        return stopper['bad_evaluations'] >= args.patience

    started = time.perf_counter()
    validate(0, verify_cache=True)  # S0 is always an eligible best checkpoint.
    iterator = iter(loader)
    step = 0
    clipped_updates = 0
    for step in range(1, args.steps + 1):
        try:
            images, targets, records = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets, records = next(iterator)
        images = images.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        native = policy.native_loss(images, targets)
        distributions, _, _, _ = policy.distributions(images, dataset.policy_config)
        with torch.no_grad():
            reference_distributions, _, _, _ = reference.distributions(images, dataset.policy_config)
        preference, diagnostic = pair_losses(distributions, reference_distributions, records, args.beta)
        if step == 1 and diagnostic['relative_margin_max_abs'] > 1e-5:
            raise ValueError('Initial policy/reference mismatch despite same-batch scoring; inspect numerical precision')
        total = native + args.dpo_weight * preference
        if not torch.isfinite(total):
            raise ValueError('Nonfinite total loss')
        if step == 1 or step % args.gradient_interval == 0:
            diagnostic.update(gradient_diagnostics(native, preference, parameters, args.dpo_weight))
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip, error_if_nonfinite=True)
        clipped_updates += int(float(norm) > args.grad_clip)
        optimizer.step()
        if step == 1 or step % args.log_interval == 0:
            record = {'event': 'train', 'step': step, 'native_loss': float(native.detach()),
                      'dpo_loss': float(preference.detach()), 'total_loss': float(total.detach()),
                      'grad_norm_before_clip': float(norm), 'clipped_update_fraction': clipped_updates / step,
                      **diagnostic}
            append_jsonl(metrics_path, record)
            print(json.dumps(record), flush=True)
        if step % args.eval_interval == 0 or step == args.steps:
            if validate(step):
                break
    if tensor_digest(reference.model) != reference_digest:
        raise RuntimeError('Reference parameters/buffers changed')
    atomic_checkpoint(output / 'last.pt', checkpoint(step))
    atomic_json(output / 'completed.json', {
        'status': 'complete', 'pilot': pilot, 'steps': step,
        'termination': 'budget' if step == args.steps else 'early_stop',
        'best_step': stopper['best_step'], 'best_mAP50_95': stopper['best'],
        'best_sha256': sha256(output / 'best.pt'), 'last_sha256': sha256(output / 'last.pt'),
        'manifest_sha256': sha256(output / 'manifest.json'),
        'reference_immutable': True, 'wall_seconds': time.perf_counter() - started,
        'cuda_peak_allocated_bytes': (torch.cuda.max_memory_allocated(policy.device)
                                      if policy.device.type == 'cuda' else 0)})
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=SUPPORTED_MODELS, default='yolov8n')
    parser.add_argument('--preferences', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--val-batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--dpo-weight', type=float, default=.01)
    parser.add_argument('--beta', type=float, default=.1)
    parser.add_argument('--grad-clip', type=float, default=1.)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--gradient-interval', type=int, default=100)
    parser.add_argument('--audit-images', type=int, default=8)
    parser.add_argument('--eval-limit', type=int, default=0, help='partial validation; marks run as PILOT')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(argv)


if __name__ == '__main__':
    train(parse_args())
