"""Controlled pilot: fixed full-train native stream, separate DPO sampler."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .. import OBJECTIVE_VERSION
from ..adapter import DFLAdapter
from ..artifacts import (append_jsonl, assert_provenance, atomic_checkpoint, atomic_json,
                         seed_everything, sha256, source_checkpoint, source_provenance, tensor_digest)
from ..data import DetectionDataset, collate, detection_loader
from ..metrics import evaluate_adapter
from ..preferences import PreferenceDataset, collate_preferences
from ..train_dpo import pair_losses, preference_audit, gradient_diagnostics
from . import VERSION, code_fingerprints
from .diagnostics import choose_probes, probe_audit
from .selection import balanced_probabilities


def train(root, arm, device='cuda', steps=2000):
    root = Path(root).resolve()
    output = root / f'pilot_{arm}'
    output.mkdir(exist_ok=False)
    seed_everything(42)
    provenance = source_provenance('yolov8n')
    extra_code = code_fingerprints()
    audit = json.loads((root / 'audit.json').read_text())
    assert_provenance(audit['provenance'], provenance)
    assert audit['recovery_code_sha256'] == extra_code
    selector = 'control' if arm == 'control' else 'selected'
    dataset = PreferenceDataset(root / selector, provenance)
    assert dataset.manifest['recovery_code_sha256'] == extra_code
    training_indices = [i for i, r in enumerate(dataset.records) if not r['probe']]
    records = [dataset.records[i] for i in training_indices]
    native_data = DetectionDataset('train', 640)
    native_loader = DataLoader(native_data, batch_size=4, shuffle=True, num_workers=0,
                               collate_fn=collate, generator=torch.Generator().manual_seed(20042))
    dpo_generator = torch.Generator().manual_seed(42)
    sampler_config = {'type': 'shuffle_without_replacement'}
    if arm == 'balanced':
        weights, sampler_config = balanced_probabilities(records)
        sampler_config['type'] = 'bounded_recovery_class_mixture'
        sampler = WeightedRandomSampler(weights, len(records), replacement=True,
                                         generator=dpo_generator)
        pref_loader = DataLoader(Subset(dataset, training_indices), batch_size=4, sampler=sampler,
                                 num_workers=0, collate_fn=collate_preferences,
                                 generator=torch.Generator().manual_seed(30042))
    else:
        pref_loader = DataLoader(Subset(dataset, training_indices), batch_size=4, shuffle=True,
                                 num_workers=0, collate_fn=collate_preferences, generator=dpo_generator)
    selected_dataset = PreferenceDataset(root / 'selected', provenance)
    probes = {name: choose_probes(selected_dataset.records, heldout, 64)
              for name, heldout in (('dpo_train_images', False), ('dpo_unseen_images', True))}
    policy = DFLAdapter(source_checkpoint('yolov8n'), device)
    reference = DFLAdapter(source_checkpoint('yolov8n'), device)
    reference.model.requires_grad_(False)
    reference_digest = tensor_digest(reference.model)
    assert reference_digest == dataset.manifest['reference_state_sha256']
    params = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-6, weight_decay=5e-4)
    manifest = {'provenance': provenance, 'recovery_code_sha256': extra_code,
                'reference_state_sha256': reference_digest,
                'config': {'arm': arm, 'version': VERSION, 'pilot': True, 'seed': 42,
                           'steps': steps, 'image_size': 640, 'batch_size': 4,
                           'native_stream': 'full_train_independent_seed20042',
                           'dpo_pairs': len(records), 'sampler': sampler_config,
                           'native_weight': 1., 'dpo_weight': .01, 'beta': .1,
                           'lr': 1e-6, 'weight_decay': 5e-4, 'eval_interval': 1000,
                           'pairs_sha256': dataset.manifest['pairs_sha256'],
                           'probe_ids': {k: [r['image_id'] for r in v] for k, v in probes.items()}}}
    atomic_json(output / 'manifest.json', manifest)
    valid = detection_loader('valid', 640, 8)
    best, best_step, last_validation = -math.inf, 0, {}

    def checkpoint(step):
        return {'objective_version': OBJECTIVE_VERSION, 'state_dict': policy.model.state_dict(),
                'optimizer': optimizer.state_dict(), 'step': step, 'manifest': manifest,
                'metrics': last_validation, 'resume_supported': False}

    def validate(step):
        nonlocal best, best_step, last_validation
        last_validation = evaluate_adapter(policy, valid, device)
        diagnostic = {k: probe_audit(policy, reference, native_data, v) for k, v in probes.items()}
        score = last_validation['mAP50_95']
        if not math.isfinite(score):
            raise ValueError('Nonfinite validation score')
        improved = score > best + 1e-4
        if improved:
            best, best_step = score, step
            atomic_checkpoint(output / 'best.pt', checkpoint(step))
        append_jsonl(output / 'metrics.jsonl', {'event': 'validation', 'step': step,
                                              'metrics': last_validation, 'probes': diagnostic,
                                              'best': improved})
        print(json.dumps({'arm': arm, 'step': step, 'best': improved,
                          **{k: last_validation[k] for k in ('mAP50_95', 'operating_recall', 'fp', 'fn')},
                          'probe': {k: {f: v[f] for f in ('relative_margin', 'dpo_loss',
                                                         'rescued_recall', 'rescued_mean_true_class_score')}
                                    for k, v in diagnostic.items()}}), flush=True)

    started = time.perf_counter()
    # Replay/cache check on actual generated records before the first update.
    replay = preference_audit(policy, reference, dataset, .1, count=8, verify_cache=True)
    atomic_json(output / 'replay_check.json', replay)
    validate(0)
    native_iterator, preference_iterator = iter(native_loader), iter(pref_loader)
    native_digest = hashlib.sha256()
    exposure, group_counts, image_counts = Counter(), Counter(), Counter()
    for step in range(1, steps+1):
        try:
            native_images, native_targets = next(native_iterator)
        except StopIteration:
            native_iterator = iter(native_loader)
            native_images, native_targets = next(native_iterator)
        try:
            images, targets, batch_records = next(preference_iterator)
        except StopIteration:
            preference_iterator = iter(pref_loader)
            images, targets, batch_records = next(preference_iterator)
        native_ids = [t['image_id'] for t in native_targets]
        native_digest.update((json.dumps(native_ids)+'\n').encode())
        for r in batch_records:
            group_counts[r['selection']['group']] += 1
            image_counts[r['image_id']] += 1
            for c in r['selection']['rescued_classes']:
                exposure[c] += 1
        optimizer.zero_grad(set_to_none=True)
        native = policy.native_loss(native_images.to(device), native_targets)
        pd, _, _, _ = policy.distributions(images.to(device), dataset.policy_config)
        with torch.no_grad():
            rd, _, _, _ = reference.distributions(images.to(device), dataset.policy_config)
        preference, diagnostic = pair_losses(pd, rd, batch_records, .1)
        if step == 1 and diagnostic['relative_margin_max_abs'] > 1e-5:
            raise ValueError('Initial same-batch policy/reference mismatch')
        loss = native + .01*preference
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        if step == 1 or step % 100 == 0:
            diagnostic.update(gradient_diagnostics(native, preference, params, .01))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            append_jsonl(output / 'metrics.jsonl', {'event': 'train', 'step': step,
                         'native_loss': float(native.detach()), 'dpo_loss': float(preference.detach()),
                         'grad_norm': float(norm), 'native_ids': native_ids, **diagnostic})
            if step == 1 or step % 500 == 0:
                print(json.dumps({'arm': arm, 'step': step, 'native_loss': float(native.detach()),
                                  'dpo_loss': float(preference.detach())}), flush=True)
        if step % 1000 == 0 or step == steps:
            validate(step)
    assert tensor_digest(reference.model) == reference_digest
    assert extra_code == code_fingerprints()
    assert_provenance(provenance, source_provenance('yolov8n'))
    atomic_checkpoint(output / 'last.pt', checkpoint(steps))
    atomic_json(output / 'exposure.json', {
        'groups': dict(group_counts), 'classes': dict(exposure),
        'unique_dpo_images': len(image_counts), 'image_draws': dict(image_counts),
        'native_sequence_sha256': native_digest.hexdigest()})
    atomic_json(output / 'completed.json', {
        'status': 'complete', 'pilot': True, 'steps': steps, 'best_step': best_step,
        'best_mAP50_95': best, 'reference_immutable': True,
        'best_sha256': sha256(output/'best.pt'), 'last_sha256': sha256(output/'last.pt'),
        'manifest_sha256': sha256(output/'manifest.json'),
        'native_sequence_sha256': native_digest.hexdigest(),
        'wall_seconds': time.perf_counter()-started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--arm', choices=('control', 'selector', 'balanced'), required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    train(args.root, args.arm, args.device, args.steps)
