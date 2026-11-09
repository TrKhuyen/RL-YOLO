"""TPE-initialized fork of the verified strength trainer, with the same updates.

Only source/provenance, bank binding and baseline labels differ. Historical
trainer files remain byte-identical so existing experiments stay replayable.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..artifacts import (append_jsonl, atomic_checkpoint, atomic_json,
                         seed_everything, sha256, tensor_digest)
from .source import source_checkpoint, source_provenance, assert_provenance
from ..data import collate, detection_loader
from ..train_dpo import gradient_diagnostics
from ..conditional import VERSION, code_fingerprints as conditional_fingerprints
from ..conditional_bg import LOSS_VERSION, code_fingerprints as background_fingerprints
from . import EXPERIMENT_VERSION, code_fingerprints
from ..conditional_strength.selection import select_pareto
from ..conditional.bank import sampling_probabilities
from .bank import Bank
from ..conditional.evaluation import choose_probes, eligible, evaluate, probe
from ..conditional.policy import ConditionalAdapter, configure_numerics, pair_loss
from ..conditional_bg.loss import BackgroundAdapter


def train(bank_path, output, arm='dpo', device='cuda', steps=2000, seed=42,
          eval_interval=1000, eval_limit=0, beta=.1, dpo_weight=.01, lr=1e-6,
          fp_tolerance=.05, ap_tolerance=.001, audit_pairs=32, background_weight=.5):
    if (arm not in ('native', 'dpo', 'bce') or min(steps, eval_interval, audit_pairs) < 1
            or eval_limit < 0 or not all(math.isfinite(v) for v in
                (beta, dpo_weight, lr, fp_tolerance, ap_tolerance, background_weight))
            or min(beta, dpo_weight, lr) <= 0 or min(fp_tolerance, ap_tolerance) < 0):
        raise ValueError('Invalid training configuration')
    if not 0 <= background_weight <= 1:
        raise ValueError('background_weight must be in [0, 1]')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a fresh output directory: {output}')
    configure_numerics()
    seed_everything(seed)
    bank = Bank(bank_path)
    provenance = bank.manifest['provenance']
    fingerprints = code_fingerprints()
    conditional_code = conditional_fingerprints()
    background_code = background_fingerprints()
    records = [r for r in bank.records if not r['probe']]
    weights, sampler_config = sampling_probabilities(records)
    pair_rng = torch.Generator().manual_seed(seed + 10000)
    # One fixed plan for every arm. No validation result can affect exposure.
    pair_plan = torch.multinomial(weights, 2 * steps, replacement=True, generator=pair_rng).reshape(steps, 2)
    native_loader = DataLoader(bank.dataset, batch_size=2, shuffle=True, drop_last=True,
                               collate_fn=collate, num_workers=0,
                               generator=torch.Generator().manual_seed(seed + 20000))
    native_iter = iter(native_loader)
    policy = BackgroundAdapter(source_checkpoint('yolov8n'), device, background_weight)
    reference = ConditionalAdapter(source_checkpoint('yolov8n'), device)
    reference.model.requires_grad_(False)
    reference_digest = tensor_digest(reference.model)
    if reference_digest != bank.manifest['reference_state_sha256']:
        raise ValueError('S0/reference differs from pair bank')
    frozen = policy.frozen_digest()
    params = [p for p in policy.model.parameters() if p.requires_grad]
    names = [n for n, p in policy.model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=5e-4)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'objective_version': VERSION, 'provenance': provenance,
                'conditional_code_sha256': conditional_code,
                'native_background_code_sha256': background_code, 'native_loss_version': LOSS_VERSION,
                'experiment_version': EXPERIMENT_VERSION, 'strength_code_sha256': fingerprints,
                'bank_path': str(bank.root),
                'bank_manifest_sha256': sha256(bank.root / 'manifest.json'),
                'pairs_sha256': bank.manifest['pairs_sha256'], 'trainable_names': names,
                'reference_state_sha256': reference_digest, 'frozen_state_sha256': frozen,
                'config': {'arm': arm, 'device': device, 'steps': steps, 'seed': seed,
                           'effective_dpo_weight': 0. if arm == 'native' else dpo_weight,
                           'native_background_weight': background_weight,
                           'native_foreground_weight': 1., 'native_denominator': 'original_target_score_sum',
                           'input': 640, 'native_batch': 4, 'preference_batch': 2,
                           'native_stream': '2 uniform full-train + 2 sampled preference images',
                           'beta': beta, 'dpo_weight': dpo_weight, 'lr': lr, 'weight_decay': 5e-4,
                           'clip_norm': 1., 'eval_interval': eval_interval, 'eval_limit': eval_limit,
                           'partial_bank': bank.manifest['partial_bank'], 'sampler': sampler_config,
                           'selection': 'max mAP50_95, min_delta=1e-4; TPE eligible; no recall gate',
                           'all_evaluation_checkpoints_saved': True,
                           'pareto_metrics': ['mAP50_95', 'precision', 'operating_recall'],
                           'fp_tolerance': fp_tolerance, 'ap_tolerance': ap_tolerance,
                           'selection_min_delta': 1e-4, 'TPE_eligible': True,
                           'pilot': True, 'resume_supported': False,
                           'precision': 'float32_tf32_off_same_batch_reference'}}
    atomic_json(output / 'manifest.json', manifest)
    initial_probes = choose_probes(bank.records, audit_pairs)
    atomic_json(output / 'replay.json', bank.replay(reference, initial_probes))
    probe_sets = {name: choose_probes([r for r in bank.records if r['probe'] == heldout], audit_pairs)
                  for name, heldout in (('dpo_train', False), ('dpo_unseen_train', True))}
    valid = detection_loader('valid', 640, 8)
    baseline_metrics, source_predictions = evaluate(reference, valid, image_limit=eval_limit)
    baseline_metrics['checkpoint_selection'] = 'mAP50_95_min_delta_1e-4_TPE_eligible_all_eval_weights_saved'
    atomic_json(output / 'source_validation.json', baseline_metrics)
    best_score, best_step, last_metrics = baseline_metrics['mAP50_95'], 0, baseline_metrics
    selection_history = []
    stream_digest = hashlib.sha256()
    image_counts, pair_counts, group_counts, class_counts, target_counts = (Counter() for _ in range(5))
    started = time.perf_counter()

    def checkpoint(step):
        return {'objective_version': VERSION, 'state_dict': policy.model.state_dict(),
                'optimizer': optimizer.state_dict(), 'step': step, 'manifest': manifest,
                'metrics': last_metrics, 'best_step': best_step, 'resume_supported': False,
                'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                        'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available() else []}}

    # TPE remains an eligible AP baseline; save each evaluation for multimetric review.
    atomic_checkpoint(output / 'best.pt', checkpoint(0))

    def validate(step):
        nonlocal best_score, best_step, last_metrics
        if step:
            last_metrics, _ = evaluate(policy, valid, source_predictions, eval_limit)
            last_metrics['checkpoint_selection'] = baseline_metrics['checkpoint_selection']
        diagnostics = {key: probe(policy, reference, bank, value, beta) for key, value in probe_sets.items()}
        if policy.frozen_digest() != frozen:
            raise ValueError('Regression/backbone/BN state changed')
        if max(d['raw_box_max_abs_delta'] for d in diagnostics.values()) > .002:
            raise ValueError('Raw geometry drift despite frozen regression')
        for key in ('mAP50_95', 'operating_recall', 'macro_recall'):
            if not math.isfinite(last_metrics[key]):
                raise ValueError(f'Nonfinite validation metric: {key}')
        legacy_eligible = eligible(last_metrics, baseline_metrics, fp_tolerance, ap_tolerance)
        accepted = True
        improved = last_metrics['mAP50_95'] > best_score + 1e-4
        if improved:
            best_score, best_step = last_metrics['mAP50_95'], step
            atomic_checkpoint(output / 'best.pt', checkpoint(step))
        atomic_checkpoint(output / f'step_{step:06d}.pt', checkpoint(step))
        selection_history.append({'step': step, 'metrics': last_metrics})
        atomic_json(output / 'selection.json', {
            'best_AP_step': best_step, 'best_AP_score': best_score, 'min_delta': 1e-4,
            'pareto_metrics': ['mAP50_95', 'precision', 'operating_recall'],
            'pareto_steps': select_pareto(selection_history),
            'all_saved_steps': [r['step'] for r in selection_history],
            'rule': 'Pareto set is diagnostic; no arbitrary weighted metric score'})
        entry = {'event': 'validation', 'step': step, 'metrics': last_metrics,
                 'legacy_recall_gate_eligible': legacy_eligible,
                 'eligible': accepted, 'selected_best': improved or step == 0, 'probes': diagnostics}
        append_jsonl(output / 'metrics.jsonl', entry)
        print(json.dumps({'arm': arm, 'step': step, 'eligible': accepted, 'best_step': best_step,
                          **{k: last_metrics[k] for k in ('mAP50_95', 'macro_recall', 'fp', 'fn')}}), flush=True)

    validate(0)
    for step in range(1, steps + 1):
        try:
            images, targets = next(native_iter)
        except StopIteration:
            native_iter = iter(native_loader)
            images, targets = next(native_iter)
        selected = [records[i] for i in pair_plan[step - 1].tolist()]
        pair_data = [bank.dataset[r['image_id']] for r in selected]
        images = torch.cat((images, torch.stack([item[0] for item in pair_data]))).to(policy.device)
        targets += [item[1] for item in pair_data]
        batch_records = [{**r, 'batch_index': i + 2} for i, r in enumerate(selected)]
        stream = {'native_image_ids': [t['image_id'] for t in targets],
                  'pair_ids': [r['pair_id'] for r in selected]}
        stream_digest.update(json.dumps(stream, sort_keys=True).encode())
        append_jsonl(output / 'stream.jsonl', {'step': step, **stream})
        for r in selected:
            image_counts[r['image_id']] += 1
            pair_counts[r['pair_id']] += 1
            group_counts[r['group']] += 1
            class_counts[r['class_id']] += 1
            target_counts[f"{r['image_id']}:{r['target_gt'] if r['target_gt'] is not None else 'slot'+str(r['slot'])}"] += 1
        optimizer.zero_grad(set_to_none=True)
        native = policy.native_loss(images, targets)
        logits, boxes = policy.raw(images)
        with torch.no_grad():
            ref_logits, ref_boxes = reference.raw(images)
        geometry_delta = float((boxes.detach() - ref_boxes).abs().max())
        if geometry_delta > .002:
            raise ValueError('Policy/reference geometry mismatch')
        dpo, margin, z, z0 = pair_loss(logits, ref_logits, batch_records, beta)
        if step == 1 and float(margin.detach().abs().max()) > 1e-6:
            raise ValueError('Initial policy/reference margin is nonzero')
        if step == 1:
            for r in batch_records:
                if not torch.allclose(boxes[r['batch_index'], r['slot']].detach().cpu(),
                                      torch.tensor(r['box']), atol=.002, rtol=0):
                    raise ValueError('Cached context box differs under training batch')
        labels = z.new_tensor([float(r['chosen_keep']) for r in selected])
        local_bce = F.binary_cross_entropy_with_logits(z, labels)
        auxiliary = dpo if arm != 'bce' else local_bce
        strength = 0. if arm == 'native' else dpo_weight
        diagnostic = {}
        if step == 1 or step % 100 == 0:
            diagnostic = gradient_diagnostics(native, auxiliary, params, strength)
        total = native + strength * auxiliary
        if not torch.isfinite(total):
            raise ValueError('Nonfinite loss')
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            entry = {'event': 'train', 'step': step, 'native_loss': float(native.detach()),
                     'dpo_loss': float(dpo.detach()), 'local_bce_loss': float(local_bce.detach()),
                     'mean_margin': float(margin.detach().mean()),
                     'clip_input_norm': float(norm), 'clipped': float(norm) > 1.,
                     'raw_box_max_abs_delta': geometry_delta, 'gradient': diagnostic,
                     'native_partition': policy.native_parts()}
            append_jsonl(output / 'metrics.jsonl', entry)
            print(json.dumps({'arm': arm, **entry}), flush=True)
        elif diagnostic:
            append_jsonl(output / 'metrics.jsonl', {'event': 'gradient', 'step': step, **diagnostic})
        if step % eval_interval == 0 or step == steps:
            validate(step)
        # Release both raw-output graphs before allocating the next native batch.
        del native, logits, boxes, ref_logits, ref_boxes, dpo, margin, z, z0, local_bce, auxiliary, total
    atomic_checkpoint(output / 'last.pt', checkpoint(steps))
    if tensor_digest(reference.model) != reference_digest or policy.frozen_digest() != frozen:
        raise ValueError('Immutable model state changed')
    assert_provenance(provenance, source_provenance('yolov8n'))
    if (code_fingerprints() != fingerprints or conditional_fingerprints() != conditional_code
            or background_fingerprints() != background_code):
        raise ValueError('Code changed during training')
    result = {'objective_version': VERSION, 'experiment_version': EXPERIMENT_VERSION,
              'selection': json.loads((output / 'selection.json').read_text()),
              'evaluation_checkpoint_sha256': {str(r['step']): sha256(output / f"step_{r['step']:06d}.pt")
                                               for r in selection_history},
              'native_loss_version': LOSS_VERSION,
              'native_background_weight': background_weight, 'arm': arm, 'steps': steps, 'best_step': best_step,
              'stream_sha256': stream_digest.hexdigest(), 'best_sha256': sha256(output / 'best.pt'),
              'last_sha256': sha256(output / 'last.pt'), 'reference_unchanged': True,
              'frozen_state_unchanged': True, 'wall_seconds': time.perf_counter() - started,
              'actual_exposure': {'group': dict(group_counts), 'class': dict(class_counts),
                                  'image': dict(image_counts), 'pair': dict(pair_counts), 'target': dict(target_counts)},
              'last_metrics': last_metrics, 'partial_evaluation': bool(eval_limit),
              'partial_bank': bank.manifest['partial_bank']}
    atomic_json(output / 'completed.json', result)
    return result
