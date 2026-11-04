"""Paired fresh AdamW steps from S0 at exactly the original pilot batches."""
from collections import Counter

import torch

from ..artifacts import append_jsonl, atomic_json, source_checkpoint, tensor_digest
from ..conditional.policy import ConditionalAdapter, pair_loss, configure_numerics
from ..conditional.pairs import matches
from ..conditional_bg.loss import BackgroundAdapter
from ..train_dpo import gradient_diagnostics


def audit(bank, stream, output, device='cuda', batches=32):
    configure_numerics()
    policy = BackgroundAdapter(source_checkpoint('yolov8n'), device, .5)
    reference = ConditionalAdapter(source_checkpoint('yolov8n'), device)
    reference.model.requires_grad_(False)
    before_digest = tensor_digest(policy.model)
    snapshot = {k: v.detach().cpu().clone() for k, v in policy.model.state_dict().items()}
    records = {r['pair_id']: r for r in bank.records}
    params = [p for p in policy.model.parameters() if p.requires_grad]
    totals = {str(weight): Counter() for weight in (0., .01, .03, .1)}
    for entry in stream[:batches]:
        data = [bank.dataset[i] for i in entry['native_image_ids']]
        images = torch.stack([x[0] for x in data]).to(device)
        targets = [x[1] for x in data]
        selected = [{**records[i], 'batch_index': b + 2} for b, i in enumerate(entry['pair_ids'])]
        with torch.no_grad():
            ref_logits, _ = reference.raw(images)
            ordinary = reference.predict(images)
            initial = [matches(p, t, .25) for p, t in zip(ordinary, targets)]
        for weight in (0., .01, .03, .1):
            policy.model.load_state_dict(snapshot, strict=True)
            optimizer = torch.optim.AdamW(params, lr=1e-6, weight_decay=5e-4)
            optimizer.zero_grad(set_to_none=True)
            native = policy.native_loss(images, targets)
            logits, _ = policy.raw(images)
            loss, margin, z, _ = pair_loss(logits, ref_logits, selected, .1)
            assert float(margin.detach().abs().max()) < 1e-6
            diagnostic = gradient_diagnostics(native, loss, params, weight)
            old_scores = z.detach().sigmoid().cpu()
            (native + weight * loss).backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                after_logits, _ = policy.raw(images)
                ordinary = policy.predict(images)
                after = [matches(p, t, .25) for p, t in zip(ordinary, targets)]
            for i, r in enumerate(selected):
                score = float(after_logits[r['batch_index'], r['slot'], r['class_id']].sigmoid())
                old_score = float(old_scores[i])
                totals[str(weight)][r['group'] + '_decisions'] += 1
                totals[str(weight)][r['group'] + '_score_increased'] += int(score > old_score)
                append_jsonl(output / 'step_decisions.jsonl', {
                    'batch_step': entry['step'], 'lambda': weight, 'pair_id': r['pair_id'],
                    'group': r['group'], 'image_id': r['image_id'], 'target_gt': r['target_gt'],
                    'score_before': old_score, 'score_after': score, 'delta': score-old_score,
                    'desired_direction': 'up' if r['chosen_keep'] else 'down'})
            lost = sum(len(a[0]-b[0]) for a, b in zip(initial, after))
            rescued = sum(len(b[0]-a[0]) for a, b in zip(initial, after))
            fp_delta = sum(b[1]-a[1] for a, b in zip(initial, after))
            totals[str(weight)].update(lost_GT=lost, rescued_GT=rescued, FP_delta=fp_delta)
            append_jsonl(output / 'step_batches.jsonl', {
                'batch_step': entry['step'], 'lambda': weight, 'gradient': diagnostic,
                'clip_input_norm': float(norm), 'lost_GT': lost, 'rescued_GT': rescued, 'FP_delta': fp_delta})
            del native, logits, loss, margin, z, optimizer
        if entry['step'] % 8 == 0:
            print({'event': 'paired_step_audit', 'batches': entry['step']}, flush=True)
    policy.model.load_state_dict(snapshot, strict=True)
    assert tensor_digest(policy.model) == before_digest == tensor_digest(reference.model)
    result = {'batches': batches, 'summary': {k: dict(v) for k, v in totals.items()},
              'scope': 'Repeated independent first AdamW steps at S0; train-batch mechanistic audit, not validation',
              'model_state_restored': True}
    atomic_json(output / 'step_audit.json', result)
    return result
