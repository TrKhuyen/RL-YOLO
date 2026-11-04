"""Full-train opportunity audit for choosing tighter existing boxes; no training."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from kb2_preference_optimization.artifacts import (
    append_jsonl, assert_provenance, atomic_json, sha256, source_checkpoint, source_provenance, tensor_digest,
)
from kb2_preference_optimization.candidate_rank import VERSION, code_fingerprints
from kb2_preference_optimization.candidate_rank.pairs import RankConfig, build_rank_context
from kb2_preference_optimization.conditional.bank import Bank, probe_image
from kb2_preference_optimization.conditional.pairs import matches
from kb2_preference_optimization.conditional.policy import ConditionalAdapter, configure_numerics
from kb2_preference_optimization.data import DetectionDataset, collate


@torch.no_grad()
def audit(output, device='cpu', limit=0):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    provenance, fingerprint = source_provenance('yolov8n'), code_fingerprints()
    source_hash = sha256(source_checkpoint('yolov8n'))
    bank = Bank('kb2_preference_optimization/outputs/yolov8n_conditional_bank_20261031_v1/preferences')
    assert not bank.manifest['partial_bank'] and bank.manifest['processed_images'] == 2411
    assert bank.config.operating_conf == .25 and bank.config.nms_iou == .6 and bank.config.max_det == 300
    digest = bank.manifest['reference_state_sha256']
    dataset = bank.dataset
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate, num_workers=0)
    config = RankConfig()
    totals, classes, density = Counter(), defaultdict(Counter), defaultdict(Counter)
    (output/'pairs.jsonl').touch()
    (output/'contexts.jsonl').touch()
    seen = pair_id = 0
    for images, targets in loader:
        if limit and seen >= limit:
            break
        if limit:
            images, targets = images[:limit-seen], targets[:limit-seen]
        for b, target in enumerate(targets):
            source_context = bank.contexts[target['image_id']]
            assert source_context['image_path'] == target['image_path']
            context, pairs, diagnostics, gt_rows = build_rank_context(source_context, target, config)
            d = '6+' if len(target['labels']) >= 6 else ('3-5' if len(target['labels']) >= 3 else '1-2')
            totals.update(images=1, pair_images=int(bool(pairs)), pairs=len(pairs),
                          **{k: v for k, v in diagnostics.items() if k != 's0_gt'})
            overlaps = box_iou(target['boxes'], target['boxes'])
            overlaps.fill_diagonal_(0)
            if pairs:
                context.update(image_id=target['image_id'], image_path=target['image_path'])
                append_jsonl(output/'contexts.jsonl', context)
            for row in gt_rows:
                g = row['target_gt']
                row.update(image_id=target['image_id'], image_path=target['image_path'], density=d,
                           probe=probe_image(target['image_path']), max_GT_overlap=float(overlaps[g].max()))
                append_jsonl(output/'gt_audit.jsonl', row)
                classes[str(row['class_id'])].update(gt=1, s0_detected=int(row['s0_detected']),
                                                    **{row['status']: 1})
            density[d].update(images=1, gt=len(target['labels']), pairs=len(pairs), pair_images=int(bool(pairs)))
            for pair in pairs:
                pair.update(pair_id=pair_id, image_id=target['image_id'], image_path=target['image_path'],
                            probe=probe_image(target['image_path']))
                append_jsonl(output/'pairs.jsonl', pair)
                totals.update(train_pairs=int(not pair['probe']), probe_pairs=int(pair['probe']))
                pair_id += 1
            seen += 1
        if seen % 100 == 0:
            print(json.dumps({'event': 'candidate_rank_audit', 'images': seen, 'pairs': pair_id}), flush=True)
    assert sha256(source_checkpoint('yolov8n')) == source_hash
    assert_provenance(provenance, source_provenance('yolov8n'))
    assert fingerprint == code_fingerprints()
    for c in range(dataset.classes):
        classes[str(c)].update(gt=0, accepted=0)
    result = {'version': VERSION, 'config': config.manifest(), 'provenance': provenance,
              'code_sha256': fingerprint, 'script_sha256': sha256(Path(__file__)),
              'reference_state_sha256': digest, 'generation_batch': bank.manifest['numerics']['generation_batch'],
              'source_bank': str(bank.root), 'source_context_sha256': bank.manifest['contexts_sha256'],
              'source_manifest_sha256': sha256(bank.root/'manifest.json'),
              'method': 'Reuse complete verified S0 pre-NMS contexts, CPU rejudge against unchanged train GT',
              'totals': dict(totals), 'per_class': {k: dict(v) for k, v in classes.items()},
              'density': {k: dict(v) for k, v in density.items()},
              'pairs_sha256': sha256(output/'pairs.jsonl'), 'contexts_sha256': sha256(output/'contexts.jsonl'),
              'partial': seen != len(dataset), 'reference_unchanged': True,
              'scope': 'Train only; existing model boxes and scores; no GT box injection, no training or test'}
    atomic_json(output/'completed.json', result)
    print(json.dumps({'event': 'audit_completed', 'totals': result['totals'], 'output': str(output)}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu'], default='cpu', help='Reuses verified contexts; no GPU inference required')
    parser.add_argument('--limit-images', type=int, default=0)
    args = parser.parse_args()
    audit(args.output, args.device, args.limit_images)
