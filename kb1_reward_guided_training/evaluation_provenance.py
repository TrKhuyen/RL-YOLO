"""Fail-closed input validation for canonical KB1 evaluation."""
import json
import math
import yaml
from pathlib import Path

from run_provenance import (assert_dataset_identity, dataset_manifest,
                            sha256, verify_supervised)


def inspect_checkpoint(model, checkpoint, source=None, expected_seed=None, expected_mode=None):
    """Validate completed artifacts before any detector is instantiated."""
    checkpoint = Path(checkpoint).resolve()
    source = Path(source or checkpoint).resolve()
    current = dataset_manifest()
    supervised = verify_supervised(model, source, current)
    record = {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
              'source': str(source), 'source_sha256': sha256(source),
              'source_manifest': supervised, 'dataset': current}
    seed_config = next((source.parent.parent / name for name in ('args.yaml', 'opt.yaml')
                        if (source.parent.parent / name).is_file()), None)
    record['supervised_seed'] = None
    if seed_config is not None:
        record['supervised_seed'] = yaml.safe_load(seed_config.read_text(encoding='utf-8')).get('seed')
        record['supervised_seed_config'] = {'path': str(seed_config), 'sha256': sha256(seed_config)}
    if checkpoint == source:
        return record, {}
    import torch
    folder = checkpoint.parent
    manifest_path, completed_path = folder / 'manifest.json', folder / 'completed.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    completed = json.loads(completed_path.read_text(encoding='utf-8'))
    final = checkpoint.name == 'last.pt'
    if final and (not manifest['config'].get('fixed_budget') or completed.get('termination') != 'budget'):
        raise ValueError('Final checkpoint comparison requires a completed fixed-budget run')
    if final and not completed.get('last_sha256'):
        raise ValueError('Final checkpoint hash missing')
    hash_key = 'last_sha256' if final else 'best_sha256'
    if completed.get(hash_key, record['checkpoint_sha256']) != record['checkpoint_sha256']:
        raise ValueError('Completed best checkpoint hash mismatch')
    if completed.get('manifest_sha256', sha256(manifest_path)) != sha256(manifest_path):
        raise ValueError('Completed manifest hash mismatch')
    if completed.get('status') != 'complete':
        raise ValueError(f'Incomplete training run: {folder}')
    if manifest['model'] != model or manifest['source_sha256'] != record['source_sha256']:
        raise ValueError('Guided source/model mismatch')
    if expected_seed is not None and manifest['seed'] != expected_seed:
        raise ValueError('Continuation seed mismatch')
    if expected_mode is not None and manifest['mode'] != expected_mode:
        raise ValueError('Continuation mode mismatch')
    assert_dataset_identity(manifest['source_manifest']['dataset'], current)
    if manifest['source_manifest'] != supervised:
        raise ValueError('Embedded supervised provenance differs from source manifest')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('metadata') != manifest:
        raise ValueError('Checkpoint metadata differs from run manifest')
    selected_step = completed['steps_run'] if final else completed['best_step']
    selected_metrics = completed['last'] if final else completed['best']
    if payload.get('step') != selected_step:
        raise ValueError('Checkpoint is not the completed validation-selected best')
    for key in ('mAP50', 'mAP50_95', 'AR300'):
        a, b = float(payload['metrics'][key]), float(selected_metrics[key])
        if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > 1e-6:
            raise ValueError(f'Best checkpoint metric mismatch: {key}')
    protocol = manifest['protocol']
    for key, value in {'conf': .001, 'nms_iou': .60, 'max_det': 300,
                       'operating_conf': .25, 'operating_iou': .5,
                       'image_size': 640, 'score': 'mAP50_95'}.items():
        if protocol.get(key) != value:
            raise ValueError(f'Unsupported training evaluation protocol: {key}')
    record.update(seed=manifest['seed'], mode=manifest['mode'], step=payload['step'],
                  run_manifest=manifest, run_manifest_sha256=sha256(manifest_path),
                  completed_sha256=sha256(completed_path))
    return record, payload
