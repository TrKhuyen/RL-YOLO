"""Resolve the completed KB3 winner by local trial ID and recorded SHA256.

Historical Windows paths remain untouched. This verifies the saved selection,
not the historical training environment or the accuracy of its stored metrics.
"""
import json
import math
from pathlib import Path

from ..artifacts import (REPO, assert_provenance as assert_base_provenance,
                         sha256, source_provenance as supervised_provenance)
from ..conditional import code_fingerprints as conditional_fingerprints
from ..conditional_bg import code_fingerprints as background_fingerprints
from ..conditional_strength import code_fingerprints as strength_fingerprints
from . import EXPERIMENT_VERSION, code_fingerprints

TPE_ROOT = (REPO / 'kb3_hyperparameter_optimization' /
            'checkpoint_hyperparameter_optimization/quality/tpe_kb1_pretrained_v2/yolov8n')


def resolve_tpe(root=TPE_ROOT):
    root = Path(root).resolve()
    evaluation = json.loads((root / 'evaluation.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = evaluation['trials']
    count = evaluation['requested_trials']
    if (evaluation['protocol'] != 'kb3_tpe_v2_kb1_pretrained'
            or manifest['protocol'] != evaluation['protocol']
            or manifest['model'] != 'yolov8n'
            or evaluation['metrics_split'] != 'validation'
            or count < 1 or evaluation['completed_trials'] != count or len(rows) != count
            or {r['trial'] for r in rows} != set(range(count))):
        raise ValueError('Expected a complete pretrained YOLOv8n TPE validation search')
    for row in rows:
        score = row['scores']['mAP50_95']
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid TPE validation score')
    winner = evaluation['best_trial']
    row = next(r for r in rows if r['trial'] == winner)
    if row['scores']['mAP50_95'] != max(r['scores']['mAP50_95'] for r in rows):
        raise ValueError('Recorded TPE winner is not a maximum validation score')
    trial = root / 'tpe' / f'trial_{winner:04d}'
    result = json.loads((trial / 'result.json').read_text())
    spec = json.loads((trial / 'trial_spec.json').read_text())
    checkpoint = trial / 'selected_best.pt'
    expected_sha = evaluation['best_checkpoint_sha256']
    if (result.get('failed') is not False or result['initialization'] != 'pretrained'
            or not result['pretrained_backbone_verified'] or spec['number'] != winner
            or spec['parameters'] != result['hyperparameters']
            or row['parameters'] != result['hyperparameters']
            or row['seed'] != result['seed'] or spec['seed'] != result['seed']
            or row['checkpoint_sha256'] != expected_sha
            or result['checkpoint_sha256'] != expected_sha
            or result['canonical_best_scores']['mAP50_95'] != row['scores']['mAP50_95']
            or sha256(checkpoint) != expected_sha):
        raise ValueError('TPE selected checkpoint/result/hash disagreement')
    return {'checkpoint': str(checkpoint), 'checkpoint_sha256': expected_sha,
            'trial': winner, 'historical_validation': row['scores'],
            'hyperparameters': result['hyperparameters'], 'seed': result['seed'],
            'selection': result['checkpoint_selection'],
            'dataset_sha256': manifest['settings']['kb1_reference']['dataset_sha256'],
            'evaluation_sha256': sha256(root / 'evaluation.json'),
            'manifest_sha256': sha256(root / 'manifest.json'),
            'result_sha256': sha256(trial / 'result.json'),
            'spec_sha256': sha256(trial / 'trial_spec.json')}


def source_checkpoint(model):
    if model != 'yolov8n':
        raise ValueError('Hybrid currently supports YOLOv8n only')
    return Path(resolve_tpe()['checkpoint'])


def source_provenance(model):
    if model != 'yolov8n':
        raise ValueError('Hybrid currently supports YOLOv8n only')
    baseline = supervised_provenance(model)
    source = resolve_tpe()
    if source['dataset_sha256'] != baseline['dataset']['sha256']:
        raise ValueError('TPE and current supervised dataset identities disagree')
    return {**baseline, 'source_checkpoint': source['checkpoint'],
            'source_sha256': source['checkpoint_sha256'], 'source_kind': 'kb3_tpe_winner',
            'source_tpe': source, 'hybrid_version': EXPERIMENT_VERSION,
            'shared_S0': {'checkpoint': baseline['source_checkpoint'],
                          'sha256': baseline['source_sha256']},
            'hybrid_code_sha256': code_fingerprints(),
            'conditional_code_sha256': conditional_fingerprints(),
            'background_code_sha256': background_fingerprints(),
            'strength_code_sha256': strength_fingerprints()}


def assert_provenance(recorded, current):
    assert_base_provenance(recorded, current)
    for key in ('source_kind', 'source_tpe', 'hybrid_version', 'shared_S0',
                'hybrid_code_sha256', 'conditional_code_sha256',
                'background_code_sha256', 'strength_code_sha256'):
        if recorded.get(key) != current.get(key):
            raise ValueError(f'Hybrid provenance mismatch: {key}; use a fresh hybrid bank')
