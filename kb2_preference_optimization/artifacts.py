"""Versioned, fail-closed experiment artifacts and shared S0 provenance."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from kb1_reward_guided_training.run_provenance import (
    dataset_manifest, runtime_manifest, sha256, verify_supervised,
)
from . import OBJECTIVE_VERSION
from .adapter import SUPPORTED_MODELS
from .data import TRANSFORM_VERSION

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_checkpoint(model):
    if model not in SUPPORTED_MODELS:
        raise ValueError(f'Unsupported structured DPO model: {model}; supported: {SUPPORTED_MODELS}')
    return REPO / 'kb1_reward_guided_training' / 'checkpoint_based' / model / 'weights' / 'best.pt'


def code_fingerprints():
    return {p.name: sha256(p) for p in sorted(ROOT.glob('*.py'))}


def source_provenance(model):
    source = source_checkpoint(model)
    dataset = dataset_manifest()
    supervised = verify_supervised(model, source, dataset)
    runtime = runtime_manifest()
    runtime['versions']['scipy'] = importlib.metadata.version('scipy')
    return {'model': model, 'source_checkpoint': str(source), 'source_sha256': sha256(source),
            'dataset': dataset, 'supervised': supervised, 'runtime': runtime,
            'code_sha256': code_fingerprints(), 'objective_version': OBJECTIVE_VERSION,
            'transform_version': TRANSFORM_VERSION}


def assert_provenance(recorded, current):
    for key in ('model', 'source_sha256', 'objective_version', 'transform_version', 'code_sha256'):
        if recorded.get(key) != current.get(key):
            raise ValueError(f'Artifact provenance mismatch: {key}')
    if recorded['dataset']['sha256'] != current['dataset']['sha256']:
        raise ValueError('Dataset content changed')
    for key in ('versions', 'ultralytics_code_sha256'):
        if recorded['runtime'][key] != current['runtime'][key]:
            raise ValueError(f'Runtime mismatch: {key}; regenerate/revalidate preferences explicitly')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def append_jsonl(path, record):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def tensor_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        # Cached anchor/stride tensors are plain attributes, not state_dict entries.
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()
