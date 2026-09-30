"""Dataset and supervised checkpoint provenance for KB1."""
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / 'pre-data' / 'data' / 'v2i_cleanned'
DATA_CONFIG = ROOT / 'configs' / 'pest.yaml'
AUGMENTED = re.compile(r'_aug\d+_(?:hflip|vflip|hvflip|rot90cw|rot90ccw|rot180)(?:_\d+)?$')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dataset_manifest():
    if not DATA_CONFIG.is_file():
        raise FileNotFoundError(DATA_CONFIG)
    config = DATA_CONFIG.read_text(encoding='utf-8')
    if str(DATA_ROOT).replace('\\', '/') not in config.replace('\\', '/'):
        raise ValueError('KB1 data config does not point to the clean dataset')
    digest = hashlib.sha256()
    counts = {}
    source_sets = {}
    for split in ('train', 'valid', 'test'):
        images = DATA_ROOT / split / 'images'
        labels = DATA_ROOT / split / 'labels'
        if not images.is_dir() or not labels.is_dir():
            raise FileNotFoundError(f'Missing {split} images or labels')
        image_files = sorted(images.glob('*.jpg'))
        label_files = sorted(labels.glob('*.txt'))
        if not image_files or {p.stem for p in image_files} != {p.stem for p in label_files}:
            raise ValueError(f'Unpaired or missing images/labels in {split}')
        augmented = sum(bool(AUGMENTED.search(p.stem)) for p in image_files)
        if split != 'train' and augmented:
            raise ValueError(f'Known augmented images found in {split}')
        counts[split] = {'images': len(image_files), 'labels': len(label_files),
                         'known_augmented_images': augmented}
        source_sets[split] = {AUGMENTED.sub('', p.stem) for p in image_files}
        for path in image_files + label_files:
            digest.update(str(path.relative_to(DATA_ROOT)).replace('\\', '/').encode())
            digest.update(bytes.fromhex(sha256(path)))
    for left, right in (('train', 'valid'), ('train', 'test'), ('valid', 'test')):
        overlap = source_sets[left] & source_sets[right]
        if overlap:
            raise ValueError(f'{left}/{right} share {len(overlap)} known source IDs')
    digest.update(DATA_CONFIG.read_bytes())
    return {'root': str(DATA_ROOT.resolve()), 'config': str(DATA_CONFIG.resolve()),
            'sha256': digest.hexdigest(), 'counts': counts,
            'source_id_policy': 'known flip/rotation suffix removed; identical full Roboflow IDs rejected'}


def supervised_manifest_path(model):
    return ROOT / 'checkpoint_based' / model / 'supervised_manifest.json'


def record_supervised(model, pretrained_name, dataset_before, dp_flags):
    current = dataset_manifest()
    if current != dataset_before:
        raise RuntimeError('Dataset changed while supervised training was running')
    checkpoint = ROOT / 'checkpoint_based' / model / 'weights' / 'best.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Supervised best checkpoint missing: {checkpoint}')
    candidates = [ROOT / pretrained_name, ROOT.parent / pretrained_name]
    pretrained = next((p for p in candidates if p.is_file()), None)
    metadata = {'model': model, 'checkpoint': str(checkpoint.resolve()),
                'checkpoint_sha256': sha256(checkpoint), 'dataset': current,
                'pretrained_name': pretrained_name,
                'pretrained_sha256': sha256(pretrained) if pretrained else None,
                'data_config_sha256': sha256(DATA_CONFIG), 'dp_flags': dp_flags}
    path = supervised_manifest_path(model)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, path)
    return metadata


def verify_supervised(model, checkpoint, current_dataset=None):
    path = supervised_manifest_path(model)
    if not path.is_file():
        raise FileNotFoundError(f'Supervised provenance missing: {path}; retrain stage 1')
    metadata = json.loads(path.read_text(encoding='utf-8'))
    source = Path(checkpoint).resolve()
    if metadata['model'] != model or metadata['checkpoint'] != str(source):
        raise ValueError(f'Wrong supervised checkpoint for {model}')
    if sha256(source) != metadata['checkpoint_sha256']:
        raise ValueError(f'Supervised checkpoint hash mismatch for {model}')
    if metadata['dataset'] != (current_dataset or dataset_manifest()):
        raise ValueError(f'Dataset changed since supervised training of {model}')
    return metadata
