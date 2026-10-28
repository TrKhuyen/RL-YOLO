"""Validated replay of immutable, train-only, whole-output preference pairs."""
import json
import math
from pathlib import Path

from torch.utils.data import Dataset

from .artifacts import assert_provenance, sha256
from .data import DetectionDataset
from .policy import PolicyConfig
from .quality import QualityConfig, quality_score


class PreferenceDataset(Dataset):
    def __init__(self, folder, provenance):
        import torch
        self.folder = Path(folder).resolve()
        self.manifest = json.loads((self.folder / 'manifest.json').read_text(encoding='utf-8'))
        manifest = self.manifest
        if manifest['status'] != 'complete' or manifest['split'] != 'train':
            raise ValueError('Require completed TRAIN preference artifacts')
        assert_provenance(manifest['provenance'], provenance)
        path = self.folder / 'pairs.jsonl'
        if sha256(path) != manifest['pairs_sha256']:
            raise ValueError('Preference JSONL hash mismatch')
        self.policy_config = PolicyConfig(**manifest['policy'])
        self.quality_config = QualityConfig(**manifest['quality'])
        self.base = DetectionDataset('train', manifest['image_size'])
        self.records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not self.records or len(self.records) != manifest['summary']['pairs_kept']:
            raise ValueError('Preference count mismatch/empty dataset')
        if len({r['image_id'] for r in self.records}) != len(self.records):
            raise ValueError('Duplicate image IDs in offline pair dataset')
        for record in self.records:
            index = record['image_id']
            if type(index) is not int or not 0 <= index < len(self.base):
                raise ValueError('Invalid preference image ID')
            expected = str(self.base.paths[index].relative_to(self.base.root))
            if record['image_path'] != expected:
                raise ValueError('Preference image ID/path mismatch')
            if not all(math.isfinite(record[k]) for k in ('reference_chosen_logp', 'reference_rejected_logp')):
                raise ValueError('Invalid cached reference likelihood')
            scores = []
            for side in ('chosen', 'rejected'):
                pred = record[f'{side}_detection']
                if len(pred['boxes']) != len(pred['labels']) or len(pred['labels']) != len(pred['scores']):
                    raise ValueError('Malformed rendered detection')
                scores.append(record[f'{side}_quality']['score'])
            if scores[0] - scores[1] < self.quality_config.min_gap:
                raise ValueError('Near-tied/reversed preference labels')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image, target = self.base[record['image_id']]
        return image, target, record


def collate_preferences(batch):
    import torch
    images, targets, records = zip(*batch)
    return torch.stack(images), list(targets), list(records)
