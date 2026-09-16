'''Read-only split audit; writes evidence, never moves or changes dataset files.'''
import csv
import json
import re
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'pre-data/data/v2i_cleanned'
OUT = ROOT / 'results/dataset_audit'


def source_id(path):
    return re.sub(r'_aug\d+_(?:hflip|vflip|hvflip)(?:_\d+)?$', '', path.stem)


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    splits = {s: sorted((DATA / s / 'images').glob('*.jpg')) for s in ('train', 'valid', 'test')}
    groups = {}
    for split, paths in splits.items():
        groups[split] = defaultdict(list)
        for p in paths:
            groups[split][source_id(p)].append(p)
    all_sources = defaultdict(list)
    for paths in splits.values():
        for p in paths:
            all_sources[source_id(p)].append(p)
    originals = sum(any(p.stem == key for p in paths) for key, paths in all_sources.items())
    rows, summary = [], {'dataset': str(DATA), 'counts': {s: len(p) for s,p in splits.items()},
                         'unique_full_source_ids': len(all_sources),
                         'source_groups_with_original_file': originals,
                         'method': 'same full Roboflow identifier after removing known flip augmentation suffix'}
    for left, right in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        shared = sorted(set(groups[left]) & set(groups[right]))
        summary[left + '_' + right] = {
            'source_groups': len(shared),
            'left_images': sum(len(groups[left][k]) for k in shared),
            'right_images': sum(len(groups[right][k]) for k in shared)}
        for key in shared:
            for p in groups[right][key]:
                rows.append({'pair': left + '_' + right, 'source_id': key,
                             'left_example': str(groups[left][key][0]), 'right_image': str(p)})
    summary['pixel_checks'] = []
    for key in ['000gb_jpg.rf.eeb741072e585dc54b4332616793266f',
                '0042241_jpg.rf.d6fe889f49a4ef1883f3a37402c9833a']:
        original = DATA / 'train/images' / (key + '.jpg')
        candidates = list((DATA / 'valid/images').glob(key + '_aug*_hvflip.jpg'))
        if original.exists() and candidates:
            a = cv2.flip(cv2.imread(str(original)), -1)
            b = cv2.imread(str(candidates[0]))
            summary['pixel_checks'].append({
                'train': str(original), 'validation': str(candidates[0]),
                'transformation': 'horizontal and vertical flip',
                'mae_0_255': float(np.abs(a.astype(float)-b.astype(float)).mean()),
                'pixel_correlation': float(np.corrcoef(a.ravel(), b.ravel())[0,1])})
    summary['status'] = 'leakage_detected' if rows else 'no_known_source_overlap'
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (OUT / 'overlap_pairs.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['pair', 'source_id', 'left_example', 'right_image'])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    audit()
