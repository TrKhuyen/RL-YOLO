"""Whole-action pair selection and bounded class-balanced sampling."""
import hashlib
import itertools

import torch

from ..quality import match_boxes


def is_probe_image(image_id):
    return int(hashlib.sha256(f'kb2-recovery:{image_id}'.encode()).hexdigest()[:8], 16) % 10 == 0


def matched_ids(prediction, target):
    return {g for _, g, _ in match_boxes(prediction, target, .5)}


def action_distance(a, b):
    """Differences in categorical choices and DFL sides, not rendered GT IDs."""
    left = {i: (c, bins) for i, c, bins in zip(a['indices'], a['classes'], a['bins'])}
    right = {i: (c, bins) for i, c, bins in zip(b['indices'], b['classes'], b['bins'])}
    distance = 0
    for index in left.keys() | right.keys():
        x, y = left.get(index), right.get(index)
        if x is None or y is None:
            distance += 5  # background/emission decision and four new box sides
        else:
            distance += int(x[0] != y[0]) + sum(u != v for u, v in zip(x[1], y[1]))
    return distance


def pair_metadata(chosen, rejected, qualities, matches, labels, actions):
    rescued = matches[chosen] - matches[rejected]
    lost = matches[rejected] - matches[chosen]
    return {'rescued_gt': sorted(rescued), 'lost_gt': sorted(lost),
            'rescued_classes': sorted({int(labels[g]) for g in rescued}),
            'fp_delta': qualities[chosen]['fp'] - qualities[rejected]['fp'],
            'quality_gap': qualities[chosen]['score'] - qualities[rejected]['score'],
            'action_distance': action_distance(actions[chosen], actions[rejected])}


def eligible_pairs(qualities, matches, labels, actions, min_gap=.02):
    """Inspect all unordered output pairs; orient by quality before GT gates."""
    recovery, maintenance = [], []
    for a, b in itertools.combinations(range(len(actions)), 2):
        chosen, rejected = (a, b) if qualities[a]['score'] >= qualities[b]['score'] else (b, a)
        meta = pair_metadata(chosen, rejected, qualities, matches, labels, actions)
        if meta['quality_gap'] < min_gap:
            continue
        if meta['rescued_gt'] and not meta['lost_gt'] and meta['fp_delta'] <= 0:
            recovery.append({'chosen_index': chosen, 'rejected_index': rejected,
                             'group': 'recovery', **meta})
        elif matches[chosen] == matches[rejected]:
            maintenance.append({'chosen_index': chosen, 'rejected_index': rejected,
                                'group': 'maintenance', **meta})
    recovery.sort(key=lambda r: (-len(r['rescued_gt']), r['action_distance'],
                                 -r['quality_gap'], r['chosen_index'], r['rejected_index']))
    maintenance.sort(key=lambda r: (-r['quality_gap'], r['action_distance'],
                                    r['chosen_index'], r['rejected_index']))
    return recovery, maintenance


def balanced_probabilities(records, recovery_fraction=.5, class_mix=.5, max_repeat=5.):
    """Mix uniform-recovery and uniform-class sampling; cap each image probability.

    Multi-class pairs get their exact marginal probability, not duplicated rows.
    Capping can change the requested group mix; report the achieved expectation.
    """
    if not records or not 0 < recovery_fraction < 1 or not 0 <= class_mix <= 1 or max_repeat < 1:
        raise ValueError('Invalid balancing parameters')
    groups = [r['selection'] for r in records]
    rescue = [i for i, r in enumerate(groups) if r['group'] == 'recovery']
    maintain = [i for i, r in enumerate(groups) if r['group'] == 'maintenance']
    if len(rescue) + len(maintain) != len(records):
        raise ValueError('Balanced dataset must contain only recovery/maintenance pairs')
    if not rescue or not maintain:
        raise ValueError('Need both groups for a recovery/maintenance pilot')
    by_class = {}
    for i in rescue:
        if not groups[i]['rescued_classes']:
            raise ValueError('Recovery pair without a rescued class')
        for c in set(groups[i]['rescued_classes']):
            by_class.setdefault(c, []).append(i)
    desired = torch.zeros(len(records), dtype=torch.float64)
    desired[maintain] = (1 - recovery_fraction) / len(maintain)
    desired[rescue] = recovery_fraction * (1 - class_mix) / len(rescue)
    for indices in by_class.values():
        desired[indices] += recovery_fraction * class_mix / len(by_class) / len(indices)
    cap = min(1., max_repeat / len(records))
    lo, hi = 0., 1.
    while float(torch.clamp(desired * hi, max=cap).sum()) < 1 - 1e-14:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if float(torch.clamp(desired * mid, max=cap).sum()) < 1:
            lo = mid
        else:
            hi = mid
    probabilities = torch.clamp(desired * hi, max=cap)
    probabilities /= probabilities.sum()
    return probabilities, {
        'requested_recovery_fraction': recovery_fraction,
        'expected_recovery_fraction': float(probabilities[rescue].sum()),
        'class_mix': class_mix, 'max_relative_repeat': float(probabilities.max() * len(records)),
        'cap_relative_repeat': max_repeat,
        'classes_with_recovery': sorted(by_class),
        'expected_class_exposure': {str(c): float(probabilities[indices].sum())
                                    for c, indices in by_class.items()},
    }
