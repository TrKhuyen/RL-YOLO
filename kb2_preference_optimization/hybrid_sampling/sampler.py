"""Change group exposure while retaining within-group class/target balancing."""
from collections import Counter, defaultdict
import math

import numpy as np
import torch

GROUP_MASS = {'preservation': .35, 'recovery': .25, 'background': .20, 'duplicate': .20}
KEEP = {'preservation': True, 'recovery': True, 'background': False, 'duplicate': False}


def sampling_probabilities(records, class_mix=.5, max_image_repeat=5., group_mass=None):
    """Requested group mass precedes the unchanged expected image-exposure cap.

    The cap can change realized group totals. Never force a scarce group at the
    expense of image diversity; record requested AND achieved exposure instead.
    Probe records are forbidden here, and no validation metric enters the plan.
    """
    mass_by_group = dict(GROUP_MASS if group_mass is None else group_mass)
    if (not records or not math.isfinite(class_mix) or not 0 <= class_mix <= 1
            or not math.isfinite(max_image_repeat) or max_image_repeat < 1
            or set(mass_by_group) != set(KEEP)
            or any(not math.isfinite(v) or v <= 0 for v in mass_by_group.values())
            or not math.isclose(sum(mass_by_group.values()), 1., abs_tol=1e-12)):
        raise ValueError('Invalid sampling configuration')
    groups = defaultdict(list)
    for i, record in enumerate(records):
        group = record['group']
        if (group not in KEEP or bool(record['chosen_keep']) != KEEP[group]
                or record.get('probe', False)):
            raise ValueError('Unknown group, inconsistent action, or probe leakage')
        groups[group].append(i)
    if set(groups) != set(mass_by_group):
        raise ValueError('Missing requested group; do not silently renormalize')
    weight = np.zeros(len(records), dtype=np.float64)
    for group, indices in groups.items():
        class_units = defaultdict(lambda: defaultdict(list))
        for i in indices:
            r = records[i]
            unit = (r['image_id'], r['target_gt'] if r['target_gt'] is not None else f"slot{r['slot']}")
            class_units[r['class_id']][unit].append(i)
        for units in class_units.values():
            for members in units.values():
                for i in members:
                    balanced = 1 / (len(class_units) * len(units) * len(members))
                    weight[i] = ((1 - class_mix) / len(indices) + class_mix * balanced) * mass_by_group[group]
    images = sorted({r['image_id'] for r in records})
    position = {image: i for i, image in enumerate(images)}
    mapping = np.array([position[r['image_id']] for r in records])
    image_weight = np.bincount(mapping, weights=weight, minlength=len(images))
    cap = min(1., max_image_repeat / len(images))
    result = np.zeros(len(images))
    remaining = np.ones(len(images), dtype=bool)
    mass = 1.
    while remaining.any():
        proposal = image_weight[remaining] / image_weight[remaining].sum() * mass
        hits = proposal > cap + 1e-14
        locations = np.flatnonzero(remaining)
        if not hits.any():
            result[locations] = proposal
            break
        capped = locations[hits]
        result[capped] = cap
        remaining[capped] = False
        mass -= cap * len(capped)
    weight *= result[mapping] / image_weight[mapping]
    if not np.isclose(weight.sum(), 1.) or result.max() > cap + 1e-10:
        raise ValueError('Sampler failed image exposure bound')
    group_totals, class_totals = Counter(), Counter()
    for record, w in zip(records, weight):
        group_totals[record['group']] += float(w)
        class_totals[str(record['class_id'])] += float(w)
    return torch.tensor(weight, dtype=torch.double), {
        'type': 'explicit_group_class_target_mixture_then_image_cap',
        'requested_group_probability': mass_by_group, 'class_mix': class_mix,
        'max_image_repeat_factor': max_image_repeat, 'eligible_images': len(images),
        'max_image_probability': float(result.max()), 'image_probability_cap': cap,
        'expected_keep_fraction': sum(float(w) for r, w in zip(records, weight) if r['chosen_keep']),
        'expected_group_probability': dict(group_totals), 'expected_class_probability': dict(class_totals),
        'bound_scope': 'expected image exposure; not hard per-run count or equal class gradient'}
