"""Conditional Bernoulli XOR policy: choose exactly one existing candidate."""
import torch
from torch.nn import functional as F


def rank_loss(logits, reference_logits, records, beta=.1):
    if not records or beta <= 0:
        raise ValueError('Nonempty records and positive beta required')
    differences, refs = [], []
    for r in records:
        b, c, good, bad = r['batch_index'], r['class_id'], r['good_slot'], r['bad_slot']
        if good == bad:
            raise ValueError('Distinct slots required')
        differences.append(logits[b, good, c]-logits[b, bad, c])
        refs.append(reference_logits[b, good, c]-reference_logits[b, bad, c])
    z, z0 = torch.stack(differences), torch.stack(refs).detach()
    margin = z-z0
    return F.softplus(-beta*margin).mean(), margin
