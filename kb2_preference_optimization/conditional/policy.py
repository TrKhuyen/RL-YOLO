"""Normalized Bernoulli policy with fixed geometry and frozen BN statistics."""
import hashlib

import torch
from torch.nn import functional as F

from ..adapter import DFLAdapter


def configure_numerics():
    # Same precision for generation, reference replay and ordinary inference.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ConditionalAdapter(DFLAdapter):
    def __init__(self, checkpoint, device='cuda'):
        super().__init__(checkpoint, device)
        self.model.requires_grad_(False)
        self.head.cv3.requires_grad_(True)

    def raw(self, images):
        self.model.eval()
        decoded, raw = self.model(images.to(self.device))
        xywh = decoded[:, :4].transpose(1, 2)
        boxes = torch.cat((xywh[..., :2] - xywh[..., 2:] / 2,
                           xywh[..., :2] + xywh[..., 2:] / 2), -1)
        return raw['scores'].transpose(1, 2), boxes

    def frozen_digest(self):
        trainable = {n for n, p in self.model.named_parameters() if p.requires_grad}
        digest = hashlib.sha256()
        for name, value in sorted(self.model.state_dict().items()):
            if name not in trainable:
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def pair_loss(policy_logits, reference_logits, records, beta=.1):
    """Both inputs are [batch, slots, classes] for the SAME full input batch."""
    if not records or beta <= 0:
        raise ValueError('Nonempty records and positive beta required')
    values, references, signs = [], [], []
    for record in records:
        b, j, c = record['batch_index'], record['slot'], record['class_id']
        values.append(policy_logits[b, j, c])
        references.append(reference_logits[b, j, c])
        signs.append(1. if record['chosen_keep'] else -1.)
    z = torch.stack(values)
    z0 = torch.stack(references).detach()
    sign = z.new_tensor(signs)
    margin = sign * (z - z0)
    # log sigmoid(z) - log sigmoid(-z) == z; no sigmoid/log roundoff.
    loss = F.softplus(-beta * margin).mean()
    return loss, margin, z, z0
