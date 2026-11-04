"""Isolated native-background ablation of the unchanged conditional V0 policy."""
from pathlib import Path

from ..artifacts import sha256

LOSS_VERSION = 'native_background_slot_weight_v1'


def code_fingerprints():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
