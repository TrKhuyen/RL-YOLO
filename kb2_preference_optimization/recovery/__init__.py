"""Experimental GT-recovery selection; base F2 implementation stays unchanged."""
from pathlib import Path
from ..artifacts import sha256

VERSION = 'gt_recovery_pilot_v1'


def code_fingerprints():
    root = Path(__file__).resolve().parent
    return {p.name: sha256(p) for p in sorted(root.glob('*.py'))}
