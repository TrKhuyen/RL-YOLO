"""TPE checkpoint -> conditional DPO, isolated from historical S0 experiments."""
from pathlib import Path

from ..artifacts import sha256

EXPERIMENT_VERSION = 'hybrid_tpe_conditional_strength_v1'


def code_fingerprints():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
