"""DPO-strength ablation isolated from previous immutable experiment code."""
from pathlib import Path
from ..artifacts import sha256

EXPERIMENT_VERSION = 'conditional_dpo_strength_v1'


def code_fingerprints():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
