"""Isolated preference-sampling trial; historical trainers remain immutable."""
from pathlib import Path

from ..artifacts import sha256

TRAINING_VARIANT = 'kb2_dpo_preservation_sampling_v1'


def code_fingerprints():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
