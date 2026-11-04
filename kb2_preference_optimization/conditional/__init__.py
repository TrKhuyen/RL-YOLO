"""Versioned, local keep/drop DPO experiment; independent of historical banks."""
from pathlib import Path

from ..artifacts import sha256

VERSION = 'conditional_detection_dpo_v0'


def code_fingerprints():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
