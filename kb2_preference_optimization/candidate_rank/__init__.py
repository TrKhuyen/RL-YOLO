"""Audited candidate-swap proposals; no change to old preference banks."""
from pathlib import Path
from ..artifacts import sha256

VERSION = 'conditional_candidate_swap_audit_v1'


def code_fingerprints():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
