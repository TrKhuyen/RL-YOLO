"""Scheduled KB2 DPO updates; historical engines and artifact formats stay intact."""
from pathlib import Path

from ..artifacts import sha256

TRAINING_VARIANT = 'kb2_dpo_step_lr_dense_eval_v1'


def code_fingerprints():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
