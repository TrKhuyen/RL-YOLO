"""Shared runner for fixed-hyperparameter traditional HPO trials."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..adapters.base import TrainerAdapter
from ..config import KB3Config


@dataclass(frozen=True)
class FixedTrialResult:
    trial: int
    seed: int
    hyperparameters: dict[str, float]
    failed: bool
    failure_reason: str | None
    checkpoint: str | None
    best_checkpoint: str | None
    metrics: dict[str, float | int]
    final_metrics: dict[str, float | int]
    run_dir: str

    def to_dict(self) -> dict:
        return asdict(self)


def _unique_directory(root: Path, name: str) -> Path:
    target = root / name
    if target.exists() and any(target.iterdir()):
        target = root / f"{name}_{time.time_ns()}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _best_from_last(checkpoint: str | None) -> str | None:
    if not checkpoint:
        return None
    last = Path(checkpoint)
    candidate = last.with_name("best.pt") if last.name == "last.pt" else None
    return str(candidate) if candidate is not None and candidate.is_file() else None


def run_fixed_trial(
    trainer: TrainerAdapter, config: KB3Config, *, trial: int, seed: int,
    hyperparameters: dict[str, float], method: str,
) -> FixedTrialResult:
    expected = set(config.search_space.parameters)
    if set(hyperparameters) != expected:
        raise ValueError(f"hyperparameters must be exactly {sorted(expected)}")
    for name, value in hyperparameters.items():
        spec = config.search_space.parameters[name]
        if not spec.minimum <= float(value) <= spec.maximum:
            raise ValueError(f"{name}={value} is outside [{spec.minimum}, {spec.maximum}]")

    run_dir = _unique_directory(
        Path(config.experiment.output_dir) / "traditional_hpo",
        f"{method}_trial_{trial:04d}_seed_{seed}",
    )
    trainer.reset(
        seed=seed, hyperparameters=dict(hyperparameters),
        output_dir=str(run_dir), evaluate=False,
    )
    result = trainer.train_segment(
        epochs=config.experiment.total_epochs,
        hyperparameters=dict(hyperparameters), output_dir=str(run_dir),
    )
    if not result.failed:
        result.metrics.validate()
    best_metrics = result.metadata.get("best_metrics") or result.metrics.to_dict()
    best_checkpoint = result.metadata.get("best_checkpoint") or _best_from_last(result.checkpoint)
    return FixedTrialResult(
        trial=trial, seed=seed, hyperparameters=dict(hyperparameters),
        failed=result.failed, failure_reason=result.failure_reason,
        checkpoint=result.checkpoint, best_checkpoint=best_checkpoint,
        metrics=best_metrics, final_metrics=result.metrics.to_dict(), run_dir=str(run_dir),
    )


def result_document(method: str, config: KB3Config, trials: list[FixedTrialResult]) -> dict:
    return {
        "schema_version": 2, "scenario": "KB3-A",
        "optimization": "traditional_hpo", "method": method,
        "budget": {
            "trials": len(trials), "epochs_per_trial": config.experiment.total_epochs,
            "maximum_total_epochs": len(trials) * config.experiment.total_epochs,
        },
        "trials": [trial.to_dict() for trial in trials],
    }
