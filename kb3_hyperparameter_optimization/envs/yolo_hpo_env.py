"""Framework-independent RL environment for segmented detector training."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Sequence

from ..adapters.base import TrainerAdapter
from ..config import KB3Config
from ..core import Metrics
from ..reward import calculate_reward
from ..search_space import DiscreteSearchSpace

OBSERVATION_NAMES = (
    "progress", "train_loss", "val_loss", "precision", "recall", "map50",
    "map50_95", "ap_small", "generalization_gap", "delta_map50_95",
    "delta_recall", "delta_ap_small", "stale_ratio", "time_ratio",
)


class YoloHPOEnv:
    def __init__(self, trainer: TrainerAdapter, config: KB3Config):
        config.validate()
        self.trainer, self.config = trainer, config
        self.space = DiscreteSearchSpace(config.search_space)
        self.hyperparameters = self.space.initial_values()
        self.metrics: Metrics | None = None
        self.previous_metrics: Metrics | None = None
        self.best_metrics: Metrics | None = None
        self.best_checkpoint: str | None = None
        self.best_map = -math.inf
        self.stale_segments = 0
        self.started_at = 0.0
        self.run_dir: Path | None = None
        self.trajectory: list[dict] = []

    @property
    def observation_size(self) -> int:
        return len(OBSERVATION_NAMES) + len(self.space.names)

    def _unique_run_dir(self, suffix: str) -> Path:
        root = Path(self.config.experiment.output_dir)
        candidate = root / suffix
        if candidate.exists() and any(candidate.iterdir()):
            candidate = root / f"{suffix}_{time.time_ns()}"
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def reset(self, *, seed: int | None = None, run_name: str | None = None):
        seed = self.config.experiment.seed if seed is None else seed
        self.hyperparameters = self.space.initial_values()
        self.previous_metrics = None
        self.run_dir = self._unique_run_dir(run_name or f"seed_{seed}_{time.time_ns()}")
        self.metrics = self.trainer.reset(
            seed=seed, hyperparameters=dict(self.hyperparameters),
            output_dir=str(self.run_dir), evaluate=True,
        )
        self.metrics.validate()
        self.best_map = self.metrics.map50_95
        self.best_metrics = self.metrics
        self.best_checkpoint = None
        self.stale_segments = 0
        self.started_at = time.monotonic()
        self.trajectory = []
        return self._observation(), {"metrics": self.metrics.to_dict(), "seed": seed}

    def _observation(self) -> tuple[float, ...]:
        if self.metrics is None:
            raise RuntimeError("environment must be reset first")
        old = self.previous_metrics or self.metrics
        max_seconds = self.config.experiment.max_seconds
        raw = (
            min(1.0, self.metrics.epoch / self.config.experiment.total_epochs),
            math.tanh(self.metrics.train_loss), math.tanh(self.metrics.val_loss),
            self.metrics.precision, self.metrics.recall, self.metrics.map50,
            self.metrics.map50_95, self.metrics.ap_small,
            math.tanh(self.metrics.val_loss - self.metrics.train_loss),
            self.metrics.map50_95 - old.map50_95, self.metrics.recall - old.recall,
            self.metrics.ap_small - old.ap_small,
            min(1.0, self.stale_segments / self.config.experiment.patience_segments),
            0.0 if not max_seconds else min(1.0, self.metrics.elapsed_seconds / max_seconds),
        )
        if any(not math.isfinite(value) for value in raw):
            raise ValueError("observation contains a non-finite value")
        return tuple(float(x) for x in raw) + self.space.normalized(self.hyperparameters)

    def step(self, action: Sequence[int]):
        if self.metrics is None or self.run_dir is None:
            raise RuntimeError("environment must be reset before step")
        observation_before, before = self._observation(), self.metrics
        applied = self.space.apply(self.hyperparameters, action)
        self.hyperparameters = applied.values
        remaining = self.config.experiment.total_epochs - before.epoch
        if remaining <= 0:
            raise RuntimeError("episode has already ended")
        result = self.trainer.train_segment(
            epochs=min(self.config.experiment.segment_epochs, remaining),
            hyperparameters=dict(self.hyperparameters), output_dir=str(self.run_dir),
        )
        self.previous_metrics, self.metrics = before, result.metrics
        if not result.failed:
            self.metrics.validate()

        # A concrete worker may evaluate its persisted best.pt and return the
        # corresponding metrics. Prefer that candidate so reported metrics and
        # the deployable checkpoint always refer to the same model.
        candidate = self.metrics
        if not result.failed and result.metadata.get("best_metrics") is not None:
            candidate = Metrics(**result.metadata["best_metrics"])
            candidate.validate()
        improved = not result.failed and candidate.map50_95 > self.best_map + 1e-9
        if improved:
            self.best_map, self.best_metrics = candidate.map50_95, candidate
            self.best_checkpoint = result.metadata.get("best_checkpoint") or result.checkpoint
            self.stale_segments = 0
        else:
            self.stale_segments += 1

        terminated = result.failed or self.metrics.epoch >= self.config.experiment.total_epochs
        timed_out = (
            self.config.experiment.max_seconds is not None
            and (time.monotonic() - self.started_at >= self.config.experiment.max_seconds
                 or self.metrics.elapsed_seconds >= self.config.experiment.max_seconds)
        )
        early_stopped = self.stale_segments >= self.config.experiment.patience_segments
        truncated = (timed_out or early_stopped) and not terminated
        breakdown = calculate_reward(
            before, self.metrics, self.config.reward,
            time_budget_seconds=self.config.experiment.max_seconds,
            clipped_actions=applied.clipped_count,
            terminated=terminated and not result.failed, failed=result.failed,
        )
        next_observation = self._observation()
        info = {
            "metrics": self.metrics.to_dict(), "hyperparameters": dict(self.hyperparameters),
            "decisions": applied.decisions, "clipped_actions": applied.clipped_count,
            "reward": breakdown.__dict__, "checkpoint": result.checkpoint,
            "best_checkpoint": self.best_checkpoint,
            "failed": result.failed, "failure_reason": result.failure_reason,
            "early_stopped": early_stopped, "timed_out": timed_out,
        }
        self.trajectory.append({
            "step": len(self.trajectory), "observation": list(observation_before),
            "action": list(self.space.validate_action(action)),
            "scalar_reward": breakdown.total, "next_observation": list(next_observation),
            "terminated": terminated, "truncated": truncated, **info,
        })
        self._write_trajectory()
        return next_observation, breakdown.total, terminated, truncated, info

    def _write_trajectory(self) -> None:
        if self.run_dir is None:
            return
        path = self.run_dir / "trajectory.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.trajectory, indent=2), encoding="utf-8")
        temporary.replace(path)

    def close(self) -> None:
        self.trainer.close()
