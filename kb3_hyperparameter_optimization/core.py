"""Shared domain objects used by the environment and trainer adapters."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Metrics:
    """Training/validation metrics only; test metrics are excluded by design."""

    epoch: int
    train_loss: float
    val_loss: float
    precision: float
    recall: float
    map50: float
    map50_95: float
    ap_small: float
    elapsed_seconds: float = 0.0
    peak_vram_mb: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def validate(self) -> None:
        values = self.to_dict()
        if self.epoch < 0 or int(self.epoch) != self.epoch:
            raise ValueError("epoch must be a non-negative integer")
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("metrics must contain only finite values")
        for name in ("precision", "recall", "map50", "map50_95", "ap_small"):
            if not 0.0 <= float(values[name]) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("train_loss", "val_loss", "elapsed_seconds", "peak_vram_mb"):
            if float(values[name]) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class SegmentResult:
    metrics: Metrics
    checkpoint: str | None = None
    failed: bool = False
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    observation: tuple[float, ...]
    action: tuple[int, ...]
    reward: float
    next_observation: tuple[float, ...]
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
