"""Deterministic lightweight trainer for tests and end-to-end smoke runs."""

from __future__ import annotations

import math
import random

from ..core import Metrics, SegmentResult


class SimulatedTrainerAdapter:
    def __init__(self):
        self.rng = random.Random(0)
        self.epoch = 0
        self.elapsed = 0.0
        self.quality = 0.05

    def reset(
        self, *, seed: int, hyperparameters: dict[str, float],
        output_dir: str | None = None, evaluate: bool = False,
    ) -> Metrics:
        self.rng.seed(seed)
        self.epoch = 0
        self.elapsed = 0.0
        self.quality = 0.05
        return self._metrics()

    def _metrics(self) -> Metrics:
        quality = min(0.95, max(0.0, self.quality))
        return Metrics(
            epoch=self.epoch,
            train_loss=max(0.03, 1.1 - quality),
            val_loss=max(0.04, 1.15 - quality + max(0, self.epoch - 80) * 0.002),
            precision=max(0.0, quality - 0.03),
            recall=max(0.0, quality - 0.06),
            map50=min(1.0, quality + 0.06),
            map50_95=quality,
            ap_small=max(0.0, quality - 0.12),
            elapsed_seconds=self.elapsed,
            peak_vram_mb=1024.0,
        )

    def train_segment(
        self, *, epochs: int, hyperparameters: dict[str, float], output_dir: str,
    ) -> SegmentResult:
        lr = hyperparameters["lr0"]
        wd = hyperparameters["weight_decay"]
        momentum = hyperparameters["momentum"]
        augmentation = hyperparameters["augmentation_strength"]
        lr_score = math.exp(-abs(math.log10(lr) - math.log10(0.006)))
        wd_score = math.exp(-abs(math.log10(wd) - math.log10(0.0007)))
        momentum_score = math.exp(-abs(momentum - 0.92) * 8)
        aug_score = math.exp(-abs(augmentation - 0.6) * 2)
        gain = epochs * 0.007 * (
            0.4 * lr_score + 0.2 * wd_score + 0.2 * momentum_score + 0.2 * aug_score
        )
        gain += self.rng.uniform(-0.001, 0.001)
        self.quality += gain * max(0.1, 1.0 - self.quality)
        self.epoch += epochs
        self.elapsed += epochs * 0.1
        return SegmentResult(self._metrics(), checkpoint=f"{output_dir}/simulated.pt")

    def close(self) -> None:
        return None
