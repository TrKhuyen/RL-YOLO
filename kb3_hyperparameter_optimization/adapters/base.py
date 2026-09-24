"""Interface between KB3 and a concrete detector trainer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core import Metrics, SegmentResult


@runtime_checkable
class TrainerAdapter(Protocol):
    """A stateful trainer capable of continuing for a bounded epoch segment."""

    def reset(
        self,
        *,
        seed: int,
        hyperparameters: dict[str, float],
        output_dir: str | None = None,
        evaluate: bool = False,
    ) -> Metrics:
        """Reset state and optionally evaluate the pretrained model.

        Adaptive RL needs a real epoch-zero validation baseline so its first
        reward is meaningful. Traditional HPO may skip this extra evaluation.
        """

    def train_segment(
        self,
        *,
        epochs: int,
        hyperparameters: dict[str, float],
        output_dir: str,
    ) -> SegmentResult:
        """Continue the same run for exactly ``epochs`` additional epochs."""

    def close(self) -> None:
        """Release resources. Implementations should make this idempotent."""
