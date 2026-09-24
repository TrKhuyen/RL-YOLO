"""Safe discrete hyperparameter action space."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .config import SearchSpaceConfig


@dataclass(frozen=True)
class AppliedAction:
    values: dict[str, float]
    clipped_count: int
    decisions: dict[str, float]


class DiscreteSearchSpace:
    def __init__(self, config: SearchSpaceConfig):
        config.validate()
        self.config = config
        self.names = tuple(config.parameters)
        self.action_sizes = tuple(len(config.parameters[name].multipliers) for name in self.names)

    def initial_values(self) -> dict[str, float]:
        return {name: spec.initial for name, spec in self.config.parameters.items()}

    def sample(self, rng: random.Random) -> tuple[int, ...]:
        return tuple(rng.randrange(size) for size in self.action_sizes)

    def validate_action(self, action: Sequence[int]) -> tuple[int, ...]:
        if len(action) != len(self.names):
            raise ValueError(f"expected {len(self.names)} action dimensions, got {len(action)}")
        result = tuple(int(value) for value in action)
        for name, value, size in zip(self.names, result, self.action_sizes):
            if value < 0 or value >= size:
                raise ValueError(f"action for {name} must be in [0, {size - 1}]")
        return result

    def apply(self, current: Mapping[str, float], action: Sequence[int]) -> AppliedAction:
        action = self.validate_action(action)
        updated: dict[str, float] = {}
        decisions: dict[str, float] = {}
        clipped = 0
        for name, index in zip(self.names, action):
            spec = self.config.parameters[name]
            decision = spec.multipliers[index]
            old = float(current.get(name, spec.initial))
            candidate = old + decision if spec.additive else old * decision
            value = min(spec.maximum, max(spec.minimum, candidate))
            clipped += int(abs(value - candidate) > 1e-12)
            updated[name] = value
            decisions[name] = decision
        return AppliedAction(updated, clipped, decisions)

    def normalized(self, values: Mapping[str, float]) -> tuple[float, ...]:
        normalized = []
        for name in self.names:
            spec = self.config.parameters[name]
            value = float(values.get(name, spec.initial))
            width = spec.maximum - spec.minimum
            normalized.append(0.0 if width == 0 else 2.0 * (value - spec.minimum) / width - 1.0)
        return tuple(normalized)

