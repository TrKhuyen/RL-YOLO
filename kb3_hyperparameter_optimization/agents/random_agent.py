from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence


class RandomAgent:
    def __init__(self, action_sizes: Sequence[int], seed: int = 42):
        self.action_sizes = tuple(int(x) for x in action_sizes)
        self.rng = random.Random(seed)

    def act(self, observation: Sequence[float], *, deterministic: bool = False) -> tuple[int, ...]:
        if deterministic:
            return tuple(size // 2 for size in self.action_sizes)
        return tuple(self.rng.randrange(size) for size in self.action_sizes)

    def observe(self, reward: float, *, done: bool, learn: bool = True) -> None:
        return None

    def update(self) -> dict[str, float]:
        return {}

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({"action_sizes": self.action_sizes}), encoding="utf-8")

