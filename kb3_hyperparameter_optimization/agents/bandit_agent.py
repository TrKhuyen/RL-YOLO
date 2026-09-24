"""Independent epsilon-greedy bandit, useful as a low-data RL baseline."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


class IndependentBanditAgent:
    def __init__(self, action_sizes: Sequence[int], seed: int = 42, epsilon: float = 0.15):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        self.action_sizes = tuple(int(x) for x in action_sizes)
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.counts = [[0 for _ in range(size)] for size in self.action_sizes]
        self.values = [[0.0 for _ in range(size)] for size in self.action_sizes]
        self.pending: tuple[int, ...] | None = None

    def act(self, observation: Sequence[float], *, deterministic: bool = False) -> tuple[int, ...]:
        actions = []
        for size, values in zip(self.action_sizes, self.values):
            if not deterministic and self.rng.random() < self.epsilon:
                action = self.rng.randrange(size)
            else:
                maximum = max(values)
                candidates = [i for i, value in enumerate(values) if value == maximum]
                action = candidates[0] if deterministic else self.rng.choice(candidates)
            actions.append(action)
        self.pending = tuple(actions)
        return self.pending

    def observe(self, reward: float, *, done: bool, learn: bool = True) -> None:
        if self.pending is None:
            raise RuntimeError("act must be called before observe")
        if learn:
            for dimension, action in enumerate(self.pending):
                self.counts[dimension][action] += 1
                count = self.counts[dimension][action]
                old = self.values[dimension][action]
                self.values[dimension][action] = old + (float(reward) - old) / count
        self.pending = None

    def update(self) -> dict[str, float]:
        return {"mean_action_value": sum(map(sum, self.values)) / sum(self.action_sizes)}

    def save(self, path: str) -> None:
        if self.pending is not None:
            raise RuntimeError("bandit can only be checkpointed between decisions")
        payload = {
            "action_sizes": self.action_sizes, "epsilon": self.epsilon,
            "counts": self.counts, "values": self.values,
            "rng_state": self.rng.getstate(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "IndependentBanditAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(payload["action_sizes"], epsilon=payload["epsilon"])
        agent.counts = payload["counts"]
        agent.values = payload["values"]
        if "rng_state" in payload:
            agent.rng.setstate(_tuples(payload["rng_state"]))
        return agent
