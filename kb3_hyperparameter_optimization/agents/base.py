from __future__ import annotations

from typing import Protocol, Sequence


class Agent(Protocol):
    def act(self, observation: Sequence[float], *, deterministic: bool = False) -> tuple[int, ...]: ...

    def observe(self, reward: float, *, done: bool, learn: bool = True) -> None: ...

    def update(self) -> dict[str, float]: ...

    def save(self, path: str) -> None: ...

