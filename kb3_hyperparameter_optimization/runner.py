"""Shared episode runner used by adaptive-policy training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agents.base import Agent
from .envs.yolo_hpo_env import YoloHPOEnv


@dataclass(frozen=True)
class EpisodeSummary:
    episode: int
    seed: int
    steps: int
    total_reward: float
    best_map50_95: float
    final_map50_95: float
    failed: bool
    run_dir: str
    checkpoint: str | None
    best_checkpoint: str | None
    metrics: dict[str, float | int]
    final_metrics: dict[str, float | int]


def _best_from_last(checkpoint: str | None) -> str | None:
    if not checkpoint:
        return None
    last = Path(checkpoint)
    candidate = last.with_name("best.pt") if last.name == "last.pt" else None
    return str(candidate) if candidate is not None and candidate.is_file() else None


def run_episode(
    env: YoloHPOEnv, agent: Agent, *, episode: int, seed: int,
    deterministic: bool = False, learn: bool = True,
) -> tuple[EpisodeSummary, dict[str, float]]:
    observation, _ = env.reset(seed=seed, run_name=f"episode_{episode:04d}_seed_{seed}")
    total_reward, steps, failed = 0.0, 0, False
    last_info: dict[str, Any] = {}
    while True:
        action = agent.act(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, last_info = env.step(action)
        done = terminated or truncated
        agent.observe(reward, done=done, learn=learn)
        total_reward += reward
        steps += 1
        failed = failed or bool(last_info["failed"])
        if done:
            break
    update_stats = agent.update() if learn else {}
    assert env.metrics is not None and env.best_metrics is not None and env.run_dir is not None
    checkpoint = last_info.get("checkpoint")
    summary = EpisodeSummary(
        episode=episode, seed=seed, steps=steps, total_reward=total_reward,
        best_map50_95=env.best_map, final_map50_95=env.metrics.map50_95,
        failed=failed, run_dir=str(env.run_dir), checkpoint=checkpoint,
        best_checkpoint=env.best_checkpoint or _best_from_last(checkpoint),
        metrics=env.best_metrics.to_dict(), final_metrics=env.metrics.to_dict(),
    )
    return summary, update_stats


def summary_dict(summary: EpisodeSummary, update: dict[str, Any]) -> dict[str, Any]:
    return {**summary.__dict__, "agent_update": update}
