"""Typed configuration and validation for KB3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    total_epochs: int = 100
    segment_epochs: int = 5
    patience_segments: int = 6
    max_seconds: float | None = None
    output_dir: str = "kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization"

    def validate(self) -> None:
        if self.total_epochs <= 0 or self.segment_epochs <= 0:
            raise ValueError("total_epochs and segment_epochs must be positive")
        if self.segment_epochs > self.total_epochs:
            raise ValueError("segment_epochs cannot exceed total_epochs")
        if self.patience_segments < 1:
            raise ValueError("patience_segments must be positive")
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive when provided")


@dataclass(frozen=True)
class RewardConfig:
    delta_map50_95: float = 0.45
    delta_recall: float = 0.25
    delta_ap_small: float = 0.20
    delta_map50: float = 0.10
    time_penalty: float = 0.01
    overfit_penalty: float = 0.10
    clipped_action_penalty: float = 0.02
    failure_penalty: float = 1.0
    terminal_map_bonus: float = 0.20

    def validate(self) -> None:
        metric_sum = (
            self.delta_map50_95
            + self.delta_recall
            + self.delta_ap_small
            + self.delta_map50
        )
        if abs(metric_sum - 1.0) > 1e-6:
            raise ValueError("reward metric weights must sum to 1.0")
        if min(
            self.time_penalty,
            self.overfit_penalty,
            self.clipped_action_penalty,
            self.failure_penalty,
            self.terminal_map_bonus,
        ) < 0:
            raise ValueError("reward penalties and bonus must be non-negative")


@dataclass(frozen=True)
class ParameterConfig:
    initial: float
    minimum: float
    maximum: float
    multipliers: tuple[float, ...] = (0.8, 1.0, 1.2)
    additive: bool = False

    def validate(self, name: str) -> None:
        if self.minimum > self.initial or self.initial > self.maximum:
            raise ValueError(f"{name}: initial must be inside [minimum, maximum]")
        if not self.multipliers:
            raise ValueError(f"{name}: actions cannot be empty")


def _default_parameters() -> dict[str, ParameterConfig]:
    return {
        "lr0": ParameterConfig(0.01, 0.0001, 0.02, (0.5, 0.8, 1.0, 1.2, 1.5)),
        "weight_decay": ParameterConfig(0.0005, 0.0001, 0.002, (0.8, 1.0, 1.2)),
        "momentum": ParameterConfig(0.937, 0.80, 0.98, (-0.02, 0.0, 0.02), True),
        "augmentation_strength": ParameterConfig(0.5, 0.0, 1.0, (-0.1, 0.0, 0.1), True),
    }


@dataclass(frozen=True)
class SearchSpaceConfig:
    parameters: dict[str, ParameterConfig] = field(default_factory=_default_parameters)

    def validate(self) -> None:
        if not self.parameters:
            raise ValueError("search space cannot be empty")
        for name, parameter in self.parameters.items():
            parameter.validate(name)


@dataclass(frozen=True)
class PPOConfig:
    hidden_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 8

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.learning_rate <= 0 or self.update_epochs <= 0:
            raise ValueError("invalid PPO size, learning rate, or update epochs")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if not 0 < self.clip_ratio < 1:
            raise ValueError("clip_ratio must be in (0, 1)")


@dataclass(frozen=True)
class KB3Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    search_space: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def validate(self) -> None:
        self.experiment.validate()
        self.reward.validate()
        self.search_space.validate()
        self.ppo.validate()


def _construct(data: dict[str, Any]) -> KB3Config:
    experiment = ExperimentConfig(**data.get("experiment", {}))
    reward = RewardConfig(**data.get("reward", {}))
    ppo = PPOConfig(**data.get("ppo", {}))
    raw_parameters = data.get("search_space", {}).get("parameters", {})
    parameters = _default_parameters()
    for name, raw in raw_parameters.items():
        if name not in parameters:
            raise ValueError(f"unsupported hyperparameter: {name}")
        values = dict(raw)
        if "multipliers" in values:
            values["multipliers"] = tuple(float(x) for x in values["multipliers"])
        parameters[name] = ParameterConfig(**values)
    config = KB3Config(experiment, reward, SearchSpaceConfig(parameters), ppo)
    config.validate()
    return config


def load_config(path: str | Path) -> KB3Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return _construct(data)

