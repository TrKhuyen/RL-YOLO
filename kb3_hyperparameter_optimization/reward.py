"""Reward calculation based only on training and validation information."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import RewardConfig
from .core import Metrics


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    metric_gain: float
    time_cost: float
    overfit_cost: float
    action_cost: float
    terminal_bonus: float


def calculate_reward(
    previous: Metrics,
    current: Metrics,
    config: RewardConfig,
    *,
    time_budget_seconds: float | None,
    clipped_actions: int = 0,
    terminated: bool = False,
    failed: bool = False,
) -> RewardBreakdown:
    if failed:
        return RewardBreakdown(-config.failure_penalty, 0.0, 0.0, 0.0, 0.0, 0.0)

    metric_gain = (
        config.delta_map50_95 * (current.map50_95 - previous.map50_95)
        + config.delta_recall * (current.recall - previous.recall)
        + config.delta_ap_small * (current.ap_small - previous.ap_small)
        + config.delta_map50 * (current.map50 - previous.map50)
    )
    time_ratio = 0.0
    if time_budget_seconds:
        time_ratio = max(0.0, current.elapsed_seconds - previous.elapsed_seconds) / time_budget_seconds
    time_cost = config.time_penalty * time_ratio

    overfit_signal = 0.0
    if current.train_loss < previous.train_loss:
        map_decline = max(0.0, previous.map50_95 - current.map50_95)
        val_loss_increase = math.tanh(max(0.0, current.val_loss - previous.val_loss))
        overfit_signal = max(map_decline, val_loss_increase)
    overfit_cost = config.overfit_penalty * overfit_signal
    action_cost = config.clipped_action_penalty * max(0, clipped_actions)
    terminal_bonus = config.terminal_map_bonus * current.map50_95 if terminated else 0.0
    total = metric_gain - time_cost - overfit_cost - action_cost + terminal_bonus
    return RewardBreakdown(total, metric_gain, time_cost, overfit_cost, action_cost, terminal_bonus)
