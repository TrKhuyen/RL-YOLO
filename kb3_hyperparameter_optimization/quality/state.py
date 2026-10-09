"""Stateless observations and feasible actions; no evaluation-time fitting."""

import math

from ..search_space import DiscreteSearchSpace

STATE_VERSION = "log_loss_trends_best_map_v1"
STATE_NAMES = (
    "progress", "losses_available", "log_train_loss", "log_val_loss",
    "log_loss_gap", "train_loss_trend", "val_loss_trend", "precision",
    "ar300", "map50", "map50_95", "ap_small", "delta_map50_95",
    "best_trained_map50_95", "stale_fraction", "schedule_factor",
)


def observation(current, previous, *, epochs, best, stale, parameters, space, schedule_factor):
    old = previous or current
    available = current.epoch > 0
    train = math.log1p(current.train_loss) / 4.0 if available else 0.0
    val = math.log1p(current.val_loss) / 4.0
    old_train = math.log1p(old.train_loss) / 4.0
    old_val = math.log1p(old.val_loss) / 4.0
    result = (
        current.epoch / epochs, float(available), train, val,
        val - train if available else 0.0,
        train - old_train if available and old.epoch > 0 else 0.0,
        val - old_val, current.precision, current.recall, current.map50,
        current.map50_95, current.ap_small, current.map50_95 - old.map50_95,
        best, stale / epochs, schedule_factor,
    ) + space.normalized(parameters)
    if not all(math.isfinite(v) for v in result):
        raise ValueError("Non-finite policy observation")
    return result


def action_masks(space: DiscreteSearchSpace, parameters):
    masks = []
    for name in space.names:
        spec, value = space.config.parameters[name], parameters[name]
        valid = []
        for decision in spec.multipliers:
            candidate = value + decision if spec.additive else value * decision
            valid.append(spec.minimum - 1e-12 <= candidate <= spec.maximum + 1e-12)
        if not any(valid):
            raise ValueError(f"No feasible action for {name}; include a hold action")
        masks.append(tuple(valid))
    return tuple(masks)


def best_gain(previous_best, current_best):
    # gamma=1 makes the episode return exactly 100*(final_best-prefix_best).
    return 100.0 * (current_best - previous_best)
