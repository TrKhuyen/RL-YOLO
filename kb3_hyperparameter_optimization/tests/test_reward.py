import unittest

from kb3_hyperparameter_optimization.config import RewardConfig
from kb3_hyperparameter_optimization.core import Metrics
from kb3_hyperparameter_optimization.reward import calculate_reward


def metrics(map_value, train_loss=1.0, val_loss=1.0, elapsed=0.0):
    return Metrics(0, train_loss, val_loss, map_value, map_value, map_value, map_value, map_value, elapsed)


class RewardTests(unittest.TestCase):
    def test_improvement_has_positive_reward(self):
        reward = calculate_reward(
            metrics(0.2), metrics(0.3, train_loss=0.9, val_loss=0.9), RewardConfig(),
            time_budget_seconds=None,
        )
        self.assertGreater(reward.total, 0)

    def test_failure_has_fixed_negative_reward(self):
        config = RewardConfig()
        reward = calculate_reward(metrics(0.2), metrics(0.3), config, time_budget_seconds=None, failed=True)
        self.assertEqual(reward.total, -config.failure_penalty)

    def test_clipped_action_is_penalized(self):
        base = calculate_reward(metrics(0.2), metrics(0.3), RewardConfig(), time_budget_seconds=None)
        clipped = calculate_reward(
            metrics(0.2), metrics(0.3), RewardConfig(), time_budget_seconds=None, clipped_actions=2
        )
        self.assertLess(clipped.total, base.total)

    def test_rising_validation_loss_is_overfit_signal(self):
        reward = calculate_reward(
            metrics(0.4, train_loss=1.0, val_loss=1.0),
            metrics(0.4, train_loss=0.8, val_loss=1.2),
            RewardConfig(),
            time_budget_seconds=None,
        )
        self.assertGreater(reward.overfit_cost, 0)
        self.assertLess(reward.total, 0)


if __name__ == "__main__":
    unittest.main()
