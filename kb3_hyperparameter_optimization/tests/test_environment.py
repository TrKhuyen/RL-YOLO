import tempfile
import unittest
from dataclasses import replace

from kb3_hyperparameter_optimization.adapters import SimulatedTrainerAdapter
from kb3_hyperparameter_optimization.agents import RandomAgent
from kb3_hyperparameter_optimization.config import ExperimentConfig, KB3Config
from kb3_hyperparameter_optimization.core import Metrics, SegmentResult
from kb3_hyperparameter_optimization.envs import YoloHPOEnv
from kb3_hyperparameter_optimization.runner import run_episode


class PeakingTrainer:
    def reset(self, *, seed, hyperparameters, output_dir=None, evaluate=False):
        self.index = 0
        return Metrics(0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1)

    def train_segment(self, *, epochs, hyperparameters, output_dir):
        self.index += 1
        score = 0.8 if self.index == 1 else 0.4
        epoch = self.index * epochs
        return SegmentResult(Metrics(epoch, 0.5, 0.6, score, score, score, score, score))

    def close(self):
        return None


class EnvironmentTests(unittest.TestCase):
    def make_env(self, directory):
        experiment = ExperimentConfig(
            seed=7, total_epochs=6, segment_epochs=2, patience_segments=10, output_dir=directory
        )
        return YoloHPOEnv(SimulatedTrainerAdapter(), replace(KB3Config(), experiment=experiment))

    def test_episode_reaches_exact_epoch_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.make_env(directory)
            observation, info = env.reset(seed=7, run_name="test")
            self.assertEqual(len(observation), env.observation_size)
            action = tuple(size // 2 for size in env.space.action_sizes)
            done = False
            steps = 0
            while not done:
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            self.assertEqual(steps, 3)
            self.assertEqual(info["metrics"]["epoch"], 6)
            self.assertFalse(info["failed"])

    def test_trajectory_contains_no_test_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.make_env(directory)
            env.reset(seed=7, run_name="test")
            action = tuple(size // 2 for size in env.space.action_sizes)
            env.step(action)
            self.assertNotIn("test_", str(env.trajectory).lower())

    def test_summary_keeps_best_metrics_when_final_segment_declines(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = ExperimentConfig(
                seed=7, total_epochs=4, segment_epochs=2,
                patience_segments=10, output_dir=directory,
            )
            config = replace(KB3Config(), experiment=experiment)
            env = YoloHPOEnv(PeakingTrainer(), config)
            agent = RandomAgent(env.space.action_sizes, seed=7)
            summary, _ = run_episode(env, agent, episode=0, seed=7, learn=False)
            self.assertEqual(summary.best_map50_95, 0.8)
            self.assertEqual(summary.metrics["map50_95"], 0.8)
            self.assertEqual(summary.final_metrics["map50_95"], 0.4)


if __name__ == "__main__":
    unittest.main()
