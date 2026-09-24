import tempfile
import unittest
from pathlib import Path

from kb3_hyperparameter_optimization.agents import IndependentBanditAgent, PPOAgent
from kb3_hyperparameter_optimization.config import PPOConfig


class AgentTests(unittest.TestCase):
    def test_bandit_learns_rewarded_action(self):
        agent = IndependentBanditAgent((3,), seed=1, epsilon=0.0)
        for _ in range(5):
            agent.act((0.0,))
            agent.observe(1.0, done=True)
        self.assertEqual(agent.act((0.0,), deterministic=True), (0,))

    def test_bandit_evaluation_does_not_learn(self):
        agent = IndependentBanditAgent((2,), seed=1, epsilon=0.0)
        agent.act((0.0,), deterministic=True)
        agent.observe(100.0, done=True, learn=False)
        self.assertEqual(agent.counts, [[0, 0]])

    def test_bandit_resume_restores_random_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bandit.json"
            agent = IndependentBanditAgent((3, 2), seed=11, epsilon=1.0)
            agent.save(str(path))
            expected = agent.act((0.0,))
            restored = IndependentBanditAgent.load(str(path))
            self.assertEqual(restored.act((0.0,)), expected)

    def test_ppo_update_clears_rollout(self):
        agent = PPOAgent(4, (3, 2), PPOConfig(update_epochs=2), seed=1)
        for step in range(4):
            action = agent.act((0.0, 0.1, 0.2, 0.3))
            self.assertEqual(len(action), 2)
            agent.observe(0.1, done=step == 3)
        stats = agent.update()
        self.assertIn("loss", stats)
        self.assertEqual(agent.buffer, [])

    def test_ppo_evaluation_does_not_fill_rollout(self):
        agent = PPOAgent(2, (2,), PPOConfig(update_epochs=1), seed=1)
        agent.act((0.0, 0.0), deterministic=True)
        agent.observe(1.0, done=True, learn=False)
        self.assertEqual(agent.buffer, [])
        self.assertIsNone(agent.pending)


if __name__ == "__main__":
    unittest.main()
