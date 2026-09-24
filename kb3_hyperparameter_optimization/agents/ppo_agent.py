"""Compact PPO implementation with one categorical policy head per parameter."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from ..config import PPOConfig


class _ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_sizes: Sequence[int], hidden_size: int):
        super().__init__()
        self.action_sizes = tuple(action_sizes)
        self.encoder = nn.Sequential(
            nn.Linear(observation_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.policy = nn.Linear(hidden_size, sum(self.action_sizes))
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor):
        hidden = self.encoder(observations)
        return self.policy(hidden).split(self.action_sizes, dim=-1), self.value(hidden).squeeze(-1)

    def evaluate(self, observations: torch.Tensor, actions: torch.Tensor):
        logits, values = self(observations)
        distributions = [Categorical(logits=head) for head in logits]
        log_prob = torch.stack(
            [distribution.log_prob(actions[:, index]) for index, distribution in enumerate(distributions)],
            dim=-1,
        ).sum(dim=-1)
        entropy = torch.stack([distribution.entropy() for distribution in distributions], dim=-1).sum(dim=-1)
        return log_prob, entropy, values


class PPOAgent:
    def __init__(
        self,
        observation_size: int,
        action_sizes: Sequence[int],
        config: PPOConfig,
        seed: int = 42,
        device: str = "cpu",
    ):
        torch.manual_seed(seed)
        self.config = config
        self.device = torch.device(device)
        self.model = _ActorCritic(observation_size, action_sizes, config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.buffer: list[dict] = []
        self.pending: dict | None = None

    def act(self, observation: Sequence[float], *, deterministic: bool = False) -> tuple[int, ...]:
        if self.pending is not None:
            raise RuntimeError("observe must be called before selecting another action")
        obs = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(obs)
            distributions = [Categorical(logits=head) for head in logits]
            tensors = [d.probs.argmax(-1) if deterministic else d.sample() for d in distributions]
            actions = torch.stack(tensors, dim=-1)
            log_prob = torch.stack(
                [distribution.log_prob(tensor) for distribution, tensor in zip(distributions, tensors)], dim=-1
            ).sum(dim=-1)
        self.pending = {
            "observation": tuple(float(x) for x in observation),
            "action": tuple(int(x) for x in actions.squeeze(0).tolist()),
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
        }
        return self.pending["action"]

    def observe(self, reward: float, *, done: bool, learn: bool = True) -> None:
        if self.pending is None:
            raise RuntimeError("act must be called before observe")
        if learn:
            self.pending.update(reward=float(reward), done=bool(done))
            self.buffer.append(self.pending)
        self.pending = None

    def _advantages(self):
        advantages = []
        gae = 0.0
        next_value = 0.0
        for transition in reversed(self.buffer):
            mask = 0.0 if transition["done"] else 1.0
            delta = transition["reward"] + self.config.gamma * next_value * mask - transition["value"]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages.append(gae)
            next_value = transition["value"]
        advantages.reverse()
        values = torch.tensor([x["value"] for x in self.buffer], dtype=torch.float32, device=self.device)
        advantage_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantage_tensor + values
        if len(advantages) > 1:
            advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
                advantage_tensor.std(unbiased=False) + 1e-8
            )
        return advantage_tensor, returns

    def update(self) -> dict[str, float]:
        if self.pending is not None:
            raise RuntimeError("cannot update with a pending action")
        if not self.buffer:
            return {}
        observations = torch.tensor([x["observation"] for x in self.buffer], dtype=torch.float32, device=self.device)
        actions = torch.tensor([x["action"] for x in self.buffer], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([x["log_prob"] for x in self.buffer], dtype=torch.float32, device=self.device)
        advantages, returns = self._advantages()
        stats = {}
        for _ in range(self.config.update_epochs):
            log_probs, entropy, values = self.model.evaluate(observations, actions)
            ratio = (log_probs - old_log_probs).exp()
            policy_loss = -torch.minimum(
                ratio * advantages,
                ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages,
            ).mean()
            value_loss = (returns - values).pow(2).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_mean
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            stats = {
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy_mean.detach()),
            }
        self.buffer.clear()
        return stats

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "observation_size": self.model.encoder[0].in_features,
            "action_sizes": self.model.action_sizes,
            "config": self.config.__dict__,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, target)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PPOAgent":
        payload = torch.load(path, map_location=device, weights_only=False)
        config = PPOConfig(**payload["config"])
        agent = cls(payload["observation_size"], payload["action_sizes"], config, device=device)
        agent.model.load_state_dict(payload["model"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        if payload.get("torch_rng_state") is not None:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        return agent
