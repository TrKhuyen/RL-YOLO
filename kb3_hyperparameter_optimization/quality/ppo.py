"""Masked PPO with complete-episode rollouts and isolated policy RNG."""

from pathlib import Path

import torch
from torch.distributions import Categorical
from torch import nn

from ..agents.ppo_agent import _ActorCritic
from . import PROTOCOL
from .state import STATE_VERSION


class QualityPPO:
    def __init__(self, observation_size, action_sizes, *, seed=42, hidden_size=128,
                 learning_rate=3e-4, rollout_steps=256, minibatch_size=64,
                 update_epochs=10, target_kl=0.02):
        self.settings = dict(observation_size=observation_size, action_sizes=tuple(action_sizes),
                             seed=seed, hidden_size=hidden_size, learning_rate=learning_rate,
                             rollout_steps=rollout_steps, minibatch_size=minibatch_size,
                             update_epochs=update_epochs, target_kl=target_kl)
        # torch.manual_seed also seeds CUDA: preserve those streams as well.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.model = _ActorCritic(observation_size, action_sizes, hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)
        self.rng = torch.Generator().manual_seed(seed)
        self.buffer, self.pending = [], None

    def _evaluate(self, observations, actions, masks):
        heads, values = self.model(observations)
        distributions = [Categorical(logits=head.masked_fill(~mask, -1e9))
                         for head, mask in zip(heads, masks)]
        log_prob = sum(d.log_prob(actions[:, i]) for i, d in enumerate(distributions))
        entropy = sum(d.entropy() for d in distributions)
        return log_prob, entropy, values

    def act(self, observation, masks, *, deterministic=False, learn=True):
        if self.pending is not None:
            raise RuntimeError("Previous policy action has no reward")
        obs = torch.tensor([observation], dtype=torch.float32)
        mask_tensors = [torch.tensor([mask], dtype=torch.bool) for mask in masks]
        with torch.no_grad():
            heads, value = self.model(obs)
            ds = [Categorical(logits=h.masked_fill(~m, -1e9)) for h, m in zip(heads, mask_tensors)]
            actions = [d.probs.argmax(-1) if deterministic else
                       torch.multinomial(d.probs, 1, generator=self.rng).squeeze(-1) for d in ds]
            log_prob = sum(d.log_prob(a) for d, a in zip(ds, actions))
        action = tuple(int(a.item()) for a in actions)
        if learn:
            self.pending = dict(observation=tuple(observation), masks=tuple(masks), action=action,
                                value=float(value.item()), log_prob=float(log_prob.item()))
        return action

    def observe(self, reward, done):
        if self.pending is None:
            raise RuntimeError("Reward received without a policy action")
        self.buffer.append({**self.pending, "reward": float(reward), "done": bool(done)})
        self.pending = None

    def update(self, *, force=False):
        if self.pending is not None or (self.buffer and not self.buffer[-1]["done"]):
            raise RuntimeError("PPO updates require a complete episode")
        n = len(self.buffer)
        if not n or (n < self.settings["rollout_steps"] and not force):
            return {}
        obs = torch.tensor([t["observation"] for t in self.buffer], dtype=torch.float32)
        actions = torch.tensor([t["action"] for t in self.buffer], dtype=torch.long)
        masks = [torch.tensor([t["masks"][i] for t in self.buffer], dtype=torch.bool)
                 for i in range(len(self.settings["action_sizes"]))]
        old_lp = torch.tensor([t["log_prob"] for t in self.buffer])
        old_values = torch.tensor([t["value"] for t in self.buffer])
        advantages = torch.empty(n)
        gae, next_value = 0.0, 0.0
        for i in reversed(range(n)):
            t = self.buffer[i]
            mask = float(not t["done"])
            # Finite-horizon undiscounted return, lambda=0.95.
            delta = t["reward"] + next_value * mask - t["value"]
            gae = delta + 0.95 * mask * gae
            advantages[i], next_value = gae, t["value"]
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        completed_epochs = 0
        for _ in range(self.settings["update_epochs"]):
            indices = torch.randperm(n, generator=self.rng)
            for batch in indices.split(self.settings["minibatch_size"]):
                lp, entropy, values = self._evaluate(obs[batch], actions[batch], [m[batch] for m in masks])
                ratio = (lp - old_lp[batch]).exp()
                policy_loss = -torch.minimum(ratio * advantages[batch],
                                            ratio.clamp(0.8, 1.2) * advantages[batch]).mean()
                value_loss = (returns[batch] - values).square().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
            completed_epochs += 1
            with torch.no_grad():
                lp, entropy, values = self._evaluate(obs, actions, masks)
                log_ratio = lp - old_lp
                kl = ((log_ratio.exp() - 1) - log_ratio).mean()
            if float(kl) > self.settings["target_kl"]:
                break
        variance = returns.var(unbiased=False)
        stats = dict(transitions=n, update_epochs=completed_epochs, approx_kl=float(kl),
                     clip_fraction=float(((log_ratio.exp() - 1).abs() > 0.2).float().mean()),
                     entropy=float(entropy.mean()), value_loss=float((returns - values).square().mean()),
                     explained_variance=float(1 - (returns - values).var(unbiased=False) / variance)
                     if float(variance) > 1e-8 else None)
        self.buffer.clear()
        return stats

    def save(self, path, *, metadata=None):
        if self.pending is not None:
            raise RuntimeError("Policy checkpoints require an episode boundary")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        torch.save(dict(protocol=PROTOCOL, state_version=STATE_VERSION, settings=self.settings,
                        model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                        rng=self.rng.get_state(), buffer=self.buffer, metadata=metadata or {}), temporary)
        temporary.replace(target)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("protocol") != PROTOCOL or payload.get("state_version") != STATE_VERSION:
            raise ValueError("Policy state/reward protocol differs; legacy policies cannot be reused")
        agent = cls(**payload["settings"])
        agent.model.load_state_dict(payload["model"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.rng.set_state(payload["rng"])
        agent.buffer = payload["buffer"]
        return agent, payload["metadata"]
