"""Train PPO, bandit, or random policies on the KB3-B environment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .agents import IndependentBanditAgent, PPOAgent, RandomAgent
from .cli import add_backend_arguments, build_trainer
from .config import load_config
from .envs import YoloHPOEnv
from .runner import run_episode, summary_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="kb3_hyperparameter_optimization/configs/kb3_default.yaml")
    parser.add_argument("--agent", choices=("ppo", "bandit", "random"), default="ppo")
    parser.add_argument("--episodes", type=int, default=20, help="Total target episode count")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", help="Output policy checkpoint")
    parser.add_argument("--resume-checkpoint", help="Resume policy and episode counter")
    add_backend_arguments(parser)
    return parser.parse_args()


def _metadata_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(checkpoint.suffix + ".meta.json")


def _config_snapshot(config) -> dict:
    """Return the JSON representation so tuples compare correctly after reload."""
    return json.loads(json.dumps(asdict(config)))


def _save_state(agent, checkpoint: Path, metadata: dict) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    agent.save(str(temporary))
    temporary.replace(checkpoint)
    metadata_path = _metadata_path(checkpoint)
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata_tmp.replace(metadata_path)


def _new_agent(args, env, config):
    if args.agent == "ppo":
        return PPOAgent(env.observation_size, env.space.action_sizes, config.ppo, config.experiment.seed, args.device)
    if args.agent == "bandit":
        return IndependentBanditAgent(env.space.action_sizes, config.experiment.seed)
    return RandomAgent(env.space.action_sizes, config.experiment.seed)


def _resume_agent(args, env, config):
    checkpoint = Path(args.resume_checkpoint)
    metadata_path = _metadata_path(checkpoint)
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("resume checkpoint and its .meta.json sidecar are both required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("agent") != args.agent:
        raise ValueError("resume checkpoint agent type does not match --agent")
    if metadata.get("config_snapshot") not in (None, _config_snapshot(config)):
        raise ValueError("current KB3 config differs from the checkpoint config")
    if args.agent == "ppo":
        agent = PPOAgent.load(str(checkpoint), args.device)
        if agent.model.encoder[0].in_features != env.observation_size:
            raise ValueError("resume checkpoint observation size does not match current environment")
        if tuple(agent.model.action_sizes) != tuple(env.space.action_sizes):
            raise ValueError("resume checkpoint action space does not match current configuration")
    elif args.agent == "bandit":
        agent = IndependentBanditAgent.load(str(checkpoint))
        if tuple(agent.action_sizes) != tuple(env.space.action_sizes):
            raise ValueError("resume checkpoint action space does not match current configuration")
    else:
        raise ValueError("random policy resume is not meaningful; start a new run")
    return agent, metadata


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    config = load_config(args.config)
    env = YoloHPOEnv(build_trainer(args, config), config)
    output_dir = Path(config.experiment.output_dir) / "adaptive_rl"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_checkpoint:
        agent, metadata = _resume_agent(args, env, config)
        start_episode = int(metadata["next_episode"])
        history = list(metadata.get("history", []))
        checkpoint = Path(args.checkpoint or args.resume_checkpoint)
    else:
        agent, start_episode, history = _new_agent(args, env, config), 0, []
        checkpoint = Path(args.checkpoint or output_dir / f"{args.agent}_agent.pt")
    if start_episode >= args.episodes:
        raise ValueError("checkpoint has already reached or exceeded --episodes")
    try:
        for episode in range(start_episode, args.episodes):
            summary, update = run_episode(env, agent, episode=episode, seed=config.experiment.seed + episode)
            history.append(summary_dict(summary, update))
            metadata = {
                "schema_version": 2, "scenario": "KB3-B", "agent": args.agent,
                "next_episode": episode + 1, "target_episodes": args.episodes,
                "config": str(Path(args.config).resolve()),
                "config_snapshot": _config_snapshot(config), "history": history,
            }
            _save_state(agent, checkpoint, metadata)
            (output_dir / "training_history.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(json.dumps(history[-1], ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
