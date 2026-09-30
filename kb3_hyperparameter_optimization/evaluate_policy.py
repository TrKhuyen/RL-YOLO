"""Evaluate a frozen KB3 policy without updating it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import IndependentBanditAgent, PPOAgent
from .cli import add_backend_arguments, build_trainer
from .config import load_config
from .envs import YoloHPOEnv
from .runner import run_episode, summary_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="kb3_hyperparameter_optimization/configs/kb3_default.yaml")
    parser.add_argument("--agent", choices=("ppo", "bandit"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/adaptive_rl/evaluation.json")
    add_backend_arguments(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    trainer = build_trainer(args, config)
    env = YoloHPOEnv(trainer, config)
    agent = (
        PPOAgent.load(args.checkpoint, args.device)
        if args.agent == "ppo"
        else IndependentBanditAgent.load(args.checkpoint)
    )
    summaries = []
    try:
        for index, seed in enumerate(args.seeds):
            summary, _ = run_episode(
                env, agent, episode=index, seed=seed, deterministic=True, learn=False
            )
            summaries.append(summary_dict(summary, {}))
    finally:
        env.close()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

