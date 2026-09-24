"""KB3-B control: change hyperparameters randomly after every segment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..agents import RandomAgent
from ..cli import add_backend_arguments, build_trainer
from ..config import load_config
from ..envs import YoloHPOEnv
from ..runner import run_episode, summary_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="kb3_hyperparameter_optimization/configs/kb3_default.yaml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", default="runs/kb3/adaptive_random_schedule.json")
    add_backend_arguments(parser)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    config = load_config(args.config)
    trainer = build_trainer(args, config)
    env = YoloHPOEnv(trainer, config)
    agent = RandomAgent(env.space.action_sizes, config.experiment.seed)
    results = []
    try:
        for episode in range(args.episodes):
            summary, update = run_episode(
                env,
                agent,
                episode=episode,
                seed=config.experiment.seed + episode,
                learn=False,
            )
            results.append(summary_dict(summary, update))
    finally:
        env.close()
    document = {
        "schema_version": 2,
        "scenario": "KB3-B",
        "optimization": "adaptive_control",
        "method": "random_schedule",
        "budget": {
            "episodes": args.episodes,
            "epochs_per_episode": config.experiment.total_epochs,
            "segment_epochs": config.experiment.segment_epochs,
            "maximum_total_epochs": args.episodes * config.experiment.total_epochs,
        },
        "episodes": results,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

