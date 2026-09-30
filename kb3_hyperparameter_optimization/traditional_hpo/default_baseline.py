"""KB3-A default baseline using the initial configuration for every seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..cli import add_backend_arguments, build_trainer
from ..config import load_config
from .runner import result_document, run_fixed_trial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="kb3_hyperparameter_optimization/configs/kb3_default.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", default="kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/traditional_default.json")
    add_backend_arguments(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    trainer = build_trainer(args, config)
    hyperparameters = {
        name: spec.initial for name, spec in config.search_space.parameters.items()
    }
    results = []
    try:
        for trial, seed in enumerate(args.seeds):
            results.append(run_fixed_trial(
                trainer,
                config,
                trial=trial,
                seed=seed,
                hyperparameters=hyperparameters,
                method="default",
            ))
    finally:
        trainer.close()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result_document("default", config, results), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
