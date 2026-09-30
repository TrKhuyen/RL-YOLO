"""KB3-A Random Search: one fixed configuration per full training trial."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from ..cli import add_backend_arguments, build_trainer
from ..config import ParameterConfig, load_config
from .runner import result_document, run_fixed_trial


def _sample(rng: random.Random, name: str, spec: ParameterConfig) -> float:
    if name in {"lr0", "weight_decay"} and spec.minimum > 0:
        return math.exp(rng.uniform(math.log(spec.minimum), math.log(spec.maximum)))
    return rng.uniform(spec.minimum, spec.maximum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="kb3_hyperparameter_optimization/configs/kb3_default.yaml")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", default="kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/traditional_random_search.json")
    add_backend_arguments(parser)
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    config = load_config(args.config)
    trainer = build_trainer(args, config)
    rng = random.Random(config.experiment.seed)
    results = []
    try:
        for trial in range(args.trials):
            hyperparameters = {
                name: _sample(rng, name, spec)
                for name, spec in config.search_space.parameters.items()
            }
            results.append(run_fixed_trial(
                trainer,
                config,
                trial=trial,
                seed=config.experiment.seed,
                hyperparameters=hyperparameters,
                method="random_search",
            ))
    finally:
        trainer.close()
    document = result_document("random_search", config, results)
    successful = [result for result in results if not result.failed]
    best = max(successful, key=lambda item: float(item.metrics["map50_95"])) if successful else None
    document["objective"] = "validation_map50_95"
    document["best_value"] = float(best.metrics["map50_95"]) if best else None
    document["best_params"] = best.hyperparameters if best else None
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
