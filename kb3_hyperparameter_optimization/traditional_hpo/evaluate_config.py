"""Retrain a frozen KB3-A configuration on unseen evaluation seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..cli import add_backend_arguments, build_trainer
from ..config import load_config
from .runner import result_document, run_fixed_trial


def _load_hyperparameters(path: str) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "best_params" in payload:
        payload = payload["best_params"]
    if not isinstance(payload, dict):
        raise ValueError("hyperparameter JSON must be a mapping or contain best_params")
    return {str(name): float(value) for name, value in payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="kb3_hyperparameter_optimization/configs/kb3_default.yaml")
    parser.add_argument("--hyperparameters-json", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--method-name", default="traditional_hpo_best")
    parser.add_argument("--output", default="kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/traditional_best_evaluation.json")
    add_backend_arguments(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    hyperparameters = _load_hyperparameters(args.hyperparameters_json)
    trainer = build_trainer(args, config)
    results = []
    try:
        for trial, seed in enumerate(args.seeds):
            results.append(run_fixed_trial(
                trainer,
                config,
                trial=trial,
                seed=seed,
                hyperparameters=hyperparameters,
                method=args.method_name,
            ))
    finally:
        trainer.close()
    document = result_document(args.method_name, config, results)
    document["phase"] = "frozen_configuration_evaluation"
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
