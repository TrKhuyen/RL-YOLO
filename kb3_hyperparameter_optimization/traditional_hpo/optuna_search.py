"""KB3-A Optuna/TPE: one fixed configuration per full training trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..cli import add_backend_arguments, build_trainer
from ..config import load_config
from .runner import result_document, run_fixed_trial


def main() -> None:
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Optuna is required for KB3-A TPE; install the optional dependency first") from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="kb3_hyperparameter_optimization/configs/kb3_default.yaml")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", default="kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/traditional_optuna.json")
    add_backend_arguments(parser)
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    config = load_config(args.config)
    trainer = build_trainer(args, config)
    completed = []

    def objective(trial):
        hyperparameters = {}
        for name, spec in config.search_space.parameters.items():
            use_log = name in {"lr0", "weight_decay"} and spec.minimum > 0
            hyperparameters[name] = trial.suggest_float(name, spec.minimum, spec.maximum, log=use_log)
        result = run_fixed_trial(
            trainer,
            config,
            trial=trial.number,
            seed=config.experiment.seed,
            hyperparameters=hyperparameters,
            method="optuna_tpe",
        )
        completed.append(result)
        if result.failed:
            raise RuntimeError(result.failure_reason or "detector trial failed")
        value = float(result.metrics["map50_95"])
        trial.set_user_attr("result", result.to_dict())
        return value

    try:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.experiment.seed),
        )
        study.optimize(objective, n_trials=args.trials, catch=(RuntimeError,))
    finally:
        trainer.close()
    document = result_document("optuna_tpe", config, completed)
    document["objective"] = "validation_map50_95"
    document["best_value"] = study.best_value if any(t.value is not None for t in study.trials) else None
    document["best_params"] = study.best_params if document["best_value"] is not None else None
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

