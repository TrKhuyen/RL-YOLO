"""KB1 reference, fixed HPO, adaptive PPO, and held-out scratch retraining."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import statistics
from types import SimpleNamespace

import yaml

from ..run_all import MODELS, ROOT, _preflight
from ..search_space import DiscreteSearchSpace
from . import PROTOCOL
from .detector import train_detector, atomic_json, file_hash, training_overrides, isolated_evaluation_rng
from .ppo import QualityPPO
from .state import STATE_NAMES, STATE_VERSION
from .reference import read_recipe, prepare_reference, evaluate_reference, evaluation_spec
from ..adapters.ultralytics_worker import _canonical_metrics
from .report import checkpoint_rows, write_checkpoint_comparison


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="kb3_hyperparameter_optimization/configs/kb3_quality.yaml")
    parser.add_argument("--space-config", default="kb3_hyperparameter_optimization/configs/kb3_default.yaml")
    parser.add_argument("--model", choices=("all", *MODELS), default="yolov8n")
    parser.add_argument("--stage", choices=("baseline", "pilot", "search", "evaluate", "test", "all"), default="baseline",
                        help="baseline (also pilot alias) scores KB1 only, without training")
    parser.add_argument("--data-root", default="pre-data/data/v2i_cleanned")
    parser.add_argument("--data-config", default="kb1_reward_guided_training/configs/pest.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true", help="Continue matching protocol at detector/episode boundaries")
    parser.add_argument("--search-target", type=int,
                        help="Cumulative trials AND episodes to complete; default is one search batch")
    parser.add_argument("--finalize", action="store_true",
                        help="For --stage all, select/retrain/test after the requested batch; locks search")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Real 3-epoch integration only, not research results")
    return parser.parse_args(argv)


def settings_from(args):
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if settings.get("protocol") != PROTOCOL:
        raise ValueError("Quality protocol version mismatch")
    if settings.get("recipe_source") != "kb1_supervised":
        raise ValueError("This protocol requires the actual KB1 supervised args.yaml recipe")
    if settings.get("initialization") != "scratch":
        raise ValueError("This protocol uses KB1 weights only as a comparison reference; KB3 initialization must be scratch")
    raw, recipe, _ = read_recipe(args.model)
    settings["trainer_recipe"] = recipe
    settings["smoke"] = args.smoke
    for target, source in (("warmup_epochs", "warmup_epochs"), ("lrf", "lrf"),
                           ("batch_size", "batch"), ("img_size", "imgsz")):
        if settings.get(target) is None:
            settings[target] = raw[source]
    if args.smoke:
        settings.update(epochs=3, segment_epochs=1, warmup_epochs=0, trials=2, episodes=2,
                        rollout_steps=2, minibatch_size=2, policy_validation_every=1,
                        tuning_seeds=[5001], eval_seeds=[10001],
                        search_batch_size=2, search_patience=0, search_min_epochs=0, search_min_delta=0.0)
        settings["trainer_recipe"] = {**recipe, "nbs": settings["batch_size"], "close_mosaic": 1}
    for name in ("epochs", "segment_epochs", "trials", "episodes", "rollout_steps", "minibatch_size",
                 "policy_validation_every", "batch_size", "img_size", "search_batch_size"):
        if not isinstance(settings.get(name), int) or settings[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if settings["trials"] != settings["episodes"]:
        raise ValueError("Staged A/B search requires equal automatic trial and episode targets")
    interval = settings["policy_validation_every"]
    if settings["search_batch_size"] % interval or settings["episodes"] % interval:
        raise ValueError("Search batch and automatic target must align with saved policy boundaries")
    if args.search_target is not None and (args.search_target <= 0 or args.search_target % interval):
        raise ValueError("--search-target must be positive and align with a policy boundary")
    for name in ("search_patience", "search_min_epochs"):
        if not isinstance(settings.get(name), int) or not 0 <= settings[name] <= settings["epochs"]:
            raise ValueError(f"{name} must be an integer in [0, epochs]")
    if settings["search_patience"] and settings["search_min_epochs"] <= settings["segment_epochs"]:
        raise ValueError("Search stopping must allow an RL decision after the shared prefix")
    if not math.isfinite(settings["search_min_delta"]) or settings["search_min_delta"] < 0:
        raise ValueError("search_min_delta must be finite and nonnegative")
    if settings["epochs"] <= settings["segment_epochs"]:
        raise ValueError("Need at least one decision after the shared fixed prefix")
    if not 0 <= settings["warmup_epochs"] <= settings["segment_epochs"]:
        raise ValueError("Warmup must finish within the shared prefix")
    if not 0 < settings["lrf"] <= 1 or settings["workers"] != 0:
        raise ValueError("lrf must be in (0,1], workers must be zero for live transforms")
    settings["device"] = str(settings["device"])
    if settings["device"] != "cpu" and not settings["device"].isdigit():
        raise ValueError("device must be cpu or a single GPU index")
    search_start = settings["seed"]
    search_end = search_start + max(settings["episodes"], settings["trials"], args.search_target or 0)
    tuning, final = settings["tuning_seeds"], settings["eval_seeds"]
    if not tuning or not final or any(s < 0 for s in [settings["seed"], *tuning, *final]):
        raise ValueError("Need nonnegative search/tuning/evaluation seeds")
    if (len(set(tuning)) != len(tuning) or len(set(final)) != len(final)
            or any(search_start <= s < search_end for s in tuning + final) or set(tuning) & set(final)):
        raise ValueError("Search, tuning and final seeds must be unique and disjoint")
    return settings


def resolved_start(args, settings, space, data_path):
    return dict(model=MODELS[args.model], data=str(Path(data_path).resolve()), seed=settings["seed"],
                **training_overrides(settings, DiscreteSearchSpace(space).initial_values()))


def budget(settings, target=None):
    target = settings["episodes"] if target is None else target
    candidates = math.ceil(target / settings["policy_validation_every"])
    hpo_tuning = candidates * len(settings["tuning_seeds"])
    policy_tuning = candidates * len(settings["tuning_seeds"])
    search = 2 * target + hpo_tuning + policy_tuning
    final = 4 * len(settings["eval_seeds"])
    decisions = math.ceil(settings["epochs"] / settings["segment_epochs"]) - 1
    return dict(pilot_detectors=0, kb1_reference_training_runs=0,
                kb1_reference_inference_runs=0 if settings.get("smoke") else 1,
                search_detectors=search, evaluation_detectors=final,
                hpo_search_detectors=target, ppo_learning_detectors=target,
                hpo_tuning_detectors=hpo_tuning, ppo_tuning_detectors=policy_tuning,
                epochs_per_detector=settings["epochs"], maximum_all_epochs=(search + final) * settings["epochs"],
                maximum_ppo_training_transitions=target * decisions,
                canonical_evaluations_per_detector=settings["epochs"] + 1,
                first_search_batch_detectors=2 * settings["search_batch_size"],
                first_search_batch_maximum_epochs=2 * settings["search_batch_size"] * settings["epochs"],
                search_early_stopping=dict(patience=settings["search_patience"],
                                          min_epochs=settings["search_min_epochs"],
                                          min_delta=settings["search_min_delta"]),
                final_retraining_early_stopping=False)


def fingerprint(args, settings, space):
    # Hash bytes, not timestamps: changed labels/images/code invalidate resume.
    sources = []
    for package in ("kb3_hyperparameter_optimization", "kb1_reward_guided_training"):
        for path in sorted((ROOT / package).rglob("*.py")):
            if "tests" not in path.parts and "yolov5" not in path.parts:
                sources.append((str(path.relative_to(ROOT)), file_hash(path)))
    root = Path(args.data_root).resolve()
    dataset = []
    for split in ("train", "valid", "test"):
        for path in sorted((root / split).rglob("*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".txt"}:
                dataset.append((str(path.relative_to(root)), file_hash(path)))
    dependencies = {name: version(name) for name in
                    ("torch", "torchvision", "ultralytics", "torchmetrics", "faster-coco-eval",
                     "pycocotools", "numpy", "albumentations")}
    from dataclasses import asdict
    identity = dict(protocol=PROTOCOL, state_version=STATE_VERSION, settings=settings,
                    model=args.model, architecture=MODELS[args.model],
                    space=asdict(space), data_config_sha256=file_hash(args.data_config),
                    sources=sources, dataset=dataset, dependencies=dependencies,
                    objective="canonical_mAP50_95_best_all_trained_epochs", scratch=True,
                    smoke=args.smoke, test_used_for_selection=False)
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def validated_result(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(payload["best_checkpoint"])
    if payload.get("failed") or not checkpoint.is_file() or file_hash(checkpoint) != payload["checkpoint_sha256"]:
        raise RuntimeError(f"Invalid completed result or checkpoint: {path}")
    return payload


class Pipeline:
    def __init__(self, args, settings, space, output, data_path):
        self.args, self.settings, self.space = args, settings, space
        self.output, self.data_path = output, data_path

    def baseline(self, *, split="val"):
        return evaluate_reference(self.settings, self.args.data_root, self.output, split=split)

    def detector(self, name, seed, *, parameters=None, agent=None, learn=False, random_schedule=False):
        directory = self.output / name
        result_path = directory / "result.json"
        if result_path.exists() and not learn:
            print(f"SKIP verified completed detector: {name}", flush=True)
            return validated_result(result_path)
        if directory.exists() and any(directory.iterdir()):
            # Interrupted detector / uncommitted PPO episode stays available for audit.
            archived = directory.with_name(directory.name + ".interrupted_" +
                                           datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
            if not directory.resolve().is_relative_to(self.output.resolve()):
                raise ValueError("Archive target must remain inside this run")
            directory.rename(archived)
        detector_settings = dict(self.settings)
        detector_settings["early_stopping"] = (dict(patience=self.settings["search_patience"],
                                                    min_epochs=self.settings["search_min_epochs"],
                                                    min_delta=self.settings["search_min_delta"])
                                               if name.startswith(("hpo/trial_", "ppo/episode_")) else None)
        return train_detector(settings=detector_settings, data_yaml=self.data_path,
                              data_root=self.args.data_root, model=MODELS[self.args.model], seed=seed,
                              output_dir=directory, space_config=self.space, parameters=parameters,
                              agent=agent, learn=learn, random_schedule=random_schedule)

    def search(self):
        settings = self.settings
        progress_path = self.output / "search_progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
        # Tuning/final seeds are held out from all decisions to extend search.
        # Once selection starts, the search population must remain frozen.
        lock_path = self.output / "selection_lock.json"
        if lock_path.exists():
            locked = json.loads(lock_path.read_text(encoding="utf-8"))["target"]
            requested = getattr(self.args, "search_target", None)
            if requested is not None and requested != locked:
                raise ValueError("Selection already started: cannot extend search after seeing held-out results")
            print(f"SKIP frozen search at {locked} trials/episodes per branch", flush=True)
            return locked
        completed = progress.get("completed_target", 0)
        requested = getattr(self.args, "search_target", None)
        target = (requested if requested is not None else
                  progress["target"] if progress.get("status") == "running" else
                  completed if completed and getattr(self.args, "finalize", False) else
                  min(completed + settings["search_batch_size"], max(completed, settings["episodes"])))
        if target < completed:
            raise ValueError("Cannot lower --search-target below completed search")
        atomic_json(progress_path, dict(status="running", target=target, completed_target=completed))
        seed = settings["seed"]
        space = DiscreteSearchSpace(self.space)
        rng = random.Random(seed)
        trials = []
        for i in range(target):
            hp = space.initial_values()
            if i > 0:  # Include the exact KB1 starting configuration before exploring.
                for name, spec in self.space.parameters.items():
                    hp[name] = (math.exp(rng.uniform(math.log(spec.minimum), math.log(spec.maximum)))
                                if name in {"lr0", "weight_decay"} else rng.uniform(spec.minimum, spec.maximum))
            trials.append(self.detector(f"hpo/trial_{i:04d}", seed + i, parameters=hp))
        policy_path = self.output / "ppo" / "latest.pt"
        if policy_path.exists():
            agent, meta = QualityPPO.load(policy_path)
            next_episode, history = meta["next_episode"], meta["history"]
        else:
            agent = QualityPPO(len(STATE_NAMES) + len(space.names), space.action_sizes,
                               seed=seed, rollout_steps=settings["rollout_steps"],
                               minibatch_size=settings["minibatch_size"])
            next_episode, history = 0, []
        if next_episode > target:
            raise ValueError("Requested target precedes saved PPO progress")
        for i in range(next_episode, target):
            record = self.detector(f"ppo/episode_{i:04d}", seed + i, agent=agent, learn=True)
            candidate_boundary = (i + 1) % settings["policy_validation_every"] == 0
            # This boundary is identical in direct and staged execution.
            update = agent.update(force=candidate_boundary)
            history.append({**record, "agent_update": update})
            meta = dict(next_episode=i + 1, history=history)
            if candidate_boundary:
                agent.save(self.output / "ppo" / f"candidate_{i + 1:04d}.pt", metadata=meta)
            # Candidate and episode history commit before advancing the resume cursor.
            agent.save(policy_path, metadata=meta)
        atomic_json(progress_path, dict(status="complete", target=target, completed_target=target,
                                       budget_if_finalized=budget(settings, target),
                                       search_detector_hours=sum(r["elapsed_seconds"] for r in [*trials, *history]) / 3600,
                                       search_actual_epochs=sum(r["final_metrics"]["epoch"] for r in [*trials, *history]),
                                       hpo_best_validation=max(r["metrics"]["map50_95"] for r in trials),
                                       ppo_learning_validation=[r["metrics"]["map50_95"] for r in history],
                                       held_out_seeds_used=False, test_used=False,
                                       interpretation="Exploratory search; learning episodes are not a frozen-policy comparison"))
        print(f"SEARCH BATCH COMPLETE: A={target}, B={target}; inspect search_progress.json. "
              "No held-out tuning/final/test performed.", flush=True)
        return target

    def select(self):
        settings = self.settings
        progress = json.loads((self.output / "search_progress.json").read_text(encoding="utf-8"))
        if progress["status"] != "complete":
            raise ValueError("Finish the current search batch before selection")
        target = progress["completed_target"]
        lock_path = self.output / "selection_lock.json"
        if lock_path.exists():
            if json.loads(lock_path.read_text(encoding="utf-8"))["target"] != target:
                raise ValueError("Frozen search population changed")
        else:
            atomic_json(lock_path, dict(target=target, reason="Held-out tuning starts; no further search extension"))
        trials = [validated_result(self.output / "hpo" / f"trial_{i:04d}" / "result.json") for i in range(target)]
        checkpoints = sorted((self.output / "ppo").glob("candidate_*.pt"))
        expected = [f"candidate_{i:04d}.pt" for i in range(settings["policy_validation_every"], target + 1,
                                                         settings["policy_validation_every"])]
        if [p.name for p in checkpoints] != expected:
            raise ValueError("Saved policy candidates do not match the frozen search population")
        ranked = sorted(enumerate(trials), key=lambda pair: pair[1]["metrics"]["map50_95"], reverse=True)
        tuned = []
        # Match HPO and PPO candidate/seed tuning budgets at each search boundary.
        for index, trial in ranked[:len(checkpoints)]:
            records = [self.detector(f"hpo/tuning_{index:04d}/seed_{s}", s, parameters=trial["hyperparameters"])
                       for s in settings["tuning_seeds"]]
            tuned.append(dict(trial=index, parameters=trial["hyperparameters"],
                              tuning_mean=statistics.mean(r["metrics"]["map50_95"] for r in records)))
        atomic_json(self.output / "hpo_selection.json", dict(selected=max(tuned, key=lambda r: r["tuning_mean"]),
                                                             candidates=tuned, trials=trials, search_target=target))
        policies = []
        for checkpoint in checkpoints:
            frozen, _ = QualityPPO.load(checkpoint)
            records = [self.detector(f"ppo/tuning_{checkpoint.stem}/seed_{s}", s, agent=frozen)
                       for s in settings["tuning_seeds"]]
            policies.append(dict(checkpoint=str(checkpoint.resolve()), sha256=file_hash(checkpoint),
                                 tuning_mean=statistics.mean(r["metrics"]["map50_95"] for r in records)))
        winner = max(policies, key=lambda r: r["tuning_mean"])
        atomic_json(self.output / "ppo_selection.json", dict(selected=winner, candidates=policies, search_target=target))

    def evaluate(self):
        reference = self.baseline()
        self.select()
        hpo = json.loads((self.output / "hpo_selection.json").read_text(encoding="utf-8"))["selected"]
        policy_selection = json.loads((self.output / "ppo_selection.json").read_text(encoding="utf-8"))
        selection = policy_selection["selected"]
        if file_hash(selection["checkpoint"]) != selection["sha256"]:
            raise RuntimeError("Selected frozen policy changed")
        agent, _ = QualityPPO.load(selection["checkpoint"])
        records = {}
        for method in ("default", "random-search", "random-schedule", "ppo"):
            records[method] = [self.detector(f"evaluation/{method}/seed_{s}", s,
                                            parameters=hpo["parameters"] if method == "random-search" else None,
                                            agent=agent if method == "ppo" else None,
                                            random_schedule=method == "random-schedule")
                               for s in self.settings["eval_seeds"]]
        summary = {}
        for method, results in records.items():
            values = [r["metrics"]["map50_95"] for r in results]
            summary[method] = dict(mean=statistics.mean(values),
                                   sample_std=statistics.stdev(values) if len(values) > 1 else None,
                                   seeds=self.settings["eval_seeds"], map50_95=values,
                                   detector_hours=sum(r["elapsed_seconds"] for r in results) / 3600,
                                   actual_epochs=sum(r["final_metrics"]["epoch"] for r in results))
        paired = [a - b for a, b in zip(summary["ppo"]["map50_95"], summary["random-search"]["map50_95"])]
        checkpoints = checkpoint_rows(reference, records)
        costs = {}
        for category, pattern in (("hpo_search", "hpo/trial_*/result.json"),
                                  ("hpo_tuning", "hpo/tuning_*/seed_*/result.json"),
                                  ("ppo_learning", "ppo/episode_*/result.json"),
                                  ("ppo_tuning", "ppo/tuning_*/seed_*/result.json"),
                                  ("final_retraining", "evaluation/*/seed_*/result.json")):
            completed = [json.loads(p.read_text(encoding="utf-8")) for p in self.output.glob(pattern)
                         if not any(".interrupted_" in part for part in p.parts)]
            costs[category] = dict(completed_detectors=len(completed),
                                   detector_hours=sum(r["elapsed_seconds"] for r in completed) / 3600,
                                   actual_epochs=sum(r["final_metrics"]["epoch"] for r in completed))
        atomic_json(self.output / "comparison.json", dict(protocol=PROTOCOL, metrics_split="validation",
                                                         methods=summary, paired_ppo_minus_hpo=paired,
                                                         budget=budget(self.settings, policy_selection["search_target"]), records=records,
                                                         completed_detector_costs=costs,
                                                         independent_policy_training_seeds=1,
                                                         kb1_pretrained_reference=reference,
                                                         best_checkpoint_comparison=checkpoints,
                                                         best_seed_selection_split="validation",
                                                         primary_conclusion_uses="paired final seed results and their means, not the best seed",
                                                         mean_minus_kb1_reference={
                                                             k: v["mean"] - reference["metrics"]["mAP50_95"]
                                                             for k, v in summary.items()} if reference else None,
                                                         kb1_comparison_caveat="Different initialization and historical training budget; not an isolated HPO/RL effect",
                                                         test_evaluated=False))
        write_checkpoint_comparison(self.output, checkpoints)

    def evaluate_test(self):
        """Score frozen validation-selected artifacts; never train or select on test."""
        comparison_path = self.output / "comparison.json"
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        if comparison["protocol"] != PROTOCOL or comparison["metrics_split"] != "validation":
            raise ValueError("Test requires this protocol's completed validation comparison")
        comparison_hash = file_hash(comparison_path)
        spec = evaluation_spec(self.settings, "test")
        reference = self.baseline(split="test")
        records, summary = {}, {}
        cache_root = self.output / "test_scores"
        cache_root.mkdir(exist_ok=True)
        for method, validation_records in comparison["records"].items():
            records[method] = []
            for original in validation_records:
                checkpoint = Path(original["best_checkpoint"])
                expected = original["checkpoint_sha256"]
                if file_hash(checkpoint) != expected:
                    raise RuntimeError("Frozen detector changed before test evaluation")
                cache = cache_root / f"{method}_seed_{original['seed']}.json"
                identity = dict(checkpoint_sha256=expected, evaluation=spec,
                                validation_comparison_sha256=comparison_hash)
                if cache.exists():
                    cached = json.loads(cache.read_text(encoding="utf-8"))
                    if cached["identity"] != identity:
                        raise RuntimeError("Frozen test score identity changed")
                    scores = cached["metrics"]
                else:
                    args = SimpleNamespace(dataset_root=str(Path(self.args.data_root).resolve()),
                                           batch=self.settings["batch_size"], imgsz=self.settings["img_size"],
                                           workers=0, device=self.settings["device"], split="test")
                    with isolated_evaluation_rng():
                        scores = _canonical_metrics(checkpoint, args, original["seed"])
                    if file_hash(checkpoint) != expected:
                        raise RuntimeError("Frozen detector changed during test evaluation")
                    atomic_json(cache, dict(identity=identity, metrics=scores))
                records[method].append({**original, "validation_metrics": original["metrics"],
                                        "canonical_best_scores": scores,
                                        "metrics": {**original["metrics"], "map50_95": scores["mAP50_95"]}})
            values = [r["metrics"]["map50_95"] for r in records[method]]
            summary[method] = dict(mean=statistics.mean(values),
                                   sample_std=statistics.stdev(values) if len(values) > 1 else None,
                                   seeds=[r["seed"] for r in records[method]], map50_95=values)
        # Keep the exact best-seed choices made on validation, even if test reverses their ranking.
        selected = {method: max(values, key=lambda r: r["metrics"]["map50_95"])["best_checkpoint"]
                    for method, values in comparison["records"].items()}
        checkpoints = checkpoint_rows(reference, records, selected_checkpoints=selected)
        atomic_json(self.output / "test_comparison.json", dict(
            protocol=PROTOCOL, metrics_split="test", checkpoint_selection_split="validation",
            training_performed=False, test_used_for_selection=False,
            validation_comparison_sha256=comparison_hash, methods=summary, records=records,
            kb1_pretrained_reference=reference, best_checkpoint_comparison=checkpoints,
            paired_ppo_minus_hpo=[a - b for a, b in zip(summary["ppo"]["map50_95"],
                                                       summary["random-search"]["map50_95"])],
            comparison_caveat=comparison["kb1_comparison_caveat"], independent_policy_training_seeds=1,
        ))
        write_checkpoint_comparison(self.output, checkpoints, filename="checkpoint_comparison_test.csv")


def main(argv=None):
    args = parse_args(argv)
    if args.model == "all":
        from .matrix import run_models
        return run_models(args)
    settings = settings_from(args)
    print(json.dumps({"protocol": PROTOCOL, "model": args.model, "stage": args.stage,
                      "default_all_behavior": "one search batch, then stop for review",
                      "budget_if_finalized_at_target": budget(settings, args.search_target)}, indent=2), flush=True)
    preflight_args = SimpleNamespace(config=args.space_config, with_optuna=False, backend="command",
                                     device=settings["device"], data_root=args.data_root, data_config=args.data_config)
    config, data = _preflight(preflight_args, [args.model])
    if len(data["names"]) != 28:
        raise ValueError("Current canonical pest evaluator requires exactly 28 classes")
    space = prepare_reference(args, settings, config.search_space)
    for name, spec in space.parameters.items():
        hold = 0.0 if spec.additive else 1.0
        if hold not in spec.multipliers:
            raise ValueError(f"{name} requires a hold action")
    identity = fingerprint(args, settings, space)
    if args.resume and not args.output_dir:
        raise ValueError("--resume requires the original --output-dir")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output_dir or ROOT / "kb3_hyperparameter_optimization" /
                  "checkpoint_hyperparameter_optimization" / "quality" / stamp).resolve()
    manifest = output / "manifest.json"
    if args.resume:
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old["sha256"] != identity["sha256"]:
            raise ValueError("Code/data/dependencies/config changed; preserve this run and create a new protocol run")
        if yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8")) != data:
            raise ValueError("Generated dataset.yaml changed since this run was created")
        for filename, expected in (("kb1_recipe.yaml", settings["trainer_recipe"]),
                                   ("starting_train.yaml", resolved_start(args, settings, space, output / "dataset.yaml"))):
            if yaml.safe_load((output / filename).read_text(encoding="utf-8")) != expected:
                raise ValueError(f"{filename} changed since this run was created")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory or --resume")
    if args.check_only:
        print(f"Quality preflight PASS, identity={identity['sha256']}; no training started.", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        atomic_json(manifest, identity)
        (output / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        (output / "kb1_recipe.yaml").write_text(yaml.safe_dump(settings["trainer_recipe"], sort_keys=False), encoding="utf-8")
        (output / "starting_train.yaml").write_text(
            yaml.safe_dump(resolved_start(args, settings, space, output / "dataset.yaml"), sort_keys=False), encoding="utf-8")
        atomic_json(output / "starting_parameters.json", {
            "initial_hyperparameters": DiscreteSearchSpace(space).initial_values(),
            "augmentation_reference_strength": 0.5, "kb1_reference": settings.get("kb1_reference"),
            "checkpoint_role": "comparison_only", "detector_initialization": "scratch",
            "recipe_exceptions": {"maximum_horizon": settings["epochs"],
                                  "search_early_stopping": budget(settings)["search_early_stopping"],
                                  "final_retraining_early_stopping": False,
                                  "independent_seeds": True, "canonical_best_selection": True},
        })
    pipeline = Pipeline(args, settings, space, output, output / "dataset.yaml")
    stages = ("baseline", "search", "evaluate", "test") if args.stage == "all" else (args.stage,)
    for stage in stages:
        atomic_json(output / "status.json", dict(stage=stage, status="running"))
        try:
            if stage in ("baseline", "pilot"):
                pipeline.baseline()
            elif stage == "search":
                pipeline.search()
            elif stage == "evaluate":
                pipeline.evaluate()
            else:
                pipeline.evaluate_test()
        except BaseException as error:
            atomic_json(output / "status.json", dict(stage=stage, status="interrupted", reason=str(error)))
            raise
        atomic_json(output / "status.json", dict(stage=stage, status="complete"))
        if (stage == "search" and args.stage == "all" and not args.finalize
                and not (output / "selection_lock.json").exists()):
            atomic_json(output / "status.json", dict(stage="search", status="awaiting_search_review",
                                                      next_choices=["resume search", "evaluate to freeze search, then test"]))
            print("STOP AFTER SEARCH BATCH. Resume to extend, or --stage evaluate to select and retrain. "
                  "--finalize explicitly enables held-out selection/retraining/test in this command.", flush=True)
            break
    print(f"QUALITY stage complete: {output}", flush=True)


if __name__ == "__main__":
    main()
