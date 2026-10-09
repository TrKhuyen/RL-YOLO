"""Run KB3 scratch baselines, HPO, adaptive RL, and frozen-seed evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

from .config import load_config

MODELS = {
    "yolov8n": "yolov8n.yaml", "yolov8s": "yolov8s.yaml",
    "yolov11n": "yolo11n.yaml", "yolov11s": "yolo11s.yaml",
    "yolo26n": "yolo26n.yaml",
}
METHODS = ("default", "random-search", "random-schedule", "bandit", "ppo")
ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", *MODELS), default="all")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--with-optuna", action="store_true", help="Also run optional Optuna/TPE (requires optuna)")
    parser.add_argument("--config", default="kb3_hyperparameter_optimization/configs/kb3_default.yaml")
    parser.add_argument("--data-root", default="pre-data/data/v2i_cleanned")
    parser.add_argument("--data-config", default="kb1_reward_guided_training/configs/pest.yaml")
    parser.add_argument("--device", default="0", help="YOLO device: 0, cuda, cuda:0, or cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--segment-epochs", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", help="New empty directory; default is a unique timestamped directory")
    parser.add_argument("--backend", choices=("command", "simulated"), default="command")
    parser.add_argument("--smoke", action="store_true", help="2 epochs, 1 epoch/segment, 1 trial/episode, one evaluation seed")
    parser.add_argument("--check-only", action="store_true", help="Check configuration, dependencies, dataset and device without training")
    args = parser.parse_args(argv)
    if args.smoke:
        args.epochs, args.segment_epochs, args.trials, args.episodes = 2, 1, 1, 1
        args.eval_seeds = args.eval_seeds[:1]
    positive = (args.epochs, args.segment_epochs, args.trials, args.episodes, args.batch_size, args.img_size)
    if min(positive) <= 0 or args.segment_epochs > args.epochs or args.workers < 0:
        parser.error("budgets/batch/image size must be positive, segment <= epochs, workers >= 0")
    if args.seed < 0 or min(args.eval_seeds) < 0 or len(set(args.eval_seeds)) != len(args.eval_seeds):
        parser.error("seeds must be non-negative and evaluation seeds unique")
    if set(args.eval_seeds) & set(range(args.seed, args.seed + args.episodes)):
        parser.error("evaluation seeds must not overlap policy training/search seeds")
    if args.device == "cuda":
        args.device = "0"
    elif args.device.startswith("cuda:"):
        args.device = args.device.split(":", 1)[1]
    if args.device != "cpu" and not args.device.isdigit():
        parser.error("use a single GPU index or cpu (multi-GPU workers are not supported)")
    return args


def _preflight(args, models):
    config = load_config(args.config)
    if args.with_optuna and importlib.util.find_spec("optuna") is None:
        raise SystemExit("Optuna is missing. Install with: uv pip install --python .venv optuna")
    import torch
    if args.backend == "simulated":
        print("SIMULATED: verifies software flow only; no detector training or research metrics.", flush=True)
        return config, None
    if args.device != "cpu" and (not torch.cuda.is_available() or int(args.device) >= torch.cuda.device_count()):
        raise SystemExit("Requested CUDA device is unavailable; use --device cpu or configure GPU PyTorch.")
    for module in ("pycocotools", "faster_coco_eval", "torchmetrics"):
        if importlib.util.find_spec(module) is None:
            raise SystemExit(f"Canonical evaluation dependency missing: {module}")
    data = yaml.safe_load(Path(args.data_config).read_text(encoding="utf-8"))
    root = Path(args.data_root).resolve()
    data.update(path=root.as_posix(), train="train/images", val="valid/images", test="test/images")
    if not data.get("names") or int(data.get("nc", len(data["names"]))) != len(data["names"]):
        raise SystemExit("Dataset class names/nc are missing or inconsistent")
    for split in ("train", "valid", "test"):
        images, labels = root / split / "images", root / split / "labels"
        stems = {p.stem for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}} if images.is_dir() else set()
        label_stems = {p.stem for p in labels.glob("*.txt")}
        if not stems or stems != label_stems:
            raise SystemExit(f"Missing or unpaired images/labels: {split}")
        print(f"PASS dataset {split}: {len(stems)} images", flush=True)
    from ultralytics import YOLO
    for model in models:
        detector = YOLO(MODELS[model])
        if detector.ckpt:
            raise RuntimeError("Scratch model unexpectedly loaded a pretrained checkpoint")
        print(f"PASS {model}: random initialization from {MODELS[model]}", flush=True)
    return config, data


def _run(module, arguments, result=None):
    command = [sys.executable, "-u", "-m", f"kb3_hyperparameter_optimization.{module}", *map(str, arguments)]
    print("RUN", shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if result is not None:
        payload = json.loads(Path(result).read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("trials", payload.get("episodes", payload.get("history", [])))
        if not records or any(record.get("failed", False) for record in records):
            raise RuntimeError(f"Detector run failed; inspect {result} and segment request/response files")
        if isinstance(payload, dict) and "best_params" in payload and not payload["best_params"]:
            raise RuntimeError(f"No successful HPO configuration: {result}")


def main():
    args = parse_args()
    # Resolve relative input paths from the repository, including Windows Python under Bash.
    models = list(MODELS) if args.model == "all" else [args.model]
    config, data = _preflight(args, models)
    if args.check_only:
        print("KB3 preflight PASS; training was not started.", flush=True)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output_dir or f"kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/run_all/{stamp}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "dataset.yaml"
    if data is not None:
        data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (output / "run_manifest.json").write_text(json.dumps({
        **vars(args), "models": models, "initialization": "scratch", "selection_split": "val",
        "test_evaluated": False, "architectures": {m: MODELS[m] for m in models},
    }, indent=2), encoding="utf-8")
    methods = list(dict.fromkeys(args.methods)) + (["optuna"] if args.with_optuna else [])
    for model in models:
        comparison = []
        for method in methods:
            method_dir = output / model / method
            method_dir.mkdir(parents=True)
            experiment = replace(config.experiment, seed=args.seed, total_epochs=args.epochs,
                                 segment_epochs=args.segment_epochs, output_dir=str(method_dir),
                                 patience_segments=args.epochs + 1, max_seconds=None)
            run_config = replace(config, experiment=experiment)
            if args.smoke:
                run_config = replace(run_config, ppo=replace(config.ppo, update_epochs=1))
            config_path = method_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(asdict(run_config), sort_keys=False), encoding="utf-8")
            common = ["--config", config_path, "--backend", args.backend]
            if args.backend == "command":
                worker = ["python", "-m", "kb3_hyperparameter_optimization.adapters.ultralytics_worker",
                          "--model", MODELS[model], "--data", str(data_path.resolve()),
                          "--total-epochs", str(args.epochs), "--imgsz", str(args.img_size),
                          "--batch", str(args.batch_size), "--workers", str(args.workers),
                          "--device", args.device, "--dataset-root", str(Path(args.data_root).resolve()), "--canonical-eval"]
                common += ["--trainer-command", shlex.join(worker)]
            final = method_dir / "evaluation.json"
            seeds = ["--seeds", *args.eval_seeds]
            if method == "default":
                _run("traditional_hpo.default_baseline", [*common, *seeds, "--output", final], final)
            elif method in {"random-search", "optuna"}:
                search = method_dir / "search.json"
                module = "random_search" if method == "random-search" else "optuna_search"
                _run(f"traditional_hpo.{module}", [*common, "--trials", args.trials, "--output", search], search)
                _run("traditional_hpo.evaluate_config", [*common, "--hyperparameters-json", search,
                     "--method-name", method, *seeds, "--output", final], final)
            elif method == "random-schedule":
                # No policy search; sample schedules on the same unseen detector seeds.
                random_results = []
                for index, seed in enumerate(args.eval_seeds):
                    seed_config = replace(run_config, experiment=replace(experiment, seed=seed))
                    seed_path = method_dir / f"config_seed_{seed}.yaml"
                    seed_path.write_text(yaml.safe_dump(asdict(seed_config), sort_keys=False), encoding="utf-8")
                    result = method_dir / f"seed_{seed}.json"
                    seed_common = [*common]
                    seed_common[1] = seed_path
                    _run("adaptive_rl.random_schedule", [*seed_common, "--episodes", 1, "--output", result], result)
                    random_results.extend(json.loads(result.read_text(encoding="utf-8"))["episodes"])
                final.write_text(json.dumps(random_results, indent=2), encoding="utf-8")
            else:
                checkpoint = method_dir / f"{method}_agent.pt"
                _run("adaptive_rl.train_agent", [*common, "--agent", method, "--episodes", args.episodes,
                     "--checkpoint", checkpoint], method_dir / "adaptive_rl" / "training_history.json")
                _run("adaptive_rl.evaluate_policy", [*common, "--agent", method, "--checkpoint", checkpoint,
                     *seeds, "--output", final], final)
            comparison.append(f"{method}={final}")
        _run("compare_hpo", [*comparison, "--output", output / model / "comparison.csv"])
    print(f"KB3 COMPLETE: {output.resolve()}\nFrozen methods compared on validation; test set remains reserved.", flush=True)


if __name__ == "__main__":
    main()
