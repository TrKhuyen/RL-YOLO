"""Ultralytics worker that preserves optimizer state across KB3 segments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[2] / "yolov8n.pt"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--total-epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dataset-root", help="Required with --canonical-eval")
    parser.add_argument("--canonical-eval", action="store_true")
    parser.add_argument("request_json")
    parser.add_argument("response_json")
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            {key.strip(): float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise RuntimeError(f"no metrics found in {path}")
    return rows


def _value(row: dict[str, float], key: str) -> float:
    if key not in row:
        raise KeyError(f"Ultralytics results.csv is missing {key!r}")
    return row[key]


def _canonical_metrics(checkpoint: Path, args: argparse.Namespace, seed: int) -> dict[str, float]:
    if not args.dataset_root:
        raise ValueError("--dataset-root is required with --canonical-eval")
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "kb1_reward_guided_training"))
    from adapters.ultralytics_adapter import UltralyticsAdapter
    from canonical_eval import evaluate_adapter
    from dataloader import get_pest_dataloader

    loader = get_pest_dataloader(
        args.dataset_root, split="val", batch_size=args.batch, img_size=args.imgsz,
        num_workers=args.workers, shuffle=False, seed=seed,
    )
    device = f"cuda:{args.device}" if str(args.device).isdigit() else str(args.device)
    return evaluate_adapter(
        UltralyticsAdapter(str(checkpoint), device=device), loader, device=device,
    )


def _apply_eval(metrics: dict, evaluated: dict) -> None:
    metrics.update({
        "precision": float(evaluated["precision"]),
        "recall": float(evaluated["recall"]),
        "map50": float(evaluated["mAP50"]),
        "map50_95": float(evaluated["mAP50_95"]),
        "ap_small": float(evaluated["APs"]),
    })
    if metrics["ap_small"] < 0:
        raise RuntimeError("canonical evaluator did not return valid AP_small")


def _row_metrics(row: dict[str, float], epoch: int, elapsed: float, peak_vram: float) -> dict:
    return {
        "epoch": epoch,
        "train_loss": sum(_value(row, k) for k in ("train/box_loss", "train/cls_loss", "train/dfl_loss")),
        "val_loss": sum(_value(row, k) for k in ("val/box_loss", "val/cls_loss", "val/dfl_loss")),
        "precision": _value(row, "metrics/precision(B)"),
        "recall": _value(row, "metrics/recall(B)"),
        "map50": _value(row, "metrics/mAP50(B)"),
        "map50_95": _value(row, "metrics/mAP50-95(B)"),
        "ap_small": 0.0,
        "elapsed_seconds": elapsed,
        "peak_vram_mb": peak_vram,
    }


def _initial_evaluation(args: argparse.Namespace, request: dict, response_path: Path) -> None:
    seed = int(request["seed"])
    if args.canonical_eval:
        measured = _canonical_metrics(Path(args.model), args, seed)
        metrics = {
            "epoch": 0, "train_loss": 0.0, "val_loss": 0.0,
            "precision": 0.0, "recall": 0.0, "map50": 0.0,
            "map50_95": 0.0, "ap_small": 0.0,
            "elapsed_seconds": 0.0, "peak_vram_mb": 0.0,
        }
        _apply_eval(metrics, measured)
    else:
        from ultralytics import YOLO
        result = YOLO(args.model).val(
            data=str(Path(args.data).resolve()), imgsz=args.imgsz, batch=args.batch,
            workers=args.workers, device=args.device, plots=False, verbose=False,
        )
        values = result.results_dict
        metrics = {
            "epoch": 0, "train_loss": 0.0, "val_loss": 0.0,
            "precision": float(values["metrics/precision(B)"]),
            "recall": float(values["metrics/recall(B)"]),
            "map50": float(values["metrics/mAP50(B)"]),
            "map50_95": float(values["metrics/mAP50-95(B)"]),
            "ap_small": 0.0, "elapsed_seconds": 0.0, "peak_vram_mb": 0.0,
        }
    response_path.write_text(json.dumps({"metrics": metrics}, indent=2), encoding="utf-8")


def main() -> None:
    args = _args()
    if args.total_epochs <= 0:
        raise ValueError("--total-epochs must be positive")
    request_path = Path(args.request_json).resolve()
    response_path = Path(args.response_json).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema_version") != 1:
        raise ValueError("unsupported request schema")
    if request.get("mode") == "evaluate_initial":
        _initial_evaluation(args, request, response_path)
        return

    start_epoch = int(request["start_epoch"])
    segment_epochs = int(request["epochs"])
    target_epoch = start_epoch + segment_epochs
    if not 0 <= start_epoch < target_epoch <= args.total_epochs:
        raise ValueError("segment epoch range is outside total training budget")

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    class SegmentDetectionTrainer(DetectionTrainer):
        def final_eval(self):
            # Upstream strips optimizer state; intermediate segments must resume it.
            return None

    run_dir = Path(request["run_dir"]).resolve()
    state_path = run_dir / "ultralytics_worker_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    elapsed_before = float(state.get("elapsed_seconds", 0.0))
    last_checkpoint = Path(state["last_checkpoint"]) if state.get("last_checkpoint") else None
    if start_epoch == 0:
        if last_checkpoint is not None:
            raise RuntimeError("new episode unexpectedly contains a previous checkpoint")
        model = YOLO(args.model)
    else:
        if last_checkpoint is None or not last_checkpoint.is_file():
            raise FileNotFoundError("resume checkpoint is missing for a non-zero segment")
        model = YOLO(str(last_checkpoint))

    hyperparameters = {key: float(value) for key, value in request["hyperparameters"].items()}
    strength = hyperparameters["augmentation_strength"]
    optimizer_applied = False

    def configure_before_data(trainer):
        trainer.args.lr0 = hyperparameters["lr0"]
        trainer.args.weight_decay = hyperparameters["weight_decay"]
        trainer.args.momentum = hyperparameters["momentum"]
        trainer.args.degrees = 20.0 * strength
        trainer.args.translate = 0.2 * strength
        trainer.args.scale = 1.0 * strength
        trainer.args.fliplr = min(1.0, strength)
        trainer.args.flipud = min(1.0, 0.6 * strength)
        trainer.args.mosaic = min(1.0, 2.0 * strength)
        trainer.args.mixup = min(1.0, 0.2 * strength)

    def configure_optimizer_once(trainer):
        nonlocal optimizer_applied
        if optimizer_applied:
            return
        for group in trainer.optimizer.param_groups:
            group["lr"] = hyperparameters["lr0"]
            group["initial_lr"] = hyperparameters["lr0"]
            if float(group.get("weight_decay", 0.0)) > 0:
                group["weight_decay"] = hyperparameters["weight_decay"]
            if "momentum" in group:
                group["momentum"] = hyperparameters["momentum"]
            if "betas" in group:
                group["betas"] = (hyperparameters["momentum"], group["betas"][1])
        optimizer_applied = True

    def stop_at_boundary(trainer):
        if trainer.epoch + 1 >= target_epoch:
            trainer.stop = True

    model.add_callback("on_pretrain_routine_start", configure_before_data)
    model.add_callback("on_train_batch_start", configure_optimizer_once)
    model.add_callback("on_train_epoch_end", stop_at_boundary)
    started = time.perf_counter()
    common = dict(
        data=str(Path(args.data).resolve()), epochs=args.total_epochs, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, device=args.device, project=str(run_dir),
        name="detector", exist_ok=True, optimizer="SGD", seed=int(request["seed"]),
        deterministic=True, patience=args.total_epochs + 1, save=True, val=True, plots=False,
    )
    if start_epoch == 0:
        model.train(trainer=SegmentDetectionTrainer, **common)
    else:
        model.train(trainer=SegmentDetectionTrainer, resume=str(last_checkpoint), **common)
    elapsed = elapsed_before + time.perf_counter() - started
    trainer = model.trainer
    checkpoint = Path(trainer.last).resolve()
    best_checkpoint = Path(trainer.best).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Ultralytics did not create {checkpoint}")
    rows = _read_rows(Path(trainer.csv).resolve())
    last_row = rows[-1]
    csv_epoch = int(round(_value(last_row, "epoch")))
    if csv_epoch not in {target_epoch, target_epoch - 1}:
        raise RuntimeError(f"unexpected final CSV epoch {csv_epoch}; expected {target_epoch}")

    peak_vram = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    metrics = _row_metrics(last_row, target_epoch, elapsed, peak_vram)
    best_row = max(rows, key=lambda row: _value(row, "metrics/mAP50-95(B)"))
    best_epoch = int(round(_value(best_row, "epoch"))) + 1
    best_metrics = _row_metrics(best_row, best_epoch, elapsed, peak_vram)
    if args.canonical_eval:
        _apply_eval(metrics, _canonical_metrics(checkpoint, args, int(request["seed"])))
        if best_checkpoint.is_file():
            _apply_eval(best_metrics, _canonical_metrics(best_checkpoint, args, int(request["seed"])))

    response_path.write_text(json.dumps({
        "metrics": metrics, "checkpoint": str(checkpoint),
        "best_metrics": best_metrics,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint.is_file() else None,
    }, indent=2), encoding="utf-8")
    state_path.write_text(json.dumps({
        "last_checkpoint": str(checkpoint), "elapsed_seconds": elapsed,
        "epoch": target_epoch,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
