"""Ultralytics worker that preserves optimizer state across KB3 segments."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import sys
import time
from pathlib import Path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.yaml", help="Architecture YAML; pretrained weights are not used")
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


def _validate_model(model: str) -> None:
    if Path(model).suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("KB3 trains from scratch: --model must be an architecture YAML, not a .pt checkpoint")


def _training_arguments(args, request) -> dict:
    return dict(
        model=args.model, pretrained=False,
        data=str(Path(args.data).resolve()), epochs=args.total_epochs, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, device=args.device,
        project=str(Path(request["run_dir"]).resolve()), name="detector", exist_ok=True,
        optimizer="SGD", seed=int(request["seed"]), deterministic=True,
        patience=args.total_epochs + 1, save=True, val=True, plots=False,
        # RL owns the actual learning rate/momentum and augmentation schedule.
        warmup_epochs=0.0, lrf=1.0, cos_lr=False, close_mosaic=0,
        # No pending accumulated gradients may be dropped at segment boundaries.
        nbs=args.batch, amp=False,
    )


def _configure_arguments(trainer, hyperparameters) -> None:
    strength = hyperparameters["augmentation_strength"]
    for name in ("lr0", "weight_decay", "momentum"):
        setattr(trainer.args, name, hyperparameters[name])
    trainer.args.warmup_epochs = 0.0
    trainer.args.lrf = 1.0
    trainer.args.cos_lr = False
    trainer.args.close_mosaic = 0
    trainer.args.degrees = 20.0 * strength
    trainer.args.translate = 0.2 * strength
    trainer.args.scale = strength
    trainer.args.fliplr = strength
    trainer.args.flipud = 0.6 * strength
    trainer.args.mosaic = min(1.0, 2.0 * strength)
    trainer.args.mixup = 0.2 * strength


def _configure_optimizer(trainer, hyperparameters) -> None:
    # Apply after resume has restored optimizer state, before scheduler.step().
    for group in trainer.optimizer.param_groups:
        group["lr"] = group["initial_lr"] = hyperparameters["lr0"]
        if float(group.get("weight_decay", 0.0)) > 0:
            group["weight_decay"] = hyperparameters["weight_decay"]
        if "momentum" in group:
            group["momentum"] = hyperparameters["momentum"]
        if "betas" in group:
            group["betas"] = (hyperparameters["momentum"], group["betas"][1])
    trainer.scheduler.base_lrs = [hyperparameters["lr0"]] * len(trainer.optimizer.param_groups)


def _segment_trainer_class():
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    class SegmentDetectionTrainer(DetectionTrainer):
        def final_eval(self):
            # Upstream strips optimizer state; intermediate segments must resume it.
            return None

        def save_model(self):
            saved = super().save_model()
            if saved is False or not self.last.is_file():
                return saved
            payload = torch.load(self.last, map_location="cpu", weights_only=False)
            # Upstream saves EMA as the model and fp16 optimizer buffers. Keep
            # the live training weights and full precision state for continuity.
            payload["kb3_training_state"] = {
                "model": deepcopy(unwrap_model(self.model).state_dict()),
                "optimizer": deepcopy(self.optimizer.state_dict()),
                "ema": deepcopy(self.ema.ema.state_dict()),
                "scheduler": self.scheduler.state_dict(),
            }
            torch.save(payload, self.last)
            if self.best_fitness == self.fitness:
                torch.save(payload, self.best)
            return saved

        def _load_checkpoint_state(self, ckpt):
            super()._load_checkpoint_state(ckpt)
            state = ckpt.get("kb3_training_state")
            if state is None:
                raise RuntimeError("KB3 resume checkpoint lacks the live training state")
            unwrap_model(self.model).load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.ema.ema.load_state_dict(state["ema"])
            self.scheduler.load_state_dict(state["scheduler"])

    return SegmentDetectionTrainer


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
        args.dataset_root, split=getattr(args, "split", "val"), batch_size=args.batch, img_size=args.imgsz,
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
    # YOLO26 uses L1 regression instead of DFL. Require the recorded loss;
    # silently treating a missing component as zero would bias HPO rewards.
    regression = 'l1_loss' if 'train/l1_loss' in row else 'dfl_loss'
    return {
        "epoch": epoch,
        "train_loss": sum(_value(row, f'train/{k}') for k in ('box_loss', 'cls_loss', regression)),
        "val_loss": sum(_value(row, f'val/{k}') for k in ('box_loss', 'cls_loss', regression)),
        "precision": _value(row, "metrics/precision(B)"),
        "recall": _value(row, "metrics/recall(B)"),
        "map50": _value(row, "metrics/mAP50(B)"),
        "map50_95": _value(row, "metrics/mAP50-95(B)"),
        "ap_small": 0.0,
        "elapsed_seconds": elapsed,
        "peak_vram_mb": peak_vram,
    }


def _initial_evaluation(args: argparse.Namespace, request: dict, response_path: Path) -> None:
    import torch

    seed = int(request["seed"])
    trainer = _segment_trainer_class()(overrides=_training_arguments(args, request))
    _configure_arguments(trainer, request["hyperparameters"])
    # Build the seeded random model with the dataset's nc (not COCO's nc).
    trainer._setup_train()
    trainer.epoch = -1
    started = time.perf_counter()
    # Validator expects the loss structure normally created by the first train
    # batch. Measure one validation batch without gradients or a weight update.
    trainer.ema.ema.eval()
    with torch.no_grad():
        batch = trainer.preprocess_batch(next(iter(trainer.test_loader)))
        losses, trainer.loss_items = trainer.ema.ema.loss(batch)
        trainer.loss = losses.sum()
    trainer.loss_names = tuple(trainer.loss_items)
    values, _ = trainer.validate()
    metrics = {
        "epoch": 0, "train_loss": 0.0,
        "val_loss": sum(float(value) for key, value in values.items() if key.startswith("val/") and key.endswith("_loss")),
        "precision": float(values["metrics/precision(B)"]),
        "recall": float(values["metrics/recall(B)"]),
        "map50": float(values["metrics/mAP50(B)"]),
        "map50_95": float(values["metrics/mAP50-95(B)"]),
        "ap_small": 0.0, "elapsed_seconds": 0.0, "peak_vram_mb": 0.0,
    }
    if args.canonical_eval:
        initial_path = Path(request["run_dir"]) / "initial_model.pt"
        torch.save({"model": deepcopy(trainer.model).half(), "train_args": vars(trainer.args)}, initial_path)
        _apply_eval(metrics, _canonical_metrics(initial_path, args, seed))
    metrics["elapsed_seconds"] = time.perf_counter() - started
    state_path = Path(request["run_dir"]) / "ultralytics_worker_state.json"
    state_path.write_text(json.dumps({"epoch": 0, "elapsed_seconds": metrics["elapsed_seconds"]}), encoding="utf-8")
    response_path.write_text(json.dumps({"metrics": metrics}, indent=2), encoding="utf-8")


def main() -> None:
    args = _args()
    _validate_model(args.model)
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
    SegmentDetectionTrainer = _segment_trainer_class()

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
        if int(state.get("epoch", -1)) != start_epoch:
            raise RuntimeError("resume state epoch does not match the requested segment")
        if last_checkpoint is None or not last_checkpoint.is_file():
            raise FileNotFoundError("resume checkpoint is missing for a non-zero segment")
        model = YOLO(str(last_checkpoint))

    hyperparameters = {key: float(value) for key, value in request["hyperparameters"].items()}
    def configure_before_data(trainer):
        _configure_arguments(trainer, hyperparameters)

    def configure_optimizer(trainer):
        _configure_optimizer(trainer, hyperparameters)

    def stop_at_boundary(trainer):
        if trainer.epoch + 1 >= target_epoch:
            trainer.stop = True

    model.add_callback("on_pretrain_routine_start", configure_before_data)
    model.add_callback("on_pretrain_routine_end", configure_optimizer)
    model.add_callback("on_train_epoch_end", stop_at_boundary)
    started = time.perf_counter()
    common = _training_arguments(args, request)
    common.pop("model")
    if start_epoch == 0:
        model.train(trainer=SegmentDetectionTrainer, **common)
    else:
        model.train(trainer=SegmentDetectionTrainer, resume=str(last_checkpoint), **common)
    trainer = model.trainer
    checkpoint = Path(trainer.last).resolve()
    best_checkpoint = Path(trainer.best).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Ultralytics did not create {checkpoint}")
    rows = _read_rows(Path(trainer.csv).resolve())
    last_row = rows[-1]
    csv_epoch = int(round(_value(last_row, "epoch")))
    if csv_epoch != target_epoch:
        raise RuntimeError(f"unexpected final CSV epoch {csv_epoch}; expected {target_epoch}")

    peak_vram = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    elapsed = elapsed_before + time.perf_counter() - started
    metrics = _row_metrics(last_row, target_epoch, elapsed, peak_vram)
    # Use the checkpoint's actual epoch; rounded CSV scores may have ties.
    import torch
    best_epoch = int(torch.load(best_checkpoint, map_location="cpu", weights_only=False)["epoch"]) + 1
    best_row = next(row for row in rows if int(round(_value(row, "epoch"))) == best_epoch)
    best_metrics = _row_metrics(best_row, best_epoch, elapsed, peak_vram)
    if args.canonical_eval:
        _apply_eval(metrics, _canonical_metrics(checkpoint, args, int(request["seed"])))
        if best_checkpoint.is_file():
            _apply_eval(best_metrics, _canonical_metrics(best_checkpoint, args, int(request["seed"])))

    elapsed = elapsed_before + time.perf_counter() - started
    metrics["elapsed_seconds"] = best_metrics["elapsed_seconds"] = elapsed

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
