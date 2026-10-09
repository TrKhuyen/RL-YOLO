"""Train one live scratch detector; evaluate and select the EMA each epoch."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..adapters.ultralytics_worker import _canonical_metrics, _apply_eval, _validate_model
from ..core import Metrics
from ..search_space import DiscreteSearchSpace
from .state import observation, action_masks, best_gain


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weights_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def preserve_transform_rng(old, new, seen=None):
    """Carry private Albumentations RNG streams into the rebuilt same family."""
    seen = set() if seen is None else seen
    pair = (id(old), id(new))
    if pair in seen or type(old) is not type(new):
        return
    seen.add(pair)
    if hasattr(old, "random_generator") and hasattr(new, "random_generator"):
        new.random_generator.bit_generator.state = deepcopy(old.random_generator.bit_generator.state)
    if hasattr(old, "py_random") and hasattr(new, "py_random"):
        new.py_random.setstate(old.py_random.getstate())
    for attribute in ("transforms", "pre_transform", "transform"):
        before, after = getattr(old, attribute, None), getattr(new, attribute, None)
        if before is None or after is None:
            continue
        if isinstance(before, (list, tuple)) and isinstance(after, (list, tuple)):
            for left, right in zip(before, after):
                preserve_transform_rng(left, right, seen)
        else:
            preserve_transform_rng(before, after, seen)


@contextmanager
def isolated_evaluation_rng():
    """Shared FP32 arithmetic for reference/trained weights, without altering training state."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = torch.backends.cudnn
    backend_state = (cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=True)
            cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32 = False, True, False
            torch.backends.cuda.matmul.allow_tf32 = False
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = backend_state
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def configure(trainer, parameters, *, rebuild=False):
    strength = parameters["augmentation_strength"]
    for name in ("lr0", "momentum", "weight_decay"):
        setattr(trainer.args, name, parameters[name])
    # strength=0.5 reproduces the actual KB1 geometric/mixing recipe.
    # Color jitter stays fixed, as in the reference recipe.
    recipe = trainer._quality_recipe
    probabilities = {"flipud", "fliplr", "mosaic", "mixup", "cutmix", "copy_paste"}
    for name in ("degrees", "translate", "scale", "shear", "perspective", *sorted(probabilities)):
        value = float(recipe.get(name, 0.0)) * strength / 0.5
        setattr(trainer.args, name, min(1.0, value) if name in probabilities else value)
    apply_close_mosaic(trainer)
    if hasattr(trainer, "optimizer"):
        # Match native decay normalization; do not tie it to warmup's changing accumulation.
        decay_scale = trainer.batch_size * max(round(trainer.args.nbs / trainer.batch_size), 1) / trainer.args.nbs
        for group in trainer.optimizer.param_groups:
            group["initial_lr"] = parameters["lr0"]
            if group.get("weight_decay", 0.0) > 0:
                group["weight_decay"] = parameters["weight_decay"] * decay_scale
            if "momentum" in group:
                group["momentum"] = parameters["momentum"]
        # Scheduler applies the shared KB1 linear factor after on_train_epoch_start.
        trainer.scheduler.base_lrs = [parameters["lr0"]] * len(trainer.optimizer.param_groups)
    if rebuild:
        dataset = trainer.train_loader.dataset
        transforms = dataset.build_transforms(hyp=trainer.args)
        preserve_transform_rng(dataset.transforms, transforms)
        dataset.transforms = transforms


def apply_close_mosaic(trainer):
    """A policy action must not reopen mixing in the recipe's final phase."""
    close = trainer.args.close_mosaic
    if close and getattr(trainer, "epoch", -1) >= trainer.epochs - close:
        for name in ("mosaic", "mixup", "cutmix", "copy_paste"):
            setattr(trainer.args, name, 0.0)


def training_overrides(settings, parameters):
    """The same resolved starting recipe used for training and its YAML snapshot."""
    return {
        **settings["trainer_recipe"],
        "pretrained": False, "resume": False,
        "epochs": settings["epochs"], "imgsz": settings["img_size"], "batch": settings["batch_size"],
        "device": settings["device"], "workers": 0,
        "warmup_epochs": settings["warmup_epochs"], "lrf": settings["lrf"],
        "deterministic": True, "patience": settings["epochs"] + 1,
        "plots": False, "save": False, "val": True,
        **{name: parameters[name] for name in ("lr0", "momentum", "weight_decay")},
    }


class CanonicalSearchStopper:
    """Patience on canonical validation mAP; native fitness cannot terminate training."""

    def __init__(self, settings):
        spec = settings.get("early_stopping") or {}
        self.patience = spec.get("patience", 0)
        self.min_epochs = spec.get("min_epochs", 0)
        self.min_delta = spec.get("min_delta", 0.0)
        self.anchor, self.last_improvement = None, 0

    def __call__(self, epoch, score):
        if self.anchor is None or score > self.anchor + self.min_delta:
            self.anchor, self.last_improvement = score, epoch
        return bool(self.patience and epoch >= self.min_epochs
                    and epoch - self.last_improvement >= self.patience)


def train_detector(*, settings, data_yaml, data_root, model, seed, output_dir,
                   space_config, parameters=None, agent=None, learn=False, random_schedule=False):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    _validate_model(model)
    recipe = dict(settings["trainer_recipe"])
    if recipe.get("optimizer") != "SGD":
        raise ValueError("Quality live actions require SGD")
    if settings["workers"] != 0:
        raise ValueError("Quality protocol requires workers=0: live augmentation updates must reach the loader")
    space = DiscreteSearchSpace(space_config)
    hp = dict(space.initial_values() if parameters is None else parameters)
    if set(hp) != set(space.names) or any(
        not np.isfinite(hp[name]) or not spec.minimum <= hp[name] <= spec.maximum
        for name, spec in space_config.parameters.items()
    ):
        raise ValueError("Initial hyperparameters must be complete, finite and within the search bounds")
    if learn and agent is None:
        raise ValueError("Learning requires a policy agent")
    initial_hp = dict(hp)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise ValueError("Completed detector result already exists")
    epochs, segment = settings["epochs"], settings["segment_epochs"]
    started = time.perf_counter()
    history, decisions = [], []
    current = previous = None
    best_metrics, best_path, initial_hash = None, output / "selected_best.pt", None
    current_canonical, best_canonical = None, None
    initial_weights_hash = None
    best, stale, reference = 0.0, 0, 0.0
    schedule_rng = random.Random(seed + 987654)
    current_decision = None
    stopper = CanonicalSearchStopper(settings)
    stopped_early = False
    eval_args = SimpleNamespace(dataset_root=str(Path(data_root).resolve()), batch=settings["batch_size"],
                                imgsz=settings["img_size"], workers=0, device=settings["device"])

    class QualityTrainer(DetectionTrainer):
        def setup_model(self):
            # First-use framework imports/callback integrations may consume RNG
            # after BaseTrainer.__init__ seeds it. Seed at model construction too.
            from ultralytics.utils.torch_utils import init_seeds
            init_seeds(seed, deterministic=True)
            return super().setup_model()

        def save_model(self):
            # Canonical checkpoint selection below replaces native-fitness selection.
            return False

        def final_eval(self):
            return None

    def evaluate(trainer, epoch, *, initial=False):
        nonlocal initial_hash, current_canonical
        temporary = output / "candidate.pt"
        # Canonical score and saved artifact use the same full-precision EMA.
        clone = deepcopy(trainer.ema.ema).float().cpu().eval()
        clone.requires_grad_(False)
        torch.save({"model": clone, "epoch": epoch - 1, "train_args": vars(trainer.args)}, temporary)
        if initial:
            initial_hash = file_hash(temporary)
        native = trainer.metrics
        train_loss = 0.0 if initial else sum(float(v) for v in trainer.tloss.values())
        values = dict(epoch=epoch, train_loss=train_loss,
                      val_loss=sum(float(v) for k, v in native.items() if k.startswith("val/") and k.endswith("_loss")),
                      precision=0.0, recall=0.0, map50=0.0, map50_95=0.0, ap_small=0.0,
                      elapsed_seconds=0.0, peak_vram_mb=0.0)
        with isolated_evaluation_rng():
            evaluated = _canonical_metrics(temporary, eval_args, seed)
        current_canonical = dict(evaluated)
        _apply_eval(values, evaluated)
        values["elapsed_seconds"] = time.perf_counter() - started
        if trainer.device.type == "cuda":
            values["peak_vram_mb"] = torch.cuda.max_memory_allocated(trainer.device) / (1024 * 1024)
        metrics = Metrics(**values)
        metrics.validate()
        return metrics, temporary

    def setup(trainer):
        nonlocal current, reference, initial_weights_hash
        from ultralytics.utils.torch_utils import init_seeds
        # Freeze the start of the training random stream after pipeline setup.
        # It then advances continuously through all epochs and RL boundaries.
        init_seeds(seed, deterministic=True)
        initial_weights_hash = weights_hash(trainer.model)
        # YOLO.train reloads trainer.best on return. Point it at our canonical
        # selection instead of an unscored native-fitness checkpoint.
        trainer.best = best_path
        configure(trainer, hp, rebuild=True)
        # Initial score comes from this exact live model, without rebuilding it.
        # Native losses are unavailable before the first batch; the state has a mask.
        current, candidate = evaluate(trainer, 0, initial=True)
        candidate.unlink()
        reference = current.map50_95
        atomic_json(output / "initial.json", {"metrics": current.to_dict(), "checkpoint_sha256": initial_hash,
                                             "live_weights_sha256": initial_weights_hash,
                                             "initialization": "scratch", "hyperparameters": initial_hp,
                                             "training_overrides": training_overrides(settings, initial_hp)})

    def epoch_start(trainer):
        nonlocal hp, current_decision, reference
        epoch = trainer.epoch
        apply_close_mosaic(trainer)
        # Upstream's last warmup interpolation is at nw-1, so momentum can
        # remain slightly below the requested target after warmup. Set the
        # target explicitly for every method before regular training starts.
        if epoch >= settings["warmup_epochs"]:
            for group in trainer.optimizer.param_groups:
                if "momentum" in group:
                    group["momentum"] = hp["momentum"]
        # Shared fixed prefix: complete warmup before the first RL decision.
        if epoch == 0 or epoch % segment or (agent is None and not random_schedule):
            return
        reference = best
        masks = action_masks(space, hp)
        state = observation(current, previous, epochs=epochs, best=best, stale=stale,
                            parameters=hp, space=space, schedule_factor=float(trainer.lf(epoch)))
        if random_schedule:
            action = tuple(schedule_rng.choice([i for i, valid in enumerate(mask) if valid]) for mask in masks)
        else:
            action = agent.act(state, masks, deterministic=not learn, learn=learn)
        applied = space.apply(hp, action)
        if applied.clipped_count:
            raise RuntimeError("Masked action unexpectedly clipped")
        augmentation_changed = applied.values["augmentation_strength"] != hp["augmentation_strength"]
        hp = applied.values
        configure(trainer, hp, rebuild=augmentation_changed)
        current_decision = dict(start_epoch=epoch, observation=state, action=action,
                                masks=masks, hyperparameters=dict(hp), prefix_best=reference)

    def epoch_end(trainer):
        nonlocal current, previous, best, stale, best_metrics, current_decision, best_canonical, stopped_early
        epoch = trainer.epoch + 1
        if trainer.batch_size != settings["batch_size"]:
            raise RuntimeError("Trainer changed batch size after OOM; rerun with a frozen smaller batch")
        previous = current
        current, candidate = evaluate(trainer, epoch)
        improved = best_metrics is None or current.map50_95 > best
        if improved:
            best, best_metrics, stale = current.map50_95, current, 0
            best_canonical = dict(current_canonical)
            candidate.replace(best_path)
            atomic_json(output / "selected_best.json", {
                "metrics": current.to_dict(), "checkpoint": str(best_path),
                "sha256": file_hash(best_path), "selection": "canonical_mAP50_95_all_trained_epochs",
                "canonical_scores": best_canonical,
            })
        else:
            candidate.unlink()
            stale += 1
        # Validate the LR that actually reached optimizer steps, not just args.
        rates = [float(g["lr"]) for g in trainer.optimizer.param_groups]
        if epoch > settings["warmup_epochs"]:
            expected_lr = hp["lr0"] * trainer.lf(trainer.epoch)
            if any(abs(rate - expected_lr) > 1e-8 for rate in rates):
                raise RuntimeError(f"Applied learning rate {rates} differs from action/schedule {expected_lr}")
            if any(abs(g.get("momentum", hp["momentum"]) - hp["momentum"]) > 1e-8
                   for g in trainer.optimizer.param_groups):
                raise RuntimeError("Optimizer momentum differs from the policy action")
        decay_scale = trainer.batch_size * max(round(trainer.args.nbs / trainer.batch_size), 1) / trainer.args.nbs
        if any(abs(g["weight_decay"] - hp["weight_decay"] * decay_scale) > 1e-8
               for g in trainer.optimizer.param_groups if g.get("weight_decay", 0.0) > 0):
            raise RuntimeError("Optimizer weight decay differs from the policy action")
        history.append({"metrics": current.to_dict(), "best_map50_95": best,
                        "hyperparameters": dict(hp), "optimizer_lr": rates,
                        "optimizer_momentum": [g.get("momentum") for g in trainer.optimizer.param_groups],
                        "optimizer_weight_decay": [g.get("weight_decay", 0.0) for g in trainer.optimizer.param_groups],
                        "ema_updates": trainer.ema.updates,
                        "gradient_accumulation": trainer.accumulate, "amp_enabled": bool(trainer.amp),
                        "augmentation": {k: getattr(trainer.args, k) for k in
                                         ("degrees", "translate", "scale", "fliplr", "flipud", "mosaic", "mixup")}})
        stop_requested = stopper(epoch, current.map50_95)
        # Stop only at a completed segment, so every sampled RL action gets a
        # terminal reward. Keep the declared final close-mosaic phase intact.
        in_final_phase = bool(trainer.args.close_mosaic and epoch >= epochs - trainer.args.close_mosaic)
        stopped_early = bool(stop_requested and epoch % segment == 0 and epoch < epochs and not in_final_phase)
        done = epoch == epochs or stopped_early
        if stopped_early:
            trainer.stop = True
        if current_decision is not None and (epoch % segment == 0 or done):
            reward = best_gain(reference, best)
            if learn:
                agent.observe(reward, done)
            decisions.append({**current_decision, "end_epoch": epoch, "reward": reward, "done": done,
                              "best_map50_95": best})
            current_decision = None
        atomic_json(output / "epochs.json", history)
        atomic_json(output / "decisions.json", decisions)
        print(f"QUALITY seed={seed} epoch={epoch}/{epochs} canonical_mAP={current.map50_95:.5f} "
              f"best={best:.5f} decision={len(decisions)}", flush=True)
        if stopped_early:
            print(f"CANONICAL EARLY STOP at {epoch}: no gain > {stopper.min_delta} "
                  f"for {stopper.patience} epochs; best checkpoint retained.", flush=True)

    detector = YOLO(model)
    if detector.ckpt:
        raise RuntimeError("Quality training must start from architecture YAML")
    def prepare(trainer):
        trainer._quality_recipe = recipe
        configure(trainer, hp)

    detector.add_callback("on_pretrain_routine_start", prepare)
    detector.add_callback("on_pretrain_routine_end", setup)
    detector.add_callback("on_train_epoch_start", epoch_start)
    detector.add_callback("on_fit_epoch_end", epoch_end)
    try:
        detector.train(trainer=QualityTrainer, data=str(Path(data_yaml).resolve()), seed=seed,
                       project=str(output), name="native", exist_ok=True,
                       **training_overrides(settings, initial_hp))
        if current is None or best_metrics is None or (current.epoch != epochs and not stopped_early):
            raise RuntimeError("Detector did not finish the declared horizon")
        result = dict(seed=seed, failed=False, metrics=best_metrics.to_dict(), final_metrics=current.to_dict(),
                      canonical_best_scores=best_canonical, canonical_final_scores=current_canonical,
                      best_checkpoint=str(best_path), checkpoint_sha256=file_hash(best_path),
                      initial_checkpoint_sha256=initial_hash, decisions=len(decisions),
                      initial_weights_sha256=initial_weights_hash,
                      final_live_weights_sha256=weights_hash(detector.trainer.model),
                      final_ema_weights_sha256=weights_hash(detector.trainer.ema.ema),
                      elapsed_seconds=time.perf_counter() - started,
                      initialization="scratch", trainer_recipe=recipe,
                      maximum_horizon=epochs, actual_epochs=current.epoch,
                      early_stopping=bool(stopper.patience), stopped_early=stopped_early,
                      stopping_rule=dict(patience=stopper.patience, min_epochs=stopper.min_epochs,
                                         min_delta=stopper.min_delta, metric="canonical_validation_mAP50_95"),
                      hyperparameters=initial_hp, final_hyperparameters=dict(hp), run_dir=str(output))
        atomic_json(output / "result.json", result)
        return result
    finally:
        # Also close loaders on exceptions; upstream closes only on normal completion.
        trainer = getattr(detector, "trainer", None)
        for name in ("train_loader", "test_loader"):
            loader = getattr(trainer, name, None)
            if hasattr(loader, "close"):
                loader.close()
