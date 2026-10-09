"""Read-only KB1 supervised reference and its actual Ultralytics recipe."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import yaml

from ..run_all import ROOT, MODELS
from ..adapters.ultralytics_worker import _canonical_metrics
from .detector import atomic_json, file_hash, isolated_evaluation_rng


# Do not import model/data/resume/pretrained/output paths from a completed run.
RECIPE_KEYS = (
    "optimizer", "nbs", "amp", "cos_lr", "close_mosaic", "warmup_momentum", "warmup_bias_lr",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix", "copy_paste", "copy_paste_mode",
    "box", "cls", "dfl", "rect", "single_cls", "fraction", "multi_scale",
)


def read_recipe(model):
    directory = ROOT / "kb1_reward_guided_training" / "checkpoint_based" / model
    path = directory / "args.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw["optimizer"] != "SGD":
        raise ValueError("Live momentum/weight-decay actions currently require the KB1 SGD recipe")
    recipe = {key: raw[key] for key in RECIPE_KEYS if key in raw}
    return raw, recipe, path


def prepare_reference(args, settings, space):
    """Verify KB1 weights/data/recipe before any expensive KB3 training."""
    if args.smoke:
        # Synthetic integration is deliberately not a KB1 quality comparison.
        return space
    raw, recipe, recipe_path = read_recipe(args.model)
    sys.path.insert(0, str(ROOT / "kb1_reward_guided_training"))
    from run_provenance import verify_supervised, DATA_ROOT, DATA_CONFIG
    if Path(args.data_root).resolve() != DATA_ROOT.resolve() or file_hash(args.data_config) != file_hash(DATA_CONFIG):
        raise ValueError("The KB1 reference requires its verified dataset and class configuration")
    checkpoint = recipe_path.parent / "weights" / "best.pt"
    metadata = verify_supervised(args.model, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = payload["train_args"]
    saved_model = payload.get("ema") or payload.get("model")
    if saved_model is None or saved_model.yaml.get("nc") != 28:
        raise ValueError("KB1 reference must be a trained 28-class detector")
    source_model = Path(str(raw["model"]).replace("\\", "/")).stem
    if source_model != Path(MODELS[args.model]).stem:
        raise ValueError(f"KB1 recipe architecture does not match {args.model}")
    for key in (*RECIPE_KEYS, "lr0", "momentum", "weight_decay", "warmup_epochs", "lrf", "batch", "imgsz",
                "pretrained", "epochs", "patience", "seed"):
        if key in raw and saved.get(key) != raw[key]:
            raise ValueError(f"KB1 args.yaml differs from the checkpoint's recorded {key}")
    parameters = dict(space.parameters)
    for name in ("lr0", "momentum", "weight_decay"):
        parameters[name] = replace(parameters[name], initial=float(raw[name]))
    parameters["augmentation_strength"] = replace(parameters["augmentation_strength"], initial=0.5)
    space = replace(space, parameters=parameters)
    space.validate()
    settings["kb1_reference"] = dict(
        checkpoint=str(checkpoint.resolve()), checkpoint_sha256=metadata["checkpoint_sha256"],
        recipe_path=str(recipe_path.resolve()), recipe_sha256=file_hash(recipe_path),
        dataset_sha256=metadata["dataset"]["sha256"],
        initialization="pretrained" if raw["pretrained"] else "scratch",
        seed=raw["seed"], declared_epochs=raw["epochs"], patience=raw["patience"],
        checkpoint_selection="KB1 native best; re-evaluated with the shared canonical evaluator",
        model=args.model,
    )
    settings["trainer_recipe"] = recipe
    return space


def evaluate_reference(settings, data_root, output, *, split="val"):
    """Inference only, never fine-tune or copy the external baseline weights."""
    reference = settings.get("kb1_reference")
    if reference is None:
        if settings.get("smoke"):
            return None
        raise ValueError("KB1 reference must be verified before evaluation")
    checkpoint = Path(reference["checkpoint"])
    if file_hash(checkpoint) != reference["checkpoint_sha256"]:
        raise RuntimeError("KB1 reference checkpoint changed")
    if split not in {"val", "test"}:
        raise ValueError("Reference evaluation must use validation or test")
    path = Path(output) / ("kb1_reference.json" if split == "val" else "kb1_reference_test.json")
    if path.exists():
        import json
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["reference"] != reference or result["evaluation"] != evaluation_spec(settings, split):
            raise RuntimeError("Cached KB1 reference does not match this run")
        return result
    args = SimpleNamespace(dataset_root=str(Path(data_root).resolve()), batch=settings["batch_size"],
                           imgsz=settings["img_size"], workers=0, device=settings["device"], split=split)
    with isolated_evaluation_rng():
        metrics = _canonical_metrics(checkpoint, args, reference["seed"])
    if file_hash(checkpoint) != reference["checkpoint_sha256"]:
        raise RuntimeError("KB1 reference changed during evaluation")
    result = dict(reference=reference, evaluation=evaluation_spec(settings, split), metrics=metrics,
                  training_performed=False, independent_training_runs=1)
    atomic_json(path, result)
    return result


def evaluation_spec(settings, split="val"):
    return dict(split="validation" if split == "val" else "test", batch_size=settings["batch_size"], img_size=settings["img_size"],
                precision="float32", tf32=False, deterministic=True, conf=0.001, nms_iou=0.60, max_det=300,
                operating_conf=0.25, test_used=split == "test", test_used_for_selection=False)
