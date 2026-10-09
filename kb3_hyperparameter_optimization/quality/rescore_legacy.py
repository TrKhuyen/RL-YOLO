"""Rescore extant legacy checkpoints without detector training or log edits."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from types import SimpleNamespace

from ..adapters.ultralytics_worker import _canonical_metrics
from .detector import atomic_json, file_hash, isolated_evaluation_rng


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory outside the original run")
    parser.add_argument("--device", default="0")
    parser.add_argument("--model", help="Only rescore this model subtree, e.g. yolov8n")
    args = parser.parse_args()
    original, output = Path(args.run_dir).resolve(), Path(args.output_dir).resolve()
    if output.is_relative_to(original) or original.is_relative_to(output):
        raise ValueError("Recovery output must be separate from the original run")
    manifest = json.loads((original / "run_manifest.json").read_text(encoding="utf-8"))
    evaluator = SimpleNamespace(dataset_root=manifest["data_root"], device=args.device,
                                batch=manifest["batch_size"], imgsz=manifest["img_size"], workers=0)
    output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[2]
    dataset_root = Path(evaluator.dataset_root).resolve()
    evaluator.dataset_root = str(dataset_root)
    identity = dict(dataset=[(str(p.relative_to(dataset_root)), file_hash(p))
                             for p in sorted((dataset_root / "valid").rglob("*"))
                             if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".txt"}],
                    evaluator_sources=[(str(p.relative_to(project)), file_hash(p)) for p in
                                       (project / "kb1_reward_guided_training" / "canonical_eval.py",
                                        project / "kb1_reward_guided_training" / "dataloader.py",
                                        project / "kb1_reward_guided_training" / "adapters" / "ultralytics_adapter.py",
                                        project / "kb3_hyperparameter_optimization" / "adapters" / "ultralytics_worker.py")],
                    args=vars(evaluator), dependencies={n: version(n) for n in
                    ("torch", "torchvision", "ultralytics", "torchmetrics", "faster-coco-eval")})
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    identity_path = output / "evaluator_manifest.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8"))["sha256"] != identity["sha256"]:
            raise ValueError("Recovery evaluator/data changed; use a new output directory")
    else:
        if any(output.iterdir()):
            raise ValueError("Recovery output must be empty or contain its matching evaluator manifest")
        atomic_json(identity_path, identity)
    results, cache = [], {}
    records_path = output / "rescored_checkpoints.json"
    if records_path.exists():
        results = json.loads(records_path.read_text(encoding="utf-8"))["checkpoints"]
        cache = {r["checkpoint_sha256"]: r["canonical_metrics"] for r in results}
    roots = [original / args.model] if args.model else [original]
    checkpoints = sorted(p for root in roots for p in root.rglob("*.pt")
                         if p.name in {"best.pt", "last.pt", "selected_best.pt"})
    known = {(r["checkpoint"], r["checkpoint_sha256"]) for r in results}
    for checkpoint in checkpoints:
        digest = file_hash(checkpoint)
        if (str(checkpoint), digest) in known:
            continue
        if digest not in cache:
            with isolated_evaluation_rng():
                cache[digest] = _canonical_metrics(checkpoint, evaluator, int(manifest["seed"]))
        # Verify the file did not change during inference (e.g. an active worker).
        if file_hash(checkpoint) != digest:
            raise RuntimeError(f"Checkpoint changed while rescoring: {checkpoint}")
        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        results.append(dict(checkpoint=str(checkpoint), checkpoint_sha256=digest,
                            actual_epoch=int(payload["epoch"]) + 1 if "epoch" in payload else None,
                            canonical_metrics=cache[digest]))
        atomic_json(records_path, dict(original_run=str(original), test_evaluated=False,
                                      scope="Only currently extant weights; overwritten historical weights are unavailable",
                                      checkpoints=results))
        print(f"RESCORED {checkpoint}: mAP={cache[digest]['mAP50_95']:.6f}", flush=True)


if __name__ == "__main__":
    main()
