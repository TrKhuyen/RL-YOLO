"""Preflight every model before sequentially running independent KB3 experiments."""

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from ..run_all import MODELS, ROOT
from . import PROTOCOL
from .detector import atomic_json


def model_command(args, model, output):
    command = [sys.executable, "-u", "-m", "kb3_hyperparameter_optimization.quality.pipeline",
               "--model", model, "--stage", args.stage, "--config", args.config,
               "--space-config", args.space_config, "--data-root", args.data_root,
               "--data-config", args.data_config, "--output-dir", str(output)]
    if args.resume:
        command.append("--resume")
    if args.smoke:
        command.append("--smoke")
    if args.search_target is not None:
        command.extend(["--search-target", str(args.search_target)])
    if args.finalize:
        command.append("--finalize")
    return command


def run_models(args):
    if args.resume and not args.output_dir:
        raise ValueError("--resume requires the original model-matrix --output-dir")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output_dir or ROOT / "kb3_hyperparameter_optimization" /
                  "checkpoint_hyperparameter_optimization" / "quality" / f"all_{stamp}").resolve()
    commands = [(name, model_command(args, name, output / name)) for name in MODELS]
    # No model may begin a long training run while another lacks a valid baseline.
    for name, command in commands:
        print(f"PREFLIGHT KB1 recipe/reference and experiment: {name}", flush=True)
        subprocess.run([*command, "--check-only"], cwd=ROOT, check=True)
    if args.check_only:
        print("ALL MODELS PREFLIGHT PASS; no training or output creation.", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    status = dict(protocol=PROTOCOL, output=str(output), models=list(MODELS), stage=args.stage,
                  results={}, detector_initialization="scratch", checkpoint_role="comparison_only")
    for name, command in commands:
        status["active_model"] = name
        atomic_json(output / "models_status.json", status)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except BaseException as error:
            status["results"][name] = dict(status="interrupted", reason=str(error))
            atomic_json(output / "models_status.json", status)
            raise
        status["results"][name] = dict(status="complete", output=str(output / name))
    status["active_model"] = None
    atomic_json(output / "models_status.json", status)
    print(f"ALL MODELS stage complete: {output}", flush=True)
