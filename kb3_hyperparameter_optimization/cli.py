"""Utilities shared by KB3 command-line entry points."""

from __future__ import annotations

import argparse
import shlex
import sys

from .adapters import CommandTrainerAdapter, SimulatedTrainerAdapter
from .config import KB3Config


def add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("simulated", "command"), default="simulated")
    parser.add_argument(
        "--trainer-command",
        help="Command adapter executable; request.json and response.json are appended automatically",
    )
    parser.add_argument("--trainer-work-dir", default=".")
    parser.add_argument("--trainer-timeout", type=float)


def build_trainer(args: argparse.Namespace, config: KB3Config):
    if args.backend == "simulated":
        return SimulatedTrainerAdapter()
    if not args.trainer_command:
        raise ValueError("--trainer-command is required for command backend")
    # The command is a shell-style argument string, including quoted paths.
    # posix=False retains quote characters and breaks paths containing spaces.
    command = shlex.split(args.trainer_command, posix=True)
    if ("kb3_hyperparameter_optimization.adapters.ultralytics_worker" in command
            and config.reward.delta_ap_small > 0 and "--canonical-eval" not in command):
        raise ValueError("KB3 AP_small reward requires --canonical-eval in the Ultralytics worker command")
    if command and command[0].lower() in {"python", "python.exe", "python3", "python3.exe"}:
        # The child must use the exact uv/venv interpreter running KB3. On
        # Windows a bare `python` can otherwise resolve to uv's toolchain
        # interpreter without project dependencies.
        command[0] = sys.executable
    return CommandTrainerAdapter(command, args.trainer_work_dir, args.trainer_timeout)
