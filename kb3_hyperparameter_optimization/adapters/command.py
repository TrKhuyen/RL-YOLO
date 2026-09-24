"""Subprocess adapter for integrating any detector trainer through JSON."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

from ..core import Metrics, SegmentResult


class CommandTrainerAdapter:
    def __init__(self, command: Sequence[str], work_dir: str | Path = ".", timeout: float | None = None):
        if not command:
            raise ValueError("command cannot be empty")
        self.command = tuple(command)
        self.work_dir = Path(work_dir).resolve()
        self.timeout = timeout
        self.seed: int | None = None
        self.segment = 0
        self.epoch = 0
        self.run_dir: Path | None = None

    @staticmethod
    def _metrics(data: dict) -> Metrics:
        required = {"epoch", "train_loss", "val_loss", "precision", "recall", "map50", "map50_95", "ap_small"}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"trainer response is missing metrics: {sorted(missing)}")
        metrics = Metrics(**{field: data.get(field, 0.0) for field in Metrics.__dataclass_fields__})
        metrics.validate()
        return metrics

    def _invoke(self, request_path: Path, response_path: Path) -> tuple[dict, str]:
        completed = subprocess.run(
            [*self.command, str(request_path), str(response_path)], cwd=self.work_dir,
            check=False, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=self.timeout,
        )
        stdout, stderr = completed.stdout or "", completed.stderr or ""
        if completed.returncode != 0:
            raise RuntimeError((stderr or stdout or "trainer failed")[-4000:])
        if not response_path.is_file():
            raise RuntimeError("trainer did not create response JSON")
        return json.loads(response_path.read_text(encoding="utf-8")), stdout[-4000:]

    def reset(
        self, *, seed: int, hyperparameters: dict[str, float],
        output_dir: str | None = None, evaluate: bool = False,
    ) -> Metrics:
        self.seed, self.segment, self.epoch = seed, 0, 0
        self.run_dir = Path(output_dir).resolve() if output_dir else None
        if not evaluate:
            return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if self.run_dir is None:
            raise ValueError("output_dir is required when evaluate=True")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        request_path = self.run_dir / "initial_request.json"
        response_path = self.run_dir / "initial_response.json"
        request_path.write_text(json.dumps({
            "schema_version": 1, "mode": "evaluate_initial", "seed": seed,
            "hyperparameters": hyperparameters, "run_dir": str(self.run_dir),
        }, indent=2), encoding="utf-8")
        try:
            response, _ = self._invoke(request_path, response_path)
            metrics = self._metrics(response["metrics"])
        except (OSError, subprocess.TimeoutExpired, RuntimeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"initial model evaluation failed: {exc}") from exc
        if metrics.epoch != 0:
            raise ValueError(f"initial evaluation returned epoch {metrics.epoch}; expected 0")
        return metrics

    @staticmethod
    def _failed(epoch: int, reason: str) -> SegmentResult:
        return SegmentResult(
            Metrics(epoch, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            failed=True, failure_reason=reason,
        )

    def train_segment(self, *, epochs: int, hyperparameters: dict[str, float], output_dir: str) -> SegmentResult:
        if self.seed is None:
            raise RuntimeError("reset must be called before train_segment")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        self.run_dir = Path(output_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        request_path = self.run_dir / f"segment_{self.segment:04d}_request.json"
        response_path = self.run_dir / f"segment_{self.segment:04d}_response.json"
        request = {
            "schema_version": 1, "mode": "train_segment", "seed": self.seed,
            "segment": self.segment, "start_epoch": self.epoch, "epochs": epochs,
            "hyperparameters": hyperparameters, "run_dir": str(self.run_dir),
        }
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
        try:
            response, stdout = self._invoke(request_path, response_path)
            metrics = self._metrics(response["metrics"])
        except (OSError, subprocess.TimeoutExpired, RuntimeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return self._failed(self.epoch, str(exc))
        expected_epoch = self.epoch + epochs
        if metrics.epoch != expected_epoch:
            return self._failed(self.epoch, f"trainer returned epoch {metrics.epoch}; expected {expected_epoch}")
        metadata: dict = {"stdout": stdout}
        if response.get("best_metrics") is not None:
            best = self._metrics(response["best_metrics"])
            metadata["best_metrics"] = best.to_dict()
        if response.get("best_checkpoint") is not None:
            metadata["best_checkpoint"] = str(response["best_checkpoint"])
        self.epoch, self.segment = metrics.epoch, self.segment + 1
        return SegmentResult(metrics, checkpoint=response.get("checkpoint"), metadata=metadata)

    def close(self) -> None:
        return None
