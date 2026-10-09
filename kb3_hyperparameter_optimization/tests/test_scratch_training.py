import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from kb3_hyperparameter_optimization.adapters.ultralytics_worker import (
    _configure_arguments, _configure_optimizer, _validate_model,
)
from kb3_hyperparameter_optimization.config import ExperimentConfig, KB3Config
from kb3_hyperparameter_optimization.cli import build_trainer
from kb3_hyperparameter_optimization.core import Metrics, SegmentResult
from kb3_hyperparameter_optimization.envs import YoloHPOEnv
from kb3_hyperparameter_optimization.run_all import parse_args


class ScratchTrainingTests(unittest.TestCase):
    def test_pretrained_checkpoint_is_rejected(self):
        _validate_model("yolo11n.yaml")
        with self.assertRaisesRegex(ValueError, "scratch"):
            _validate_model("weights/best.pt")

    def test_rl_lr_survives_every_scheduler_step_after_resume(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=0.01, momentum=0.937, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
        trainer = SimpleNamespace(optimizer=optimizer, scheduler=scheduler)
        selected = {"lr0": 0.004, "weight_decay": 0.001, "momentum": 0.85}
        _configure_optimizer(trainer, selected)
        for _ in range(3):
            optimizer.step()
            scheduler.step()
            self.assertEqual(optimizer.param_groups[0]["lr"], 0.004)
            self.assertEqual(optimizer.param_groups[0]["momentum"], 0.85)
            self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.001)

    def test_warmup_and_mosaic_close_do_not_override_rl(self):
        trainer = SimpleNamespace(args=SimpleNamespace())
        _configure_arguments(trainer, {"lr0": 0.004, "weight_decay": 0.001,
                                      "momentum": 0.85, "augmentation_strength": 0.3})
        self.assertEqual(trainer.args.warmup_epochs, 0)
        self.assertEqual(trainer.args.lrf, 1)
        self.assertEqual(trainer.args.close_mosaic, 0)
        self.assertAlmostEqual(trainer.args.mosaic, 0.6)

    def test_zero_map_first_segment_still_selects_trained_checkpoint(self):
        class ZeroTrainer:
            def reset(self, **kwargs):
                return Metrics(0, 0, 0, 0, 0, 0, 0, 0)

            def train_segment(self, **kwargs):
                return SegmentResult(Metrics(1, 1, 1, 0, 0, 0, 0, 0), checkpoint="trained.pt")

        with tempfile.TemporaryDirectory() as directory:
            config = replace(KB3Config(), experiment=ExperimentConfig(
                total_epochs=1, segment_epochs=1, output_dir=directory))
            env = YoloHPOEnv(ZeroTrainer(), config)
            env.reset()
            env.step(tuple(size // 2 for size in env.space.action_sizes))
            self.assertEqual(env.best_checkpoint, "trained.pt")
            self.assertEqual(env.best_metrics.epoch, 1)

    def test_run_all_rejects_evaluation_seed_leakage(self):
        with self.assertRaises(SystemExit):
            parse_args(["--eval-seeds", "42"])

    def test_ap_small_reward_requires_canonical_ultralytics_worker(self):
        args = SimpleNamespace(backend="command", trainer_command=
                               "python -m kb3_hyperparameter_optimization.adapters.ultralytics_worker")
        with self.assertRaisesRegex(ValueError, "canonical-eval"):
            build_trainer(args, KB3Config())

    def test_quoted_worker_path_does_not_keep_quotes(self):
        args = SimpleNamespace(backend="command", trainer_command=
                               "python -m custom_worker --data 'D:/data with spaces/data.yaml'",
                               trainer_work_dir=".", trainer_timeout=None)
        trainer = build_trainer(args, KB3Config())
        self.assertEqual(trainer.command[-1], "D:/data with spaces/data.yaml")

    def test_selected_checkpoint_survives_native_best_overwrite(self):
        class ReplacingTrainer:
            def reset(self, **kwargs):
                self.index = 0
                return Metrics(0, 0, 0, 0, 0, 0, 0, 0)

            def train_segment(self, **kwargs):
                self.index += 1
                checkpoint = Path(kwargs["output_dir"]) / "best.pt"
                checkpoint.write_bytes(b"chosen" if self.index == 1 else b"replaced")
                score = .8 if self.index == 1 else .4
                return SegmentResult(Metrics(self.index, 1, 1, score, score, score, score, score),
                                     checkpoint=str(checkpoint))

        with tempfile.TemporaryDirectory() as directory:
            config = replace(KB3Config(), experiment=ExperimentConfig(
                total_epochs=2, segment_epochs=1, output_dir=directory))
            env = YoloHPOEnv(ReplacingTrainer(), config)
            env.reset()
            action = tuple(size // 2 for size in env.space.action_sizes)
            env.step(action)
            env.step(action)
            self.assertEqual(env.best_metrics.map50_95, .8)
            self.assertEqual(Path(env.best_checkpoint).read_bytes(), b"chosen")


if __name__ == "__main__":
    unittest.main()
