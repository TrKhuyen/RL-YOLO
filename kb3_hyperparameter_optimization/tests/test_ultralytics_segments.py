"""Optional real trainer integration: KB3_TEST_ULTRALYTICS=1 enables it."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

from kb3_hyperparameter_optimization.adapters.command import CommandTrainerAdapter


@unittest.skipUnless(os.environ.get("KB3_TEST_ULTRALYTICS") == "1", "optional real Ultralytics integration")
class UltralyticsSegmentTests(unittest.TestCase):
    def test_scratch_initial_validation_and_two_resumed_segments(self):
        import torch
        import yaml
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            for split in ("train", "valid", "test"):
                for kind in ("images", "labels"):
                    (data / split / kind).mkdir(parents=True)
                for index in range(2):
                    Image.new("RGB", (64, 64), (80 + 20 * index, 120, 60)).save(data / split / "images" / f"{index}.jpg")
                    (data / split / "labels" / f"{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            config = root / "data.yaml"
            config.write_text(yaml.safe_dump({"path": data.as_posix(), "train": "train/images",
                              "val": "valid/images", "test": "test/images", "nc": 28,
                              "names": [f"class_{i}" for i in range(28)]}), encoding="utf-8")
            adapter = CommandTrainerAdapter([
                sys.executable, "-m", "kb3_hyperparameter_optimization.adapters.ultralytics_worker",
                "--model", os.environ.get("KB3_TEST_MODEL", "yolov8n.yaml"), "--data", str(config), "--total-epochs", "2",
                "--batch", "2", "--imgsz", "64", "--device", "cpu", "--workers", "0",
                "--canonical-eval", "--dataset-root", str(data),
            ], work_dir=Path(__file__).resolve().parents[2], timeout=120)
            initial = {"lr0": .01, "weight_decay": .0005, "momentum": .937, "augmentation_strength": .5}
            run_dir = str(root / "run")
            metrics = adapter.reset(seed=42, hyperparameters=initial, output_dir=run_dir, evaluate=True)
            self.assertEqual(metrics.epoch, 0)
            self.assertGreater(metrics.val_loss, 0)
            init_ckpt = torch.load(Path(run_dir) / "initial_model.pt", map_location="cpu", weights_only=False)
            self.assertEqual(init_ckpt["model"].nc, 28)
            first = adapter.train_segment(epochs=1, hyperparameters=initial, output_dir=run_dir)
            self.assertFalse(first.failed, first.failure_reason)
            first_ckpt = torch.load(first.checkpoint, map_location="cpu", weights_only=False)
            self.assertIn("kb3_training_state", first_ckpt)
            second_params = {**initial, "lr0": .004, "momentum": .85, "weight_decay": .001,
                             "augmentation_strength": .3}
            second = adapter.train_segment(epochs=1, hyperparameters=second_params, output_dir=run_dir)
            self.assertFalse(second.failed, second.failure_reason)
            self.assertEqual(second.metrics.epoch, 2)
            last = torch.load(second.checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(last["epoch"], 1)
            state = last["kb3_training_state"]
            for group in state["optimizer"]["param_groups"]:
                self.assertEqual(group["lr"], .004)
                self.assertEqual(group["momentum"], .85)
            self.assertEqual(state["scheduler"]["base_lrs"], [.004] * len(state["optimizer"]["param_groups"]))
            self.assertTrue(state["optimizer"]["state"])
            self.assertGreater(last["updates"], first_ckpt["updates"])
            self.assertEqual(second.metadata["best_metrics"]["epoch"],
                             torch.load(second.metadata["best_checkpoint"], map_location="cpu", weights_only=False)["epoch"] + 1)


if __name__ == "__main__":
    unittest.main()
