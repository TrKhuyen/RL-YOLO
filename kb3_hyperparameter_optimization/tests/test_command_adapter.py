import json
import sys
import tempfile
import unittest
from pathlib import Path

from kb3_hyperparameter_optimization.adapters import CommandTrainerAdapter


WORKER = r'''
import json, sys
request_path, response_path = sys.argv[-2:]
request = json.load(open(request_path, encoding="utf-8"))
if request.get("mode") == "evaluate_initial":
    epoch, score = 0, 0.17
else:
    epoch, score = request["start_epoch"] + request["epochs"], 0.3
payload = {"metrics": {
    "epoch": epoch, "train_loss": 0.8, "val_loss": 0.9,
    "precision": 0.4, "recall": 0.5, "map50": 0.6,
    "map50_95": score, "ap_small": 0.2,
    "elapsed_seconds": float(epoch), "peak_vram_mb": 0.0
}, "checkpoint": "fake.pt"}
json.dump(payload, open(response_path, "w", encoding="utf-8"))
'''


class CommandAdapterTests(unittest.TestCase):
    def _adapter(self, root):
        worker = root / "worker.py"
        worker.write_text(WORKER, encoding="utf-8")
        return CommandTrainerAdapter([sys.executable, str(worker)], root)

    def test_request_response_contract_and_continuity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = self._adapter(root)
            adapter.reset(seed=12, hyperparameters={"lr0": 0.01})
            first = adapter.train_segment(epochs=2, hyperparameters={"lr0": 0.01}, output_dir=str(root / "run"))
            second = adapter.train_segment(epochs=3, hyperparameters={"lr0": 0.005}, output_dir=str(root / "run"))
            self.assertFalse(first.failed)
            self.assertFalse(second.failed)
            self.assertEqual(second.metrics.epoch, 5)
            request = json.loads((root / "run" / "segment_0001_request.json").read_text(encoding="utf-8"))
            self.assertEqual(request["start_epoch"], 2)
            self.assertEqual(request["hyperparameters"]["lr0"], 0.005)

    def test_adaptive_reset_uses_real_initial_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = self._adapter(root)
            initial = adapter.reset(
                seed=12, hyperparameters={"lr0": 0.01},
                output_dir=str(root / "run"), evaluate=True,
            )
            self.assertEqual(initial.epoch, 0)
            self.assertEqual(initial.map50_95, 0.17)
            request = json.loads((root / "run" / "initial_request.json").read_text(encoding="utf-8"))
            self.assertEqual(request["mode"], "evaluate_initial")


if __name__ == "__main__":
    unittest.main()
