import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from kb3_hyperparameter_optimization.adapters import SimulatedTrainerAdapter
from kb3_hyperparameter_optimization.config import ExperimentConfig, KB3Config
from kb3_hyperparameter_optimization.envs import YoloHPOEnv


class RunSafetyTests(unittest.TestCase):
    def test_nonempty_run_directory_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            stale = Path(directory) / "same"
            stale.mkdir()
            (stale / "checkpoint.pt").write_text("do not overwrite", encoding="utf-8")
            experiment = ExperimentConfig(
                total_epochs=2, segment_epochs=1, patience_segments=2, output_dir=directory
            )
            env = YoloHPOEnv(
                SimulatedTrainerAdapter(), replace(KB3Config(), experiment=experiment)
            )
            env.reset(run_name="same")
            self.assertNotEqual(env.run_dir, stale)
            self.assertEqual((stale / "checkpoint.pt").read_text(encoding="utf-8"), "do not overwrite")


if __name__ == "__main__":
    unittest.main()
