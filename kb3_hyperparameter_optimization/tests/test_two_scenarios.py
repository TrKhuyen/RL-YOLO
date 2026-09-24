import tempfile
import unittest
from dataclasses import replace

from kb3_hyperparameter_optimization.adapters import SimulatedTrainerAdapter
from kb3_hyperparameter_optimization.config import ExperimentConfig, KB3Config
from kb3_hyperparameter_optimization.core import Metrics, SegmentResult
from kb3_hyperparameter_optimization.traditional_hpo.runner import run_fixed_trial


class CountingTrainer(SimulatedTrainerAdapter):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.reset_options = []

    def reset(self, **kwargs):
        self.reset_options.append(kwargs)
        return super().reset(**kwargs)

    def train_segment(self, *, epochs, hyperparameters, output_dir):
        self.calls.append((epochs, dict(hyperparameters)))
        return super().train_segment(epochs=epochs, hyperparameters=hyperparameters, output_dir=output_dir)


class BestResultTrainer(CountingTrainer):
    def train_segment(self, *, epochs, hyperparameters, output_dir):
        self.calls.append((epochs, dict(hyperparameters)))
        final = Metrics(epochs, 0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4)
        best = Metrics(epochs - 1, 0.3, 0.4, 0.8, 0.8, 0.8, 0.8, 0.8)
        return SegmentResult(final, checkpoint="last.pt", metadata={
            "best_metrics": best.to_dict(), "best_checkpoint": "best.pt",
        })


class TwoScenarioTests(unittest.TestCase):
    def _config(self, directory, epochs=12):
        experiment = ExperimentConfig(
            total_epochs=epochs, segment_epochs=3,
            patience_segments=5, output_dir=directory,
        )
        return replace(KB3Config(), experiment=experiment)

    @staticmethod
    def _params(config):
        return {name: spec.initial for name, spec in config.search_space.parameters.items()}

    def test_traditional_hpo_uses_one_fixed_full_training_call(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            trainer = CountingTrainer()
            hyperparameters = self._params(config)
            result = run_fixed_trial(
                trainer, config, trial=0, seed=42,
                hyperparameters=hyperparameters, method="test",
            )
            self.assertFalse(result.failed)
            self.assertEqual(len(trainer.calls), 1)
            self.assertEqual(trainer.calls[0][0], 12)
            self.assertEqual(trainer.calls[0][1], hyperparameters)
            self.assertFalse(trainer.reset_options[0]["evaluate"])
            self.assertEqual(result.metrics["epoch"], 12)

    def test_traditional_objective_uses_best_not_final_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory)
            result = run_fixed_trial(
                BestResultTrainer(), config, trial=0, seed=42,
                hyperparameters=self._params(config), method="test",
            )
            self.assertEqual(result.metrics["map50_95"], 0.8)
            self.assertEqual(result.final_metrics["map50_95"], 0.4)
            self.assertEqual(result.best_checkpoint, "best.pt")

    def test_fixed_trial_rejects_out_of_range_value(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, epochs=2)
            hyperparameters = self._params(config)
            hyperparameters["lr0"] = 99.0
            with self.assertRaisesRegex(ValueError, "outside"):
                run_fixed_trial(
                    SimulatedTrainerAdapter(), config, trial=0, seed=42,
                    hyperparameters=hyperparameters, method="test",
                )


if __name__ == "__main__":
    unittest.main()
