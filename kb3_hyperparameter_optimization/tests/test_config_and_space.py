import tempfile
import unittest
from pathlib import Path

from kb3_hyperparameter_optimization.config import KB3Config, load_config
from kb3_hyperparameter_optimization.search_space import DiscreteSearchSpace


class ConfigAndSpaceTests(unittest.TestCase):
    def test_default_config_is_valid(self):
        config = KB3Config()
        config.validate()

    def test_config_rejects_unknown_parameter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("search_space:\n  parameters:\n    unknown: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported"):
                load_config(path)

    def test_action_is_clipped_and_reported(self):
        space = DiscreteSearchSpace(KB3Config().search_space)
        values = space.initial_values()
        action = tuple(size - 1 for size in space.action_sizes)
        for _ in range(20):
            result = space.apply(values, action)
            values = result.values
        self.assertLessEqual(values["lr0"], 0.02)
        self.assertLessEqual(values["momentum"], 0.98)
        self.assertGreater(result.clipped_count, 0)

    def test_wrong_action_shape_is_rejected(self):
        space = DiscreteSearchSpace(KB3Config().search_space)
        with self.assertRaises(ValueError):
            space.apply(space.initial_values(), (0,))


if __name__ == "__main__":
    unittest.main()

