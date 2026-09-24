import unittest

from kb3_hyperparameter_optimization.compare_hpo import _number


class ComparisonTests(unittest.TestCase):
    def test_quality_comes_from_best_but_cost_comes_from_final(self):
        record = {
            "metrics": {"map50_95": 0.8, "elapsed_seconds": 10.0},
            "final_metrics": {"map50_95": 0.6, "elapsed_seconds": 50.0},
        }
        self.assertEqual(_number(record, "map50_95"), 0.8)
        self.assertEqual(_number(record, "elapsed_seconds"), 50.0)


if __name__ == "__main__":
    unittest.main()
