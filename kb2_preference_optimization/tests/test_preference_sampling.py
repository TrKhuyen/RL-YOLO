import ast
from pathlib import Path
import unittest

import numpy as np
import torch

from kb2_preference_optimization.conditional.bank import sampling_probabilities as historical
from kb2_preference_optimization.hybrid_sampling.sampler import GROUP_MASS, KEEP, sampling_probabilities


def records():
    return [dict(group=g, chosen_keep=KEEP[g], probe=False, image_id=i*7+j,
                 class_id=j % 3, target_gt=j, slot=j)
            for i, g in enumerate(GROUP_MASS) for j in range(1+i*2)]


class SamplingTests(unittest.TestCase):
    def test_equal_mass_reproduces_historical_class_target_balance_and_cap(self):
        rows = records()
        old, _ = historical(rows, max_image_repeat=1.5)
        new, _ = sampling_probabilities(rows, max_image_repeat=1.5,
                                         group_mass={g: .25 for g in GROUP_MASS})
        torch.testing.assert_close(old, new, atol=1e-15, rtol=1e-14)

    def test_requested_mass_and_conditional_weights_before_cap(self):
        rows = records()
        old, _ = historical(rows, max_image_repeat=100)
        new, config = sampling_probabilities(rows, max_image_repeat=100)
        for group, expected in GROUP_MASS.items():
            ids = [i for i, r in enumerate(rows) if r['group'] == group]
            self.assertAlmostEqual(float(new[ids].sum()), expected)
            torch.testing.assert_close(new[ids]/new[ids].sum(), old[ids]/old[ids].sum())
        self.assertAlmostEqual(config['expected_keep_fraction'], .6)

    def test_cap_overrides_scarce_group_mass_and_is_reported(self):
        rows = records()
        weights, config = sampling_probabilities(rows, max_image_repeat=1)
        image_totals = {}
        for record, w in zip(rows, weights):
            image_totals[record['image_id']] = image_totals.get(record['image_id'], 0.)+float(w)
        self.assertLessEqual(max(image_totals.values()), config['image_probability_cap']+1e-12)
        self.assertAlmostEqual(float(weights.sum()), 1.)
        self.assertFalse(np.isclose(config['expected_group_probability']['preservation'], .35))

    def test_rejects_probe_leakage_missing_groups_and_bad_signs(self):
        cases = [records()[1:], [{**r, 'probe': True} for r in records()],
                 [{**r, 'chosen_keep': not r['chosen_keep']} for r in records()]]
        for rows in cases:
            with self.assertRaises(ValueError):
                sampling_probabilities(rows)

    def test_sampling_is_the_only_change_to_update_loop(self):
        root = Path(__file__).resolve().parents[1]
        def update_loop(path):
            tree = ast.parse(path.read_text())
            trainer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
            loop = next(n for n in trainer.body if isinstance(n, ast.For)
                        and isinstance(n.target, ast.Name) and n.target.id == 'step')
            return ast.dump(loop, include_attributes=False)
        self.assertEqual(update_loop(root/'hybrid_schedule/train.py'),
                         update_loop(root/'hybrid_sampling/train.py'))


if __name__ == '__main__':
    unittest.main()
