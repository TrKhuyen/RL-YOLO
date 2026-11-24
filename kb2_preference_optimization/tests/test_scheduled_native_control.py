"""Native control must remove the auxiliary gradient without changing the stream path."""
import unittest
from unittest.mock import patch
import torch
from kb2_preference_optimization.hybrid_schedule.train import train
from kb2_preference_optimization.train_dpo import gradient_diagnostics


class ScheduledNativeControlTests(unittest.TestCase):
    def test_rejects_nonzero_native_and_zero_dpo_before_reading_data(self):
        for arm, weight in [('native', .03), ('native', -.01), ('dpo', 0.), ('dpo', -.01)]:
            with self.subTest(arm=arm, weight=weight), self.assertRaises(ValueError):
                train('/missing', '/missing', arm=arm, dpo_weight=weight)

    def test_zero_weight_preserves_native_gradient_and_adamw_update(self):
        a = torch.nn.Parameter(torch.tensor([.2, -.5]))
        b = torch.nn.Parameter(a.detach().clone())
        oa = torch.optim.AdamW([a], lr=1e-6, weight_decay=5e-4)
        ob = torch.optim.AdamW([b], lr=1e-6, weight_decay=5e-4)
        native = (a - 1).square().sum()
        auxiliary = torch.nn.functional.softplus(-a).sum()
        diagnostic = gradient_diagnostics(native, auxiliary, [a], 0.)
        self.assertEqual(diagnostic['weighted_dpo_grad_norm'], 0.)
        (native + 0.*auxiliary).backward()
        (b-1).square().sum().backward()
        self.assertTrue(torch.equal(a.grad, b.grad))
        torch.nn.utils.clip_grad_norm_([a], 1.)
        torch.nn.utils.clip_grad_norm_([b], 1.)
        oa.step(); ob.step()
        self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
