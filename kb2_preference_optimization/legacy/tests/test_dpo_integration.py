"""Opt-in real YOLO/GPU check: KB2_RUN_REAL_DPO=1 enables local data/weights."""
import os
import sys
import unittest
from pathlib import Path

import torch

KB2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KB2_DIR))
from dataloader import get_pest_dataloader
from feedback_loss import compute_dpo_preference_loss, dpo_preference_loss_from_predictions
from train_feedback import make_frozen_reference, set_seed
from train_rl import CHECKPOINTS, load_adapter


@unittest.skipUnless(os.environ.get('KB2_RUN_REAL_DPO') == '1', 'opt-in real data/weights check')
class RealDpoTests(unittest.TestCase):
    def test_real_yolo_gradients_and_reference_immutability(self):
        set_seed(42)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        adapter = load_adapter('yolov8n', str(CHECKPOINTS['yolov8n']), device)
        reference = make_frozen_reference(adapter)
        before = {key: value.clone() for key, value in reference.state_dict().items()}
        loader = get_pest_dataloader(
            str(KB2_DIR.parent / 'pre-data/data/v2i_cleanned'), split='train',
            batch_size=4, img_size=640, num_workers=0, shuffle=True)
        if hasattr(loader.dataset.transforms, 'set_random_seed'):
            loader.dataset.transforms.set_random_seed(42)
        images, targets = next(iter(loader))
        loss, diagnostic = compute_dpo_preference_loss(adapter, reference, images, targets)
        self.assertGreater(diagnostic['pair_count'], 0)
        self.assertGreaterEqual(float(diagnostic['chosen_iou']), .5)
        self.assertLess(float(diagnostic['rejected_iou']), float(diagnostic['chosen_iou']))
        self.assertGreater(float(diagnostic['quality_gap']), 0.)
        self.assertAlmostEqual(float(loss.detach()), .69314718, places=5)
        params = [p for p in adapter.parameters() if p.requires_grad]
        gradients = torch.autograd.grad(.1 * loss, params, retain_graph=True, allow_unused=True)
        dpo_norm = sum(g.detach().square().sum() for g in gradients if g is not None).sqrt()
        self.assertGreater(float(dpo_norm), 0)
        native, _ = adapter.supervised_loss(images, targets)
        native_grads = torch.autograd.grad(native, params, retain_graph=True, allow_unused=True)
        native_norm = sum(g.detach().square().sum() for g in native_grads if g is not None).sqrt()
        optimizer = torch.optim.AdamW(params, lr=1e-6)
        optimizer.zero_grad(set_to_none=True)
        (native + .1 * loss).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.)
        optimizer.step()
        self.assertTrue(all(p.grad is None for p in reference.parameters()))
        for key, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        after, _ = compute_dpo_preference_loss(adapter, reference, images, targets)
        self.assertTrue(torch.isfinite(after))
        with torch.no_grad():
            current = adapter.raw_predictions(images)
            ref_predictions = reference.raw_predictions(images)
            unchanged_loss, quality = dpo_preference_loss_from_predictions(
                current, ref_predictions, targets)
            moved_reference = ref_predictions.clone()
            moved_reference[:, :2] += 10000
            moved_loss, moved_quality = dpo_preference_loss_from_predictions(
                current, moved_reference, targets)
        torch.testing.assert_close(unchanged_loss, moved_loss)
        self.assertEqual(quality['pair_count'], moved_quality['pair_count'])
        torch.testing.assert_close(quality['quality_gap'], moved_quality['quality_gap'])
        print(f'REAL DPO: pairs={diagnostic["pair_count"]} initial_loss={float(loss.detach()):.6f} '
              f'weighted_gradient_ratio={float(dpo_norm / native_norm):.6f} '
              f'loss_after_update={float(after.detach()):.6f}; reference unchanged; '
              f'current_GT_quality_gap={float(quality["quality_gap"]):.6f}; '
              'reference geometry does not affect preferences')


if __name__ == '__main__':
    unittest.main(verbosity=2)
