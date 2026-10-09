"""Optional real CPU gates for the continuous quality trainer."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from PIL import Image

from kb3_hyperparameter_optimization.config import SearchSpaceConfig
from kb3_hyperparameter_optimization.quality.detector import train_detector
from kb3_hyperparameter_optimization.quality.reference import read_recipe


@unittest.skipUnless(os.environ.get("KB3_TEST_QUALITY") == "1", "optional real quality trainer integration")
class QualityUltralyticsTests(unittest.TestCase):
    def test_canonical_early_stop_finishes_rl_episode_and_final_retraining_stays_full(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            for split in ('train', 'valid', 'test'):
                (data / split / 'images').mkdir(parents=True)
                (data / split / 'labels').mkdir()
                for i in range(2):
                    Image.new('RGB', (64, 64), (120, 130, 70)).save(data / split / 'images' / f'{i}.jpg')
                    (data / split / 'labels' / f'{i}.txt').write_text('0 .5 .5 .2 .2\n')
            data_yaml = root / 'data.yaml'
            data_yaml.write_text(yaml.safe_dump(dict(path=data.as_posix(), train='train/images', val='valid/images',
                                                    test='test/images', nc=28, names=[f'class_{i}' for i in range(28)])))
            _, recipe, _ = read_recipe('yolov8n')
            settings = dict(epochs=8, segment_epochs=2, warmup_epochs=0, lrf=.1, workers=0,
                            device='cpu', batch_size=2, img_size=64, trainer_recipe={**recipe, 'nbs': 2, 'close_mosaic': 1},
                            early_stopping=dict(patience=2, min_epochs=3, min_delta=0.0))

            class LearningHold:
                def __init__(self):
                    self.rewards = []

                def act(self, *args, **kwargs):
                    return (2, 1, 1, 1)

                def observe(self, reward, done):
                    self.rewards.append((reward, done))

            scores = [dict(precision=.5, recall=.5, mAP50=.5, mAP50_95=s, APs=.1)
                      for s in (0, .1, .2, .19, .18, .17, .16, .15, .14)]
            common = dict(data_yaml=data_yaml, data_root=data, model='yolov8n.yaml', seed=42, space_config=SearchSpaceConfig())
            agent = LearningHold()
            with patch('kb3_hyperparameter_optimization.quality.detector._canonical_metrics', side_effect=scores):
                early = train_detector(settings=settings, **common, output_dir=root / 'early', agent=agent, learn=True)
            self.assertEqual(early['actual_epochs'], 4)
            self.assertTrue(early['stopped_early'])
            self.assertEqual(early['metrics']['epoch'], 2)
            self.assertEqual(agent.rewards, [(0.0, True)])
            decisions = json.loads((root / 'early/decisions.json').read_text())
            self.assertEqual(len(decisions), 1)
            self.assertTrue(decisions[-1]['done'])
            settings = {**settings, 'early_stopping': None}
            final_agent = LearningHold()
            with patch('kb3_hyperparameter_optimization.quality.detector._canonical_metrics', side_effect=scores):
                final = train_detector(settings=settings, **common, output_dir=root / 'final', agent=final_agent, learn=True)
            self.assertEqual(final['actual_epochs'], 8)
            self.assertFalse(final['early_stopping'])
            self.assertFalse(final['stopped_early'])
            self.assertEqual([done for _, done in final_agent.rewards], [False, False, True])

    def test_kb1_recipe_start_reward_and_canonical_best_without_loading_kb1_weights(self):
        from dataclasses import replace
        from ultralytics import YOLO
        from kb3_hyperparameter_optimization.quality.detector import file_hash

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            for split in ("train", "valid", "test"):
                (data / split / "images").mkdir(parents=True)
                (data / split / "labels").mkdir()
                for i in range(2):
                    Image.new("RGB", (64, 64), (120, 130, 70)).save(data / split / "images" / f"{i}.jpg")
                    (data / split / "labels" / f"{i}.txt").write_text("0 .5 .5 .2 .2\n")
            data_yaml = root / "data.yaml"
            data_yaml.write_text(yaml.safe_dump(dict(path=data.as_posix(), train="train/images", val="valid/images",
                                                    test="test/images", nc=28, names=[f"class_{i}" for i in range(28)])))
            external = root / "kb1_best.pt"
            external.write_bytes(b"external reference must never be loaded for training")
            external_hash = file_hash(external)
            _, recipe, _ = read_recipe("yolov8n")
            recipe = {**recipe, "nbs": 2, "close_mosaic": 1, "degrees": 13, "flipud": .17}
            settings = dict(epochs=3, segment_epochs=1, warmup_epochs=0, lrf=.1, workers=0,
                            device="cpu", batch_size=2, img_size=64, trainer_recipe=recipe,
                            kb1_reference={"checkpoint": str(external)})
            space = SearchSpaceConfig()
            parameters = dict(space.parameters)
            for name, value in (("lr0", .006), ("momentum", .91), ("weight_decay", .0009)):
                parameters[name] = replace(parameters[name], initial=value)
            space = replace(space, parameters=parameters)

            class LearningHold:
                rewards = []

                def act(self, observation, masks, **kwargs):
                    return (2, 1, 1, 1)

                def observe(self, reward, done):
                    self.rewards.append((reward, done))

            scores = [dict(precision=.5, recall=.5, mAP50=.9, mAP50_95=score, APs=.1, f1=.5)
                      for score in (.8, .1, .4, .2)]
            agent = LearningHold()
            with patch("ultralytics.YOLO", wraps=YOLO) as constructor, \
                 patch("kb3_hyperparameter_optimization.quality.detector._canonical_metrics", side_effect=scores):
                result = train_detector(settings=settings, data_yaml=data_yaml, data_root=data, model="yolov8n.yaml",
                                        seed=42, output_dir=root / "detector", space_config=space, agent=agent, learn=True)
            constructor.assert_called_once_with("yolov8n.yaml")
            self.assertEqual(file_hash(external), external_hash)
            self.assertEqual(result["metrics"]["epoch"], 2)
            self.assertEqual(result["metrics"]["map50_95"], .4)
            self.assertEqual(result["canonical_best_scores"]["f1"], .5)
            self.assertEqual(result["final_metrics"]["map50_95"], .2)
            self.assertAlmostEqual(sum(r for r, _ in agent.rewards), 100 * (.4 - .1))
            self.assertEqual([done for _, done in agent.rewards], [False, True])
            import json
            initial = json.loads((root / "detector" / "initial.json").read_text())
            epochs = json.loads((root / "detector" / "epochs.json").read_text())
            self.assertEqual(initial["hyperparameters"]["lr0"], .006)
            self.assertEqual(initial["hyperparameters"]["weight_decay"], .0009)
            self.assertEqual(initial["hyperparameters"]["momentum"], .91)
            self.assertEqual(epochs[0]["augmentation"]["degrees"], 13)
            self.assertEqual(epochs[0]["augmentation"]["flipud"], .17)
            selected = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
            self.assertEqual(selected["epoch"] + 1, 2)
            self.assertFalse(selected["train_args"]["pretrained"])

    def test_live_model_continuity_hold_equivalence_and_parameter_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            for split in ("train", "valid", "test"):
                (data / split / "images").mkdir(parents=True)
                (data / split / "labels").mkdir()
                for i in range(2):
                    Image.new("RGB", (64, 64), (100 + 20 * i, 130, 70)).save(data / split / "images" / f"{i}.jpg")
                    (data / split / "labels" / f"{i}.txt").write_text("0 .5 .5 .2 .2\n")
            data_yaml = root / "data.yaml"
            data_yaml.write_text(yaml.safe_dump(dict(path=data.as_posix(), train="train/images", val="valid/images",
                                                    test="test/images", nc=28, names=[f"class_{i}" for i in range(28)])))
            settings = dict(epochs=3, segment_epochs=1, warmup_epochs=1, lrf=.1, workers=0,
                            device=os.environ.get("KB3_TEST_DEVICE", "cpu"), batch_size=2, img_size=64)
            _, recipe, _ = read_recipe("yolov8n")
            settings["trainer_recipe"] = {**recipe, "nbs": 2, "close_mosaic": 2}
            space = SearchSpaceConfig()

            class Schedule:
                def __init__(self, change=False):
                    self.change = change
                    self.actions = 0

                def act(self, state, masks, **kwargs):
                    self.actions += 1
                    return (1, 2, 0, 0) if self.change else (2, 1, 1, 1)

            common = dict(settings=settings, data_yaml=data_yaml, data_root=data,
                          model=os.environ.get("KB3_TEST_MODEL", "yolov8n.yaml"), seed=42, space_config=space)
            baseline = train_detector(**common, output_dir=root / "baseline")
            schedule = Schedule()
            hold = train_detector(**common, output_dir=root / "hold", agent=schedule)
            self.assertEqual(schedule.actions, 2)
            self.assertEqual(baseline["initial_weights_sha256"], hold["initial_weights_sha256"])
            self.assertEqual(baseline["final_live_weights_sha256"], hold["final_live_weights_sha256"])
            self.assertEqual(baseline["final_ema_weights_sha256"], hold["final_ema_weights_sha256"])
            baseline_model = torch.load(baseline["best_checkpoint"], map_location="cpu", weights_only=False)["model"]
            hold_model = torch.load(hold["best_checkpoint"], map_location="cpu", weights_only=False)["model"]
            for name, tensor in baseline_model.state_dict().items():
                self.assertTrue(torch.equal(tensor, hold_model.state_dict()[name]), name)
            self.assertEqual(baseline["metrics"]["map50_95"], hold["metrics"]["map50_95"])
            changed = train_detector(**common, output_dir=root / "changed", agent=Schedule(change=True))
            import json
            history = json.loads((root / "changed" / "epochs.json").read_text())
            self.assertEqual([h["ema_updates"] for h in history], [1, 2, 3])
            for epoch, h in enumerate(history):
                hp = h["hyperparameters"]
                expected_lr = .01 * (.8 ** epoch) * (1 - .9 * epoch / 3)
                if epoch > 0:
                    for lr in h["optimizer_lr"]:
                        self.assertAlmostEqual(lr, expected_lr)
                self.assertAlmostEqual(h["augmentation"]["scale"], .5 - .1 * epoch)
                self.assertAlmostEqual(h["augmentation"]["degrees"], 10 * (1 - .2 * epoch))
                self.assertAlmostEqual(h["augmentation"]["flipud"], .3 * (1 - .2 * epoch))
                self.assertEqual(h["gradient_accumulation"], 1)
                if settings["device"] != "cpu":
                    self.assertTrue(h["amp_enabled"])
                if epoch > 0:
                    self.assertEqual(h["augmentation"]["mosaic"], 0.0)
                    self.assertEqual(h["augmentation"]["mixup"], 0.0)
                else:
                    self.assertEqual(h["augmentation"]["mixup"], .1)
                if epoch > 0:
                    for momentum in h["optimizer_momentum"]:
                        self.assertAlmostEqual(momentum, .937 - .02 * epoch)
                for decay in h["optimizer_weight_decay"]:
                    if decay:
                        self.assertAlmostEqual(decay, .0005 * 1.2 ** epoch)
            selected = torch.load(changed["best_checkpoint"], map_location="cpu", weights_only=False)
            self.assertEqual(selected["epoch"] + 1, changed["metrics"]["epoch"])
            self.assertEqual(changed["final_metrics"]["epoch"], 3)

    def test_gradient_accumulation_continues_across_policy_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            for split in ("train", "valid", "test"):
                (data / split / "images").mkdir(parents=True)
                (data / split / "labels").mkdir()
                for i in range(2):
                    Image.new("RGB", (64, 64), (120, 130, 70)).save(data / split / "images" / f"{i}.jpg")
                    (data / split / "labels" / f"{i}.txt").write_text("0 .5 .5 .2 .2\n")
            data_yaml = root / "data.yaml"
            data_yaml.write_text(yaml.safe_dump(dict(path=data.as_posix(), train="train/images", val="valid/images",
                                                    test="test/images", nc=28, names=[f"class_{i}" for i in range(28)])))
            _, recipe, _ = read_recipe("yolov8n")
            settings = dict(epochs=7, segment_epochs=2, warmup_epochs=0, lrf=.1, workers=0,
                            device="cpu", batch_size=2, img_size=64,
                            trainer_recipe={**recipe, "nbs": 8, "close_mosaic": 2})

            class Hold:
                def act(self, *args, **kwargs):
                    return (2, 1, 1, 1)

            common = dict(settings=settings, data_yaml=data_yaml, data_root=data,
                          model="yolov8n.yaml", seed=42, space_config=SearchSpaceConfig())
            baseline = train_detector(**common, output_dir=root / "default")
            hold = train_detector(**common, output_dir=root / "hold", agent=Hold())
            self.assertEqual(hold["decisions"], 3)
            self.assertEqual(baseline["final_live_weights_sha256"], hold["final_live_weights_sha256"])
            self.assertEqual(baseline["final_ema_weights_sha256"], hold["final_ema_weights_sha256"])
            import json
            history = json.loads((root / "hold" / "epochs.json").read_text())
            self.assertEqual([r["ema_updates"] for r in history], [0, 0, 0, 1, 1, 1, 1])
            self.assertTrue(all(r["gradient_accumulation"] == 4 for r in history))
            for row in history:
                for decay in row["optimizer_weight_decay"]:
                    if decay:
                        self.assertAlmostEqual(decay, .0005)


if __name__ == "__main__":
    unittest.main()
