import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from kb3_hyperparameter_optimization.config import SearchSpaceConfig
from kb3_hyperparameter_optimization.core import Metrics
from kb3_hyperparameter_optimization.search_space import DiscreteSearchSpace
from kb3_hyperparameter_optimization.quality.detector import (
    isolated_evaluation_rng, file_hash, atomic_json, preserve_transform_rng,
)
from kb3_hyperparameter_optimization.quality.pipeline import Pipeline, budget, settings_from, parse_args, validated_result
from kb3_hyperparameter_optimization.quality.ppo import QualityPPO
from kb3_hyperparameter_optimization.quality.state import observation, action_masks, best_gain, STATE_NAMES
from kb3_hyperparameter_optimization.quality.reference import evaluate_reference, read_recipe


class QualityProtocolTests(unittest.TestCase):
    def test_real_loss_range_is_distinguishable_in_float32(self):
        space = DiscreteSearchSpace(SearchSpaceConfig())
        states = []
        for loss in (7.0, 14.0, 70.0):
            metric = Metrics(1, loss, loss + 1, .2, .2, .2, .1, .05)
            states.append(torch.tensor(observation(metric, None, epochs=100, best=.1, stale=0,
                                                   parameters=space.initial_values(), space=space,
                                                   schedule_factor=1.0))[2].item())
        self.assertEqual(len(set(states)), 3)

    def test_initial_loss_is_marked_unavailable(self):
        space = DiscreteSearchSpace(SearchSpaceConfig())
        state = observation(Metrics(0, 0, 0, 0, 0, 0, 0, 0), None, epochs=100, best=0, stale=0,
                            parameters=space.initial_values(), space=space, schedule_factor=1.0)
        self.assertEqual(state[1:5], (0.0, 0.0, 0.0, 0.0))

    def test_mask_keeps_hold_and_rejects_clipping_at_bounds(self):
        space = DiscreteSearchSpace(SearchSpaceConfig())
        for side in ("minimum", "maximum"):
            hp = {name: getattr(spec, side) for name, spec in space.config.parameters.items()}
            masks = action_masks(space, hp)
            for name, mask in zip(space.names, masks):
                spec = space.config.parameters[name]
                self.assertTrue(mask[spec.multipliers.index(0.0 if spec.additive else 1.0)])
            agent = QualityPPO(2, space.action_sizes, seed=3)
            for _ in range(50):
                action = agent.act((0.0, 1.0), masks)
                self.assertEqual(space.apply(hp, action).clipped_count, 0)
                agent.observe(0.0, True)

    def test_reward_telescopes_to_selected_checkpoint_objective(self):
        best = [.1, .12, .12, .19, .19, .25]
        self.assertAlmostEqual(sum(best_gain(a, b) for a, b in zip(best, best[1:])), 100 * (.25 - .1))

    def test_evaluation_preserves_training_random_streams(self):
        random.seed(5)
        np.random.seed(5)
        torch.manual_seed(5)
        states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        with isolated_evaluation_rng():
            random.random()
            np.random.rand(100)
            torch.rand(100)
        actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        self.assertEqual(actual, (random.random(), float(np.random.rand()), float(torch.rand(()))))

    def test_rebuilding_augmentation_preserves_its_private_rng(self):
        import albumentations as A
        old = A.Compose([A.RandomBrightnessContrast(p=1.0)], seed=7)
        new = A.Compose([A.RandomBrightnessContrast(p=1.0)], seed=7)
        image = np.full((8, 8, 3), 100, dtype=np.uint8)
        old(image=image)
        preserve_transform_rng(old, new)
        for _ in range(3):
            np.testing.assert_array_equal(old(image=image)["image"], new(image=image)["image"])

    def test_mask_used_for_both_act_and_ppo_update(self):
        agent = QualityPPO(2, (3,), rollout_steps=8, minibatch_size=4)
        mask = ((False, True, False),)
        for _ in range(8):
            self.assertEqual(agent.act((.1, .2), mask), (1,))
            agent.observe(1.0, True)
        stats = agent.update()
        self.assertEqual(stats["approx_kl"], 0.0)
        self.assertEqual(stats["entropy"], 0.0)
        self.assertEqual(stats["transitions"], 8)

    def test_resume_restores_unupdated_rollout_and_sampling_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.pt"
            agent = QualityPPO(2, (3,), rollout_steps=8)
            for _ in range(3):
                agent.act((.1, .2), ((True, True, True),))
                agent.observe(.2, True)
            self.assertEqual(agent.update(), {})
            agent.save(path, metadata={"next_episode": 3})
            restored, meta = QualityPPO.load(path)
            self.assertEqual(restored.buffer, agent.buffer)
            self.assertEqual(meta["next_episode"], 3)
            self.assertEqual(agent.act((.2, .3), ((True, True, True),)),
                             restored.act((.2, .3), ((True, True, True),)))

    def test_legacy_policy_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            torch.save({"model": {}}, path)
            with self.assertRaisesRegex(ValueError, "legacy policies"):
                QualityPPO.load(path)

    def test_ppo_learns_a_state_dependent_cpu_task(self):
        agent = QualityPPO(2, (2,), seed=11, hidden_size=32, learning_rate=.003,
                           rollout_steps=128, minibatch_size=32)
        masks = ((True, True),)
        for _ in range(12):
            for i in range(128):
                target = i % 2
                state = (float(target), float(1 - target))
                action = agent.act(state, masks)[0]
                agent.observe(1.0 if action == target else -1.0, True)
            agent.update()
        self.assertEqual(agent.act((0.0, 1.0), masks, deterministic=True, learn=False), (0,))
        self.assertEqual(agent.act((1.0, 0.0), masks, deterministic=True, learn=False), (1,))

    def test_budget_and_disjoint_seed_validation(self):
        args = parse_args(["--smoke"])
        settings = settings_from(args)
        plan = budget(settings)
        self.assertEqual(plan["search_detectors"], 8)
        self.assertEqual(plan["maximum_all_epochs"], 36)
        self.assertEqual(plan["pilot_detectors"], 0)
        self.assertEqual(plan["maximum_ppo_training_transitions"], 4)
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump({**settings, "tuning_seeds": [settings["seed"]]}))
            with self.assertRaisesRegex(ValueError, "disjoint"):
                settings_from(parse_args(["--config", str(path)]))

    def test_kb1_recipe_is_the_actual_supervised_recipe(self):
        raw, recipe, path = read_recipe("yolov8n")
        settings = settings_from(parse_args([]))
        self.assertEqual(path.name, "args.yaml")
        self.assertEqual(settings["batch_size"], raw["batch"])
        self.assertEqual(settings["warmup_epochs"], raw["warmup_epochs"])
        for key in ("nbs", "cos_lr", "amp", "close_mosaic", "box", "cls", "dfl", "degrees", "flipud", "mixup"):
            self.assertEqual(settings["trainer_recipe"][key], raw[key])
        self.assertFalse(set(recipe) & {"model", "pretrained", "resume", "data", "project", "patience"})
        self.assertEqual(parse_args([]).stage, "baseline")

    def test_external_reference_is_inference_only_cached_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "kb1_best.pt"
            checkpoint.write_bytes(b"verified external baseline")
            settings = dict(kb1_reference=dict(checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint), seed=0),
                            batch_size=16, img_size=640, device="cpu")
            with patch("kb3_hyperparameter_optimization.quality.reference._canonical_metrics",
                       return_value={"mAP50_95": .38}) as score, \
                 patch("kb3_hyperparameter_optimization.quality.pipeline.train_detector") as train:
                first = evaluate_reference(settings, root, root)
                self.assertEqual(evaluate_reference(settings, root, root), first)
                score.assert_called_once()
                train.assert_not_called()
                self.assertFalse(first["training_performed"])
                self.assertEqual(first["independent_training_runs"], 1)
                self.assertEqual(checkpoint.read_bytes(), b"verified external baseline")
                checkpoint.write_bytes(b"modified")
                with self.assertRaisesRegex(RuntimeError, "checkpoint changed"):
                    evaluate_reference(settings, root, root)

    def test_full_pipeline_skip_resume_preserves_completed_detectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = settings_from(parse_args(["--smoke"]))
            args = SimpleNamespace(data_root="unused", model="yolov8n")
            space_config = SearchSpaceConfig()
            space = DiscreteSearchSpace(space_config)
            calls = []

            def fake_train(**kwargs):
                calls.append(str(kwargs["output_dir"]))
                output = Path(kwargs["output_dir"])
                output.mkdir(parents=True, exist_ok=True)
                checkpoint = output / "selected_best.pt"
                checkpoint.write_bytes(str(kwargs["seed"]).encode())
                agent = kwargs["agent"]
                if agent is not None and kwargs["learn"]:
                    state = (0.0,) * (len(STATE_NAMES) + len(space.names))
                    agent.act(state, action_masks(space, space.initial_values()))
                    agent.observe(1.0, True)
                result = dict(failed=False, best_checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
                              metrics={"map50_95": .1}, final_metrics={"epoch": 3},
                              hyperparameters=kwargs["parameters"] or space.initial_values(),
                              elapsed_seconds=1.0)
                atomic_json(output / "result.json", result)
                return result

            pipeline = Pipeline(args, settings, space_config, root, root / "data.yaml")
            with patch("kb3_hyperparameter_optimization.quality.pipeline.train_detector", side_effect=fake_train):
                pipeline.search()
                first = json.loads((root / "hpo" / "trial_0000" / "result.json").read_text())
                self.assertEqual(first["hyperparameters"], space.initial_values())
                pipeline.evaluate()
                first_count = len(calls)
                pipeline.search()
                pipeline.evaluate()
                self.assertEqual(len(calls), first_count)
            checkpoint = root / "evaluation" / "ppo" / "seed_10001" / "selected_best.pt"
            checkpoint.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "Invalid completed"):
                validated_result(checkpoint.parent / "result.json")


if __name__ == "__main__":
    unittest.main()
