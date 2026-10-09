"""Regression gates for the user's KB1 recipe / external checkpoint contract."""

import csv
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import yaml

from kb3_hyperparameter_optimization.config import SearchSpaceConfig
from kb3_hyperparameter_optimization.quality.detector import training_overrides, file_hash, atomic_json, isolated_evaluation_rng
from kb3_hyperparameter_optimization.quality.matrix import run_models
from kb3_hyperparameter_optimization.quality.pipeline import parse_args, settings_from, resolved_start, Pipeline
from kb3_hyperparameter_optimization.quality import PROTOCOL
from kb3_hyperparameter_optimization.quality.ppo import QualityPPO
from kb3_hyperparameter_optimization.quality.reference import prepare_reference, read_recipe
from kb3_hyperparameter_optimization.quality.report import checkpoint_rows, write_checkpoint_comparison
from kb3_hyperparameter_optimization.run_all import MODELS, ROOT


class QualityContractTests(unittest.TestCase):
    def test_canonical_math_flags_are_shared_and_restored_even_after_failure(self):
        def state():
            return (torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled(),
                    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)

        before = state()
        with self.assertRaisesRegex(RuntimeError, "evaluation failure"):
            with isolated_evaluation_rng():
                self.assertEqual(state(), (True, True, False, True, False, False))
                raise RuntimeError("evaluation failure")
        self.assertEqual(state(), before)

    def test_every_supported_model_inherits_its_own_recipe_and_batch(self):
        for model in MODELS:
            with self.subTest(model=model):
                raw, recipe, path = read_recipe(model)
                settings = settings_from(parse_args(["--model", model]))
                self.assertEqual(path.parent.name, model)
                self.assertEqual(settings["batch_size"], raw["batch"])
                self.assertEqual(settings["img_size"], raw["imgsz"])
                self.assertEqual(settings["trainer_recipe"], recipe)

    def test_nondefault_kb1_hyperparameters_replace_search_space_defaults(self):
        args = parse_args([])
        settings = settings_from(args)
        raw, recipe, path = read_recipe(args.model)
        raw = {**raw, "lr0": .006, "momentum": .91, "weight_decay": .0009}
        metadata = dict(checkpoint_sha256="verified", dataset={"sha256": "verified_dataset"})
        provenance = SimpleNamespace(verify_supervised=Mock(return_value=metadata),
                                     DATA_ROOT=ROOT / args.data_root, DATA_CONFIG=ROOT / args.data_config)
        checkpoint = dict(train_args=raw, model=SimpleNamespace(yaml={"nc": 28}))
        with patch.dict("sys.modules", {"run_provenance": provenance}), \
             patch("kb3_hyperparameter_optimization.quality.reference.read_recipe", return_value=(raw, recipe, path)), \
             patch("kb3_hyperparameter_optimization.quality.reference.torch.load", return_value=checkpoint):
            space = prepare_reference(args, settings, SearchSpaceConfig())
        for key in ("lr0", "momentum", "weight_decay"):
            self.assertEqual(space.parameters[key].initial, raw[key])
        self.assertEqual(space.parameters["augmentation_strength"].initial, .5)
        provenance.verify_supervised.assert_called_once()
        self.assertTrue(settings["kb1_reference"]["checkpoint"].endswith("weights\\best.pt"))

    def test_starting_yaml_contains_actual_training_inputs_without_reference_weights(self):
        args = parse_args([])
        settings = settings_from(args)
        space = SearchSpaceConfig()
        settings["kb1_reference"] = dict(checkpoint="comparison_only/best.pt")
        config = resolved_start(args, settings, space, "dataset.yaml")
        roundtrip = yaml.safe_load(yaml.safe_dump(config))
        self.assertEqual(roundtrip["model"], "yolov8n.yaml")
        self.assertFalse(roundtrip["pretrained"])
        self.assertFalse(roundtrip["resume"])
        self.assertEqual(roundtrip["lr0"], space.parameters["lr0"].initial)
        self.assertEqual(roundtrip["momentum"], space.parameters["momentum"].initial)
        self.assertEqual(roundtrip["weight_decay"], space.parameters["weight_decay"].initial)
        self.assertEqual(roundtrip["nbs"], 64)
        self.assertEqual(roundtrip["degrees"], 10)
        self.assertNotIn("comparison_only", yaml.safe_dump(roundtrip))
        self.assertEqual(training_overrides(settings, {k: v.initial for k, v in space.parameters.items()}),
                         {k: v for k, v in roundtrip.items() if k not in {"model", "data", "seed"}})

    def test_best_checkpoint_comparison_selects_by_validation_and_preserves_kb1(self):
        reference = dict(reference=dict(checkpoint="kb1/best.pt", checkpoint_sha256="kb1hash", seed=0,
                                        initialization="pretrained", checkpoint_selection="historical native best"),
                         metrics={"mAP50_95": .38, "mAP50": .49, "f1": .52})

        def record(seed, score):
            return dict(seed=seed, best_checkpoint=f"seed_{seed}/best.pt", checkpoint_sha256=f"hash{seed}",
                        metrics={"epoch": 20, "map50_95": score})

        rows = checkpoint_rows(reference, {"random-search": [record(1, .4), record(2, .39)],
                                           "ppo": [record(1, .41), record(2, .42)]})
        self.assertEqual([r["checkpoint"] for r in rows], ["kb1/best.pt", "seed_1/best.pt", "seed_2/best.pt"])
        self.assertAlmostEqual(rows[1]["delta_kb1_mAP50_95_points"], 2)
        self.assertAlmostEqual(rows[2]["delta_kb1_mAP50_95_points"], 4)
        self.assertEqual(reference["metrics"]["mAP50_95"], .38)
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint_comparison(directory, rows)
            with (Path(directory) / "checkpoint_comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 3)
            self.assertEqual(csv_rows[1]["f1"], "")  # unavailable metrics must not become fictitious zeros

    def test_all_models_are_preflighted_before_any_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--model", "all", "--stage", "baseline", "--output-dir", directory])
            commands = []

            def fake_run(command, **kwargs):
                commands.append(command)

            with patch("kb3_hyperparameter_optimization.quality.matrix.subprocess.run", side_effect=fake_run):
                run_models(args)
            self.assertEqual(len(commands), 2 * len(MODELS))
            self.assertTrue(all("--check-only" in c for c in commands[:len(MODELS)]))
            self.assertTrue(all("--check-only" not in c for c in commands[len(MODELS):]))
            for name, command in zip(MODELS, commands[len(MODELS):]):
                self.assertEqual(command[command.index("--model") + 1], name)
                self.assertEqual(Path(command[command.index("--output-dir") + 1]).name, name)

    def test_invalid_model_reference_prevents_every_model_from_training(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not_created"
            args = parse_args(["--model", "all", "--stage", "all", "--output-dir", str(output)])
            calls = []

            def reject_second(command, **kwargs):
                calls.append(command)
                if len(calls) == 2:
                    raise subprocess.CalledProcessError(1, command)

            with patch("kb3_hyperparameter_optimization.quality.matrix.subprocess.run", side_effect=reject_second):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_models(args)
            self.assertTrue(all("--check-only" in c for c in calls))
            self.assertFalse(output.exists())

    def test_test_evaluation_never_reselects_best_seed_or_trains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = {}
            for method in ("random-search", "ppo"):
                records[method] = []
                for seed, val_score in ((1, .4), (2, .3)):
                    checkpoint = root / f"{method}_{seed}.pt"
                    checkpoint.write_bytes(f"frozen {method} {seed}".encode())
                    records[method].append(dict(seed=seed, best_checkpoint=str(checkpoint),
                                                checkpoint_sha256=file_hash(checkpoint),
                                                metrics=dict(map50_95=val_score, epoch=7)))
            atomic_json(root / "comparison.json", dict(protocol=PROTOCOL, metrics_split="validation", records=records,
                                                       kb1_comparison_caveat="historical pretrained reference"))
            before = (root / "comparison.json").read_bytes()
            settings = dict(batch_size=16, img_size=640, device="cpu")
            pipeline = Pipeline(SimpleNamespace(data_root=root), settings, SearchSpaceConfig(), root, root / "dataset.yaml")
            reference = dict(reference=dict(checkpoint="kb1/best.pt", checkpoint_sha256="external", seed=0,
                                            initialization="pretrained", checkpoint_selection="native best"),
                             metrics={"mAP50_95": .35})

            def test_scores(checkpoint, args, seed):
                self.assertEqual(args.split, "test")
                return dict(mAP50_95=.1 if seed == 1 else .9)

            with patch.object(pipeline, "baseline", return_value=reference), \
                 patch.object(pipeline, "detector") as train, \
                 patch("kb3_hyperparameter_optimization.quality.pipeline._canonical_metrics", side_effect=test_scores) as score:
                pipeline.evaluate_test()
                first = (root / "test_comparison.json").read_bytes()
                pipeline.evaluate_test()
                self.assertEqual((root / "test_comparison.json").read_bytes(), first)
                self.assertEqual(score.call_count, 4)
                train.assert_not_called()
                selected = json.loads(first)["best_checkpoint_comparison"]
                self.assertTrue(all(row["seed"] == 1 for row in selected[1:]))
                self.assertTrue(all(row["scores"]["mAP50_95"] == .1 for row in selected[1:]))
                self.assertFalse(json.loads(first)["test_used_for_selection"])
                self.assertEqual((root / "comparison.json").read_bytes(), before)
                (root / "ppo_1.pt").write_bytes(b"changed weights")
                with self.assertRaisesRegex(RuntimeError, "Frozen detector changed"):
                    pipeline.evaluate_test()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA RNG gate requires CUDA")
    def test_policy_construction_and_load_preserve_cuda_and_cpu_rng(self):
        cpu_state, cuda_state = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        agent = QualityPPO(2, (2,), seed=987)
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_state))
        self.assertTrue(torch.equal(torch.cuda.get_rng_state(), cuda_state))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            agent.save(checkpoint)
            QualityPPO.load(checkpoint)
            self.assertTrue(torch.equal(torch.get_rng_state(), cpu_state))
            self.assertTrue(torch.equal(torch.cuda.get_rng_state(), cuda_state))


if __name__ == "__main__":
    unittest.main()
