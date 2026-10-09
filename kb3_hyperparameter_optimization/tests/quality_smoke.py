"""Explicit tiny CPU end-to-end quality pipeline gate (no real dataset training)."""

from pathlib import Path
import tempfile

from PIL import Image
import yaml

from kb3_hyperparameter_optimization.quality.pipeline import main


def run():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = root / "data"
        for split in ("train", "valid", "test"):
            (dataset / split / "images").mkdir(parents=True)
            (dataset / split / "labels").mkdir()
            for i in range(2):
                Image.new("RGB", (64, 64), (100 + i * 20, 100, 60)).save(dataset / split / "images" / f"{i}.jpg")
                (dataset / split / "labels" / f"{i}.txt").write_text("0 .5 .5 .3 .3\n")
        data_yaml = root / "dataset.yaml"
        data_yaml.write_text(yaml.safe_dump(dict(nc=28, names=[f"class_{i}" for i in range(28)])))
        config = yaml.safe_load(Path("kb3_hyperparameter_optimization/configs/kb3_quality.yaml").read_text())
        config.update(device="cpu", batch_size=2, img_size=64)
        config_yaml = root / "quality.yaml"
        config_yaml.write_text(yaml.safe_dump(config))
        output = root / "results"
        common = ["--config", str(config_yaml), "--data-root", str(dataset), "--data-config", str(data_yaml),
                  "--output-dir", str(output), "--smoke", "--stage", "all"]
        main([*common, "--search-target", "1"])
        assert not (output / "comparison.json").exists()
        assert not (output / "selection_lock.json").exists()
        first_detector = (output / "hpo/trial_0000/result.json").read_bytes()
        first_policy_episode = (output / "ppo/episode_0000/result.json").read_bytes()
        common += ["--search-target", "2", "--finalize"]
        main([*common, "--resume"])
        assert (output / "hpo/trial_0000/result.json").read_bytes() == first_detector
        assert (output / "ppo/episode_0000/result.json").read_bytes() == first_policy_episode
        comparison = (output / "comparison.json").read_bytes()
        test_comparison = (output / "test_comparison.json").read_bytes()
        snapshot = (output / "starting_train.yaml").read_bytes()
        tampered = yaml.safe_load(snapshot)
        tampered["pretrained"] = True
        (output / "starting_train.yaml").write_text(yaml.safe_dump(tampered))
        try:
            main([*common, "--resume", "--check-only"])
        except ValueError as error:
            assert "starting_train.yaml changed" in str(error)
        else:
            raise AssertionError("Tampered initialization was accepted")
        (output / "starting_train.yaml").write_bytes(snapshot)
        # The second invocation must only validate/skip, not train again.
        main([arg for arg in [*common, "--resume"] if arg != "--finalize"])
        assert (output / "comparison.json").read_bytes() == comparison
        assert (output / "test_comparison.json").read_bytes() == test_comparison
        print("REAL QUALITY SMOKE PASS: staged search, no discarded work, both scenarios, tuning, frozen validation/test and resume.")


if __name__ == "__main__":
    run()
