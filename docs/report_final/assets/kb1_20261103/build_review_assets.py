"""Build KB1 report tables/figures from saved artifacts, without inference."""

import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kb1-report-matplotlib")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
KB = ROOT / "kb1_reward_guided_training"
RUN = KB / "checkpoint_reward_guide_trainning/remaining_seed43_44_ablation_v1"
MODELS = {
    "yolov5s": "YOLOv5s", "yolov8n": "YOLOv8n", "yolov8s": "YOLOv8s",
    "yolov11n": "YOLOv11n", "yolov11s": "YOLOv11s",
    "yolo26n": "YOLO26n", "dp_yolo": "DP-YOLO",
}
SOURCES = set()


def read_json(path):
    SOURCES.add(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def curve(experiment, seed, model, mode):
    directory = RUN / experiment / f"seed{seed}" / model / mode
    done = read_json(directory / "completed.json")
    if done["status"] != "complete":
        raise ValueError(f"Incomplete run: {directory}")
    if experiment == "ablation" and done["steps_run"] != 10000:
        raise ValueError(f"Unexpected ablation budget: {directory}")
    initial = read_json(directory / "baseline.json")
    log = directory / "validation.jsonl"
    SOURCES.add(log)
    # A resumed run may append repeated steps; retain the last saved observation.
    points = {0: initial["mAP50_95"] * 100}
    for line in log.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["step"] <= done["steps_run"]:
            points[record["step"]] = record["metrics"]["mAP50_95"] * 100
    if max(points) != done["steps_run"]:
        raise ValueError(f"Missing final validation: {directory}")
    if abs(points[max(points)] - done["last"]["mAP50_95"] * 100) > 1e-6:
        raise ValueError(f"Final metric mismatch: {directory}")
    if abs(points[done["best_step"]] - done["best"]["mAP50_95"] * 100) > 1e-6:
        raise ValueError(f"Best metric mismatch: {directory}")
    return done, sorted(points.items())


def main():
    source_csv = RUN / "results_all.csv"
    SOURCES.add(source_csv)
    with source_csv.open(encoding="utf-8", newline="") as stream:
        results = list(csv.DictReader(stream))
    summary = []
    for model in MODELS.values():
        row = {"Model": model}
        for stage in ("supervised", "native_only", "kb1b"):
            values = [float(v["mAP50_95"]) * 100 for v in results
                      if v["experiment"] == "continuation" and v["split"] == "val"
                      and v["Model"] == model and v["Stage"] == stage]
            if len(values) != 2:
                raise ValueError(f"Expected two continuation seeds: {model}/{stage}")
            row[f"{stage}_mean_pct"] = statistics.mean(values)
            row[f"{stage}_sample_sd_pct"] = statistics.stdev(values)
        summary.append(row)
    write_csv("validation_summary.csv", summary)

    steps, observations = [], []
    curves = {}
    for model, label in MODELS.items():
        for seed in (43, 44):
            for mode in ("native_only", "kb1b"):
                done, points = curve("continuation", seed, model, mode)
                curves[model, seed, mode] = done, points
                steps.append({"Model": label, "seed": seed, "mode": mode,
                              "best_step": done["best_step"], "steps_run": done["steps_run"],
                              "optimizer_updates": done["optimizer_updates"],
                              "termination": done["termination"]})
                observations.extend({"experiment": "continuation", "Model": label,
                                     "seed": seed, "mode": mode, "step": step,
                                     "validation_map5095_pct": ap} for step, ap in points)
    write_csv("continuation_steps.csv", steps)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "kb1-report-20261103"})
    colors = {"native_only": "#3474a8", "kb1b": "#c45d19"}
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    for ax, (model, label) in zip(axes.flat, MODELS.items()):
        for seed in (43, 44):
            for mode in ("native_only", "kb1b"):
                done, points = curves[model, seed, mode]
                ax.plot(*zip(*points), color=colors[mode],
                        linestyle="-" if seed == 43 else "--", linewidth=1.6)
                ax.scatter(done["best_step"], done["best"]["mAP50_95"] * 100,
                           color=colors[mode], marker="o" if seed == 43 else "s", s=30, zorder=4)
        s0 = curves[model, 43, "kb1b"][1][0][1]
        ax.axhline(s0, color="#777777", linestyle=":", linewidth=1)
        ax.set(title=label, xlabel="Micro-step", ylabel="Validation mAP50–95 (%)", xlim=(0, 10000))
        ax.grid(alpha=0.18)
    axes.flat[-1].axis("off")
    legend = [Line2D([0], [0], color=colors[mode], linestyle=style, label=f"{name}, seed {seed}")
              for mode, name in (("native_only", "S1"), ("kb1b", "G"))
              for seed, style in ((43, "-"), (44, "--"))]
    legend.extend([Line2D([0], [0], color="#777777", linestyle=":", label="S0"),
                   Line2D([0], [0], color="#333333", marker="o", linestyle="", label="Best, seed 43"),
                   Line2D([0], [0], color="#333333", marker="s", linestyle="", label="Best, seed 44")])
    axes.flat[-1].legend(handles=legend, loc="center", frameon=False, fontsize=12)
    fig.suptitle("KB1 continuation: saved validation observations", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "continuation_validation.svg", metadata={"Date": None})
    # PNG is a local preview; the report embeds the SVG.
    fig.savefig(OUT / "continuation_validation.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    modes = {"native_only": ("S1", "#3474a8"), "proxy_only": ("P", "#238b45"),
             "reward_only": ("R", "#8a60b0"), "kb1b": ("G", "#c45d19")}
    for ax, model in zip(axes, ("yolov11n", "yolov8s")):
        for mode, (label, color) in modes.items():
            seed_curves = []
            for seed in (42, 43, 44):
                _, points = curve("ablation", seed, model, mode)
                seed_curves.append(dict(points))
                observations.extend({"experiment": "ablation", "Model": MODELS[model],
                                     "seed": seed, "mode": mode, "step": step,
                                     "validation_map5095_pct": ap} for step, ap in points)
            if any(set(c) != set(seed_curves[0]) for c in seed_curves):
                raise ValueError("Unaligned ablation validation steps")
            xs = sorted(seed_curves[0])
            means = [statistics.mean(c[x] for c in seed_curves) for x in xs]
            sds = [statistics.stdev(c[x] for c in seed_curves) for x in xs]
            ax.plot(xs, means, label=label, color=color, linewidth=1.8)
            ax.fill_between(xs, [a-b for a, b in zip(means, sds)],
                            [a+b for a, b in zip(means, sds)], color=color, alpha=0.10)
        ax.axhline(seed_curves[0][0], color="#777777", linestyle=":", linewidth=1, label="S0")
        ax.set(title=MODELS[model], xlabel="Micro-step", ylabel="Validation mAP50–95 (%)", xlim=(0, 10000))
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, ncol=3)
    fig.suptitle("KB1 fixed-budget ablation: mean ± sample SD, seeds 42/43/44", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "ablation_validation.svg", metadata={"Date": None})
    fig.savefig(OUT / "ablation_validation.png", dpi=130)
    plt.close(fig)
    write_csv("validation_observations.csv", observations)

    sys.path.insert(0, str(KB))
    from dataloader import get_train_transforms, get_val_transforms
    transforms = {"train": get_train_transforms().to_dict(), "val_test": get_val_transforms().to_dict()}
    (OUT / "augmentation_effective.json").write_text(
        json.dumps(transforms, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SOURCES.add(KB / "dataloader.py")
    SOURCES.add(KB / "run_verified.py")
    manifest = {"review_date": "2026-11-03", "inference_rerun": False,
                "continuation_runs": len(steps), "validation_observations": len(observations),
                "matplotlib_version": matplotlib.__version__,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(SOURCES)}}
    (OUT / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved tables, figures and effective augmentation to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
