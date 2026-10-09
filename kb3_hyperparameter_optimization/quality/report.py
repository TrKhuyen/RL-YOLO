"""Compare the external KB1 checkpoint with validation-selected A/B artifacts."""

import csv
from pathlib import Path


def checkpoint_rows(reference, records, *, selected_checkpoints=None):
    rows = []
    if reference:
        rows.append(dict(method="KB1 supervised", checkpoint=reference["reference"]["checkpoint"],
                         checkpoint_sha256=reference["reference"]["checkpoint_sha256"],
                         seed=reference["reference"]["seed"], epoch=None,
                         initialization=reference["reference"]["initialization"],
                         selection=reference["reference"]["checkpoint_selection"],
                         scores=reference["metrics"]))
    for label, method in (("A: fixed HPO", "random-search"), ("B: adaptive PPO", "ppo")):
        result = (next(r for r in records[method] if r["best_checkpoint"] == selected_checkpoints[method])
                  if selected_checkpoints is not None else
                  max(records[method], key=lambda r: r["metrics"]["map50_95"]))
        # Real detectors retain the complete canonical score. Fallback also permits
        # reading previous training summaries; absent metrics stay absent, never zero.
        scores = result.get("canonical_best_scores", {})
        scores = {**dict(mAP50_95=result["metrics"]["map50_95"],
                         mAP50=result["metrics"].get("map50"), APs=result["metrics"].get("ap_small"),
                         precision=result["metrics"].get("precision"), AR300=result["metrics"].get("recall")),
                  **scores}
        rows.append(dict(method=label, checkpoint=result["best_checkpoint"],
                         checkpoint_sha256=result["checkpoint_sha256"], seed=result.get("seed"),
                         epoch=result["metrics"].get("epoch"), initialization=result.get("initialization", "scratch"),
                         selection="canonical validation best epoch; highest validation score among final seeds",
                         scores=scores))
    baseline = reference["metrics"]["mAP50_95"] if reference else None
    for row in rows:
        row["delta_kb1_mAP50_95_points"] = (
            100 * (row["scores"]["mAP50_95"] - baseline) if baseline is not None else None
        )
    return rows


def write_checkpoint_comparison(output, rows, *, filename="checkpoint_comparison.csv"):
    """Best artifacts are useful for deployment; seed means remain the main comparison."""
    columns = ("method", "checkpoint", "checkpoint_sha256", "seed", "epoch", "initialization",
               "mAP50_95", "mAP50", "APs", "precision", "operating_recall", "f1", "AR300",
               "delta_kb1_mAP50_95_points", "selection")
    path = Path(output) / filename
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, row["scores"].get(key)) for key in columns})
