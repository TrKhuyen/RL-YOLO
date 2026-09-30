"""Flatten trajectory JSON files into CSV for analysis and plotting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--output", default="kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/adaptive_rl/trajectories.csv")
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.run_root).rglob("trajectory.json")):
        for record in json.loads(path.read_text(encoding="utf-8")):
            row = {
                "run": str(path.parent),
                "step": record["step"],
                "reward": record["scalar_reward"],
                "terminated": record["terminated"],
                "truncated": record["truncated"],
                "failed": record["failed"],
            }
            row.update({f"metric_{key}": value for key, value in record["metrics"].items()})
            row.update({f"hp_{key}": value for key, value in record["hyperparameters"].items()})
            rows.append(row)
    if not rows:
        raise SystemExit("No trajectory.json files found")
    columns = sorted({key for row in rows for key in row})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

