"""Compare KB3-A fixed HPO and KB3-B adaptive policies from result JSON."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def _flatten(payload) -> tuple[str, str, list[dict]]:
    if isinstance(payload, list):
        return "KB3-B", "adaptive_policy", payload
    if not isinstance(payload, dict):
        raise ValueError("result root must be an object or list")
    scenario, method = str(payload.get("scenario", "unknown")), str(payload.get("method", "unknown"))
    if "trials" in payload:
        records = payload["trials"]
    elif "episodes" in payload:
        records = payload["episodes"]
    else:
        raise ValueError("result document must contain trials or episodes")
    return scenario, method, records


def _finite(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _number(record: dict, name: str) -> float | None:
    # Quality belongs to the persisted best checkpoint. Compute cost belongs to
    # the completed run, not to the epoch at which that checkpoint was found.
    if name == "elapsed_seconds":
        return _finite(record.get("final_metrics", {}).get(name, record.get(name)))
    if name == "total_reward":
        return _finite(record.get(name))
    aliases = {
        "map50_95": ("map50_95", "best_map50_95", "final_map50_95"),
        "map50": ("map50",), "recall": ("recall",),
        "precision": ("precision",), "ap_small": ("ap_small",),
    }
    metrics = record.get("metrics", {})
    for key in aliases[name]:
        value = metrics.get(key, record.get(key))
        if value is not None:
            return _finite(value)
    return None


def _aggregate(records: list[dict], metric: str) -> tuple[float, float]:
    clean = [value for record in records if (value := _number(record, metric)) is not None]
    if not clean:
        return float("nan"), float("nan")
    return statistics.fmean(clean), statistics.pstdev(clean) if len(clean) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="OptionalLabel=path/to/results.json or just a path")
    parser.add_argument("--output", default="runs/kb3/comparison.csv")
    args = parser.parse_args()
    rows = []
    for item in args.inputs:
        label, raw_path = item.split("=", 1) if "=" in item else (None, item)
        scenario, method, records = _flatten(json.loads(Path(raw_path).read_text(encoding="utf-8")))
        valid = [record for record in records if not record.get("failed", False)]
        row = {"scenario": scenario, "method": label or method, "runs": len(records), "successful_runs": len(valid)}
        for metric in ("map50_95", "map50", "precision", "recall", "ap_small", "elapsed_seconds", "total_reward"):
            mean, std = _aggregate(valid, metric)
            row[f"mean_{metric}"], row[f"std_{metric}"] = mean, std
        rows.append(row)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
