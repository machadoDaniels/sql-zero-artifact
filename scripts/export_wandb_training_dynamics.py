#!/usr/bin/env python3
"""Export all W&B training histories with paper-relevant dynamics metrics.

The export is deliberately lossless: it inventories every run in the project that
logged a target metric, including interrupted attempts, and writes one tidy CSV
plus per-run ``.dat`` files for the paper appendix. Figures may facet completed
runs separately from retries, but no run is omitted silently.

Authentication is delegated to the standard WANDB_API_KEY environment variable.
Run with a key injected only for this process, e.g. from the macOS Keychain.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import wandb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = "ANON-ENTITY/dr-zero-t2sql"
TARGET_METRICS = (
    "critic/score/mean",
    "challenger/difficulty_eligible",
    "challenger/difficulty",
    "actor/entropy",
    "response_length/mean",
)
PLOT_METRICS = {
    "critic/score/mean": "solver_reward",
    "challenger/difficulty_eligible": "challenger_difficulty",
    "challenger/difficulty": "challenger_difficulty",
}


def classify(name: str, state: str) -> dict[str, str]:
    """Return conservative metadata inferred from stable W&B run names."""
    lower = name.lower()
    role = "challenger" if "challenger" in lower else "solver"
    size = "7B" if "7b" in lower or "omnisql" in lower else "3B"
    if "goldbird" in lower:
        family = "matched-bird-gold"
    elif "gold" in lower:
        family = "spider-gold"
    elif "omni" in lower:
        family = "omnisql-init"
    elif "bird" in lower:
        family = "bird-self-play"
    elif "iter" in lower:
        family = "spider-self-play"
    else:
        family = "other"
    iteration = ""
    for token in ("iter0", "iter1", "iter2", "iter3", "bird1", "bird2", "bird3"):
        if token in lower:
            iteration = token[-1]
            break
    return {
        "role": role,
        "model_size": size,
        "family": family,
        "iteration": iteration,
        "state": state,
    }


def finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "ara/evidence/data/wandb_training_dynamics_all.csv",
    )
    parser.add_argument(
        "--dat-dir",
        type=Path,
        default=ROOT / "paper/data/wandb",
    )
    args = parser.parse_args()

    api = wandb.Api()
    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for run in api.runs(args.project, per_page=500):
        summary = dict(run.summary)
        available = [metric for metric in TARGET_METRICS if metric in summary]
        if not available:
            continue
        meta = classify(run.name, run.state)
        manifest.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "role": meta["role"],
                "model_size": meta["model_size"],
                "family": meta["family"],
                "iteration": meta["iteration"],
                "available_metrics": ";".join(available),
            }
        )
        history = run.scan_history(keys=["_step", *available], page_size=1000)
        latest: dict[tuple[str, int], float] = {}
        for item in history:
            step_value = finite(item.get("_step"))
            if step_value is None:
                continue
            step = int(step_value)
            for metric in available:
                value = finite(item.get(metric))
                if value is not None:
                    latest[(metric, step)] = value
        for (metric, step), value in latest.items():
            rows.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "state": run.state,
                    "created_at": run.created_at,
                    "role": meta["role"],
                    "model_size": meta["model_size"],
                    "family": meta["family"],
                    "iteration": meta["iteration"],
                    "metric": metric,
                    "step": step,
                    "value": f"{value:.12g}",
                }
            )

    fields = [
        "run_id", "run_name", "state", "created_at", "role", "model_size",
        "family", "iteration", "metric", "step", "value",
    ]
    rows.sort(key=lambda row: (row["metric"], row["run_name"], int(row["step"])))
    write_rows(args.output, rows, fields)
    write_rows(args.output.with_name("wandb_training_run_manifest.csv"), manifest, list(manifest[0]) if manifest else [])

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["metric"] in PLOT_METRICS:
            grouped[(row["run_id"], row["metric"])].append(row)
    args.dat_dir.mkdir(parents=True, exist_ok=True)
    for (run_id, metric), series in grouped.items():
        stem = f"{PLOT_METRICS[metric]}__{run_id}.dat"
        write_rows(args.dat_dir / stem, series, fields)

    completed = sum(row["state"] == "finished" for row in manifest)
    print(f"exported {len(rows)} observations from {len(manifest)} runs ({completed} finished)")
    print(f"tidy CSV: {args.output.relative_to(ROOT)}")
    print(f"manifest: {args.output.with_name('wandb_training_run_manifest.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
