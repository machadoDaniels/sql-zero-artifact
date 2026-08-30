#!/usr/bin/env python3
"""Render paper-ready, faceted W&B training-dynamics figures.

Each completed and interrupted run with a target metric appears in exactly one
role/family/state figure. The manifest is the coverage record; figures keep
interrupted retries separate from completed experiments so their short horizons
are not mistaken for comparable final outcomes.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # AAAI forbids Type 3 fonts
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "ara/evidence/data/wandb_training_dynamics_all.csv"
OUT = ROOT / "paper/figures/wandb"
COLORS = {"finished": "#0072B2", "crashed": "#D55E00"}
LINESTYLES = {"finished": "-", "crashed": "--"}
METRIC_LABELS = {
    "critic/score/mean": "mean training reward",
    "challenger/difficulty_eligible": "eligible challenger difficulty",
    "challenger/difficulty": "challenger difficulty",
}


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open() as handle:
        return list(csv.DictReader(handle))


def preferred_metric(role: str, rows: list[dict[str, str]]) -> str | None:
    candidates = (
        ("challenger/difficulty_eligible", "challenger/difficulty")
        if role == "challenger"
        else ("critic/score/mean",)
    )
    available = {row["metric"] for row in rows}
    return next((metric for metric in candidates if metric in available), None)


def clean_name(name: str) -> str:
    return (name.replace("t2sql_", "").replace("_grpo_group8", "")
            .replace("_grpo_batch_group1-5", "").replace("_qwen2.5-coder-3b-instruct", "")
            .replace("_qwen2.5-coder-7b-instruct", "").replace("_omnisql-7b", ""))


def plot_group(key: tuple[str, str, str], rows: list[dict[str, str]]) -> Path | None:
    role, family, state = key
    metric = preferred_metric(role, rows)
    if metric is None:
        return None
    series: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["metric"] == metric:
            series[row["run_id"]].append(row)
    if not series:
        return None

    ordered = sorted(series.items(), key=lambda pair: pair[1][0]["created_at"])
    ncols = 3
    nrows = math.ceil(len(ordered) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.3, max(2.2, 1.9 * nrows)), squeeze=False,
                             sharex=False, sharey=True)
    all_values = [float(row["value"]) for _, values in ordered for row in values]
    low, high = min(all_values), max(all_values)
    padding = max(0.02, 0.08 * (high - low or 1))
    for axis, (run_id, values) in zip(axes.flat, ordered):
        values.sort(key=lambda row: int(row["step"]))
        axis.plot(
            [int(row["step"]) for row in values],
            [float(row["value"]) for row in values],
            color=COLORS[state], linestyle=LINESTYLES[state], linewidth=1.25,
        )
        axis.scatter(int(values[0]["step"]), float(values[0]["value"]), s=12,
                     color=COLORS[state], marker="o", zorder=3)
        axis.scatter(int(values[-1]["step"]), float(values[-1]["value"]), s=14,
                     color=COLORS[state], marker="s", zorder=3)
        start, end = float(values[0]["value"]), float(values[-1]["value"])
        axis.set_title(f"{clean_name(values[0]['run_name'])}\n"
                       f"steps {values[0]['step']}–{values[-1]['step']}; Δ={end-start:+.2f}",
                       fontsize=7, loc="left")
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.45)
        axis.tick_params(labelsize=6)
        axis.set_ylim(low - padding, high + padding)
    for axis in axes.flat[len(ordered):]:
        axis.set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel(METRIC_LABELS[metric], fontsize=7)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("W&B step", fontsize=7)
    title_state = "completed runs" if state == "finished" else "interrupted or crashed runs"
    fig.suptitle(f"{METRIC_LABELS[metric].capitalize()}: {family.replace('-', ' ')} ({title_state})",
                 fontsize=10, y=0.995)
    fig.text(0.5, 0.005, "Solid/dashed style encodes run completion; circles and squares mark first and last logged values.",
             ha="center", fontsize=6.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{role}_{family}_{state}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in load_rows():
        grouped[(row["role"], row["family"], row["state"])].append(row)
    paths = []
    for key, rows in sorted(grouped.items()):
        path = plot_group(key, rows)
        if path:
            paths.append(path)
    print("\n".join(str(path.relative_to(ROOT)) for path in paths))


if __name__ == "__main__":
    main()
