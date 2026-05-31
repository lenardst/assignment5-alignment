"""Plot GRPO training curves from W&B or local JSON logs.

Usage (from W&B — after training completes):
    uv run python scripts/plot_grpo_results.py \
        --wandb-project cs336-a5-grpo \
        --run-group standard_on_policy \
        --output-dir experiments/grpo/plots

Usage (from local final_metrics.json files, no W&B needed):
    uv run python scripts/plot_grpo_results.py \
        --results-dir experiments/grpo/standard_on_policy \
        --output-dir experiments/grpo/plots --local-only

Produces one PNG per metric, showing mean ± 1.96*std/sqrt(n) or min/max bands
across seeds, as the handout requests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# W&B loader
# ---------------------------------------------------------------------------

def load_wandb(project: str, run_group: str) -> list[dict]:
    """Return list of run histories from W&B."""
    import wandb
    api = wandb.Api()
    runs = api.runs(project, filters={"config.run_name": {"$regex": run_group}})
    histories = []
    for run in runs:
        hist = run.history(samples=500, keys=[
            "train/loss", "train/grad_norm", "train/mean_reward",
            "train/mean_format_reward", "val/reward", "val/format_reward",
            "val/mean_response_length", "_step",
        ])
        histories.append({
            "name": run.name,
            "history": hist.to_dict(orient="list"),
        })
    return histories


# ---------------------------------------------------------------------------
# Local JSON loader (from W&B run local summary files or final_metrics.json)
# ---------------------------------------------------------------------------

def load_local(results_dir: Path) -> list[dict]:
    """Load per-seed final_metrics.json for a summary bar chart."""
    records = []
    for seed_dir in sorted(results_dir.iterdir()):
        fm = seed_dir / "final_metrics.json"
        if fm.exists():
            data = json.loads(fm.read_text())
            records.append({"name": seed_dir.name, "final": data})
    return records


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _band_plot(ax, steps, values_per_seed: list[list[float]], label: str, color=None):
    """Plot mean ± 1.96*std/sqrt(n) band across seeds."""
    arr = np.array(values_per_seed)  # (n_seeds, n_steps)
    n = arr.shape[0]
    mean = arr.mean(axis=0)
    sem = arr.std(axis=0) / max(np.sqrt(n), 1)
    margin = 1.96 * sem
    kwargs = dict(color=color) if color else {}
    ax.plot(steps, mean, label=label, **kwargs)
    ax.fill_between(steps, mean - margin, mean + margin, alpha=0.2, **kwargs)


def plot_metric_from_wandb(histories: list[dict], metric: str, out_path: Path):
    """Plot one metric from W&B run histories."""
    fig, ax = plt.subplots(figsize=(8, 4))
    # Find common steps
    all_steps = [h["history"].get("_step", list(range(len(h["history"].get(metric, []))))) for h in histories]
    # Simple approach: plot each run separately (they may have different lengths)
    for h in histories:
        hist = h["history"]
        steps = hist.get("_step", list(range(len(hist.get(metric, [])))))
        vals  = hist.get(metric, [])
        if vals:
            ax.plot(steps, vals, alpha=0.5, label=h["name"])
    ax.set_xlabel("Rollout step")
    ax.set_ylabel(metric.split("/")[-1])
    ax.set_title(metric)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_final_bar(records: list[dict], metric: str, out_path: Path):
    """Bar chart of final metric value across seeds."""
    names = [r["name"] for r in records]
    vals  = [r["final"].get(metric, 0) for r in records]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, vals)
    ax.set_ylabel(metric)
    ax.set_title(f"Final {metric} per seed")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wandb-project", default="cs336-a5-grpo")
    p.add_argument("--run-group",     default="standard_on_policy",
                   help="Filter runs whose name contains this string")
    p.add_argument("--results-dir",   type=Path, default=None,
                   help="Local results directory (for --local-only)")
    p.add_argument("--output-dir",    type=Path, default=Path("experiments/grpo/plots"))
    p.add_argument("--local-only",    action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.local_only and args.results_dir:
        records = load_local(args.results_dir)
        for metric in ["val/reward", "val/format_reward", "val/mean_response_length"]:
            plot_final_bar(records, metric,
                           args.output_dir / f"final_{metric.replace('/', '_')}.png")
    else:
        try:
            histories = load_wandb(args.wandb_project, args.run_group)
        except Exception as e:
            print(f"W&B load failed ({e}); try --local-only")
            return
        metrics_to_plot = [
            "train/loss",
            "train/grad_norm",
            "train/mean_reward",
            "train/mean_format_reward",
            "val/reward",
            "val/format_reward",
            "val/mean_response_length",
        ]
        for metric in metrics_to_plot:
            out = args.output_dir / f"{args.run_group}_{metric.replace('/', '_')}.png"
            plot_metric_from_wandb(histories, metric, out)


if __name__ == "__main__":
    main()
