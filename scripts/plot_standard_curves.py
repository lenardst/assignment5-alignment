"""Plot per-metric training curves (mean +/- std over seeds) for the standard
on-policy GRPO run, from the per-step metrics.jsonl written by scripts/grpo.py.

Usage:
    uv run python scripts/plot_standard_curves.py \
        --runs-dir experiments/grpo_standard_on_policy \
        --out-dir experiments/writeup/plots
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_runs(runs_dir: Path) -> list[list[dict]]:
    """Each run -> list of per-step record dicts."""
    runs = []
    for seed_dir in sorted(runs_dir.glob("seed*")):
        mp = seed_dir / "metrics.jsonl"
        if not mp.exists():
            continue
        records = [json.loads(line) for line in mp.read_text().splitlines() if line.strip()]
        if records:
            runs.append(records)
    return runs


def series(runs: list[list[dict]], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, mean, std) over seeds for `key`, using only steps where the
    key is present (val metrics are logged sparsely)."""
    by_step: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        for rec in run:
            if key in rec and rec[key] is not None:
                by_step[rec["step"]].append(float(rec[key]))
    steps = sorted(by_step)
    mean = np.array([np.mean(by_step[s]) for s in steps])
    std = np.array([np.std(by_step[s]) for s in steps])
    return np.array(steps), mean, std


def band(ax, runs, key, label, color, logy=False):
    s, m, sd = series(runs, key)
    if len(s) == 0:
        return False
    ax.plot(s, m, color=color, label=label, lw=2)
    ax.fill_between(s, m - sd, m + sd, color=color, alpha=0.2)
    if logy:
        ax.set_yscale("log")
    return True


def one_metric(runs, key, title, ylabel, out_path, logy=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    if band(ax, runs, key, key.split("/")[-1], "C0", logy=logy):
        ax.set_xlabel("rollout step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    plt.close(fig)


def two_metrics(runs, keys_labels, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ok = False
    for (key, label), color in zip(keys_labels, ["C0", "C1"]):
        ok = band(ax, runs, key, label, color) or ok
    if ok:
        ax.set_xlabel("rollout step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="experiments/grpo_standard_on_policy")
    ap.add_argument("--out-dir", default="experiments/writeup/plots")
    ap.add_argument("--prefix", default="standard")
    args = ap.parse_args()

    runs = load_runs(Path(args.runs_dir))
    if not runs:
        raise SystemExit(f"No metrics.jsonl found under {args.runs_dir}")
    print(f"Loaded {len(runs)} runs")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = args.prefix

    one_metric(runs, "train/loss", "Training loss", "loss", out / f"{p}_curve_loss.png")
    one_metric(runs, "train/grad_norm", "Gradient norm", "grad norm", out / f"{p}_curve_grad_norm.png", logy=True)
    one_metric(runs, "train/token_entropy", "Token entropy", "entropy (nats)", out / f"{p}_curve_entropy.png")
    two_metrics(runs, [("train/mean_reward", "total"), ("train/mean_format_reward", "format")],
                "Train reward", "reward", out / f"{p}_curve_train_reward.png")
    two_metrics(runs, [("val/reward", "total"), ("val/format_reward", "format")],
                "Validation reward", "reward", out / f"{p}_curve_val_reward.png")
    one_metric(runs, "val/mean_response_length", "Val mean response length", "tokens",
               out / f"{p}_curve_val_length.png")
    print("Done.")


if __name__ == "__main__":
    main()
