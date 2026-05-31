"""Generate comparison bar charts from local final_metrics.json files.

Each experiment group gets one figure comparing variants (mean ± std across seeds).
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "experiments" / "modal_results"
OUT = REPO / "experiments" / "writeup" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# Also include the already-local standard_on_policy run
STANDARD_DIR = REPO / "experiments" / "grpo_standard_on_policy"


def load_seeds(variant_dir: Path) -> list[dict]:
    """Load all final_metrics.json files under variant_dir/seed*."""
    records = []
    for seed_dir in sorted(variant_dir.iterdir()):
        fm = seed_dir / "final_metrics.json"
        if fm.exists():
            records.append(json.loads(fm.read_text()))
    return records


def mean_std(records: list[dict], key: str) -> tuple[float, float]:
    vals = [r[key] for r in records if key in r]
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def bar_comparison(
    groups: dict[str, list[dict]],   # label → list of per-seed dicts
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    labels = list(groups.keys())
    means, stds = zip(*[mean_std(v, metric) for v in groups.values()])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
    bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Annotate bar tops
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Standard on-policy (4 seeds, already local)
# ──────────────────────────────────────────────────────────────────────────────
standard_records = load_seeds(STANDARD_DIR)
if standard_records:
    bar_comparison(
        {"standard\n(r1_zero, seq-norm)": standard_records},
        "val/reward", "Standard on-policy GRPO — final val reward", "val reward",
        OUT / "standard_on_policy_reward.png",
    )

# ──────────────────────────────────────────────────────────────────────────────
# 2. LR sweep
# ──────────────────────────────────────────────────────────────────────────────
lr_dir = RESULTS / "lr_sweep"
if lr_dir.exists():
    groups: dict[str, list[dict]] = {}
    for variant_dir in sorted(lr_dir.iterdir()):
        records = load_seeds(variant_dir)
        if records:
            groups[variant_dir.name.replace("lr_", "lr=")] = records
    if groups:
        bar_comparison(groups, "val/reward", "LR sweep — final val reward", "val reward",
                       OUT / "lr_sweep_reward.png")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Prompt ablation  (include standard r1_zero as reference)
# ──────────────────────────────────────────────────────────────────────────────
pa_dir = RESULTS / "prompt_ablation"
if pa_dir.exists():
    groups = {}
    if standard_records:
        groups["r1_zero\n(standard)"] = standard_records
    for variant_dir in sorted(pa_dir.iterdir()):
        records = load_seeds(variant_dir)
        if records:
            groups[variant_dir.name] = records
    if groups:
        bar_comparison(groups, "val/reward", "Prompt ablation — final val reward", "val reward",
                       OUT / "prompt_ablation_reward.png")

# ──────────────────────────────────────────────────────────────────────────────
# 4. On-policy variants
# ──────────────────────────────────────────────────────────────────────────────
var_dir = RESULTS / "variants_on_policy"
if var_dir.exists():
    VARIANT_ORDER = ["grpo_constant", "dr_grpo", "rft", "maxrl"]
    groups = {}
    for name in VARIANT_ORDER:
        d = var_dir / name
        if d.exists():
            records = load_seeds(d)
            if records:
                groups[name] = records
    if groups:
        bar_comparison(groups, "val/reward", "On-policy variants — final val reward", "val reward",
                       OUT / "variants_on_policy_reward.png")
        bar_comparison(groups, "val/mean_response_length",
                       "On-policy variants — response length", "mean tokens",
                       OUT / "variants_on_policy_length.png")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Off-policy variants
# ──────────────────────────────────────────────────────────────────────────────
op_dir = RESULTS / "off_policy"
if op_dir.exists():
    OP_ORDER = ["offpolicy_naive", "offpolicy_noclip", "offpolicy_clip", "offpolicy_gspo"]
    groups = {}
    for name in OP_ORDER:
        d = op_dir / name
        if d.exists():
            records = load_seeds(d)
            if records:
                groups[name] = records
    if groups:
        bar_comparison(groups, "val/reward", "Off-policy variants — final val reward", "val reward",
                       OUT / "off_policy_reward.png")

# ──────────────────────────────────────────────────────────────────────────────
# 6. LOO-GRPO vs Dr.GRPO
# ──────────────────────────────────────────────────────────────────────────────
loo_dir = RESULTS / "loo_grpo"
if loo_dir.exists():
    groups = {}
    for name in ["loo_grpo", "dr_grpo_baseline"]:
        d = loo_dir / name
        if d.exists():
            records = load_seeds(d)
            if records:
                groups[name] = records
    if groups:
        bar_comparison(groups, "val/reward", "LOO-GRPO vs Dr.GRPO — final val reward", "val reward",
                       OUT / "loo_grpo_reward.png")

print("All plots done.")
