#!/usr/bin/env bash
# Pull all experiment results from the Modal volume, then plot everything.
set -e
SUNET_ID="lenardst"
ENV="cs336-lenardst"

echo "=== Pulling from Modal volume ==="
MODAL_ENVIRONMENT=$ENV modal volume get a5-grpo-$SUNET_ID . ./experiments/modal_results

echo "=== Plotting from W&B ==="
GROUPS=(
  "standard_on_policy"
  "lr_sweep"
  "prompt_ablation"
  "variants_on_policy"
  "off_policy"
  "loo_grpo"
)
for g in "${GROUPS[@]}"; do
  uv run python scripts/plot_grpo_results.py \
    --wandb-project cs336-a5-grpo \
    --run-group "$g" \
    --output-dir "experiments/writeup/plots/$g"
done

echo "=== Plots written to experiments/writeup/plots/ ==="
