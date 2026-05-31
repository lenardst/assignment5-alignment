"""Modal launcher for the GRPO training script.

Entrypoints:
    standard_on_policy   -- 4 seeds, standard GRPO (r1_zero prompt, sequence norm)
    learning_rate_sweep  -- sweep over 3 LRs, 2 seeds each
    prompt_ablation      -- question_only + r1_zero_three_shot, 2 seeds each
    variants_on_policy   -- GRPO_constant / Dr_GRPO / RFT / MaxRL, 4 seeds each

Usage:
    # Standard run (4 seeds, r1_zero, sequence norm)
    modal run scripts/modal_grpo.py::standard_on_policy

    # LR sweep
    modal run scripts/modal_grpo.py::learning_rate_sweep

    # Prompt ablation
    modal run scripts/modal_grpo.py::prompt_ablation

    # All on-policy variants
    modal run scripts/modal_grpo.py::variants_on_policy

Pull results afterwards:
    MODAL_ENVIRONMENT=cs336-lenardst modal volume get a5-grpo-lenardst experiments ./experiments
"""

from __future__ import annotations

import sys
from itertools import product

import modal

from cs336_alignment.modal_utils import (
    GPU,
    RUN_TIMEOUT_SECONDS,
    SUNET_ID,
    app,
    image,
    quote_command,
)

RESULTS_MOUNT = "/root/grpo_results"
GRPO_VOLUME = modal.Volume.from_name(f"a5-grpo-{SUNET_ID}", create_if_missing=True)

WANDB_SECRET_NAME = "my-wandb-secret"  # actual name in cs336-lenardst env


def _grpo_function(**extra_kwargs):
    """Decorator factory: creates a Modal function with the grpo volume mounted."""
    return app.function(
        image=image,
        gpu=GPU,
        timeout=RUN_TIMEOUT_SECONDS,
        volumes={RESULTS_MOUNT: GRPO_VOLUME},
        secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
        **extra_kwargs,
    )


def _make_args(
    *,
    seed: int,
    prompt: str = "r1_zero",
    output_subdir: str,
    learning_rate: float = 1e-5,
    baseline: str = "mean",
    advantage_normalizer: str = "std",
    loss_normalization: str = "sequence",
    num_rollout_steps: int = 200,
    importance_reweighting_method: str = "none",
    cliprange: float | None = None,
    run_name: str | None = None,
) -> list[str]:
    """Build CLI arguments (no interpreter prefix; run_grpo prepends it)."""
    out_dir = f"{RESULTS_MOUNT}/{output_subdir}/seed{seed}"
    args = [
        "scripts/grpo.py",
        "--model",     "allenai/OLMo-2-0425-1B",
        "--prompt",    prompt,
        "--output-dir", out_dir,
        "--seed",      str(seed),
        "--learning-rate", str(learning_rate),
        "--baseline",  baseline,
        "--advantage-normalizer", advantage_normalizer,
        "--loss-normalization",   loss_normalization,
        "--num-rollout-steps",    str(num_rollout_steps),
        "--importance-reweighting-method", importance_reweighting_method,
        "--wandb-project", "cs336-a5-grpo",
    ]
    if cliprange is not None:
        args += ["--cliprange", str(cliprange)]
    if run_name:
        args += ["--run-name", run_name]
    return args


# ---------------------------------------------------------------------------
# Run function (one per job)
# ---------------------------------------------------------------------------

import subprocess


@app.function(
    image=image,
    gpu=GPU,
    timeout=RUN_TIMEOUT_SECONDS,
    volumes={RESULTS_MOUNT: GRPO_VOLUME},
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
)
def run_grpo(script_args: list[str]) -> str:
    """Runs inside the container: prepend the container's Python interpreter."""
    import sys as _sys
    command = [_sys.executable, "-u"] + script_args
    cmd_str = quote_command(command)
    print(cmd_str, flush=True)
    subprocess.run(command, check=True)
    GRPO_VOLUME.commit()
    return cmd_str


# ---------------------------------------------------------------------------
# 1. Standard on-policy GRPO: 4 seeds, r1_zero, sequence norm
# ---------------------------------------------------------------------------

@app.local_entrypoint(name="standard_on_policy")
def standard_on_policy(
    seeds: str = "0,1,2,3",
    learning_rate: float = 1e-5,
    num_rollout_steps: int = 200,
) -> None:
    commands = [
        _make_args(
            seed=int(s),
            prompt="r1_zero",
            output_subdir="standard_on_policy",
            learning_rate=learning_rate,
            num_rollout_steps=num_rollout_steps,
            run_name=f"standard_seed{s}",
        )
        for s in seeds.split(",")
    ]
    print(f"Submitting {len(commands)} jobs: standard_on_policy", flush=True)
    failures = []
    for idx, result in enumerate(run_grpo.map(commands, return_exceptions=True)):
        if isinstance(result, BaseException):
            print(f"FAILED [{idx}]: {result!r}", flush=True)
            failures.append(idx)
        else:
            print(f"Done [{idx}]: {result}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed")
    print(f"All done. Pull with:\n  MODAL_ENVIRONMENT=cs336-lenardst modal volume get a5-grpo-{SUNET_ID} standard_on_policy ./experiments/grpo/standard_on_policy", flush=True)


# ---------------------------------------------------------------------------
# 2. Learning-rate sweep
# ---------------------------------------------------------------------------

@app.local_entrypoint(name="learning_rate_sweep")
def learning_rate_sweep(
    lrs: str = "3e-6,1e-5,3e-5",
    seeds: str = "0,1",
    num_rollout_steps: int = 200,
) -> None:
    commands = [
        _make_args(
            seed=int(s),
            prompt="r1_zero",
            output_subdir=f"lr_sweep/lr_{lr}",
            learning_rate=float(lr),
            num_rollout_steps=num_rollout_steps,
            run_name=f"lr{lr}_seed{s}",
        )
        for lr, s in product(lrs.split(","), seeds.split(","))
    ]
    print(f"Submitting {len(commands)} jobs: lr_sweep", flush=True)
    failures = []
    for idx, result in enumerate(run_grpo.map(commands, return_exceptions=True)):
        if isinstance(result, BaseException):
            print(f"FAILED [{idx}]: {result!r}", flush=True)
            failures.append(idx)
        else:
            print(f"Done [{idx}]: {result}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed")


# ---------------------------------------------------------------------------
# 3. Prompt ablation
# ---------------------------------------------------------------------------

@app.local_entrypoint(name="prompt_ablation")
def prompt_ablation(
    seeds: str = "0,1",
    learning_rate: float = 1e-5,
    num_rollout_steps: int = 200,
) -> None:
    commands = [
        _make_args(
            seed=int(s),
            prompt=prompt,
            output_subdir=f"prompt_ablation/{prompt}",
            learning_rate=learning_rate,
            num_rollout_steps=num_rollout_steps,
            run_name=f"{prompt}_seed{s}",
        )
        for prompt, s in product(
            ["question_only", "r1_zero_three_shot"],
            seeds.split(","),
        )
    ]
    print(f"Submitting {len(commands)} jobs: prompt_ablation", flush=True)
    failures = []
    for idx, result in enumerate(run_grpo.map(commands, return_exceptions=True)):
        if isinstance(result, BaseException):
            print(f"FAILED [{idx}]: {result!r}", flush=True)
            failures.append(idx)
        else:
            print(f"Done [{idx}]: {result}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed")


# ---------------------------------------------------------------------------
# 4. On-policy RL variants  (Section 5.4)
# ---------------------------------------------------------------------------

# Variant configs matching the assignment spec
VARIANT_CONFIGS = {
    "grpo_constant": dict(baseline="mean", advantage_normalizer="std",  loss_normalization="constant"),
    "dr_grpo":       dict(baseline="mean", advantage_normalizer="none", loss_normalization="constant"),
    "rft":           dict(baseline="none", advantage_normalizer="none", loss_normalization="constant"),
    "maxrl":         dict(baseline="mean", advantage_normalizer="mean", loss_normalization="constant"),
}


@app.local_entrypoint(name="variants_on_policy")
def variants_on_policy(
    seeds: str = "0,1,2,3",
    learning_rate: float = 1e-5,
    num_rollout_steps: int = 200,
    variants: str = "grpo_constant,dr_grpo,rft,maxrl",
) -> None:
    commands = []
    for variant, s in product(variants.split(","), seeds.split(",")):
        cfg = VARIANT_CONFIGS[variant]
        commands.append(
            _make_args(
                seed=int(s),
                prompt="r1_zero",
                output_subdir=f"variants_on_policy/{variant}",
                learning_rate=learning_rate,
                num_rollout_steps=num_rollout_steps,
                run_name=f"{variant}_seed{s}",
                **cfg,
            )
        )
    print(f"Submitting {len(commands)} jobs: variants_on_policy", flush=True)
    failures = []
    for idx, result in enumerate(run_grpo.map(commands, return_exceptions=True)):
        if isinstance(result, BaseException):
            print(f"FAILED [{idx}]: {result!r}", flush=True)
            failures.append(idx)
        else:
            print(f"Done [{idx}]: {result}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed")


# ---------------------------------------------------------------------------
# 5. Off-policy variants
# ---------------------------------------------------------------------------

OFF_POLICY_CONFIGS = {
    "offpolicy_naive":  dict(importance_reweighting_method="none",  cliprange=None),
    "offpolicy_noclip": dict(importance_reweighting_method="noclip", cliprange=None),
    "offpolicy_clip":   dict(importance_reweighting_method="grpo",   cliprange=0.2),
    "offpolicy_gspo":   dict(importance_reweighting_method="gspo",   cliprange=3e-4),
}


@app.local_entrypoint(name="off_policy")
def off_policy(
    seeds: str = "0,1,2,3",
    learning_rate: float = 1e-5,
    num_rollout_steps: int = 200,
    variants: str = "offpolicy_naive,offpolicy_noclip,offpolicy_clip,offpolicy_gspo",
) -> None:
    commands = []
    for variant, s in product(variants.split(","), seeds.split(",")):
        cfg = OFF_POLICY_CONFIGS[variant]
        commands.append(
            _make_args(
                seed=int(s),
                prompt="r1_zero",
                output_subdir=f"off_policy/{variant}",
                learning_rate=learning_rate,
                baseline="mean",
                advantage_normalizer="std",
                loss_normalization="sequence",
                num_rollout_steps=num_rollout_steps,
                run_name=f"{variant}_seed{s}",
                **cfg,
            )
        )
    print(f"Submitting {len(commands)} jobs: off_policy", flush=True)
    failures = []
    for idx, result in enumerate(run_grpo.map(commands, return_exceptions=True)):
        if isinstance(result, BaseException):
            print(f"FAILED [{idx}]: {result!r}", flush=True)
            failures.append(idx)
        else:
            print(f"Done [{idx}]: {result}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed")


# ---------------------------------------------------------------------------
# 6. LOO-GRPO vs Dr.GRPO (custom estimator)
# ---------------------------------------------------------------------------

@app.local_entrypoint(name="loo_grpo")
def loo_grpo(
    seeds: str = "0,1,2,3",
    learning_rate: float = 1e-5,
    num_rollout_steps: int = 200,
) -> None:
    commands = []
    for variant, s in product(["loo_grpo", "dr_grpo_baseline"], seeds.split(",")):
        if variant == "loo_grpo":
            cfg = dict(baseline="loo", advantage_normalizer="none", loss_normalization="constant")
        else:
            cfg = dict(baseline="mean", advantage_normalizer="none", loss_normalization="constant")
        commands.append(
            _make_args(
                seed=int(s),
                prompt="r1_zero",
                output_subdir=f"loo_grpo/{variant}",
                learning_rate=learning_rate,
                num_rollout_steps=num_rollout_steps,
                run_name=f"{variant}_seed{s}",
                **cfg,
            )
        )
    print(f"Submitting {len(commands)} jobs: loo_grpo", flush=True)
    failures = []
    for idx, result in enumerate(run_grpo.map(commands, return_exceptions=True)):
        if isinstance(result, BaseException):
            print(f"FAILED [{idx}]: {result!r}", flush=True)
            failures.append(idx)
        else:
            print(f"Done [{idx}]: {result}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed")
