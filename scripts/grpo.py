"""GRPO training loop for OLMo-2-0425-1B on GSM8K.

Usage (on a 2-GPU machine, e.g. a Modal B200:2):
    uv run python scripts/grpo.py \
        --model allenai/OLMo-2-0425-1B \
        --prompt r1_zero \
        --output-dir experiments/grpo/seed0 \
        --seed 0

The script expects two GPUs:
  - GPU 0: HuggingFace policy model + optimizer
  - GPU 1: vLLM inference server

Suggested hypers (from the assignment handout):
    n_train_examples     = 6400
    n_val_examples       = 1024
    num_rollout_steps    = 200
    learning_rate        = 1e-5
    rollout_batch_size   = 256   (= 32 prompts × 8 responses)
    group_size           = 8
    gradient_accumulation_steps = 32
    sampling_temperature = 1.0
    sampling_max_tokens  = 512
    max_grad_norm        = 1.0
    optimizer            = AdamW(betas=(0.9, 0.95), weight_decay=0.0)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Callable

import torch

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.grpo import (
    get_response_log_probs,
    grpo_train_step,
    tokenize_prompt_and_output,
)
from cs336_alignment.vllm_utils import VLLMServer

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "cs336_alignment" / "prompts"

PROMPT_FILES = {
    "r1_zero": PROMPTS_DIR / "r1_zero.prompt",
    "question_only": PROMPTS_DIR / "question_only.prompt",
    "r1_zero_three_shot": PROMPTS_DIR / "r1_zero_three_shot_gsm8k.prompt",
}

REWARD_FNS: dict[str, Callable] = {
    "r1_zero": r1_zero_reward_fn,
    "question_only": question_only_reward_fn,
    "r1_zero_three_shot": r1_zero_reward_fn,
}

# Prompts that need the </answer> stop string
STOP_PROMPTS = {"r1_zero", "r1_zero_three_shot"}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_gsm8k(path: Path, n: int | None, seed: int) -> list[dict]:
    """Load GSM8K JSONL, extract ground-truth from the #### line."""
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gt = row["answer"].split("####")[-1].strip()
            rows.append({"question": row["question"], "ground_truth": gt})
    if n is not None:
        rng = random.Random(seed)
        rows = rng.sample(rows, min(n, len(rows)))
    return rows


def render_prompt(template: str, question: str) -> str:
    return template.replace("{question}", question)


# ---------------------------------------------------------------------------
# Eval helper
# ---------------------------------------------------------------------------

def evaluate(
    server: VLLMServer,
    examples: list[dict],
    template: str,
    reward_fn: Callable,
    use_stop: bool,
    sampling_params_base: dict,
    batch_size: int = 256,
) -> dict[str, float]:
    """Generate one response per example and compute val rewards."""
    prompts = [render_prompt(template, ex["question"]) for ex in examples]
    params = dict(sampling_params_base, n=1)
    if use_stop:
        params["stop"] = ["</answer>"]
        params["include_stop_str_in_output"] = True
    else:
        params["stop"] = None

    completions = server.generate_completions(
        prompts=prompts,
        sampling_params=params,
        batch_size=batch_size,
    )

    rewards, fmt_rewards, lengths = [], [], []
    for ex, comp in zip(examples, completions):
        info = reward_fn(comp.text, ex["ground_truth"])
        rewards.append(info["reward"])
        fmt_rewards.append(info.get("format_reward", 0.0))
        lengths.append(len(comp.token_ids))

    return {
        "val/reward": sum(rewards) / len(rewards),
        "val/format_reward": sum(fmt_rewards) / len(fmt_rewards),
        "val/mean_response_length": sum(lengths) / len(lengths),
        "val/n_examples": len(examples),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ------------------------------------------------------------------
    # W&B (optional)
    # ------------------------------------------------------------------
    use_wandb = bool(os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_DISABLED") is None) and args.wandb_project
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.run_name or f"{args.prompt}_seed{args.seed}",
                config=vars(args),
                reinit=True,
            )
        except Exception as e:
            print(f"W&B init failed ({e}), running without W&B.", flush=True)
            wandb_run = None

    def log(metrics: dict, step: int) -> None:
        print(f"[step {step}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)), flush=True)
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_data = load_gsm8k(Path(args.train_path), args.n_train_examples, args.seed)
    val_data   = load_gsm8k(Path(args.val_path),   args.n_val_examples,   args.seed)
    print(f"Train: {len(train_data)} examples, Val: {len(val_data)} examples", flush=True)

    template = PROMPTS_DIR / (PROMPT_FILES[args.prompt].name)
    template_str = PROMPT_FILES[args.prompt].read_text()
    reward_fn = REWARD_FNS[args.prompt]
    use_stop = args.prompt in STOP_PROMPTS

    # ------------------------------------------------------------------
    # Model + optimizer  (GPU 0)
    # ------------------------------------------------------------------
    policy_device = f"cuda:{args.gpu_policy}"
    print(f"Loading model {args.model} on {policy_device}...", flush=True)
    policy, tokenizer = get_model_and_tokenizer(args.model, policy_device)
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    # ------------------------------------------------------------------
    # vLLM server  (GPU 1)
    # ------------------------------------------------------------------
    print(f"Starting vLLM server on GPU {args.gpu_vllm}...", flush=True)
    server = VLLMServer(
        model_id=args.model,
        gpu=args.gpu_vllm,
        seed=args.seed,
    )
    server.start()

    # NCCL weight-sync group between GPU 0 and vLLM GPU 1
    server.init_weight_sync(policy_device)

    # Use seed only for val eval (reproducibility), NOT for rollout generation.
    # During training we need diverse rollouts per prompt; passing a fixed seed
    # with n=1 to repeated copies of the same prompt gives identical outputs.
    sampling_params_base = {
        "temperature": args.sampling_temperature,
        "max_tokens": args.sampling_max_tokens,
        "n": 1,
        "seed": args.seed,  # used for val eval
    }

    # ------------------------------------------------------------------
    # On-policy vs off-policy batching
    #   - train_batch_size == rollout_batch_size  -> one optimizer step per
    #     inference batch (on-policy).
    #   - train_batch_size <  rollout_batch_size  -> rollout/train optimizer
    #     steps per inference batch, with old_log_probs frozen at the sampling
    #     policy, so the policy drifts off-policy across the inner steps.
    # ------------------------------------------------------------------
    train_batch_size = args.train_batch_size or args.rollout_batch_size
    assert args.rollout_batch_size % train_batch_size == 0, (
        f"rollout_batch_size {args.rollout_batch_size} not divisible by "
        f"train_batch_size {train_batch_size}"
    )
    assert train_batch_size % args.group_size == 0, (
        f"train_batch_size {train_batch_size} must be a multiple of group_size "
        f"{args.group_size} so each optimizer step holds whole groups"
    )
    n_train_steps = args.rollout_batch_size // train_batch_size

    # ------------------------------------------------------------------
    # Normalization constant for constant-normalization variants.
    # Z is per optimizer step: train_batch_size * L (= rollout_batch_size * L
    # in the on-policy case, matching the previous behaviour).
    # ------------------------------------------------------------------
    n_prompts_per_batch = args.rollout_batch_size // args.group_size
    normalization_constant = train_batch_size * args.sampling_max_tokens

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_dir = out_dir / "rollouts"
    rollouts_dir.mkdir(exist_ok=True)
    # Per-step metric history for offline plotting (mean/variance over seeds).
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    rng = random.Random(args.seed)
    global_step = 0

    for rollout_step in range(args.num_rollout_steps):
        t0 = time.monotonic()

        # 1. Sync policy weights into vLLM
        server.sync_policy_weights(policy)

        # 2. Sample a batch of questions
        batch_questions = rng.choices(train_data, k=n_prompts_per_batch)
        unique_prompts = [
            render_prompt(template_str, ex["question"]) for ex in batch_questions
        ]
        unique_ground_truths = [ex["ground_truth"] for ex in batch_questions]

        # 3. Generate G diverse rollouts per prompt using n=group_size.
        # Do NOT use a fixed seed here: passing seed+n=1 with repeated prompts
        # would produce identical outputs per group, collapsing all advantages to 0.
        rollout_params = {
            "temperature": args.sampling_temperature,
            "max_tokens": args.sampling_max_tokens,
            "n": args.group_size,
            # Vary seed each step so repeated calls to the same prompt produce
            # different outputs.  Fixed seed would collapse all group samples.
            "seed": rollout_step,
        }
        if use_stop:
            rollout_params["stop"] = ["</answer>"]
            rollout_params["include_stop_str_in_output"] = True

        # generate_completions returns B*G completions sorted by index:
        # completions[i*G + j] is the j-th rollout for prompt i.
        completions = server.generate_completions(
            prompts=unique_prompts,
            sampling_params=rollout_params,
        )
        rollout_responses = [c.text for c in completions]

        # Build the B*G repeated lists grpo_train_step expects
        repeated_prompts = [
            unique_prompts[i]
            for i in range(n_prompts_per_batch)
            for _ in range(args.group_size)
        ]
        repeated_ground_truths = [
            unique_ground_truths[i]
            for i in range(n_prompts_per_batch)
            for _ in range(args.group_size)
        ]

        # 4a. Freeze old log-probs at the sampling policy, per train minibatch.
        # These are computed BEFORE any optimizer step so that across the inner
        # off-policy steps the ratio pi_theta / pi_theta_old drifts away from 1.
        # We tokenize each minibatch exactly as grpo_train_step will, so the
        # padded shapes line up when we slice.
        need_old = args.importance_reweighting_method != "none"
        old_log_probs_per_step: list[torch.Tensor | None] = [None] * n_train_steps
        if need_old:
            policy.eval()
            with torch.no_grad():
                for k in range(n_train_steps):
                    s, e = k * train_batch_size, (k + 1) * train_batch_size
                    old_batch = tokenize_prompt_and_output(
                        prompt_strs=repeated_prompts[s:e],
                        output_strs=rollout_responses[s:e],
                        tokenizer=tokenizer,
                    )
                    old_out = get_response_log_probs(
                        model=policy,
                        input_ids=old_batch["input_ids"].to(policy_device),
                        labels=old_batch["labels"].to(policy_device),
                    )
                    old_log_probs_per_step[k] = old_out["log_probs"]
            policy.train()

        # 4b. Inner training loop: n_train_steps optimizer steps per inference
        # batch (== 1 on-policy, == rollout/train off-policy).
        cliprange = args.cliprange if args.importance_reweighting_method in ("grpo", "gspo") else None
        loss_sum = 0.0
        metric_sums: dict[str, float] = {}
        for k in range(n_train_steps):
            s, e = k * train_batch_size, (k + 1) * train_batch_size
            loss, metadata = grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                reward_fn=reward_fn,
                repeated_prompts=repeated_prompts[s:e],
                rollout_responses=rollout_responses[s:e],
                repeated_ground_truths=repeated_ground_truths[s:e],
                group_size=args.group_size,
                baseline=args.baseline,
                advantage_eps=args.advantage_eps,
                advantage_normalizer=args.advantage_normalizer,
                importance_reweighting_method=args.importance_reweighting_method,
                old_log_probs=old_log_probs_per_step[k],
                cliprange=cliprange,
                loss_normalization=args.loss_normalization,
                normalization_constant=normalization_constant if args.loss_normalization == "constant" else None,
            )
            loss_sum += loss.item()
            for key in ("grad_norm", "mean_reward", "mean_format_reward", "std_reward", "token_entropy"):
                metric_sums[key] = metric_sums.get(key, 0.0) + float(metadata.get(key, 0.0))
            if "clip_fraction" in metadata:
                cf = metadata["clip_fraction"]
                metric_sums["clip_fraction"] = metric_sums.get("clip_fraction", 0.0) + float(
                    cf.item() if hasattr(cf, "item") else cf
                )

        dt = time.monotonic() - t0
        step_metrics = {
            "train/loss": loss_sum / n_train_steps,
            "train/grad_norm": metric_sums.get("grad_norm", 0.0) / n_train_steps,
            "train/token_entropy": metric_sums.get("token_entropy", 0.0) / n_train_steps,
            "train/mean_reward": metric_sums.get("mean_reward", 0.0) / n_train_steps,
            "train/mean_format_reward": metric_sums.get("mean_format_reward", 0.0) / n_train_steps,
            "train/std_reward": metric_sums.get("std_reward", 0.0) / n_train_steps,
            "train/step_time_s": dt,
        }
        if "clip_fraction" in metric_sums:
            step_metrics["train/clip_fraction"] = metric_sums["clip_fraction"] / n_train_steps
        log(step_metrics, global_step)

        # Per-step record for offline plotting (train metrics always; val merged
        # in when an eval runs this step).
        record = {"step": global_step, **step_metrics}

        # 5. Periodic validation
        if rollout_step % args.eval_interval == 0:
            policy.eval()
            with torch.no_grad():
                val_metrics = evaluate(
                    server=server,
                    examples=val_data,
                    template=template_str,
                    reward_fn=reward_fn,
                    use_stop=use_stop,
                    sampling_params_base=sampling_params_base,
                )
            policy.train()
            log(val_metrics, global_step)
            record.update(val_metrics)

        with metrics_path.open("a") as _mfh:
            _mfh.write(json.dumps(record) + "\n")

        # 6. Periodic rollout logging
        if rollout_step % args.log_rollouts_interval == 0:
            sample_idx = rng.randrange(n_prompts_per_batch)
            base = sample_idx * args.group_size
            sample_rollouts = []
            for g in range(args.group_size):
                sample_rollouts.append({
                    "prompt": unique_prompts[sample_idx],
                    "response": rollout_responses[base + g],
                    "ground_truth": unique_ground_truths[sample_idx],
                })
            rollout_path = rollouts_dir / f"step_{global_step:05d}.json"
            rollout_path.write_text(json.dumps(sample_rollouts, indent=2))
            if wandb_run is not None:
                table_rows = [[r["ground_truth"], r["response"][:300]] for r in sample_rollouts[:4]]
                import wandb as _wandb
                wandb_run.log({"rollout_table": _wandb.Table(columns=["gt", "response"], data=table_rows)}, step=global_step)

        global_step += 1

    # ------------------------------------------------------------------
    # Final eval + checkpoint
    # ------------------------------------------------------------------
    policy.eval()
    with torch.no_grad():
        final_val = evaluate(
            server=server,
            examples=val_data,
            template=template_str,
            reward_fn=reward_fn,
            use_stop=use_stop,
            sampling_params_base=sampling_params_base,
        )
    print("Final val metrics:", final_val, flush=True)
    log({f"final/{k.split('/')[-1]}": v for k, v in final_val.items()}, global_step)

    ckpt_dir = out_dir / "checkpoint"
    policy.save_pretrained(save_directory=str(ckpt_dir))
    tokenizer.save_pretrained(save_directory=str(ckpt_dir))
    print(f"Checkpoint saved to {ckpt_dir}", flush=True)

    # Save final metrics JSON
    (out_dir / "final_metrics.json").write_text(json.dumps({**final_val, "seed": args.seed}, indent=2))

    server.stop()
    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO training on GSM8K")
    # Model / data
    p.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    p.add_argument("--prompt", default="r1_zero", choices=list(PROMPT_FILES.keys()))
    p.add_argument("--train-path", default=str(REPO_ROOT / "data" / "gsm8k" / "train.jsonl"))
    p.add_argument("--val-path",   default=str(REPO_ROOT / "data" / "gsm8k" / "test.jsonl"))
    p.add_argument("--output-dir", default=str(REPO_ROOT / "experiments" / "grpo" / "run"))
    # Scale
    p.add_argument("--n-train-examples", type=int, default=6400)
    p.add_argument("--n-val-examples",   type=int, default=1024)
    p.add_argument("--num-rollout-steps",type=int, default=200)
    # Optimisation
    p.add_argument("--learning-rate",    type=float, default=1e-5)
    p.add_argument("--rollout-batch-size",type=int,  default=256)
    # Number of responses per optimizer step. When < rollout-batch-size we take
    # rollout-batch-size / train-batch-size optimizer steps per inference batch,
    # i.e. off-policy RL. Defaults to rollout-batch-size (fully on-policy: one
    # optimizer step per inference batch).
    p.add_argument("--train-batch-size", type=int, default=None)
    p.add_argument("--group-size",       type=int,   default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=32)
    p.add_argument("--max-grad-norm",    type=float, default=1.0)
    # GRPO variant knobs
    p.add_argument("--baseline",         default="mean", choices=["mean", "none", "loo"])
    p.add_argument("--advantage-normalizer", default="std", choices=["std", "none", "mean"])
    p.add_argument("--advantage-eps",    type=float, default=1e-6)
    p.add_argument("--loss-normalization", default="sequence", choices=["sequence", "constant"])
    # Off-policy importance reweighting
    p.add_argument("--importance-reweighting-method", default="none",
                   choices=["none", "noclip", "grpo", "gspo"])
    p.add_argument("--cliprange",        type=float, default=0.2)
    # Sampling
    p.add_argument("--sampling-temperature", type=float, default=1.0)
    p.add_argument("--sampling-max-tokens",  type=int,   default=512)
    # Hardware
    p.add_argument("--gpu-policy", type=int, default=0)
    p.add_argument("--gpu-vllm",   type=int, default=1)
    # Logging
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--wandb-project", default="cs336-a5-grpo")
    p.add_argument("--run-name",      default=None)
    p.add_argument("--eval-interval",          type=int, default=10)
    p.add_argument("--log-rollouts-interval",  type=int, default=40)
    return p


if __name__ == "__main__":
    args = make_parser().parse_args()
    train(args)
