"""GRPO component implementations.

All functions here are importable both locally and on Modal (since
cs336_alignment/ is mounted as a Python package in the container image).

tests/adapters.py delegates to these implementations so the test suite
can validate them.
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    """Tokenize prompts + outputs without special tokens; build response_mask."""
    assert len(prompt_strs) == len(output_strs)
    batch_size = len(prompt_strs)

    prompt_ids_list = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_ids_list = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]
    full_ids_list = [p + o for p, o in zip(prompt_ids_list, output_ids_list)]
    max_len = max(len(x) for x in full_ids_list)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    is_response = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, (p_ids, o_ids) in enumerate(zip(prompt_ids_list, output_ids_list)):
        seq = p_ids + o_ids
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        is_response[i, len(p_ids) : len(p_ids) + len(o_ids)] = 1

    input_ids = padded[:, :-1].contiguous()
    labels = padded[:, 1:].contiguous()
    response_mask = is_response[:, 1:].contiguous()

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


# ---------------------------------------------------------------------------
# Log-probabilities
# ---------------------------------------------------------------------------

def _compute_entropy(logits: Tensor) -> Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Per-token conditional log-probs log π_θ(y_t | x, y_{<t})."""
    logits = model(input_ids).logits           # (B, T, V)
    log_probs_all = F.log_softmax(logits, dim=-1)
    log_probs = log_probs_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    out: dict[str, Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        out["token_entropy"] = _compute_entropy(logits)
    return out


# ---------------------------------------------------------------------------
# Rollout rewards
# ---------------------------------------------------------------------------

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[Tensor, dict[str, float]]:
    assert len(rollout_responses) == len(repeated_ground_truths)
    rewards, fmt_rewards, ans_rewards = [], [], []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        info = reward_fn(response, gt)
        rewards.append(float(info["reward"]))
        fmt_rewards.append(float(info.get("format_reward", 0.0)))
        ans_rewards.append(float(info.get("answer_reward", info["reward"])))

    raw = torch.tensor(rewards, dtype=torch.float32)
    metadata = {
        "mean_reward": float(sum(rewards) / max(len(rewards), 1)),
        "mean_format_reward": float(sum(fmt_rewards) / max(len(fmt_rewards), 1)),
        "mean_answer_reward": float(sum(ans_rewards) / max(len(ans_rewards), 1)),
    }
    return raw, metadata


# ---------------------------------------------------------------------------
# Advantage normalisation
# ---------------------------------------------------------------------------

def compute_group_normalized_rewards(
    raw_rewards: Tensor,
    group_size: int,
    baseline: Literal["mean", "none", "loo"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[Tensor, dict[str, float]]:
    rewards = raw_rewards.float().view(-1, group_size)
    group_mean = rewards.mean(dim=-1, keepdim=True)
    group_std = rewards.std(dim=-1, keepdim=True)

    if baseline == "mean":
        centered = rewards - group_mean
    elif baseline == "none":
        centered = rewards.clone()
    elif baseline == "loo":
        G = group_size
        # leave-one-out mean for sample j: (G*mean - r_j) / (G-1)
        loo_baseline = (G * group_mean - rewards) / max(G - 1, 1)
        centered = rewards - loo_baseline
    else:
        raise NotImplementedError(f"baseline={baseline!r}")

    if advantage_normalizer == "std":
        advantages = centered / (group_std + advantage_eps)
    elif advantage_normalizer == "mean":
        advantages = centered / (group_mean.abs() + advantage_eps)
    elif advantage_normalizer == "none":
        advantages = centered
    else:
        raise NotImplementedError(f"advantage_normalizer={advantage_normalizer!r}")

    advantages = advantages.reshape(-1)
    metadata = {
        "mean_reward": float(rewards.mean().item()),
        "std_reward": float(rewards.std().item()),
        "max_reward": float(rewards.max().item()),
        "min_reward": float(rewards.min().item()),
    }
    return advantages, metadata


# ---------------------------------------------------------------------------
# Policy-gradient loss
# ---------------------------------------------------------------------------

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    response_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    adv = raw_rewards_or_advantages
    if adv.dim() == 1:
        adv = adv.unsqueeze(-1)
    adv = adv.to(policy_log_probs.dtype)

    metadata: dict[str, Tensor] = {}

    if importance_reweighting_method == "none":
        per_token = -adv * policy_log_probs
    elif importance_reweighting_method == "noclip":
        assert old_log_probs is not None
        ratio = (policy_log_probs - old_log_probs).exp()
        per_token = -adv * ratio
    elif importance_reweighting_method == "grpo":
        assert old_log_probs is not None and cliprange is not None
        ratio = (policy_log_probs - old_log_probs).exp()
        clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
        unclipped = adv * ratio
        clipped = adv * clipped_ratio
        per_token = -torch.minimum(unclipped, clipped)
        metadata["clip_fraction"] = (unclipped != clipped).float().mean().detach()
    elif importance_reweighting_method == "gspo":
        assert old_log_probs is not None and cliprange is not None
        log_ratio = policy_log_probs - old_log_probs
        if response_mask is not None:
            mask = response_mask.to(log_ratio.dtype)
            denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            seq_log_ratio = (log_ratio * mask).sum(dim=-1, keepdim=True) / denom
        else:
            seq_log_ratio = log_ratio.mean(dim=-1, keepdim=True)
        seq_ratio = seq_log_ratio.exp()
        clipped_seq_ratio = torch.clamp(seq_ratio, 1.0 - cliprange, 1.0 + cliprange)
        unclipped = adv * seq_ratio
        clipped = adv * clipped_seq_ratio
        seq_loss = -torch.minimum(unclipped, clipped)
        per_token = seq_loss.expand_as(policy_log_probs)
        metadata["clip_fraction"] = (unclipped != clipped).float().mean().detach()
    else:
        raise NotImplementedError(f"importance_reweighting_method={importance_reweighting_method!r}")

    return per_token, metadata


# ---------------------------------------------------------------------------
# Loss aggregation
# ---------------------------------------------------------------------------

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: Tensor,
    mask: Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> Tensor:
    mask_f = mask.to(per_token_policy_gradient_loss.dtype)
    masked = per_token_policy_gradient_loss * mask_f

    if loss_normalization == "sequence":
        per_seq_sum = masked.sum(dim=-1)
        per_seq_count = mask_f.sum(dim=-1).clamp(min=1.0)
        return (per_seq_sum / per_seq_count).mean()
    elif loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError("normalization_constant is required for 'constant' mode")
        return masked.sum() / float(normalization_constant)
    else:
        raise NotImplementedError(f"loss_normalization={loss_normalization!r}")


# ---------------------------------------------------------------------------
# Full GRPO train step
# ---------------------------------------------------------------------------

def grpo_train_step(
    model: torch.nn.Module,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none", "loo"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[Tensor, dict]:
    # Tokenize
    batch = tokenize_prompt_and_output(
        prompt_strs=repeated_prompts,
        output_strs=rollout_responses,
        tokenizer=tokenizer,
    )
    device = next(model.parameters()).device
    input_ids     = batch["input_ids"].to(device)
    labels        = batch["labels"].to(device)
    response_mask = batch["response_mask"].to(device)

    # Rewards → advantages
    raw_rewards, reward_meta = compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
    )
    raw_rewards = raw_rewards.to(device)

    old_log_probs_filtered = old_log_probs.to(device) if old_log_probs is not None else None

    advantages, adv_meta = compute_group_normalized_rewards(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )

    batch_size = input_ids.shape[0]
    assert batch_size % gradient_accumulation_steps == 0, (
        f"batch_size {batch_size} not divisible by gradient_accumulation_steps "
        f"{gradient_accumulation_steps}"
    )
    micro_size = batch_size // gradient_accumulation_steps

    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    aggregated_metadata: dict = {}

    for mb_idx in range(gradient_accumulation_steps):
        s, e = mb_idx * micro_size, (mb_idx + 1) * micro_size
        mb_input_ids = input_ids[s:e]
        mb_labels    = labels[s:e]
        mb_mask      = response_mask[s:e]
        mb_adv       = advantages[s:e]

        out = get_response_log_probs(model=model, input_ids=mb_input_ids, labels=mb_labels)
        mb_log_probs = out["log_probs"]

        mb_old = old_log_probs_filtered[s:e] if old_log_probs_filtered is not None else None

        per_token_loss, loss_meta = compute_policy_gradient_loss(
            raw_rewards_or_advantages=mb_adv,
            policy_log_probs=mb_log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=mb_old,
            cliprange=cliprange,
            response_mask=mb_mask,
        )

        mb_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=mb_mask,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )

        scaled = mb_loss if loss_normalization == "constant" else mb_loss / gradient_accumulation_steps
        scaled.backward()
        total_loss = total_loss + scaled.detach()

        for k, v in loss_meta.items():
            aggregated_metadata[k] = aggregated_metadata.get(k, 0) + v

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm if max_grad_norm is not None else float("inf"),
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metadata: dict = {
        "grad_norm": grad_norm.detach() if isinstance(grad_norm, Tensor) else float(grad_norm),
        **reward_meta,
        **adv_meta,
    }
    for k, v in aggregated_metadata.items():
        metadata[k] = v / gradient_accumulation_steps if isinstance(v, Tensor) else v

    return total_loss, metadata
