from __future__ import annotations

import math
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer) -> dict[str, Tensor]:
    assert len(prompt_strs) == len(output_strs)
    batch_size = len(prompt_strs)

    prompt_ids_list = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_ids_list = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]
    full_ids_list = [p + o for p, o in zip(prompt_ids_list, output_ids_list)]
    max_len = max(len(x) for x in full_ids_list)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    is_response = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (prompt_ids, output_ids) in enumerate(zip(prompt_ids_list, output_ids_list)):
        seq = prompt_ids + output_ids
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        is_response[i, len(prompt_ids) : len(prompt_ids) + len(output_ids)] = 1

    input_ids = padded[:, :-1].contiguous()
    labels = padded[:, 1:].contiguous()
    response_mask = is_response[:, 1:].contiguous()
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def get_response_log_probs(model, input_ids, labels, return_token_entropy=False) -> dict[str, Tensor]:
    logits = model(input_ids).logits
    log_probs_all = F.log_softmax(logits, dim=-1)
    log_probs = log_probs_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    out = {"log_probs": log_probs}
    if return_token_entropy:
        out["token_entropy"] = -(log_probs_all.exp() * log_probs_all).sum(dim=-1)
    return out


def compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths):
    assert len(rollout_responses) == len(repeated_ground_truths)
    rewards, fmt_rewards, ans_rewards = [], [], []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        info = reward_fn(response, ground_truth)
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


def compute_group_normalized_rewards(
    raw_rewards,
    group_size,
    baseline: Literal["mean", "none", "loo"] = "mean",
    advantage_eps=1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
):
    rewards = raw_rewards.float().view(-1, group_size)
    group_mean = rewards.mean(dim=-1, keepdim=True)
    group_std = rewards.std(dim=-1, keepdim=True)

    if baseline == "mean":
        centered = rewards - group_mean
    elif baseline == "none":
        centered = rewards.clone()
    elif baseline == "loo":
        loo_baseline = (group_size * group_mean - rewards) / max(group_size - 1, 1)
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


def compute_policy_gradient_loss(
    raw_rewards_or_advantages,
    policy_log_probs,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs=None,
    cliprange=None,
    response_mask=None,
):
    advantages = raw_rewards_or_advantages
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)
    advantages = advantages.to(policy_log_probs.dtype)

    # Clamp log-ratio to avoid exp() overflow on extreme off-policy samples.
    LOG_RATIO_CAP = 20.0
    metadata: dict[str, Tensor] = {}

    if importance_reweighting_method == "none":
        per_token = -advantages * policy_log_probs

    elif importance_reweighting_method == "noclip":
        ratio = (policy_log_probs - old_log_probs).clamp(-LOG_RATIO_CAP, LOG_RATIO_CAP).exp()
        per_token = -advantages * ratio

    elif importance_reweighting_method == "grpo":
        ratio = (policy_log_probs - old_log_probs).clamp(-LOG_RATIO_CAP, LOG_RATIO_CAP).exp()
        clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
        unclipped = advantages * ratio
        clipped = advantages * clipped_ratio
        per_token = -torch.minimum(unclipped, clipped)
        metadata["clip_fraction"] = (unclipped != clipped).float().mean().detach()

    elif importance_reweighting_method == "gspo":
        log_ratio = policy_log_probs - old_log_probs
        if response_mask is not None:
            mask_f = response_mask.to(log_ratio.dtype)
            token_count = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            seq_log_ratio = (log_ratio * mask_f).sum(dim=-1, keepdim=True) / token_count
        else:
            seq_log_ratio = log_ratio.mean(dim=-1, keepdim=True)
        seq_ratio = seq_log_ratio.clamp(-LOG_RATIO_CAP, LOG_RATIO_CAP).exp()
        clipped_seq_ratio = torch.clamp(seq_ratio, 1.0 - cliprange, 1.0 + cliprange)
        unclipped = advantages * seq_ratio
        clipped = advantages * clipped_seq_ratio
        per_token = -torch.minimum(unclipped, clipped).expand_as(policy_log_probs)
        metadata["clip_fraction"] = (unclipped != clipped).float().mean().detach()

    else:
        raise NotImplementedError(f"importance_reweighting_method={importance_reweighting_method!r}")

    return per_token, metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss,
    mask,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant=None,
):
    mask_f = mask.to(per_token_policy_gradient_loss.dtype)
    masked = per_token_policy_gradient_loss * mask_f

    if loss_normalization == "sequence":
        per_seq_sum = masked.sum(dim=-1)
        per_seq_count = mask_f.sum(dim=-1).clamp(min=1.0)
        return (per_seq_sum / per_seq_count).mean()
    elif loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError("normalization_constant required for 'constant' mode")
        return masked.sum() / float(normalization_constant)
    else:
        raise NotImplementedError(f"loss_normalization={loss_normalization!r}")


def grpo_train_step(
    model,
    tokenizer,
    optimizer,
    gradient_accumulation_steps,
    max_grad_norm,
    reward_fn,
    repeated_prompts,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    baseline: Literal["mean", "none", "loo"] = "mean",
    advantage_eps=1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs=None,
    cliprange=None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant=None,
):
    batch = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    response_mask = batch["response_mask"].to(device)

    raw_rewards, reward_meta = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    raw_rewards = raw_rewards.to(device)
    old_log_probs_on_device = old_log_probs.to(device) if old_log_probs is not None else None

    advantages, adv_meta = compute_group_normalized_rewards(
        raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )

    batch_size = input_ids.shape[0]
    assert batch_size % gradient_accumulation_steps == 0
    micro_size = batch_size // gradient_accumulation_steps

    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    entropy_sum = torch.zeros((), dtype=torch.float32, device=device)
    token_count = torch.zeros((), dtype=torch.float32, device=device)
    aggregated_metadata: dict = {}

    for mb_idx in range(gradient_accumulation_steps):
        start, end = mb_idx * micro_size, (mb_idx + 1) * micro_size
        mb_old = old_log_probs_on_device[start:end] if old_log_probs_on_device is not None else None

        out = get_response_log_probs(model, input_ids[start:end], labels[start:end], return_token_entropy=True)
        mb_log_probs = out["log_probs"]

        with torch.no_grad():
            mb_mask_f = response_mask[start:end].to(torch.float32)
            entropy_sum += (out["token_entropy"].detach().to(torch.float32) * mb_mask_f).sum()
            token_count += mb_mask_f.sum()

        per_token_loss, loss_meta = compute_policy_gradient_loss(
            advantages[start:end], mb_log_probs, importance_reweighting_method, mb_old, cliprange, response_mask[start:end]
        )
        mb_loss = aggregate_loss_across_microbatch(
            per_token_loss, response_mask[start:end], loss_normalization, normalization_constant
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
    grad_norm_value = grad_norm.item() if isinstance(grad_norm, Tensor) else float(grad_norm)
    update_skipped = not math.isfinite(grad_norm_value)
    if not update_skipped:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    mean_entropy = (entropy_sum / token_count).item() if float(token_count) > 0 else 0.0
    metadata: dict = {
        "grad_norm": grad_norm.detach() if isinstance(grad_norm, Tensor) else float(grad_norm),
        "update_skipped": float(update_skipped),
        "token_entropy": mean_entropy,
        **reward_meta,
        **adv_meta,
    }
    for k, v in aggregated_metadata.items():
        metadata[k] = v / gradient_accumulation_steps if isinstance(v, Tensor) else v

    return total_loss, metadata
