from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from cs336_alignment.grpo import (
    aggregate_loss_across_microbatch as _aggregate_loss,
    compute_group_normalized_rewards as _compute_group_rewards,
    compute_policy_gradient_loss as _compute_pg_loss,
    compute_rollout_rewards as _compute_rollout_rewards,
    get_response_log_probs as _get_log_probs,
    grpo_train_step as _grpo_train_step,
    tokenize_prompt_and_output as _tokenize,
)
from cs336_alignment.rlhf import (
    compute_per_instance_dpo_loss as _dpo_loss,
    get_packed_sft_dataset as _get_packed_sft,
    iterate_batches as _iterate_batches,
    masked_normalize as _masked_normalize,
    parse_gsm8k_response as _parse_gsm8k,
    parse_mmlu_response as _parse_mmlu,
    sft_microbatch_train_step as _sft_step,
)


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    return _tokenize(prompt_strs=prompt_strs, output_strs=output_strs, tokenizer=tokenizer)


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    return _get_log_probs(model=model, input_ids=input_ids, labels=labels, return_token_entropy=return_token_entropy)


def run_compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    return _compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
    )


def run_compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none", "loo"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    return _compute_group_rewards(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )


def run_compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _compute_pg_loss(
        raw_rewards_or_advantages=raw_rewards_or_advantages,
        policy_log_probs=policy_log_probs,
        importance_reweighting_method=importance_reweighting_method,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
        response_mask=response_mask,
    )


def run_aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    return _aggregate_loss(
        per_token_policy_gradient_loss=per_token_policy_gradient_loss,
        mask=mask,
        loss_normalization=loss_normalization,
        normalization_constant=normalization_constant,
    )


def run_grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
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
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    return _grpo_train_step(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        reward_fn=reward_fn,
        repeated_prompts=repeated_prompts,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
        importance_reweighting_method=importance_reweighting_method,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
        loss_normalization=loss_normalization,
        normalization_constant=normalization_constant,
    )


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    return _masked_normalize(tensor=tensor, mask=mask, dim=dim, normalize_constant=normalize_constant)


def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _sft_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        normalize_constant=normalize_constant,
    )


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    return _get_packed_sft(tokenizer=tokenizer, dataset_path=dataset_path, seq_length=seq_length, shuffle=shuffle)


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    return _iterate_batches(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    return _parse_mmlu(mmlu_example=mmlu_example, model_output=model_output)


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    return _parse_gsm8k(model_output=model_output)


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    return _dpo_loss(
        lm=lm,
        lm_ref=lm_ref,
        tokenizer=tokenizer,
        beta=beta,
        prompt=prompt,
        response_chosen=response_chosen,
        response_rejected=response_rejected,
    )
