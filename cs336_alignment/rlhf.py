from __future__ import annotations

import json
import re
import random
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from cs336_alignment.grpo import tokenize_prompt_and_output, get_response_log_probs


def masked_normalize(tensor: Tensor, mask: Tensor, dim=None, normalize_constant=1.0) -> Tensor:
    return (tensor * mask).sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant=1.0):
    if normalize_constant is None:
        normalize_constant = 1.0
    loss = -(policy_log_probs * response_mask).sum(dim=-1).mean() / (normalize_constant * gradient_accumulation_steps)
    loss.backward()
    return loss, {}


_ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    "\n\n### Instruction:\n{prompt}\n\n### Response:\n{response}"
)


class _PackedSFTDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def get_packed_sft_dataset(tokenizer, dataset_path, seq_length, shuffle):
    with open(dataset_path) as f:
        examples = [json.loads(line) for line in f]

    if shuffle:
        random.shuffle(examples)

    all_tokens = []
    for example in examples:
        text = _ALPACA_TEMPLATE.format(prompt=example["prompt"], response=example["response"])
        tokens = [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
        all_tokens.extend(tokens)

    packed = []
    for i in range(len(all_tokens) // seq_length):
        if i * seq_length + seq_length < len(all_tokens):
            input_ids = torch.tensor(all_tokens[i * seq_length : i * seq_length + seq_length], dtype=torch.long)
            labels = torch.tensor(all_tokens[i * seq_length + 1 : i * seq_length + seq_length + 1], dtype=torch.long)
            packed.append({"input_ids": input_ids, "labels": labels})

    return _PackedSFTDataset(packed)


def iterate_batches(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    match = re.search(r"\b([A-D])\b", model_output)
    return match.group(1) if match else None


def parse_gsm8k_response(model_output: str) -> str | None:
    numbers = re.findall(r"\d+", model_output)
    return numbers[-1] if numbers else None


def compute_per_instance_dpo_loss(lm, lm_ref, tokenizer, beta, prompt, response_chosen, response_rejected) -> Tensor:
    def sequence_log_prob(model, response):
        batch = tokenize_prompt_and_output([prompt], [response], tokenizer)
        log_probs = get_response_log_probs(model, batch["input_ids"], batch["labels"])["log_probs"]
        return (log_probs * batch["response_mask"]).sum()

    chosen_log_prob = sequence_log_prob(lm, response_chosen)
    rejected_log_prob = sequence_log_prob(lm, response_rejected)

    with torch.no_grad():
        ref_chosen_log_prob = sequence_log_prob(lm_ref, response_chosen)
        ref_rejected_log_prob = sequence_log_prob(lm_ref, response_rejected)

    chosen_reward = beta * (chosen_log_prob - ref_chosen_log_prob)
    rejected_reward = beta * (rejected_log_prob - ref_rejected_log_prob)
    return -F.logsigmoid(chosen_reward - rejected_reward)
