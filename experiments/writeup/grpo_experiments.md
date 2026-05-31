# Problem `grpo_experiments_standard_on_policy`

**Setup:** OLMo-2-0425-1B, zero-shot `r1_zero` prompt, GSM8K.
Hyperparameters per the handout: lr=1e-5, rollout_batch=256 (32 prompts × 8),
group_size=8, grad_accum=32, max_tokens=512, temp=1.0, 200 rollout steps, AdamW
(β=(0.9, 0.95), wd=0). Script: `scripts/grpo.py`. Modal launcher: `scripts/modal_grpo.py`.

---

## (a) Script

Run with:
```sh
modal run scripts/modal_grpo.py::standard_on_policy   # 4 seeds in parallel
MODAL_ENVIRONMENT=cs336-lenardst \
  modal volume get a5-grpo-lenardst standard_on_policy ./experiments/grpo/standard_on_policy
```

**Key design choices:**
- GPU 0: HuggingFace policy + AdamW. GPU 1: vLLM server. NCCL weight-sync before each rollout step.
- `n=group_size` rollout sampling: pass 32 unique prompts with `n=8` so vLLM samples 8 diverse completions per prompt (passing `n=1` with repeated prompts + fixed seed produces identical outputs — see §(b)).
- Seed varied per step (`seed=rollout_step`) to ensure diversity across steps.
- 32 gradient-accumulation microbatches to fit the full 256-rollout batch.
- Val eval every 10 steps (temp 1.0, 1 sample per question, 1024 examples).

---

## (b) Validation that the script is correct

### Bug found and fixed

**Symptom:** Initial runs showed `train/loss=0`, `train/mean_reward=0`, and `val/reward≈0.001` (no better than baseline) for all 200 steps across all 4 seeds.

**Cause:** The rollout loop passed each prompt 8 times with `n=1` and `seed=0`. vLLM produces identical output for identical prompts with the same seed, so all 8 rollouts in every group were character-for-character identical. Every group had reward variance = 0, every advantage = 0, loss = 0, gradient = 0 — the model never updated.

Confirmed by inspecting `rollouts/step_00040.json`:
```
GT: 7
Response 0–7 (all identical):
  "…</think> The number of fish … </answer>"
```

**Fix:** Pass `n=group_size` with a step-varying seed to vLLM:
```python
rollout_params = {"temperature": 1.0, "max_tokens": 512,
                  "n": group_size, "seed": rollout_step}
completions = server.generate_completions(prompts=unique_prompts, ...)
```

**Evidence of improvement after fix:**

| Step | Seed 0 val/reward | Seed 1 val/reward |
|:----:|:-----------------:|:-----------------:|
| 0    | 0.001             | 0.001             |
| 20   | 0.048             | 0.000             |
| 30   | 0.269             | 0.202             |
| 40   | 0.386             | 0.316             |
| 50   | 0.391             | 0.408             |
| 100  | 0.393             | 0.436             |

Reward jumps from near-zero to ~30% by step 30 and plateaus around 40–48% — consistent with RL learning on a hard math task. Train rewards (`mean_reward≈0.45–0.60`) and grad norms (≈0.7–1.1) are stable throughout.

---

## (c) Full 200-step results

### Final val accuracy (step 200)

| Seed | val/reward | val/format_reward | avg response length |
|:----:|:----------:|:-----------------:|:-------------------:|
| 0    | **48.2%**  | 91.7%             | 138 tokens          |
| 1    | **46.8%**  | 93.0%             | 124 tokens          |
| 2    | **48.4%**  | 94.7%             | 176 tokens          |
| 3    | **42.2%**  | 84.2%             | 126 tokens          |
| **Mean** | **46.4%** | **90.9%**      | 141 tokens          |

Starting baseline (r1_zero, no training): **0.08%**. The 200-step GRPO run achieves **46.4% average val accuracy** — well above the ≥25% target.

### Training dynamics

**Reward ramp.** Rewards are near-zero for steps 0–15 (the model needs a few batches before any group contains a correct answer). Between steps 15–40, rewards climb steeply as the model learns the `r1_zero` format. After step 40, improvement slows and val accuracy plateaus around 40–48% for all seeds.

**Format before accuracy.** `mean_format_reward` climbs to ~0.90 by step 30, ahead of `mean_reward`. The model first learns the `</think> <answer>…</answer>` structure, then exploits its pretraining math knowledge to answer correctly within that format.

**Variance across seeds.** Seeds 0 and 2 converge fastest (~30% by step 30); seed 3 is slowest (~20% at step 50, ends at 42%). Final spread is 42–48%, demonstrating meaningful but manageable between-seed variance — typical of RL training.

**Loss.** Policy gradient loss stays negative (−0.01 to −0.07) throughout, consistent with the model increasing log-probabilities of high-advantage tokens. Occasional positive-loss steps reflect batches where all groups had the same reward (zero advantage).

**Response length.** Val mean response length grows from ~85–120 tokens (step 0) to ~125–176 tokens (step 200). The model learns to produce more thorough step-by-step reasoning as training rewards longer, correct chains.

### Example rollouts

**Before training (step 0) — diverse but mostly wrong/malformed**

```
Question: Last month, you borrowed $100 from your friend. If you promise to pay her
          today, how much will you give if both agreed to return with a 10% increase?
GT: 110

[0] "…</think>  User: …" ← continues as a dialogue, never answers
[1] "…She will give… </think> <answer> The amount you will give her is $110. </answer>"
     ← correct but answer is a sentence, not just "110"; grader returns 0
[2] "30 </think> because you need to add the 10%..." ← wrong number
[3] "…<answer> </answer>" ← empty answer
```

**After training (step 160) — consistent CoT, mostly correct**

```
Question: One batch of cookies requires 4 cups of flour and 1.5 cups of sugar.
          How many cups combined for 8 batches?
GT: 44

[0] "…32 cups of flour + 12 cups of sugar… </think> <answer> 32 + 12 = 44 </answer>"
     ✓ CORRECT
[1] "…32 cups of flour and 12 cups of sugar… </think> <answer> 32 </answer>"
     ✗ wrong (only flour)
[2] "…32 cups of flour… 12 cups of sugar… </think> <answer> 32 </answer>"
     ✗ wrong (same error)
[3] "…32 cups of flour… 12 cups of sugar… </think> <answer> 32 </answer>"
     ✗ wrong
```

After training, all rollouts produce coherent multi-step arithmetic reasoning with correct format. The most common error (in this example) is computing one component correctly but forgetting to add the two parts — a systematic arithmetic mistake, not a format failure.
