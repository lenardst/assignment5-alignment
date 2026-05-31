# Problem `prompting_baselines` — writeup

Model: `allenai/OLMo-2-0425-1B` on the full GSM8K test split (1319 problems).
Sampling: temperature 1.0, top-p 1.0, max 512 new tokens, seed 0.
For `r1_zero` and `r1_zero_three_shot` we stop on `</answer>` and include the
stop string in the output; `question_only` runs without a stop string.

Reward functions:
- `cs336_alignment.drgrpo_grader.question_only_reward_fn` for `question_only`.
- `cs336_alignment.drgrpo_grader.r1_zero_reward_fn` for both r1_zero variants.

Categories:
- **cat1**: format_reward 1, answer_reward 1 (well-formed + correct)
- **cat2**: format_reward 1, answer_reward 0 (well-formed but wrong answer)
- **cat3**: format_reward 0, answer_reward 0 (unparseable)

## (a) Metrics

| Prompt                  | n    | cat1 (acc) | cat2 (fmt, wrong) | cat3 (no fmt) | Accuracy |
| ----------------------- | ---- | ---------- | ----------------- | ------------- | -------- |
| `question_only`         | 1319 | 1          | 305               | 1013          | 0.08%    |
| `r1_zero`               | 1319 | 1          | 817               | 501           | 0.08%    |
| `r1_zero_three_shot`    | 1319 | 246        | 1007              | 66            | 18.65%   |

**Manual audit of cat2 (n=12–15 per prompt).**

- `question_only`: 0 / 12 correct-but-misparsed — garbled LaTeX / wrong arithmetic
- `r1_zero`: 0 / 15 correct-but-misparsed — all compute wrong answers
- `r1_zero_three_shot`: 0 / 12 correct-but-misparsed — wrong reasoning (e.g., forgot ÷12 for dozens)

**Manual audit of cat3 (n=8–10 per prompt).**

- `question_only`: 0 / 8 correct-but-misparsed — total hallucinations, no \boxed{} produced
- `r1_zero`: 0 / 12 correct-but-misparsed — answer outside `<answer>` tags AND arithmetically wrong
- `r1_zero_three_shot`: 2 / 10 correct-but-misparsed — model gives right answer inside `</think>`
  but omits the `<answer>...</answer>` wrapper

## (b) Behavior characterization

**`question_only`**: Base model treats the prompt as a cue to generate additional
math word problems or educational content rather than to answer the given problem.
The \boxed{} convention is essentially absent (76.8% cat3). The lone cat1 success
produced well-formatted LaTeX with \boxed{350}.

**`r1_zero`**: Pre-seeding `<think>` and the assistant role lifts format rate to 62%.
The model produces plausible-looking reasoning chains but is almost always
arithmetically wrong. Cat3 failures: answer written outside `<answer>` tag,
appended "User:" turns, or `</think>` never closed at 512-token limit.

**`r1_zero_three_shot`**: Three worked examples sharply constrain output to the
expected trajectory style: short step-by-step arithmetic + single number in
`<answer>`. Format rate = 95%, accuracy = 18.65%. Cat3 (5%) is mostly outputs
that close `</think>` without opening `<answer>`, or truncated by the 512-token
limit. ~20% of those cat3 cases are actually correct but misparsed.
