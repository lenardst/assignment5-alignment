"""Evaluate OLMo-2-0425-1B on GSM8K under the three baseline prompts.

Runs the `question_only`, `r1_zero`, and `r1_zero_three_shot_gsm8k` prompts on the
GSM8K test split, scores each generation with the matching reward function in
`cs336_alignment.drgrpo_grader`, buckets every generation into one of three
categories, and writes per-prompt summaries + per-example JSONL records to
`experiments/prompting_baselines/`.

Categories (from the assignment handout):
    cat1 = format_reward 1, answer_reward 1   (well-formed and correct)
    cat2 = format_reward 1, answer_reward 0   (well-formed but wrong answer)
    cat3 = format_reward 0, answer_reward 0   (failed format / unparseable)

Run on a GPU host (or via Modal) since it spins up a vLLM server.
Example:
    uv run python scripts/eval_prompting_baselines.py \
        --model allenai/OLMo-2-0425-1B \
        --output-dir experiments/prompting_baselines
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)
from cs336_alignment.vllm_utils import VLLMServer


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "cs336_alignment" / "prompts"
DEFAULT_TEST_PATH = REPO_ROOT / "data" / "gsm8k" / "test.jsonl"


@dataclass
class PromptSpec:
    name: str
    template_path: Path
    reward_fn: callable
    use_stop: bool  # True -> stop on "</answer>" and include the stop in the output


PROMPT_SPECS: list[PromptSpec] = [
    PromptSpec(
        name="question_only",
        template_path=PROMPTS_DIR / "question_only.prompt",
        reward_fn=question_only_reward_fn,
        use_stop=False,
    ),
    PromptSpec(
        name="r1_zero",
        template_path=PROMPTS_DIR / "r1_zero.prompt",
        reward_fn=r1_zero_reward_fn,
        use_stop=True,
    ),
    PromptSpec(
        name="r1_zero_three_shot",
        template_path=PROMPTS_DIR / "r1_zero_three_shot_gsm8k.prompt",
        reward_fn=r1_zero_reward_fn,
        use_stop=True,
    ),
]


def load_gsm8k(path: Path, limit: int | None) -> list[dict]:
    examples = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # GSM8K answers are "{rationale}\n#### {answer}" -- strip to just the
            # final answer for grading, as the handout specifies.
            gt = row["answer"].split("####")[-1].strip()
            examples.append({"question": row["question"], "ground_truth": gt})
            if limit is not None and len(examples) >= limit:
                break
    return examples


def render_prompts(template: str, questions: list[str]) -> list[str]:
    return [template.replace("{question}", q) for q in questions]


def _safe_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def bucket(reward: dict) -> str:
    fr = reward["format_reward"]
    ar = reward["answer_reward"]
    if fr == 1.0 and ar == 1.0:
        return "cat1_format_correct"
    if fr == 1.0 and ar == 0.0:
        return "cat2_format_wrong"
    return "cat3_unformatted"


def run_one_prompt(
    spec: PromptSpec,
    server: VLLMServer,
    examples: list[dict],
    sampling_params: dict,
    out_dir: Path,
) -> dict:
    template = spec.template_path.read_text()
    prompts = render_prompts(template, [ex["question"] for ex in examples])

    params = dict(sampling_params)
    if spec.use_stop:
        params["stop"] = ["</answer>"]
        params["include_stop_str_in_output"] = True
    else:
        params["stop"] = None

    print(f"[{spec.name}] generating {len(prompts)} completions...", flush=True)
    completions = server.generate_completions(prompts=prompts, sampling_params=params)
    assert len(completions) == len(examples), (len(completions), len(examples))

    counter: Counter[str] = Counter()
    records = []
    for ex, comp in zip(examples, completions):
        reward = spec.reward_fn(comp.text, ex["ground_truth"])
        cat = bucket(reward)
        counter[cat] += 1
        records.append(
            {
                "question": ex["question"],
                "ground_truth": ex["ground_truth"],
                "response": comp.text,
                "finish_reason": comp.finish_reason,
                "format_reward": reward["format_reward"],
                "answer_reward": reward["answer_reward"],
                "reward": reward["reward"],
                "category": cat,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / f"{spec.name}.jsonl"
    with records_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    total = len(records)
    summary = {
        "prompt": spec.name,
        "n_examples": total,
        "accuracy": counter["cat1_format_correct"] / total,
        "format_rate": (counter["cat1_format_correct"] + counter["cat2_format_wrong"]) / total,
        "counts": {
            "cat1_format_correct": counter["cat1_format_correct"],
            "cat2_format_wrong": counter["cat2_format_wrong"],
            "cat3_unformatted": counter["cat3_unformatted"],
        },
        "records_path": _safe_relpath(records_path),
    }
    print(f"[{spec.name}] {summary}", flush=True)
    return summary


def sample_examples_per_category(records: list[dict], k: int, seed: int) -> dict:
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
    return {cat: rng.sample(rs, min(k, len(rs))) for cat, rs in by_cat.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "experiments" / "prompting_baselines")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of test examples (debugging).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--examples-per-cat", type=int, default=10,
                        help="How many random per-category examples to keep alongside the summary.")
    args = parser.parse_args()

    examples = load_gsm8k(args.test_path, args.limit)
    print(f"Loaded {len(examples)} GSM8K test examples.", flush=True)

    sampling_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "n": 1,
        "seed": args.seed,
    }

    server = VLLMServer(model_id=args.model, port=args.port, gpu=args.gpu, seed=args.seed)
    server.start()
    try:
        all_summaries = []
        for spec in PROMPT_SPECS:
            summary = run_one_prompt(spec, server, examples, sampling_params, args.output_dir)
            all_summaries.append(summary)

            records = [json.loads(line) for line in (args.output_dir / f"{spec.name}.jsonl").open()]
            picks = sample_examples_per_category(records, args.examples_per_cat, args.seed)
            with (args.output_dir / f"{spec.name}.samples.json").open("w") as fh:
                json.dump(picks, fh, indent=2)
    finally:
        server.stop()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w") as fh:
        json.dump(all_summaries, fh, indent=2)
    print("\nFinal summary:")
    print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
